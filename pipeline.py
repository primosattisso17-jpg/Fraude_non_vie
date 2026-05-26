# pipeline.py — FraudScan : chargement du pipeline au démarrage de l'API
# Exporte : hybrid_score, run_fraud_agent, run_agent_with_retry, LLM_MODEL, collection

import os, json, re, time
from pathlib import Path
from getpass import getpass

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# ─── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
VECTOR_DIR  = BASE_DIR / "vector_store"
SKILLS_DIR  = BASE_DIR / "skills"
NETWORK_DIR = BASE_DIR / "network"

# ─── Client Claude ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = getpass("Clé Anthropic (sk-ant-...) : ")

client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
LLM_MODEL = "claude-haiku-4-5-20251001"

# ─── Modèle d'embedding local ────────────────────────────────────────────────
print("Chargement du modèle d'embedding (all-MiniLM-L6-v2)...")
ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Vector store ChromaDB ───────────────────────────────────────────────────
chroma_path   = VECTOR_DIR / "chroma_db_prod"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
collection    = chroma_client.get_collection("auto_fraud_prod")
print(f"Vector store chargé : {collection.count():,} sinistres indexés")

# ─── Garages et experts suspects (depuis network/stats) ──────────────────────
_garage_stats = pd.read_csv(NETWORK_DIR / "garage_stats.csv", index_col=0)
_expert_stats = pd.read_csv(NETWORK_DIR / "expert_stats.csv", index_col=0)

suspicious_garages = set(
    _garage_stats[
        (_garage_stats["fraud_rate"] > 0.35) & (_garage_stats["n_claims"] > 10)
    ].index
)
suspicious_experts = set(
    _expert_stats[
        (_expert_stats["fraud_rate"] > 0.40) & (_expert_stats["n_claims"] > 5)
    ].index
)
print(f"Garages suspects : {len(suspicious_garages)} | Experts suspects : {len(suspicious_experts)}")

# ─── Données de référence pour enrichissement ────────────────────────────────
_ref_garages = pd.read_csv(DATA_DIR / "ref_garages.csv")
_ref_experts = pd.read_csv(DATA_DIR / "ref_experts.csv")
_ref_blacklist = pd.read_csv(DATA_DIR / "ref_blacklist.csv")
_ref_agira_ext = pd.read_csv(DATA_DIR / "ref_agira_externe.csv")

_garage_sra   = dict(zip(_ref_garages["garage_id"],  _ref_garages["sra_certified"]))
_expert_cert  = dict(zip(_ref_experts["expert_id"],   _ref_experts["tribunal_list"]))
_blacklist_set = set(_ref_blacklist["insured_id"])
_agira_counts  = _ref_agira_ext.groupby("insured_id").size().to_dict()

# ─── Skills LLM ──────────────────────────────────────────────────────────────
EXPERT_FRAUD_SKILL = (SKILLS_DIR / "expert_fraude_production.md").read_text()
JUDGE_SKILL        = (SKILLS_DIR / "juge_production.md").read_text()

# ─── Tracker de coûts LLM ────────────────────────────────────────────────────
COST_TRACKER = {"calls": [], "prompt_tokens": 0, "completion_tokens": 0}


def call_llm(messages, model=LLM_MODEL, temperature=0.2, max_tokens=1200):
    t0 = time.time()

    system_prompt = None
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            user_messages.append(msg)

    kwargs = dict(model=model, max_tokens=max_tokens, messages=user_messages)
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature > 0:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)
    latency  = time.time() - t0
    usage    = response.usage

    COST_TRACKER["calls"].append({"latency_s": round(latency, 2),
                                   "prompt_tokens": usage.input_tokens})
    COST_TRACKER["prompt_tokens"]     += usage.input_tokens
    COST_TRACKER["completion_tokens"] += usage.output_tokens
    return {"content": response.content[0].text, "latency_s": latency}


def afficher_bilan():
    input_tok  = COST_TRACKER["prompt_tokens"]
    output_tok = COST_TRACKER["completion_tokens"]
    cost = (input_tok * 0.80 + output_tok * 4.00) / 1_000_000
    print(f"\n{'='*45}")
    print(f"Appels LLM    : {len(COST_TRACKER['calls'])}")
    print(f"Tokens input  : {input_tok:,}")
    print(f"Tokens output : {output_tok:,}")
    print(f"Coût estimé   : ${cost:.5f} (Claude Haiku 4.5)")
    print(f"{'='*45}")


# ─── Enrichissement depuis les tables de référence ───────────────────────────
def enrich_claim(claim: dict) -> dict:
    """Ajoute les champs des tables de référence manquants dans la requête API."""
    c          = dict(claim)
    insured_id = c.get("insured_id", "")
    garage_id  = c.get("garage_id",  "")
    expert_id  = c.get("expert_id",  "AUCUN")
    c.setdefault("blacklisted",          insured_id in _blacklist_set)
    c.setdefault("agira_ext_count",      _agira_counts.get(insured_id, 0))
    c.setdefault("garage_sra_certified", bool(_garage_sra.get(garage_id, True)))
    c.setdefault("expert_certified",
                 bool(_expert_cert.get(expert_id, True)) if expert_id != "AUCUN" else True)
    c.setdefault("is_covered", True)
    return c


# ─── Requête vectorielle ──────────────────────────────────────────────────────
def incoming_claim_to_query(claim: dict) -> str:
    network_signals = []
    if claim.get("garage_id") in suspicious_garages:
        network_signals.append(f"Garage {claim['garage_id']} associé à des fraudes")
    if claim.get("expert_id", "AUCUN") != "AUCUN" and claim["expert_id"] in suspicious_experts:
        network_signals.append(f"Expert {claim['expert_id']} associé à des fraudes")
    if claim.get("claims_last_36m_agira", 0) >= 3:
        network_signals.append(f"{claim['claims_last_36m_agira']} sinistres en 36 mois (AGIRA)")

    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    dow  = days[int(claim.get("incident_day_of_week", 0))]
    return (
        f"Sinistre {claim.get('claim_id','NEW')} - Assure {claim.get('insured_id','?')}\n"
        f"Type : {claim.get('incident_type','?')} | Gravite : {claim.get('incident_severity','?')}\n"
        f"Region : {claim.get('region','?')} | Date : {claim.get('incident_date','?')}\n"
        f"Sinistre un {dow} à {claim.get('incident_hour',0)}h. "
        f"Declaration {claim.get('declaration_delay_days',0)} jour(s) après.\n"
        f"Vehicule : {claim.get('vehicle_brand','?')}, {claim.get('vehicle_age',0)} an(s)\n"
        f"Montant reclame : {claim.get('total_claim_amount',0):.2f} EUR\n"
        f"Conducteur : {claim.get('driver_age',0)} ans | Profession : {claim.get('profession','?')}\n"
        f"Revenu annuel : {claim.get('annual_income',0):.2f} EUR\n"
        f"Anciennete contrat : {claim.get('months_as_customer',0)} mois\n"
        f"Temoins : {claim.get('witnesses',0)} | Rapport police : {claim.get('police_report','?')}\n"
        f"Sinistres antérieurs : {claim.get('prior_claims',0)}\n"
        f"Sinistres AGIRA (36 mois) : {claim.get('claims_last_36m_agira',0)}\n"
        f"Garage : {claim.get('garage_id','?')} | Expert : {claim.get('expert_id','AUCUN')}\n"
        f"Description : {claim.get('incident_description','')}\n"
        f"Signaux reseau : {'; '.join(network_signals) if network_signals else 'Aucun'}"
    )


def search_similar_frauds(query_text: str, k: int = 5) -> list:
    query_embedding = ST_MODEL.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "document":   doc,
            "metadata":   meta,
            "distance":   float(dist),
            "similarity": float(1 - dist),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def fraud_score_weighted(neighbors: list) -> float:
    total_sim = sum(n["similarity"] for n in neighbors)
    if total_sim == 0:
        return 0.0
    return sum(n["similarity"] * n["metadata"]["fraud_label"] for n in neighbors) / total_sim


def network_score(claim: dict) -> float:
    bonus = 0.0
    if claim.get("garage_id") in suspicious_garages:
        bonus += 0.15
    if claim.get("expert_id", "AUCUN") != "AUCUN" and claim.get("expert_id") in suspicious_experts:
        bonus += 0.10
    agira_count = claim.get("claims_last_36m_agira", 0)
    if agira_count >= 3:
        bonus += min(0.05 * (agira_count - 2), 0.20)
    if not claim.get("is_covered", True):
        bonus += 0.15
    agira_ext = claim.get("agira_ext_count", 0)
    if agira_ext >= 3:
        bonus += min(0.05 * (agira_ext - 2), 0.15)
    if claim.get("blacklisted", False):
        bonus += 0.30
    if not claim.get("garage_sra_certified", True):
        bonus += 0.05
    if claim.get("expert_id", "AUCUN") != "AUCUN" and not claim.get("expert_certified", True):
        bonus += 0.05
    return min(bonus, 0.50)


def hybrid_score(claim: dict, k: int = 5) -> dict:
    claim      = enrich_claim(claim)
    query_text = incoming_claim_to_query(claim)
    neighbors  = search_similar_frauds(query_text, k=k)
    rag        = fraud_score_weighted(neighbors)
    net        = network_score(claim)
    final      = min(0.70 * rag + 0.30 * net + (0.10 if net > 0 else 0), 0.99)
    return {
        "query_text":    query_text,
        "neighbors":     neighbors,
        "score_rag":     round(rag, 4),
        "score_network": round(net, 4),
        "score_final":   round(final, 4),
        "risk_level":    ("ÉLEVÉ" if final >= 0.7 else "MODÉRÉ" if final >= 0.4 else "FAIBLE"),
    }


# ─── Agents LLM ──────────────────────────────────────────────────────────────
def build_agent_payload(claim: dict, scored: dict) -> dict:
    return {
        "sinistre": {k: claim.get(k) for k in [
            "claim_id", "insured_id", "incident_type", "incident_severity",
            "incident_date", "incident_hour", "declaration_delay_days",
            "region", "vehicle_brand", "vehicle_age", "total_claim_amount",
            "driver_age", "months_as_customer", "annual_income", "profession",
            "witnesses", "prior_claims", "police_report", "bodily_injuries",
            "garage_id", "expert_id", "claims_last_36m_agira", "incident_description",
        ]},
        "score_rag":     scored["score_rag"],
        "score_network": scored["score_network"],
        "score_final":   scored["score_final"],
        "niveau":        scored["risk_level"],
        "voisins_rag": [
            {
                "claim_id":    n["metadata"]["claim_id"],
                "fraud_label": n["metadata"]["fraud_label"],
                "montant":     n["metadata"]["claim_amount"],
                "garage_id":   n["metadata"]["garage_id"],
                "similarity":  round(n["similarity"], 3),
            }
            for n in scored["neighbors"][:5]
        ],
        "signaux_reseau": {
            "garage_suspect":   claim.get("garage_id") in suspicious_garages,
            "expert_suspect":   (claim.get("expert_id", "AUCUN") != "AUCUN"
                                 and claim.get("expert_id") in suspicious_experts),
            "agira_recurrence": claim.get("claims_last_36m_agira", 0),
        },
    }


def run_fraud_agent(claim: dict, scored: dict) -> dict:
    payload  = build_agent_payload(claim, scored)
    messages = [
        {"role": "system", "content": EXPERT_FRAUD_SKILL},
        {"role": "user",   "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    return call_llm(messages, temperature=0.1, max_tokens=1000)


def judge_agent_output(claim: dict, scored: dict, agent_output: str) -> dict:
    payload  = {
        "sinistre_resume": {
            "claim_id":    claim.get("claim_id"),
            "score_final": scored["score_final"],
            "niveau":      scored["risk_level"],
        },
        "analyse_expert": agent_output,
    }
    messages = [
        {"role": "system", "content": JUDGE_SKILL},
        {"role": "user",   "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    return call_llm(messages, temperature=0.0, max_tokens=600)


def run_agent_with_retry(claim: dict, scored: dict, max_retries: int = 2) -> dict:
    previous_feedback = None
    agent_resp = judge_resp = None
    verdict = "VALIDÉ"
    for attempt in range(1, max_retries + 1):
        messages = [{"role": "system", "content": EXPERT_FRAUD_SKILL}]
        content  = json.dumps(build_agent_payload(claim, scored), ensure_ascii=False, indent=2)
        if previous_feedback:
            content += (f"\n\n[FEEDBACK JUGE — tentative {attempt-1}]\n"
                        f"{previous_feedback}\nAméliore en tenant compte de ce feedback.")
        messages.append({"role": "user", "content": content})

        agent_resp = call_llm(messages, temperature=0.1, max_tokens=1000)
        judge_resp = judge_agent_output(claim, scored, agent_resp["content"])

        try:
            clean   = re.sub(r"```json|```", "", judge_resp["content"] or "").strip()
            jdata   = json.loads(clean)
            verdict = jdata.get("verdict", "VALIDÉ")
        except Exception:
            verdict = "VALIDÉ" if "VALIDÉ" in (judge_resp["content"] or "") else "À REVOIR"

        if verdict == "VALIDÉ":
            break
        previous_feedback = judge_resp["content"]

    return {
        "agent_output": agent_resp["content"],
        "judge_output": judge_resp["content"],
        "verdict":      verdict,
        "attempts":     attempt,
    }
