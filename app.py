# app.py — API REST FraudScan Production (FastAPI)
# Lancement : uvicorn app:app --reload --port 8000
# Documentation Swagger auto : http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn, time

from pipeline import hybrid_score, run_fraud_agent, run_agent_with_retry, LLM_MODEL, collection

app = FastAPI(
    title="FraudScan API — Production",
    description=(
        "API de détection de fraude assurance auto.\n"
        "Pipeline hybride : RAG sémantique + score réseau (garages/experts/AGIRA).\n"
        "Modèle : Claude Haiku 4.5 via Anthropic | Embeddings : all-MiniLM-L6-v2"
    ),
    version="2.0.0",
)

# CORS : autoriser les appels depuis une interface web
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class ClaimRequest(BaseModel):
    claim_id:               str   = Field(..., example="CLM_000001")
    insured_id:             str   = Field(..., example="INS_012345")
    incident_type:          str   = Field(..., example="Collision simple")
    incident_severity:      str   = Field(..., example="important")
    incident_date:          str   = Field(..., example="2024-08-15")
    incident_hour:          int   = Field(12, ge=0, le=23)
    incident_day_of_week:   int   = Field(0,  ge=0, le=6)
    declaration_delay_days: int   = Field(1,  ge=0)
    region:                 str   = Field(..., example="Île-de-France")
    vehicle_brand:          str   = Field(..., example="BMW")
    vehicle_age:            int   = Field(2,  ge=0, le=30)
    total_claim_amount:     float = Field(..., gt=0, example=19800.0)
    driver_age:             int   = Field(30, ge=18, le=90)
    months_as_customer:     int   = Field(24, ge=1)
    annual_income:          float = Field(..., gt=0, example=28000.0)
    profession:             str   = Field(..., example="commercial")
    witnesses:              int   = Field(0,  ge=0)
    prior_claims:           int   = Field(0,  ge=0)
    bodily_injuries:        int   = Field(0,  ge=0)
    police_report:          str   = Field("Oui", pattern="^(Oui|Non)$")
    garage_id:              str   = Field("GAR_0001")
    expert_id:              str   = Field("AUCUN")
    claims_last_36m_agira:  int   = Field(0, ge=0)
    incident_description:   str   = Field("", example="Collision sur parking.")


class ScoreResponse(BaseModel):
    claim_id:       str
    score_rag:      float
    score_network:  float
    score_final:    float
    risk_level:     str
    neighbors:      List[dict]
    latency_ms:     int


@app.get("/health")
def health():
    """Vérification que le service est opérationnel."""
    return {
        "status":       "ok",
        "model":        LLM_MODEL,
        "vector_store": collection.count(),
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.post("/score", response_model=ScoreResponse)
def score_claim(req: ClaimRequest):
    """Scoring hybride RAG + réseau sans appel LLM (~200ms)."""
    t0    = time.time()
    claim = req.dict()
    try:
        scored = hybrid_score(claim, k=5)
        return ScoreResponse(
            claim_id=      req.claim_id,
            score_rag=     scored["score_rag"],
            score_network= scored["score_network"],
            score_final=   scored["score_final"],
            risk_level=    scored["risk_level"],
            neighbors=     [
                {
                    "claim_id":    n["metadata"]["claim_id"],
                    "fraud_label": n["metadata"]["fraud_label"],
                    "similarity":  round(n["similarity"], 4),
                    "garage_id":   n["metadata"].get("garage_id","?"),
                }
                for n in scored["neighbors"]
            ],
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze_claim(req: ClaimRequest):
    """Analyse complète avec Agent Expert LLM (~2-4s)."""
    t0    = time.time()
    claim = req.dict()
    try:
        scored = hybrid_score(claim, k=5)
        agent  = run_fraud_agent(claim, scored)
        return {
            "claim_id":     req.claim_id,
            "score_rag":    scored["score_rag"],
            "score_network":scored["score_network"],
            "score_final":  scored["score_final"],
            "risk_level":   scored["risk_level"],
            "analysis":     agent["content"],
            "latency_ms":   int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_with_judge")
def analyze_with_judge(req: ClaimRequest):
    """Analyse complète Expert + validation Juge (~5-8s)."""
    t0    = time.time()
    claim = req.dict()
    try:
        scored = hybrid_score(claim, k=5)
        result = run_agent_with_retry(claim, scored, max_retries=2)
        return {
            "claim_id":     req.claim_id,
            "score_final":  scored["score_final"],
            "risk_level":   scored["risk_level"],
            "analysis":     result["agent_output"],
            "judge_verdict":result["verdict"],
            "attempts":     result["attempts"],
            "latency_ms":   int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)