# FraudScan — Détection de fraude assurance auto par RAG & LLM

> Projet académique — *IA & Assurance*, ISFA 2025–2026  
> Encadrants : A. Couloumy / Dylogy  
> Auteurs : **Apélété Primos ATISSO · Massile BADARO · Koami Bernardin ZANTE**

---

## Problématique métier

La fraude à l'assurance auto représente plusieurs milliards d'euros de pertes annuelles pour le secteur. Ce projet implémente un pipeline de détection automatisé combinant :

- **RAG sémantique** : recherche des sinistres historiques les plus similaires via embeddings vectoriels
- **Score réseau** : règles métier sur garages, experts, AGIRA et blacklist
- **Agent LLM** : analyse experte générée par LLaMA 3.3 70B (Groq)
- **Juge LLM** : validation automatique de la qualité des analyses avec boucle de correction

---

## Architecture du pipeline

```
Sinistre entrant
      │
      ▼
[Module 1] Génération données synthétiques (10 000 sinistres, 16 règles R1–R16)
      │
      ▼
[Module 2] Analyse exploratoire (EDA) — distributions, corrélations, stats réseau
      │
      ▼
[Module 3] RAG — claim_to_document() → embeddings all-MiniLM-L6-v2 → ChromaDB
      │
      ▼
[Module 4] Score hybride : 0.70 × score_RAG + 0.30 × score_réseau
      │                     ├─ FAIBLE  < 0,40
      │                     ├─ MODÉRÉ  0,40–0,70
      │                     └─ ÉLEVÉ   > 0,70
      ▼
[Module 5] Agent Expert LLM → JSON structuré (score, motifs, recommandation)
      │
      ▼
[Module 6] Juge LLM → évaluation 5 critères (1–5) → VALIDÉ / À REVOIR
      │         └─ boucle de correction (max 2 retries si rejeté)
      ▼
[Module 7–11] Évaluation : AUC-ROC, F1, Precision, Recall + sensibilité k
      │
      ▼
[Module 12] API FastAPI — /score · /analyze · /analyze_with_judge
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` (384 dims, CPU local) |
| Vector store | `chromadb` PersistentClient · HNSW · distance cosine · k=5 |
| LLM | `llama-3.3-70b-versatile` via Groq (SDK OpenAI, `base_url` redirigé) |
| Data | `pandas`, `numpy`, `faker` (locale fr_FR) |
| Évaluation | `scikit-learn` (AUC-ROC, F1, Precision, Recall) + LLM-as-a-Judge |
| API | `FastAPI` + `uvicorn` |
| Python | 3.11+ |

---

## Structure du dépôt

```
.
├── Notebook_Final_v2.ipynb      # Notebook principal (13 modules)
├── app.py                       # API REST FastAPI (3 endpoints)
├── pipeline.py                  # Module pipeline : RAG, scoring, agents LLM
├── fraudscan.html               # Interface démo interactive (Module 13)
├── requirements.txt
├── README.md
├── skills/
│   ├── expert_fraude_production.md   # Prompt système Agent Expert
│   └── juge_production.md            # Prompt système Juge LLM
├── data/
│   ├── claims_10k.csv           # Dataset principal (10 000 sinistres)
│   ├── ref_assures.csv          # Table assurés
│   ├── ref_contrats.csv         # Table contrats
│   ├── ref_garages.csv          # Table garages (50 garages)
│   ├── ref_experts.csv          # Table experts (20 experts)
│   ├── ref_agira_externe.csv    # Historique AGIRA
│   ├── ref_blacklist.csv        # Liste noire
│   └── ref_contrat_insure.csv   # Association contrat–assuré
├── artifacts/
│   ├── eda_production.png       # Visualisation EDA
│   ├── performances_prod.png    # Courbes ROC / Precision-Recall
│   ├── confusion_prod.png       # Matrice de confusion
│   ├── calibrage_prod.png       # Courbe de calibration
│   ├── saisonnalite_prod.png    # Analyse temporelle
│   ├── metrics_prod.json        # Métriques finales (AUC, F1, etc.)
│   └── predictions_test_prod.csv
└── network/
    ├── garage_stats.csv         # Statistiques de suspicion garages
    ├── expert_stats.csv         # Statistiques de suspicion experts
    └── fraud_network.gexf       # Graphe réseau (Gephi)

# Non versionnés (générés automatiquement) :
# data/claims_10k.json    ← Module 1
# data/documents_10k.jsonl ← Module 3
# vector_store/            ← Module 3 (ChromaDB, ~60 MB)
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://forge.univ-lyon1.fr/p2408012/fraude_auto_rag.git
cd fraude_auto_rag
```

### 2. Créer l'environnement Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé Groq

Créer un fichier `.env` à la racine (jamais commité) :

```bash
echo "GROQ_API_KEY=gsk_..." > .env
```

Ou exporter directement :

```bash
export GROQ_API_KEY="gsk_..."
```

Obtenir une clé gratuite sur [console.groq.com](https://console.groq.com).

---

## Exécution du notebook

```bash
jupyter notebook Notebook_Final_v2.ipynb
```

**Ordre d'exécution recommandé :**

| Module | Contenu | Fichiers générés |
|---|---|---|
| 1 | Génération des données synthétiques | `data/claims_10k.csv`, `data/claims_10k.json`, `data/ref_*.csv` |
| 2 | Analyse exploratoire (EDA) | `artifacts/eda_production.png`, `artifacts/saisonnalite_prod.png` |
| 3 | Construction de l'index RAG | `data/documents_10k.jsonl`, `vector_store/` |
| 4 | Scoring hybride RAG + réseau | — |
| 5 | Agent Expert LLM | — |
| 6 | Juge LLM + boucle de correction | — |
| 7–11 | Évaluation & métriques | `artifacts/performances_prod.png`, `artifacts/metrics_prod.json`, `artifacts/predictions_test_prod.csv` |
| 12 | API FastAPI | `app.py` |
| 13 | Interface HTML démo interactive | `fraudscan.html` |

> **Note :** si `data/claims_10k.csv` et `data/ref_*.csv` sont déjà présents (cas GitLab),  
> vous pouvez démarrer directement au **Module 3** pour reconstruire l'index vectoriel.

---

## Lancer l'API

```bash
export GROQ_API_KEY="gsk_..."
.venv/bin/python -m uvicorn app:app --port 8000
```

> **Note :** ne pas utiliser `--reload` — il surveille le dossier `.venv` et provoque une boucle de redémarrage.

Documentation interactive Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Endpoint | Description | Latence |
|---|---|---|
| `GET /health` | Statut du service | < 50 ms |
| `POST /score` | Score hybride RAG + réseau, sans LLM | ~200 ms |
| `POST /analyze` | Score + analyse Agent Expert LLM (Groq) | ~2–4 s |
| `POST /analyze_with_judge` | Score + Expert + validation Juge | ~5–8 s |
| `POST /proxy/anthropic` | Proxy CORS vers Anthropic API (utilisé par fraudscan.html) | ~2–4 s |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_000001",
    "insured_id": "INS_012345",
    "incident_type": "Collision simple",
    "incident_severity": "important",
    "incident_date": "2024-08-15",
    "incident_hour": 2,
    "incident_day_of_week": 6,
    "declaration_delay_days": 15,
    "region": "Île-de-France",
    "vehicle_brand": "BMW",
    "vehicle_age": 2,
    "total_claim_amount": 19800.0,
    "driver_age": 24,
    "months_as_customer": 3,
    "annual_income": 14000.0,
    "profession": "sans emploi",
    "witnesses": 0,
    "prior_claims": 3,
    "bodily_injuries": 0,
    "police_report": "Non",
    "garage_id": "GAR_0001",
    "expert_id": "AUCUN",
    "claims_last_36m_agira": 4,
    "incident_description": "Collision sur autoroute de nuit."
  }'
```

---

## Interface démo — FraudScan HTML (Module 13)

`fraudscan.html` est une interface web interactive affichant le pipeline complet sans quitter le navigateur.

### Utilisation dans le notebook (recommandé)

```python
# Cellule 94 du notebook — affiche l'interface inline dans Jupyter
display_fraudscan()
```

### Utilisation standalone

Servir le fichier via un serveur HTTP local (obligatoire pour éviter les restrictions CORS du navigateur) :

```bash
# Terminal 1 — serveur HTTP
python3 -m http.server 3000

# Terminal 2 — API FastAPI
export GROQ_API_KEY="gsk_..."
.venv/bin/python -m uvicorn app:app --port 8000
```

Ouvrir dans le navigateur : `http://localhost:3000/fraudscan.html`

### Prérequis

1. Les deux serveurs ci-dessus doivent tourner.
2. Dans la console DevTools du navigateur (**F12**) :
   ```js
   window.ANTHROPIC_API_KEY = "sk-ant-api03-..."
   ```
   *(à refaire à chaque rechargement de page)*

### Fonctionnement

| Étape | Appel | Résultat |
|---|---|---|
| 1 | `POST http://localhost:8000/score` | Jauge score final, voisins RAG, scores RAG/réseau |
| 2 | `POST http://localhost:8000/proxy/anthropic` → `api.anthropic.com` | Analyse Expert Claude Haiku : motifs, signaux, recommandation |

> **Note :** l'appel Anthropic transite par le proxy FastAPI pour éviter les blocages CORS du navigateur.

### Prévisualisation en ligne (GitLab Pages)

`https://p2408012.pages.univ-lyon1.fr/fraude_auto_rag/`

> **Note :** en ligne, le scoring RAG et l'analyse LLM nécessitent tous les deux l'API locale (`localhost:8000`).

---

## Résultats clés

| Métrique | Valeur (k=5) |
|---|---|
| AUC-ROC | 0,7748 |
| F1-score (seuil optimal) | 0,6048 |
| Taux de fraude dans les données | ~30,4 % (3 043 / 10 000) |
| Score moyen — sinistres frauduleux | ~5 399 € de montant réclamé |
| Évaluation Juge LLM | 5,0 / 5 |

**Sensibilité à k (nombre de voisins RAG) :**

| k | AUC-ROC |
|---|---|
| k=1 | 0,8796 |
| k=5 | 0,7748 |

---

## Reproductibilité

- Toutes les données sont générées avec `random_state=42` (reproductibles).
- Les embeddings sont recalculés localement (aucune dépendance externe, CPU uniquement).
- Le seul élément non déterministe est la sortie du LLM Groq (`temperature=0.2`).
- Le `COST_TRACKER` enregistre chaque appel LLM : appeler `afficher_bilan()` pour le résumé des tokens et coûts.

---

## Limites identifiées

- **Données synthétiques** : le taux de fraude (~30 %) est volontairement élevé pour les besoins pédagogiques ; en réalité il est de 6–8 %.
- **Scalabilité** : ChromaDB en mode local n'est pas adapté à des millions de sinistres ; une migration vers un serveur ChromaDB ou Qdrant serait nécessaire.
- **Coût LLM** : Groq est gratuit avec des limites de débit ; en production, prévoir un budget API ou un modèle local (Ollama).
- **Latence** : l'endpoint `/analyze_with_judge` (~5–8 s) n'est pas adapté à un usage temps réel ; à réserver aux cas à risque élevé.
- **Sécurité** : la clé API Groq ne doit jamais être versionnée ; utiliser des variables d'environnement ou un gestionnaire de secrets.
