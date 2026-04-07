# Fraud Detection — Graph Intelligence & Behavioral Risk Scoring

A hackathon-ready, in-memory fraud detection system that combines **graph analytics**,
**behavioural feature engineering**, and **machine learning** to identify suspicious
accounts and money-laundering rings in financial transaction networks.

---

## Architecture at a Glance

```
Synthetic Generator / Raw Transactions
        ↓
Ingestion Layer        (batch_loader / stream_simulator)
        ↓
Graph Builder          (NetworkX DiGraph: nodes=accounts, edges=txns)
        ↓
Feature Engineering
  ├── Structural  (degree, centrality, PageRank, clustering)
  ├── Temporal    (velocity, retention time, burst score)
  └── Behavioural (fan-out ratio, flow depth, cross-channel jumps)
        ↓
Risk Scoring Engine
  ├── Weighted scorer   (YAML-configured weights)
  ├── Isolation Forest  (anomaly boost)
  └── Normalizer        → score ∈ [0, 1], tier label
        ↓
Network Intelligence
  ├── Community detection  (greedy modularity)
  ├── Subgraph extraction  (suspicious communities)
  └── Cluster scoring      (mule ring detection)
        ↓
Explainability Layer   (human-readable per-account reports)
        ↓
FastAPI REST API       (ingest, score, cluster endpoints)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (first time only)

```bash
python data/generate_data.py
```

### 3. Run the full pipeline + start the API

```bash
python run.py
```

API docs available at: **http://localhost:8000/docs**

### 4. Pipeline only (no server)

```bash
python run.py --no-serve
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transactions` | Ingest a single transaction |
| `POST` | `/transactions/batch` | Ingest a batch of transactions |
| `GET`  | `/accounts/{id}/risk` | Risk score + explanation for one account |
| `GET`  | `/clusters/suspicious?min_risk=0.7` | List flagged suspicious clusters |
| `GET`  | `/graph/stats` | Graph-level statistics |
| `GET`  | `/health` | Liveness probe |

### Example: ingest a transaction

```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "txn_id": "t001",
    "from_account": "ACC12345",
    "to_account": "ACC67890",
    "amount": 4990.00,
    "channel": "P2P",
    "timestamp": 1700000100
  }'
```

### Example: query risk score

```bash
curl http://localhost:8000/accounts/ACC12345/risk
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/scoring_weights.yaml` | Feature weights (must sum to 1.0) |
| `config/thresholds.yaml` | Risk tier thresholds, cluster flags, anomaly boost |

Edit these files to tune sensitivity without code changes.

---

## Risk Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0.00 – 0.30 | LOW | Normal — no action |
| 0.30 – 0.60 | MEDIUM | Elevated — monitor |
| 0.60 – 0.85 | HIGH | Suspicious — manual review |
| 0.85 – 1.00 | CRITICAL | Likely fraud — block / escalate |

---

## Fraud Patterns Simulated

| Pattern | Description |
|---------|-------------|
| **Mule ring** | Circular transfers through a chain of accounts |
| **Fan-out / smurfing** | Single source → many targets below reporting thresholds |
| **Layering** | Deep hop chain to obscure trail |
| **Rapid burst** | Account firing many transactions in a short window |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Demo Notebook

Open `notebooks/demo_analysis.ipynb` for an end-to-end interactive walkthrough including
score distribution plots, community detection, and explainability reports.

---

## Project Structure

```
fraud-detection/
├── config/           # YAML weights and thresholds
├── data/             # Synthetic transaction datasets
├── src/
│   ├── ingestion/    # Batch loader, stream simulator
│   ├── graph/        # Graph builder, in-memory store
│   ├── features/     # Structural, temporal, behavioural features
│   ├── scoring/      # Weighted scorer, Isolation Forest, normalizer
│   ├── network/      # Community detection, subgraph, cluster scorer
│   ├── explainability/ # Human-readable report generator
│   └── simulation/   # Synthetic fraud + normal data generators
├── api/              # FastAPI app, routers, Pydantic schemas
├── tests/            # Pytest test suite
├── notebooks/        # Demo analysis notebook
├── requirements.txt
├── run.py            # Single entry point
└── README.md
```

---

## Stack

- **Python 3.11**
- **NetworkX 3.x** — graph construction, centrality, community detection
- **scikit-learn** — Isolation Forest anomaly detection
- **FastAPI + Uvicorn** — REST API
- **Pydantic v2** — request/response validation
- **PyYAML** — configuration

## Scale-out Path

For graphs > 10M edges, replace NetworkX with **Neo4j GDS** — all interface
contracts (feature functions, scorer, API schemas) remain unchanged.
