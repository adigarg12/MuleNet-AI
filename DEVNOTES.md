# Development Notes — Fraud Detection System

> For future Claude sessions working on this project.
> Stack: Python 3.13, NetworkX, scikit-learn, FastAPI, Pydantic v2, PyYAML.

---

## What Has Been Built (Complete)

### Infrastructure
- [x] Full folder structure created (`config/`, `data/`, `src/`, `api/`, `tests/`, `notebooks/`)
- [x] `config/scoring_weights.yaml` — feature weights (velocity, fan_out, centrality, retention_time, pagerank, cross_channel, burst_score)
- [x] `config/thresholds.yaml` — risk tiers (LOW/MEDIUM/HIGH/CRITICAL), cluster flagging, anomaly boost settings
- [x] `requirements.txt` — all dependencies pinned
- [x] `README.md` — usage, API endpoints, architecture diagram

### Simulation (`src/simulation/`)
- [x] `normal_patterns.py` — p2p payments, payroll fan-out, recurring subscriptions, `generate_mixed_normal()`
- [x] `fraud_patterns.py` — mule ring, fan-out/smurfing, layering chain, rapid burst, `generate_mixed_fraud()`
- [x] `data/generate_data.py` — one-shot script to write JSON files to `data/synthetic/`

### Ingestion (`src/ingestion/`)
- [x] `batch_loader.py` — load JSON/CSV, validate fields, yield normalised dicts
- [x] `stream_simulator.py` — async generator wrapper with configurable delay

### Graph (`src/graph/`)
- [x] `graph_builder.py` — `build_graph()`, `add_transaction()`, `graph_summary()`; parallel edges accumulated on DiGraph
- [x] `graph_store.py` — singleton in-memory store; `ingest_transaction`, `ingest_batch`, `load_fresh`, `snapshot`, `reset`

### Features (`src/features/`)
- [x] `structural.py` — in/out degree, betweenness centrality (cached), PageRank (cached), clustering coefficient
- [x] `temporal.py` — velocity (rolling window), avg retention time, burst score (CoV of inter-arrival times)
- [x] `behavioral.py` — fan-out ratio, fan-in ratio, flow depth, cross-channel jumps, amount concentration (Gini)

### Scoring (`src/scoring/`)
- [x] `weighted_scorer.py` — YAML-driven weighted sum, sigmoid normalisation per feature, contribution breakdown
- [x] `anomaly_model.py` — Isolation Forest singleton; `train()`, `anomaly_boost()` (additive score boost)
- [x] `normalizer.py` — `normalize_score()`, `assign_tier()`, `build_risk_result()` (packages full result dict)

### Network Intelligence (`src/network/`)
- [x] `community_detection.py` — `detect_communities()` using greedy modularity on undirected projection
- [x] `subgraph_extractor.py` — `extract_subgraph()`, `detect_cycles()`, `flag_suspicious_subgraphs()`
- [x] `cluster_scorer.py` — `score_cluster()` (density-weighted risk, mule ring tagging), `score_all_clusters()`

### Explainability (`src/explainability/`)
- [x] `report_generator.py` — `generate_report()` (formatted text), `generate_report_dict()` (JSON-ready); score bars, contextual hints

### API (`api/`)
- [x] `schemas.py` — Pydantic v2 models: `TransactionIn`, `BatchTransactionsIn`, `IngestResponse`, `RiskResponse`, `ClusterResponse`, `GraphStatsResponse`
- [x] `main.py` — FastAPI app with lifespan startup (auto-loads synthetic data + trains anomaly model)
- [x] `routers/transactions.py` — `POST /transactions`, `POST /transactions/batch`
- [x] `routers/accounts.py` — `GET /accounts/{account_id}/risk`
- [x] `routers/clusters.py` — `GET /clusters/suspicious`, `GET /graph/stats`

### Entry Point
- [x] `run.py` — ingest → build graph → compute features → train anomaly model → score → print high-risk reports → start API server

### Tests (`tests/`)
- [x] `test_graph_builder.py` — node/edge counts, parallel txn accumulation, cycle detection
- [x] `test_features.py` — structural, temporal, behavioural feature correctness
- [x] `test_scoring.py` — scorer output range, fraud > normal assertion, tier assignment, result structure
- [x] `test_simulation.py` — schema validation, mule ring cycle check, fan-out amounts, burst time window

### Demo
- [x] `notebooks/demo_analysis.ipynb` — 8-cell end-to-end notebook: generate → graph → features → score → distribution plot → community detection → reports → fraud/normal separation boxplot

---

## What Still Needs to Be Done

### High Priority (hackathon demo quality)
- [ ] **Run `data/generate_data.py`** to produce `data/synthetic/normal_transactions.json` and `data/synthetic/fraud_transactions.json` — these files don't exist yet (the script exists but hasn't been run)
- [ ] **Install dependencies** — `pip install -r requirements.txt` (network issue was hitting PyPI; retry or use mirror)
- [ ] **Run tests** — `pytest tests/ -v` to confirm all pass end-to-end
- [ ] **Smoke test the API** — `python run.py`, then `curl http://localhost:8000/health`
- [ ] **`tests/__init__.py`** — add empty init file so pytest discovers tests correctly on all platforms

### Medium Priority (polish)
- [ ] **API authentication** — add Bearer token / API key middleware (mentioned in plan's security section, not yet implemented)
- [ ] **Rate limiting** — add `slowapi` or similar to the ingest endpoint
- [ ] **`GET /accounts/` endpoint** — list all accounts with their risk scores (pagination-ready)
- [ ] **Streaming ingest endpoint** — `POST /transactions/stream` using `asyncio.Queue` + background task
- [ ] **`/metrics` endpoint** — expose Prometheus-style counters (total transactions, high-risk count, etc.)
- [ ] **Dockerfile** — single-stage Python 3.11-slim image as described in the plan

### Low Priority (scale-out / future)
- [ ] **Incremental centrality** — current betweenness/PageRank is cached by edge count; implement delta-recompute on graph changes
- [ ] **Parallel feature computation** — wrap `get_*_features` calls with `concurrent.futures.ThreadPoolExecutor`
- [ ] **Persist graph state** — save/load graph to disk (pickle or GraphML) for restarts
- [ ] **Neo4j adapter** — drop-in replacement for `graph_store.py` using Neo4j GDS for >10M edges
- [ ] **Fine-tune Isolation Forest** — implement incremental `partial_fit` as new transactions arrive
- [ ] **Label feedback loop** — endpoint to mark a transaction as confirmed fraud, update model
- [ ] **Stress test** — generate 100k transactions, measure graph build + scoring time (target: <30s)

---

## Key File Locations

| What | Where |
|------|-------|
| Feature weights (tune) | `config/scoring_weights.yaml` |
| Risk tier thresholds | `config/thresholds.yaml` |
| Main entry point | `run.py` |
| FastAPI app | `api/main.py` |
| Graph singleton | `src/graph/graph_store.py` → `graph_store` |
| Anomaly model singleton | `src/scoring/anomaly_model.py` → `anomaly_model` |
| Demo notebook | `notebooks/demo_analysis.ipynb` |

---

## Bugs Fixed

- `compute_flow_depth` in `src/features/behavioral.py`: `nx.single_target_shortest_path_length`
  returns a `dict` in NetworkX 3.6 (not a tuple iterator). Fixed by wrapping with `dict()` and
  using `.values()`. All 35 tests pass on Python 3.13 + NetworkX 3.6.1.

---

## Known Gotchas

- **`graph_store` is a singleton** — call `graph_store.reset()` between test runs or tests will share state
- **Betweenness centrality is O(VE)** — cached by edge count, but will recompute on any graph change; don't call in tight loops
- **Anomaly model needs training before `anomaly_boost()` returns non-zero** — it returns `0.0` if untrained (safe default)
- **Community detection runs on undirected projection** — direction information is lost; this is intentional (Louvain doesn't support directed graphs natively)
- **`data/synthetic/` JSON files must exist before starting the API** — run `python data/generate_data.py` first, or use `python run.py` which auto-generates them
