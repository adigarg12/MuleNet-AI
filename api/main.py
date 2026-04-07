"""
FastAPI application entry point.

Endpoints:
  POST /transactions              — ingest single transaction
  POST /transactions/batch        — ingest batch
  GET  /accounts/{id}/risk        — risk score + explanation
  GET  /clusters/suspicious       — flagged clusters
  GET  /graph/stats               — graph summary
  GET  /health                    — liveness probe
"""

from contextlib import asynccontextmanager
import logging
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.routers import transactions, accounts, clusters, sar

logger = logging.getLogger("fraud_detection")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _extract_account_labels(txns) -> dict:
    fraud_count = {}
    total_count = {}
    for t in txns:
        for acc in [t["from_account"], t["to_account"]]:
            total_count[acc] = total_count.get(acc, 0) + 1
            if t.get("label") in ("fraud", 1, "1"):
                fraud_count[acc] = fraud_count.get(acc, 0) + 1
    return {a: fraud_count.get(a, 0) / total_count[a] for a in total_count}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load synthetic data and train GNN."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    if os.path.isdir(data_dir):
        try:
            from src.ingestion.batch_loader import load_all
            from src.graph.graph_store import graph_store
            from src.gnn.embedder import gnn_embedder

            txns = list(load_all(data_dir))
            graph_store.load_fresh(txns)
            G = graph_store.get_graph()
            logger.info("Loaded %d transactions → %d nodes, %d edges",
                        len(txns), G.number_of_nodes(), G.number_of_edges())

            labels = _extract_account_labels(txns)
            gnn_embedder.train(G, labels=labels, epochs=200)
            logger.info("GNN trained on %d accounts", G.number_of_nodes())

            # Populate pipeline state so /clusters endpoint serves cached data
            from src.pipeline.state import pipeline_state
            from src.pipeline.retrain_worker import rescore_and_detect
            all_scores = gnn_embedder.get_all_scores()
            pipeline_state.set_scores(all_scores)
            rescore_and_detect(G, all_scores)
            logger.info("Pipeline state initialised — %d clusters cached.",
                        len(pipeline_state.get_clusters()))

            # Start stream worker if enabled via env var
            if os.getenv("ENABLE_STREAM", "").lower() in ("1", "true", "yes"):
                from src.pipeline.stream_worker import start_stream_worker
                from src.pipeline.retrain_worker import trigger_retrain
                _stream_worker = start_stream_worker(txns, on_retrain_needed=trigger_retrain)
                app.state.stream_worker = _stream_worker

        except Exception as exc:
            logger.warning("Could not pre-load data or train GNN: %s", exc)
    yield

    # Shutdown: stop stream worker if running
    worker = getattr(app.state, "stream_worker", None)
    if worker is not None:
        worker.stop()


app = FastAPI(
    title="Fraud Detection — Graph Intelligence API",
    description=(
        "Real-time and batch fraud detection using graph-based features, "
        "behavioural risk scoring, and network community analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(transactions.router)
app.include_router(accounts.router)
app.include_router(clusters.router)
app.include_router(sar.router)


# ------------------------------------------------------------------
# Health probe
# ------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    from src.graph.graph_store import graph_store
    G = graph_store.get_graph()
    return {
        "status": "ok",
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }


# ------------------------------------------------------------------
# Global error handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
