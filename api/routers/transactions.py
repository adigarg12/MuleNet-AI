"""
POST /transactions — ingest single or batch transactions into the graph.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import TransactionIn, BatchTransactionsIn, IngestResponse
from src.graph.graph_store import graph_store
from src.graph.graph_builder import graph_summary

router = APIRouter(prefix="/transactions", tags=["transactions"])


def _txn_dict(t: TransactionIn) -> dict:
    return t.model_dump()


@router.post("", response_model=IngestResponse)
def ingest_single(txn: TransactionIn):
    """Ingest a single transaction."""
    graph_store.ingest_transaction(_txn_dict(txn))
    G = graph_store.get_graph()
    stats = graph_summary(G)
    return IngestResponse(
        ingested=1,
        graph_nodes=stats["nodes"],
        graph_edges=stats["edges"],
        message="Transaction ingested successfully.",
    )


@router.post("/batch", response_model=IngestResponse)
def ingest_batch(body: BatchTransactionsIn):
    """Ingest a batch of transactions."""
    count = graph_store.ingest_batch(_txn_dict(t) for t in body.transactions)
    G = graph_store.get_graph()
    stats = graph_summary(G)
    return IngestResponse(
        ingested=count,
        graph_nodes=stats["nodes"],
        graph_edges=stats["edges"],
        message=f"Batch of {count} transactions ingested successfully.",
    )
