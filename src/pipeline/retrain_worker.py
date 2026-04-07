"""
RetrainWorker — periodically retrains the GNN and drives the rescore pipeline.

Flow after each retrain:
  GNN.train(graph_snapshot) → get_all_scores() → detect_communities()
  → score_all_clusters() → check_thresholds() → webhook if needed

Uses a single-worker ThreadPoolExecutor so retrain calls are serialised:
if a new trigger fires while one retrain is already running, it queues
rather than spawning a second concurrent training job.
"""

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import networkx as nx

from src.graph.graph_store import graph_store
from src.gnn.embedder import gnn_embedder
from src.network.community_detection import detect_communities
from src.network.cluster_scorer import score_all_clusters
from src.pipeline.state import pipeline_state
from src.pipeline.label_extractor import extract_labels_from_graph
from src.pipeline.threshold_monitor import check_thresholds

logger = logging.getLogger(__name__)

RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "50"))

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")


def trigger_retrain() -> None:
    """Submit a retrain cycle to the background executor (non-blocking)."""
    _executor.submit(_do_retrain)


def _do_retrain() -> None:
    """Full retrain + rescore cycle. Runs in background thread."""
    if not pipeline_state._retrain_lock.acquire(blocking=False):
        logger.debug("Retrain already in progress — skipping this trigger.")
        return

    try:
        pipeline_state.is_retraining = True
        logger.info("Retraining GNN (epochs=%d) ...", RETRAIN_EPOCHS)

        # Snapshot the graph so ingestion can continue during training
        with graph_store.lock:
            G_snapshot = copy.deepcopy(graph_store.get_graph())

        labels = extract_labels_from_graph(G_snapshot)
        gnn_embedder.train(G_snapshot, labels=labels, epochs=RETRAIN_EPOCHS)

        all_scores = gnn_embedder.get_all_scores()
        pipeline_state.set_scores(all_scores)
        pipeline_state.retrain_counter += 1

        import time
        pipeline_state.last_retrain_at = time.time()

        rescore_and_detect(G_snapshot, all_scores)
        logger.info("Retrain cycle #%d complete.", pipeline_state.retrain_counter)

    except Exception as exc:
        logger.error("Retrain cycle failed: %s", exc, exc_info=True)
    finally:
        pipeline_state.is_retraining = False
        pipeline_state._retrain_lock.release()


def rescore_and_detect(G: nx.DiGraph, all_scores: Dict[str, float]) -> None:
    """
    Re-detect communities, score clusters, check thresholds.
    Called after each retrain and also at API startup for initial population.
    """
    try:
        communities    = detect_communities(G)
        scored_clusters = score_all_clusters(G, communities, all_scores, min_members=3)
        pipeline_state.set_clusters(scored_clusters)
        check_thresholds(scored_clusters, G, all_scores)
    except Exception as exc:
        logger.error("Rescore/detect failed: %s", exc, exc_info=True)
