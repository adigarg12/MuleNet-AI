"""
Single entry point: ingest synthetic data → train GNN → score → serve API.

Usage:
    # Generate synthetic data (first time):
    python data/generate_data.py

    # Run the full pipeline + start API server:
    python run.py

    # Or just score without serving:
    python run.py --no-serve
"""

import argparse
import logging
import os
import sys

# Ensure stdout can handle Unicode (box-drawing chars in reports)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run")


def _extract_account_labels(txns) -> dict:
    """
    Build {account_id: float} fraud-ratio labels in [0, 1].
    An account's label = (# fraud transactions it touched) / (# total transactions).
    This gives intermediate scores to accounts that mix fraud and normal activity.
    """
    fraud_count = {}
    total_count = {}
    for t in txns:
        for acc in [t["from_account"], t["to_account"]]:
            total_count[acc] = total_count.get(acc, 0) + 1
            if t.get("label") in ("fraud", 1, "1"):
                fraud_count[acc] = fraud_count.get(acc, 0) + 1
    return {a: fraud_count.get(a, 0) / total_count[a] for a in total_count}


def run_pipeline(data_dir: str) -> dict:
    """Ingest data → build graph → train GNN → score all accounts."""
    from src.ingestion.batch_loader import load_all
    from src.graph.graph_store import graph_store
    from src.graph.graph_builder import graph_summary
    from src.gnn.embedder import gnn_embedder
    from src.scoring.normalizer import build_risk_result, assign_tier
    from src.network.community_detection import detect_communities
    from src.network.cluster_scorer import score_all_clusters
    from src.explainability.report_generator import generate_report

    # 1. Ingest
    logger.info("Loading transactions from %s ...", data_dir)
    txns = list(load_all(data_dir))
    graph_store.load_fresh(txns)
    G     = graph_store.get_graph()
    stats = graph_summary(G)
    logger.info("Graph built: %d nodes, %d edges", stats["nodes"], stats["edges"])

    # 2. Extract labels from transaction metadata
    labels = _extract_account_labels(txns)
    n_fraud  = sum(1 for v in labels.values() if v == 1)
    n_normal = sum(1 for v in labels.values() if v == 0)
    logger.info("Labels — fraud accounts: %d, normal: %d", n_fraud, n_normal)

    # 3. Train GNN (replaces get_structural/temporal/behavioral_features + WeightedScorer)
    logger.info("Training GNN on %d nodes ...", G.number_of_nodes())
    gnn_embedder.train(G, labels=labels, epochs=200)

    # 4. Score all accounts via GNN
    all_scores  = gnn_embedder.get_all_scores()
    risk_scores = {}
    for node, gnn_score in all_scores.items():
        contributions = gnn_embedder.explain_node(node)
        result = build_risk_result(
            account_id    = node,
            weighted_score= gnn_score,
            anomaly_boost = 0.0,
            contributions = contributions,
            missing       = [],
        )
        risk_scores[node] = result

    # 5. Community / cluster detection (unchanged — still uses graph topology)
    communities    = detect_communities(G)
    scored_clusters = score_all_clusters(
        G, communities, all_scores, min_members=3
    )
    cluster_map = {}
    for cluster in scored_clusters:
        for member in cluster["members"]:
            cluster_map[member] = cluster

    # 6. Print summary
    high_risk = [
        (n, r) for n, r in risk_scores.items() if r["risk_score"] >= 0.60
    ]
    high_risk.sort(key=lambda x: x[1]["risk_score"], reverse=True)

    logger.info("=" * 60)
    logger.info("HIGH / CRITICAL ACCOUNTS (%d):", len(high_risk))
    for node, result in high_risk[:10]:
        cluster_info = cluster_map.get(node)
        # Pass raw feature importances as 'features' for contextual hints
        report = generate_report(node, result, cluster_info,
                                 features=gnn_embedder.explain_node(node))
        print(report)

    logger.info("Mule rings detected: %d",
                sum(1 for c in scored_clusters if c["is_mule_ring"]))
    logger.info("Total suspicious clusters (risk ≥ 0.6): %d",
                sum(1 for c in scored_clusters if c["cluster_risk_score"] >= 0.6))

    return {
        "graph_stats":     stats,
        "total_accounts":  len(risk_scores),
        "high_risk_count": len(high_risk),
        "clusters_scored": len(scored_clusters),
        "transactions":    txns,
    }


def serve_api(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--no-serve", action="store_true",
                        help="Run pipeline only, don't start API")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "synthetic"),
        help="Directory containing transaction JSON/CSV files",
    )
    parser.add_argument("--stream", action="store_true",
                        help="Start real-time transaction stream after pipeline")
    parser.add_argument("--stream-delay", type=float, default=0.1,
                        help="Seconds between streamed transactions (default 0.1)")
    parser.add_argument("--retrain-every", type=int, default=50,
                        help="Trigger GNN retrain every N new transactions (default 50)")
    args = parser.parse_args()

    normal_path = os.path.join(args.data_dir, "normal_transactions.json")
    fraud_path  = os.path.join(args.data_dir, "fraud_transactions.json")
    if not os.path.exists(normal_path) or not os.path.exists(fraud_path):
        logger.info("Synthetic data not found — generating now ...")
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(__file__), "data", "generate_data.py"))

    summary = run_pipeline(args.data_dir)
    logger.info("Pipeline complete: %s", {k: v for k, v in summary.items() if k != "transactions"})

    stream_worker = None
    if args.stream:
        from src.pipeline.stream_worker import start_stream_worker
        from src.pipeline.retrain_worker import trigger_retrain
        txns = summary["transactions"]
        logger.info(
            "Starting stream worker — %d transactions, %.2fs delay, retrain every %d txns",
            len(txns), args.stream_delay, args.retrain_every,
        )
        stream_worker = start_stream_worker(
            transactions      = txns,
            delay_seconds     = args.stream_delay,
            retrain_every     = args.retrain_every,
            on_retrain_needed = trigger_retrain,
        )

    if not args.no_serve:
        logger.info("Starting API server at http://%s:%d", args.host, args.port)
        logger.info("API docs: http://%s:%d/docs", args.host, args.port)
        try:
            serve_api(host=args.host, port=args.port)
        except KeyboardInterrupt:
            pass
        finally:
            if stream_worker:
                stream_worker.stop()
