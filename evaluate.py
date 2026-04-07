"""
Model evaluation — 80/20 train/test split on account labels.

Trains the GNN on 80% of labeled accounts (test accounts are masked as
unlabeled during training, but still exist in the graph for message passing).
Evaluates on the held-out 20% using threshold 0.5 for binary classification.

Outputs: Precision, Recall, F1, AUC-ROC, Confusion Matrix.

Usage:
    python evaluate.py
"""

import os
import sys
import random
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluate")

THRESHOLD  = 0.5
EPOCHS     = 200
SEED       = 42
TEST_SPLIT = 0.2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    from src.ingestion.batch_loader import load_all
    from src.graph.graph_store import graph_store
    from src.gnn.model import FraudGNN
    from src.gnn.graph_to_pyg import to_pyg_data
    from src.gnn.trainer import train as gnn_train
    from src.pipeline.label_extractor import extract_labels_from_graph

    # ── 1. Load data and build graph ──────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "data", "synthetic")
    logger.info("Loading transactions from %s ...", data_dir)
    txns = list(load_all(data_dir))
    graph_store.load_fresh(txns)
    G = graph_store.get_graph()
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── 2. Get ground-truth labels ────────────────────────────────────
    all_labels = extract_labels_from_graph(G)

    # Binary ground truth: account is "fraud" if fraud_ratio > 0
    fraud_accounts  = {a for a, r in all_labels.items() if r > 0}
    normal_accounts = {a for a, r in all_labels.items() if r == 0}
    logger.info("Labeled accounts — fraud: %d, normal: %d",
                len(fraud_accounts), len(normal_accounts))

    # ── 3. 80/20 stratified split ─────────────────────────────────────
    fraud_list  = sorted(fraud_accounts)
    normal_list = sorted(normal_accounts)
    random.shuffle(fraud_list)
    random.shuffle(normal_list)

    n_fraud_test  = max(1, int(len(fraud_list)  * TEST_SPLIT))
    n_normal_test = max(1, int(len(normal_list) * TEST_SPLIT))

    test_accounts  = set(fraud_list[:n_fraud_test] + normal_list[:n_normal_test])
    train_accounts = set(fraud_list[n_fraud_test:] + normal_list[n_normal_test:])

    logger.info("Train accounts: %d | Test accounts: %d",
                len(train_accounts), len(test_accounts))

    # Train labels: hide test accounts (set to -1 = unlabeled)
    train_labels = {
        a: r for a, r in all_labels.items() if a in train_accounts
    }

    # ── 4. Train GNN on train split only ─────────────────────────────
    logger.info("Training GNN on 80%% split (%d epochs) ...", EPOCHS)
    data, node_order, node_index = to_pyg_data(G, train_labels)
    model = FraudGNN()
    gnn_train(model, data, epochs=EPOCHS)

    # ── 5. Get scores for all nodes ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
        scores     = model.classify(embeddings)

    all_scores = {
        node_order[i]: float(scores[i].item())
        for i in range(len(node_order))
    }

    # ── 6. Evaluate on test split ─────────────────────────────────────
    y_true, y_score, y_pred = [], [], []

    for acc in sorted(test_accounts):
        true_label = 1 if acc in fraud_accounts else 0
        score      = all_scores.get(acc, 0.0)
        pred_label = 1 if score >= THRESHOLD else 0
        y_true.append(true_label)
        y_score.append(score)
        y_pred.append(pred_label)

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = np.array(y_pred)

    # ── 7. Metrics ────────────────────────────────────────────────────
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    cm        = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print()
    print("=" * 50)
    print("  FRAUD DETECTION — EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Test accounts :  {len(y_true)}  "
          f"({int(y_true.sum())} fraud, {int((1-y_true).sum())} normal)")
    print(f"  Threshold     :  {THRESHOLD}")
    print()
    print(f"  Precision     :  {precision:.1%}")
    print(f"  Recall        :  {recall:.1%}")
    print(f"  F1 Score      :  {f1:.1%}")
    print(f"  AUC-ROC       :  {auc:.1%}")
    print()
    print("  Confusion Matrix:")
    print(f"               Predicted Normal  Predicted Fraud")
    print(f"  Actual Normal      {tn:>4}               {fp:>4}")
    print(f"  Actual Fraud       {fn:>4}               {tp:>4}")
    print()
    print(f"  True Positives  (caught fraud)    : {tp}")
    print(f"  False Positives (false alarms)    : {fp}")
    print(f"  True Negatives  (correct clears)  : {tn}")
    print(f"  False Negatives (missed fraud)    : {fn}")
    print("=" * 50)
    print()

    if fn > 0:
        missed = [
            (acc, all_scores.get(acc, 0.0))
            for acc in sorted(test_accounts)
            if acc in fraud_accounts and all_scores.get(acc, 0.0) < THRESHOLD
        ]
        print(f"  Missed fraud accounts ({len(missed)}):")
        for acc, sc in sorted(missed, key=lambda x: x[1]):
            print(f"    {acc}  score={sc:.3f}")
        print()


if __name__ == "__main__":
    main()
