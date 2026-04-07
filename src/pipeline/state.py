"""
PipelineState — shared singleton between background workers and the API.

Workers write scores/clusters here after each retrain cycle.
API routes read from here instead of recomputing on every request.
"""

import threading
import time
from typing import Dict, List, Optional, Set


class PipelineState:

    def __init__(self) -> None:
        self._lock                = threading.Lock()
        self._retrain_lock        = threading.Lock()

        self.latest_scores:  Dict[str, float] = {}
        self.latest_clusters: List[Dict]      = []

        # Deduplication: fingerprint -> alerted_at unix timestamp
        self._alerted: Dict[str, float] = {}
        self._alert_ttl_seconds: float  = 86400.0  # 24 hours

        self.txn_counter:     int            = 0
        self.retrain_counter: int            = 0
        self.last_retrain_at: Optional[float] = None
        self.is_retraining:   bool           = False

    # ------------------------------------------------------------------
    # Transaction counter
    # ------------------------------------------------------------------

    def increment_txn(self, n: int = 1) -> int:
        with self._lock:
            self.txn_counter += n
            return self.txn_counter

    # ------------------------------------------------------------------
    # Scores / clusters
    # ------------------------------------------------------------------

    def set_scores(self, scores: Dict[str, float]) -> None:
        with self._lock:
            self.latest_scores = scores

    def set_clusters(self, clusters: List[Dict]) -> None:
        with self._lock:
            self.latest_clusters = clusters

    def get_scores(self) -> Dict[str, float]:
        return self.latest_scores

    def get_clusters(self) -> List[Dict]:
        return self.latest_clusters

    # ------------------------------------------------------------------
    # Alert deduplication (content-based fingerprint + TTL)
    # ------------------------------------------------------------------

    @staticmethod
    def cluster_fingerprint(cluster: Dict) -> str:
        return "|".join(sorted(cluster.get("members", [])))

    def is_alerted(self, cluster: Dict) -> bool:
        fp = self.cluster_fingerprint(cluster)
        with self._lock:
            alerted_at = self._alerted.get(fp)
            if alerted_at is None:
                return False
            if time.time() - alerted_at > self._alert_ttl_seconds:
                del self._alerted[fp]
                return False
            return True

    def mark_alerted(self, cluster: Dict) -> None:
        fp = self.cluster_fingerprint(cluster)
        with self._lock:
            self._alerted[fp] = time.time()

    def reset_alerted(self) -> None:
        with self._lock:
            self._alerted.clear()


# Module-level singleton
pipeline_state = PipelineState()
