"""
SARStore — in-memory store for SAR drafts.

Tracks:
  - All SAR drafts (pending / approved / dismissed)
  - Flag counts per cluster (how many times flagged)
  - Which clusters already have a SAR (no duplicates)
"""

import threading
from typing import Dict, List, Optional


class SARStore:

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._sars:       Dict[str, dict] = {}   # sar_id → SAR dict
        self._flag_counts: Dict[str, int] = {}   # cluster_fingerprint → count
        self._sar_by_fp:  Dict[str, str]  = {}   # cluster_fingerprint → sar_id

    # ------------------------------------------------------------------
    # Flag counting
    # ------------------------------------------------------------------

    def increment_flag(self, fingerprint: str) -> int:
        """Increment flag count for a cluster. Returns new count."""
        with self._lock:
            self._flag_counts[fingerprint] = self._flag_counts.get(fingerprint, 0) + 1
            return self._flag_counts[fingerprint]

    def get_flag_count(self, fingerprint: str) -> int:
        return self._flag_counts.get(fingerprint, 0)

    def has_sar(self, fingerprint: str) -> bool:
        """Returns True only if there is a pending or approved SAR for this cluster.
        Dismissed SARs allow a new one to be generated if the cluster resurfaces."""
        sar_id = self._sar_by_fp.get(fingerprint)
        if not sar_id:
            return False
        sar = self._sars.get(sar_id)
        if not sar:
            return False
        return sar["status"] in ("pending", "approved")

    # ------------------------------------------------------------------
    # SAR CRUD
    # ------------------------------------------------------------------

    def save(self, sar: dict) -> None:
        with self._lock:
            self._sars[sar["sar_id"]] = sar
            self._sar_by_fp[sar["cluster_fingerprint"]] = sar["sar_id"]

    def get(self, sar_id: str) -> Optional[dict]:
        return self._sars.get(sar_id)

    def get_by_fingerprint(self, fingerprint: str) -> Optional[dict]:
        sar_id = self._sar_by_fp.get(fingerprint)
        return self._sars.get(sar_id) if sar_id else None

    def all(self) -> List[dict]:
        return list(self._sars.values())

    def pending(self) -> List[dict]:
        return [s for s in self._sars.values() if s["status"] == "pending"]

    def update_status(self, sar_id: str, status: str, reviewed_by: str = "compliance_officer") -> Optional[dict]:
        import datetime
        with self._lock:
            sar = self._sars.get(sar_id)
            if not sar:
                return None
            sar["status"]      = status
            sar["reviewed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            sar["reviewed_by"] = reviewed_by
            # On dismiss: remove the SAR pointer and reset flag count
            # so the cluster can generate a fresh SAR if it resurfaces
            if status == "dismissed":
                fp = sar.get("cluster_fingerprint")
                if fp:
                    self._sar_by_fp.pop(fp, None)
                    self._flag_counts[fp] = 0
            return sar


# Module-level singleton
sar_store = SARStore()
