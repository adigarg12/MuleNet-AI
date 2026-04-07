"""
TransactionStreamWorker — replays transactions in a background thread.

Simulates a real-time feed by cycling through the loaded transaction list
with a configurable delay. After every RETRAIN_EVERY transactions it calls
trigger_retrain() to kick off a GNN retrain cycle.

Uses time.sleep() (not asyncio.sleep) — this runs in a plain thread,
not inside uvicorn's event loop.
"""

import logging
import os
import threading
import time
import uuid
from typing import Callable, Dict, List

from src.graph.graph_store import graph_store
from src.pipeline.state import pipeline_state

logger = logging.getLogger(__name__)

STREAM_DELAY   = float(os.getenv("STREAM_DELAY_SECS", "0.1"))
RETRAIN_EVERY  = int(os.getenv("RETRAIN_EVERY_N_TXNS", "50"))


class TransactionStreamWorker(threading.Thread):

    def __init__(
        self,
        transactions:       List[Dict],
        delay_seconds:      float    = STREAM_DELAY,
        retrain_every:      int      = RETRAIN_EVERY,
        on_retrain_needed:  Callable = None,
    ) -> None:
        super().__init__(daemon=True, name="stream-worker")
        self._transactions      = transactions
        self._delay             = delay_seconds
        self._retrain_every     = retrain_every
        self._on_retrain_needed = on_retrain_needed or (lambda: None)
        self._stop_event        = threading.Event()

    def run(self) -> None:
        logger.info(
            "Stream worker started — %d transactions, %.2fs delay, retrain every %d txns",
            len(self._transactions), self._delay, self._retrain_every,
        )
        while not self._stop_event.is_set():
            for txn in self._transactions:
                if self._stop_event.is_set():
                    break

                # Give each replayed transaction a fresh unique ID
                replayed = dict(txn, txn_id=f"replay_{uuid.uuid4().hex[:8]}")
                graph_store.ingest_transaction(replayed)
                new_count = pipeline_state.increment_txn()

                if new_count % self._retrain_every == 0:
                    logger.info("Stream: %d transactions ingested — triggering retrain.", new_count)
                    self._on_retrain_needed()

                if self._delay > 0:
                    time.sleep(self._delay)

        logger.info("Stream worker stopped.")

    def stop(self) -> None:
        self._stop_event.set()


def start_stream_worker(
    transactions:      List[Dict],
    delay_seconds:     float    = STREAM_DELAY,
    retrain_every:     int      = RETRAIN_EVERY,
    on_retrain_needed: Callable = None,
) -> TransactionStreamWorker:
    worker = TransactionStreamWorker(
        transactions      = transactions,
        delay_seconds     = delay_seconds,
        retrain_every     = retrain_every,
        on_retrain_needed = on_retrain_needed,
    )
    worker.start()
    return worker
