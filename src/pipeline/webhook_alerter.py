"""
WebhookAlerter — fire-and-forget HTTP POST alerts to the bank's endpoint.

Configure via env var:
  FRAUD_ALERT_WEBHOOK_URL  — target URL (leave empty to log-only)
  WEBHOOK_TIMEOUT_SECS     — HTTP timeout in seconds (default 5.0)
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_URL     = os.getenv("FRAUD_ALERT_WEBHOOK_URL", "")
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT_SECS", "5.0"))


class WebhookAlerter:

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")

    def send_alert(self, payload: Dict[str, Any]) -> None:
        """Submit alert delivery as a background task (non-blocking)."""
        self._executor.submit(self._post, payload)

    def _post(self, payload: Dict[str, Any]) -> None:
        if not WEBHOOK_URL:
            logger.warning(
                "FRAUD_ALERT_WEBHOOK_URL not set — alert logged only:\n%s",
                json.dumps(payload, indent=2, default=str),
            )
            return
        try:
            resp = httpx.post(WEBHOOK_URL, json=payload, timeout=WEBHOOK_TIMEOUT)
            resp.raise_for_status()
            logger.info(
                "Alert sent for cluster %s — HTTP %d",
                payload.get("cluster_id"), resp.status_code,
            )
        except Exception as exc:
            logger.error("Webhook delivery failed: %s", exc)


# Module-level singleton
webhook_alerter = WebhookAlerter()
