"""Best-effort operator alerting for the unattended live loop.

The primary alert channel is the CI itself: any fatal guard makes the scheduled
GitHub Actions run exit non-zero, the run turns red, and GitHub e-mails the
repository owner. That channel needs no configuration and cannot be forgotten.

This module adds an OPTIONAL second channel: a webhook POst (Slack, Discord,
Mattermost, healthchecks.io, ntfy, ...) configured via the
``ALERT_WEBHOOK_URL`` environment variable (in CI: a repository secret). The
payload carries the message under both ``text`` (Slack-style) and ``content``
(Discord-style) keys so one URL works for either family without configuration.

Design constraints:
* **Never raises.** An alerting failure must never break the trading run that
  is trying to report a problem; failures are logged and swallowed.
* **No new dependencies.** Standard-library ``urllib`` only.
* **No secrets in payloads.** Only the human-readable message is sent.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

__all__ = ["send_alert", "ALERT_WEBHOOK_ENV"]

_LOGGER = logging.getLogger(__name__)

# Environment variable holding the optional webhook URL.
ALERT_WEBHOOK_ENV: str = "ALERT_WEBHOOK_URL"


def send_alert(
    message: str,
    *,
    webhook_url: str | None = None,
    timeout: float = 10.0,
) -> bool:
    """Send ``message`` to the configured webhook, best-effort.

    Parameters
    ----------
    message:
        Human-readable alert text (already fully composed by the caller).
    webhook_url:
        Explicit webhook URL. When ``None`` (default), the
        ``ALERT_WEBHOOK_URL`` environment variable is used; when that is also
        unset/empty, the alert is a silent no-op (the red CI run remains the
        primary channel).
    timeout:
        Network timeout in seconds.

    Returns
    -------
    bool
        ``True`` if the webhook accepted the POST (HTTP 2xx), ``False`` in
        every other case (no URL configured, network error, non-2xx). This
        function NEVER raises.
    """
    url = webhook_url or os.environ.get(ALERT_WEBHOOK_ENV) or ""
    if not url:
        _LOGGER.debug(
            "No %s configured; skipping webhook alert (red CI run remains "
            "the primary alert channel).", ALERT_WEBHOOK_ENV,
        )
        return False

    payload = json.dumps({"text": message, "content": message}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = int(getattr(response, "status", 200))
        ok = 200 <= code < 300
        if ok:
            _LOGGER.info("Alert webhook delivered (HTTP %d).", code)
        else:  # pragma: no cover - unusual: urlopen returns non-2xx unraised
            _LOGGER.warning("Alert webhook returned HTTP %d.", code)
        return ok
    except Exception as exc:  # noqa: BLE001 - alerting must never raise
        _LOGGER.warning("Alert webhook failed (%s); continuing.", exc)
        return False
