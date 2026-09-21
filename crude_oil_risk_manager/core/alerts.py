"""Alert management: persist alerts and dispatch them to Microsoft Teams.

AlertManager never raises. The Teams webhook URL is a secret: it is never
logged, and request exceptions are logged by type only because their messages
can embed the URL.
"""

import logging

import requests

from core.models import Alert, AlertLevel
from db.repository import Repository

logger = logging.getLogger(__name__)

WEBHOOK_SETTING_KEY = "teams_webhook_url"
ENABLED_SETTING_KEY = "teams_alerts_enabled"

_TEAMS_TIMEOUT_SECONDS = 10
_TEAMS_THEME_COLORS = {
    AlertLevel.INFO: "0076D7",
    AlertLevel.WARNING: "FF8C00",
    AlertLevel.CRITICAL: "FF0000",
}
_MAX_TITLE_LENGTH = 100
_MAX_BODY_LENGTH = 500


class AlertManager:
    """Creates, stores and dispatches alerts (in-app via the repository, and Teams)."""

    def __init__(self, repository: Repository, fallback_webhook_url: str = ""):
        """`fallback_webhook_url` is used only if no webhook is stored in the settings table."""
        self.repository = repository
        self._teams_webhook: str | None = None
        self._fallback_webhook_url = fallback_webhook_url
        # Why the last Teams send failed (exception type or HTTP status); never contains the URL.
        self.last_error: str | None = None

    def _get_teams_webhook(self) -> str | None:
        """Teams webhook URL from the settings table (falling back to the configured
        default), cached after the first successful fetch. None if not set.
        """
        if self._teams_webhook:
            return self._teams_webhook
        try:
            url = self.repository.get_setting(WEBHOOK_SETTING_KEY, "") or self._fallback_webhook_url
        except Exception as exc:  # noqa: BLE001 - alerting must never raise
            logger.error("Could not read Teams webhook setting: %s", type(exc).__name__)
            return None
        if not url:
            return None
        self._teams_webhook = url
        return url

    def clear_cache(self) -> None:
        """Forget the cached webhook so a changed setting is picked up."""
        self._teams_webhook = None

    def _teams_enabled(self) -> bool:
        """Teams sending is on unless the settings table explicitly disables it."""
        try:
            return self.repository.get_setting(ENABLED_SETTING_KEY, True) is not False
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not read Teams enabled setting: %s", type(exc).__name__)
            return True

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        body: str,
        structure_id: str | None = None,
    ) -> Alert:
        """Create an alert, dispatch it to Teams if configured, persist it, and return it.

        Title/body are truncated to the model limits (100/500 chars). Teams is
        attempted before the single save so channels_sent in the stored row
        is accurate. Never raises; failures are logged.
        """
        alert = Alert(
            level=level,
            title=title[:_MAX_TITLE_LENGTH],
            body=body[:_MAX_BODY_LENGTH],
            structure_id=structure_id,
        )

        channels = ["in_app"]
        if self._get_teams_webhook() and self._teams_enabled() and self._send_teams_alert(alert):
            channels.append("teams")
        alert.channels_sent = channels

        try:
            self.repository.save_alert(alert)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not save alert %s: %s", alert.alert_id, type(exc).__name__)
            alert.channels_sent = [c for c in channels if c != "in_app"]
        return alert

    def send_test_alert(self, webhook_url: str | None = None) -> bool:
        """Send a test message to Teams. It is not saved as an alert.

        Uses `webhook_url` if given (e.g. a value typed in Settings but not saved
        yet), else the configured webhook. Sends even if Teams alerts are
        disabled, since it is an explicit user action. On failure `last_error` says why.
        """
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert from Crude Oil Risk Manager",
            body="This is a test message.",
        )
        return self._send_teams_alert(alert, webhook_url=webhook_url)

    def _send_teams_alert(self, alert: Alert, webhook_url: str | None = None) -> bool:
        """POST the alert to the Teams incoming webhook as a MessageCard.

        `webhook_url` overrides the configured webhook. Returns True on HTTP 200,
        False (logged, never raised) otherwise.
        """
        self.last_error = None
        webhook = webhook_url or self._get_teams_webhook()
        if not webhook:
            self.last_error = "no webhook URL configured"
            return False

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": _TEAMS_THEME_COLORS[alert.level],
            "summary": alert.title,
            "sections": [
                {
                    "activityTitle": alert.title,
                    "activitySubtitle": f"Crude Oil Risk Manager — {alert.timestamp.isoformat()}",
                    "facts": [
                        {"name": "Level", "value": alert.level.value},
                        {"name": "Details", "value": alert.body},
                        {"name": "Structure", "value": alert.structure_id or "Portfolio"},
                    ],
                    "markdown": True,
                }
            ],
        }
        try:
            response = requests.post(webhook, json=payload, timeout=_TEAMS_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            logger.error("Teams alert failed: %s", type(exc).__name__)
            self.last_error = type(exc).__name__
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Teams alert failed unexpectedly: %s", type(exc).__name__)
            self.last_error = type(exc).__name__
            return False

        if response.status_code != 200:
            logger.error("Teams alert rejected with HTTP %s", response.status_code)
            self.last_error = f"Teams rejected the message (HTTP {response.status_code})"
            return False
        return True
