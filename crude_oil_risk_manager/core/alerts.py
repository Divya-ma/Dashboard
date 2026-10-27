"""Alert management: persist alerts and dispatch them to Microsoft Teams.

AlertManager never raises. The Teams webhook URL is a secret: it is never
logged, and request exceptions are logged by type only because their messages
can embed the URL.

@MENTION. Teams sends a push to a phone for a post in a channel only when the user is @mentioned (or watches the channel's every
post). With an email saved in Settings the alert is sent as an ADAPTIVE CARD that @mentions that address; the dashboard's
older MessageCard cannot mention anyone. The mention can be limited to warnings and critical alerts, or to critical ones. If
the Teams workflow does not accept the Adaptive Card, the same alert is sent again as the MessageCard, so an alert is never
lost to the mention (the reason is in `mention_problem`, and Settings > Test shows it). Without an email nothing changes.
"""

import logging
from datetime import datetime, timezone

import requests

from core.models import Alert, AlertLevel, Structure, Trade
from core.structure_view import structure_live_price
from db.repository import Repository

logger = logging.getLogger(__name__)

WEBHOOK_SETTING_KEY = "teams_webhook_url"
ENABLED_SETTING_KEY = "teams_alerts_enabled"
MENTION_EMAIL_KEY = "teams_mention_email"
MENTION_NAME_KEY = "teams_mention_name"
MENTION_LEVELS_KEY = "teams_mention_levels"
MENTION_ALL, MENTION_WARNING, MENTION_CRITICAL = "all", "warning", "critical"
MENTION_LEVEL_CHOICES = {MENTION_ALL: "All alerts", MENTION_WARNING: "Warnings and critical alerts", MENTION_CRITICAL: "Critical alerts only"}

# Stop-loss / target alerts resend every this many seconds until stopped (see
# AlertManager.stop_repeating_alert) or until the price moves back off the level.
ALERT_REPEAT_SECONDS = 300

_TEAMS_TIMEOUT_SECONDS = 10
_TEAMS_THEME_COLORS = {
    AlertLevel.INFO: "0076D7",
    AlertLevel.WARNING: "FF8C00",
    AlertLevel.CRITICAL: "FF0000",
}
_MENTION_COVERS = {
    MENTION_ALL: {AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL},
    MENTION_WARNING: {AlertLevel.WARNING, AlertLevel.CRITICAL},
    MENTION_CRITICAL: {AlertLevel.CRITICAL},
}
_ADAPTIVE_COLORS = {AlertLevel.INFO: "Accent", AlertLevel.WARNING: "Warning", AlertLevel.CRITICAL: "Attention"}
_AUTO = object()  # "decide the mention from the saved settings and the alert's level"
_MAX_TITLE_LENGTH = 100
_MAX_BODY_LENGTH = 500


def mention_display_name(email: str, name: str | None = None) -> str:
    """The name shown in the mention: the one typed in Settings, else made from the email ("jane.doe@x.com" -> "Jane Doe")."""
    typed = (name or "").replace("<", "").replace(">", "").strip()
    if typed:
        return typed
    local = (email or "").split("@")[0].replace("<", "").replace(">", "")
    return " ".join(local.replace(".", " ").replace("_", " ").replace("-", " ").split()).title() or "You"


def message_card_payload(alert: Alert) -> dict:
    """The older MessageCard: no @mention is possible in it."""
    return {
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


def mention_payload(alert: Alert, email: str, name: str) -> dict:
    """The alert as an Adaptive Card in the envelope a Teams workflow posts ("Post card in a chat or channel"), with an @mention
    of `email` in its first line, which is what makes Teams push it to the phone."""
    tag = f"<at>{name}</at>"
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {"type": "TextBlock", "text": f"{tag} {alert.title}", "weight": "Bolder", "size": "Medium", "wrap": True, "color": _ADAPTIVE_COLORS[alert.level]},
            {"type": "TextBlock", "text": alert.body, "wrap": True},
            {"type": "FactSet", "facts": [
                {"title": "Level", "value": alert.level.value},
                {"title": "Structure", "value": alert.structure_id or "Portfolio"},
                {"title": "Time (UTC)", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")},
            ]},
        ],
        "msteams": {"entities": [{"type": "mention", "text": tag, "mentioned": {"id": email, "name": name}}]},
    }
    return {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "contentUrl": None, "content": card}]}


def _trade_alert_checks(trade: Trade, live: float) -> tuple[tuple[str, float | None, bool], ...]:
    """(kind, level, hit) for 'stop' and 'target'. A buy is stopped at or below its stop and
    hits its target at or above it; a sell is the reverse."""
    buy = trade.direction == "buy"
    stop, target = trade.stop_loss_price, trade.target_price
    return (
        ("stop", stop, stop is not None and (live <= stop if buy else live >= stop)),
        ("target", target, target is not None and (live >= target if buy else live <= target)),
    )


class AlertManager:
    """Creates, stores and dispatches alerts (in-app via the repository, and Teams)."""

    def __init__(self, repository: Repository, fallback_webhook_url: str = ""):
        """`fallback_webhook_url` is used only if no webhook is stored in the settings table."""
        self.repository = repository
        self._teams_webhook: str | None = None
        self._fallback_webhook_url = fallback_webhook_url
        # Why the last Teams send failed (exception type or HTTP status); never contains the URL.
        self.last_error: str | None = None
        # Which format the last successful send used ("mention" or "card"), and why an @mention was dropped (if it was).
        self.last_format: str | None = None
        self.mention_problem: str | None = None

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
        max_body_length: int = _MAX_BODY_LENGTH,
    ) -> Alert:
        """Create an alert, dispatch it to Teams if configured, persist it, and return it.

        Title/body are truncated to the model limits (100/500 chars). Teams is
        attempted before the single save so channels_sent in the stored row
        is accurate. Never raises; failures are logged.
        """
        alert = Alert(
            level=level,
            title=title[:_MAX_TITLE_LENGTH],
            body=body[:max_body_length],
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

    def check_price_alerts(
        self,
        live_prices: dict[str, float],
        open_trades: list[Trade],
        structures: dict[str, Structure],
    ) -> list[Alert]:
        """Fire stop-loss / target alerts for trades whose structure price has crossed a level.

        `live_prices` is {symbol: price}; the structure price is the composite of its legs'
        prices (None if a leg has no price, in which case that trade is skipped). A buy is
        stopped at or below its stop and hits its target at or above it; a sell is the reverse.

        Each trade + level that is hit keeps re-alerting every ALERT_REPEAT_SECONDS (5 minutes)
        until the user stops it (see stop_repeating_alert) — the last-sent timestamp lives in
        the settings table under `alert_last_sent_{trade_id}_{stop|target}` and the stopped flag
        under `alert_stopped_{trade_id}_{stop|target}`, so both survive a restart. The stop is
        permanent for that trade + level (it is NOT cleared just because the live price ticks
        back off the level momentarily — crude prices flicker right around a level, and clearing
        on every such tick would silently un-stop an alert the user just stopped). Returns the
        alerts raised this call (already sent and saved via send_alert). Never raises.
        """
        raised: list[Alert] = []
        now = datetime.now(timezone.utc)
        for trade in open_trades:
            if trade.stop_loss_price is None and trade.target_price is None:
                continue
            structure = structures.get(trade.structure_id)
            if structure is None:
                continue
            live = structure_live_price(structure, live_prices)
            if live is None:
                continue

            for kind, level, hit in _trade_alert_checks(trade, live):
                if level is None or not hit:
                    continue
                stopped_key = f"alert_stopped_{trade.trade_id}_{kind}"
                last_sent_key = f"alert_last_sent_{trade.trade_id}_{kind}"

                try:
                    if self.repository.get_setting(stopped_key, False):
                        continue
                    last_sent = self.repository.get_setting(last_sent_key, None)
                except Exception as exc:  # noqa: BLE001 - alerting must never raise
                    logger.error("Could not read alert state %s: %s", last_sent_key, type(exc).__name__)
                    continue
                if last_sent is not None:
                    elapsed = (now - datetime.fromisoformat(last_sent)).total_seconds()
                    if elapsed < ALERT_REPEAT_SECONDS:
                        continue

                if kind == "stop":
                    alert = self.send_alert(
                        AlertLevel.CRITICAL,
                        f"🛑 Stop Loss Hit: {structure.name}",
                        f"Live price {live:g} has crossed your stop loss at {level:g}. "
                        f"Direction: {trade.direction}. Review position immediately.",
                        structure.structure_id,
                    )
                else:
                    alert = self.send_alert(
                        AlertLevel.INFO,
                        f"🎯 Target Hit: {structure.name}",
                        f"Live price {live:g} has reached your target of {level:g}. Direction: {trade.direction}.",
                        structure.structure_id,
                    )
                try:
                    self.repository.set_setting(last_sent_key, now.isoformat())
                except Exception as exc:  # noqa: BLE001
                    logger.error("Could not save alert state %s: %s", last_sent_key, type(exc).__name__)
                raised.append(alert)
        return raised

    def stop_repeating_alert(self, trade_id: str) -> None:
        """Permanently stop resending stop-loss/target alerts for this trade (called from the
        "Stop Alerts" button in the structure detail view). Remembered in the settings table so
        it survives a restart; nothing clears it automatically (see check_price_alerts)."""
        for kind in ("stop", "target"):
            try:
                self.repository.set_setting(f"alert_stopped_{trade_id}_{kind}", True)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not save alert-stopped state for %s %s: %s", trade_id, kind, type(exc).__name__)

    def active_alert_kinds(self, trade: Trade, live: float | None) -> list[str]:
        """Which of 'stop'/'target' are currently hit for `trade` at `live` and still repeating
        (not yet stopped by the user). Used to show/hide the "Stop Alerts" button."""
        if live is None:
            return []
        active: list[str] = []
        for kind, level, hit in _trade_alert_checks(trade, live):
            if level is None or not hit:
                continue
            try:
                if self.repository.get_setting(f"alert_stopped_{trade.trade_id}_{kind}", False):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Could not read alert-stopped state for %s %s: %s", trade.trade_id, kind, type(exc).__name__
                )
                continue
            active.append(kind)
        return active

    def _mention_target(self, level: AlertLevel) -> tuple[str, str] | None:
        """(email, display name) to @mention for an alert of this level, or None (no email saved, or the level is not covered)."""
        try:
            email = (self.repository.get_setting(MENTION_EMAIL_KEY, "") or "").strip()
            if not email:
                return None
            covers = _MENTION_COVERS.get(self.repository.get_setting(MENTION_LEVELS_KEY, MENTION_ALL), _MENTION_COVERS[MENTION_ALL])
            if level not in covers:
                return None
            return email, mention_display_name(email, self.repository.get_setting(MENTION_NAME_KEY, ""))
        except Exception as exc:  # noqa: BLE001 - alerting must never raise
            logger.error("Could not read the Teams mention settings: %s", type(exc).__name__)
            return None

    def send_test_alert(self, webhook_url: str | None = None, mention_email: str | None = None, mention_name: str | None = None) -> bool:
        """Send a test message to Teams. It is not saved as an alert.

        Uses `webhook_url` if given (e.g. a value typed in Settings but not saved
        yet), else the configured webhook. Sends even if Teams alerts are
        disabled, since it is an explicit user action. With a mention email (typed, else the saved one) the test @mentions it
        whatever the alert levels chosen; `last_format` and `mention_problem` then say what happened. On failure `last_error` says why.
        """
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert from Crude Oil Risk Manager",
            body="This is a test message.",
        )
        email = (mention_email or "").strip()
        if email:
            mention = (email, mention_display_name(email, mention_name))
        else:
            try:
                saved = (self.repository.get_setting(MENTION_EMAIL_KEY, "") or "").strip()
                mention = (saved, mention_display_name(saved, self.repository.get_setting(MENTION_NAME_KEY, ""))) if saved else None
            except Exception:  # noqa: BLE001
                mention = None
        return self._send_teams_alert(alert, webhook_url=webhook_url, mention=mention)

    def _send_teams_alert(self, alert: Alert, webhook_url: str | None = None, mention=_AUTO) -> bool:
        """POST the alert to the Teams incoming webhook: as an Adaptive Card that @mentions the saved email when there is one
        for this level, else (or if that is not accepted) as a MessageCard.

        `webhook_url` overrides the configured webhook. Returns True on HTTP 200 / 202,
        False (logged, never raised) otherwise.
        """
        self.last_error = None
        self.last_format = None
        self.mention_problem = None
        webhook = webhook_url or self._get_teams_webhook()
        if not webhook:
            self.last_error = "no webhook URL configured"
            return False

        if mention is _AUTO:
            mention = self._mention_target(alert.level)
        if mention:
            if self._post(webhook, mention_payload(alert, *mention)):
                self.last_format = "mention"
                return True
            self.mention_problem = self.last_error
            logger.warning("Teams did not accept the @mention format (%s); sending the alert without it", self.mention_problem)
            self.last_error = None
        if self._post(webhook, message_card_payload(alert)):
            self.last_format = "card"
            return True
        return False

    def _post(self, webhook: str, payload: dict) -> bool:
        """One POST to the webhook; sets `last_error` and returns False if it failed."""
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

        if response.status_code not in (200, 202):
            logger.error("Teams alert rejected with HTTP %s", response.status_code)
            self.last_error = f"Teams rejected the message (HTTP {response.status_code})"
            return False
        return True
