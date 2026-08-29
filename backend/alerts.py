"""
Alert dispatch: email (SendGrid API, SMTP fallback) + user webhook.

The webhook is the universal adapter: Slack, Teams, PagerDuty, a relay board —
all take an HTTP POST. We send one boring, stable JSON shape and let the
receiver translate. Every attempt is logged to the alerts table (alerts.status),
because "did the customer get warned" is an auditable business question.

No paid API in the critical path (spec): SendGrid free tier or any SMTP server;
if neither is configured, email is skipped and the webhook still fires.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

log = logging.getLogger(__name__)

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alerts@acoustic-monitor.local")


def _render(event: dict) -> tuple[str, str]:
    subject = f"⚠ Machine anomaly: {event.get('machine', event.get('device_id', '?'))}"
    score, thr = event.get("score"), event.get("threshold")
    ratio = f"{score / thr:.1f}x its learned threshold" if score and thr else "above baseline"
    body = (
        f"Machine: {event.get('machine')}\n"
        f"Time (UTC epoch): {event.get('ts')}\n"
        f"Sound/vibration has been {ratio} for "
        f"{event.get('persisted_minutes', '?')} minutes continuously.\n\n"
        "This is an indicative anomaly notification, not a fault diagnosis.\n"
        "Recommended action: visually inspect the machine and listen for\n"
        "changes at the next safe opportunity.\n"
        "If the machine was doing something unusual but legitimate (cleaning\n"
        "cycle, product changeover), press 'This was normal' on the dashboard\n"
        "and the sensor will learn it.\n"
    )
    return subject, body


def send_email(to: str, event: dict) -> tuple[str, str]:
    """Returns (status, detail). SendGrid first, SMTP second, skip third."""
    subject, body = _render(event)
    if SENDGRID_API_KEY:
        try:
            payload = {
                "personalizations": [{"to": [{"email": to}]}],
                "from": {"email": FROM_EMAIL},
                "subject": subject,
                "content": [{"type": "text/plain", "value": body}],
            }
            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                json.dumps(payload).encode(),
                {"Authorization": f"Bearer {SENDGRID_API_KEY}",
                 "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            return "sent", "sendgrid"
        except Exception as e:                               # noqa: BLE001
            log.warning("sendgrid failed: %s — trying SMTP", e)
    if SMTP_HOST:
        try:
            msg = EmailMessage()
            msg["From"], msg["To"], msg["Subject"] = FROM_EMAIL, to, subject
            msg.set_content(body)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
                s.starttls()
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            return "sent", "smtp"
        except Exception as e:                               # noqa: BLE001
            return "failed", f"smtp: {e}"
    return "failed", "no email transport configured"


def send_webhook(url: str, event: dict) -> tuple[str, str]:
    try:
        req = urllib.request.Request(url, json.dumps(event).encode(),
                                     {"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        return "sent", f"http {resp.status}"
    except Exception as e:                                   # noqa: BLE001
        return "failed", str(e)


def dispatch(event: dict, config) -> list[dict]:
    """Fan out one anomaly event to all configured channels.
    `config` is an AlertConfig row (or None). Returns dispatch records for the
    alerts table."""
    records = []
    if config is None or not config.enabled:
        return records
    if config.email:
        status, detail = send_email(config.email, event)
        records.append({"channel": "email", "target": config.email,
                        "status": status, "detail": detail})
    if config.webhook_url:
        status, detail = send_webhook(config.webhook_url, event)
        records.append({"channel": "webhook", "target": config.webhook_url,
                        "status": status, "detail": detail})
    return records
