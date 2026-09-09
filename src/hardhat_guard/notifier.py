"""Email alerts for confirmed violations.

The one rule this module obeys above all others: **a delivery failure never
stops monitoring**. An unreachable SMTP server, expired credentials or a
network outage are all logged and swallowed. A camera that stops watching the
floor because a mail server went down is a worse failure than a missed email.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

from hardhat_guard.config import settings
from hardhat_guard.rules import Violation

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10

_BODY = """\
A PPE violation was confirmed by automated monitoring.

Type      : {violation_type}
Camera    : {camera_id}
Location  : {location}
Time (UTC): {detected_at:%Y-%m-%d %H:%M:%S}
Confidence: {confidence:.0%}

The attached snapshot has been face-blurred. It documents a rule breach and is
not intended to identify an individual.
"""


class EmailNotifier:
    def __init__(
        self,
        enabled: bool | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        recipient: str | None = None,
    ) -> None:
        self.enabled = settings.notifications_enabled if enabled is None else enabled
        self._host = host or settings.smtp_host
        self._port = port or settings.smtp_port
        self._user = user or settings.smtp_user
        self._password = password or settings.smtp_password
        self._recipient = recipient or settings.alert_recipient

        if not self.enabled:
            logger.info("Email notifications are disabled")

    def notify(self, violation: Violation, snapshot_path: Path | None = None) -> bool:
        """Send one alert. Returns whether it was delivered.

        The return value exists for tests and metrics; callers in the pipeline
        deliberately ignore it, because there is nothing useful they could do.
        """
        if not self.enabled:
            return False

        message = self._build(violation, snapshot_path)
        try:
            # SMTP_SSL, not SMTP + starttls: port 465 is implicit TLS, so the
            # connection is encrypted before any credential crosses it.
            with smtplib.SMTP_SSL(self._host, self._port, timeout=_TIMEOUT_SECONDS) as smtp:
                smtp.login(self._user, self._password)
                smtp.send_message(message)
        except (smtplib.SMTPException, OSError):
            # OSError covers refused connections, DNS failures and timeouts.
            logger.exception("Could not send alert for %s on %s",
                             violation.violation_type, violation.camera_id)
            return False

        logger.info("Alert sent for %s on %s", violation.violation_type, violation.camera_id)
        return True

    def _build(self, violation: Violation, snapshot_path: Path | None) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = (
            f"[PPE] {violation.violation_type} - {violation.location or violation.camera_id}"
        )
        message["From"] = self._user
        message["To"] = self._recipient
        message.set_content(
            _BODY.format(
                violation_type=violation.violation_type,
                camera_id=violation.camera_id,
                location=violation.location or "-",
                detected_at=violation.detected_at,
                confidence=violation.confidence,
            )
        )

        if snapshot_path is not None and snapshot_path.exists():
            try:
                message.add_attachment(
                    snapshot_path.read_bytes(),
                    maintype="image",
                    subtype="jpeg",
                    filename=snapshot_path.name,
                )
            except OSError:
                # Send the alert without the image rather than not at all.
                logger.warning("Could not attach %s", snapshot_path, exc_info=True)

        return message
