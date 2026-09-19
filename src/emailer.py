from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

LOG = logging.getLogger(__name__)


def send(subject: str, html: str, text: str) -> bool:
    user, password, recipient = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASSWORD"), os.getenv("EMAIL_TO")
    if not all((user, password, recipient)):
        LOG.warning("Email skipped: EMAIL_USER, EMAIL_PASSWORD and EMAIL_TO are required")
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, recipient
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    host, port = os.getenv("SMTP_HOST", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    LOG.info("Email sent successfully")
    return True

