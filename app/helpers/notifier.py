import collections
import json
import smtplib
from email.mime.text import MIMEText
from helpers.logging import Logging

from typing import Dict, Deque, TYPE_CHECKING

if TYPE_CHECKING:
    from helpers.settings import Settings
    from helpers.outlier import Outlier

from configparser import NoOptionError


class Notifier:

    settings: 'Settings'
    smtp_server: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    notification_email: str
    last_seen_ignore_queue: Deque[str]
    # logging = None

    def __init__(self, settings: 'Settings', logging: Logging) -> None:
        self.settings = settings
        self.logging: Logging = logging

        self.smtp_server = self.settings.config.get("notifier", "smtp_server")
        self.smtp_port = self.settings.config.getint("notifier", "smtp_port")
        self.smtp_user = self.settings.config.get("notifier", "smtp_user")
        self.smtp_pass = self.settings.config.get("notifier", "smtp_pass")
        self.notification_email = self.settings.config.get("notifier", "notification_email")

        try:
            nr_max_que_len = settings.config.getint("notifier", "max_cache_ignore")
        except NoOptionError:
            nr_max_que_len = 1000

        self.last_seen_ignore_queue = collections.deque(maxlen=nr_max_que_len)

    def notify_on_outlier(self, outlier: 'Outlier') -> None:
        if outlier.outlier_dict["summary"] in self.last_seen_ignore_queue:
            self.logging.logger.debug("not notifying on outlier because it is still in the ignore queue - " +
                                      outlier.outlier_dict["summary"])
        else:
            self.last_seen_ignore_queue.append(outlier.outlier_dict["summary"])
            email_dict: Dict[str, str] = dict()
            email_dict["subject"] = "Eagle Eye - outlier alert: " + outlier.outlier_dict["summary"]
            email_dict["body"] = str(outlier) + "\n\n\n========RAW EVENT========\n\n\n\n" + json.dumps(outlier.doc,
                                                                                                       sort_keys=True,
                                                                                                       indent=4)
            self.send_email(email_dict)

    def send_email(self, email_dict: Dict[str, str]) -> None:
        try:
            msg: MIMEText = MIMEText(email_dict["body"])
            msg['Subject'] = email_dict["subject"]
            msg['From'] = self.smtp_user
            msg['To'] = self.notification_email

            s: smtplib.SMTP_SSL = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            s.login(self.smtp_user, self.smtp_pass)
            s.send_message(msg)
            s.quit()
        except Exception:
            self.logging.logger.error("something went wrong sending notification e-mail", exc_info=True)
