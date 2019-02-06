import collections
import json
import smtplib
from email.mime.text import MIMEText


class Notifier:

    settings = None
    smtp_server = None
    smtp_port = None
    smtp_user = None
    smtp_pass = None
    notification_email = None
    last_seen_ignore_queue = None
    logging = None

    def __init__(self, settings=None, logging=None):
        self.settings = settings
        self.logging = logging

        self.smtp_server = self.settings.config.get("notifier", "smtp_server")
        self.smtp_port = self.settings.config.getint("notifier", "smtp_port")
        self.smtp_user = self.settings.config.get("notifier", "smtp_user")
        self.smtp_pass = self.settings.config.get("notifier", "smtp_pass")
        self.notification_email = self.settings.config.get("notifier", "notification_email")

        self.last_seen_ignore_queue = collections.deque(maxlen=1000)

    def notify_on_outlier(self, doc=None, outlier=None):
        if outlier.get_summary() in self.last_seen_ignore_queue:
            self.logging.logger.debug("not notifying on outlier because it is still in the ignore queue - " + outlier.get_summary())
        else:
            self.last_seen_ignore_queue.append(outlier.get_summary())
            email_dict = dict()
            email_dict["subject"] = "Eagle Eye - outlier alert: " + outlier.get_summary()
            email_dict["body"] = str(outlier) + "\n\n\n========RAW EVENT========\n\n\n\n" + json.dumps(doc, sort_keys=True, indent=4)

            self.send_email(email_dict)

    def send_email(self, email_dict):
        try:
            msg = MIMEText(email_dict["body"])
            msg['Subject'] = email_dict["subject"]
            msg['From'] = self.smtp_user
            msg['To'] = self.notification_email

            s = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            s.login(self.smtp_user, self.smtp_pass)
            s.send_message(msg)
            s.quit()
        except Exception as ex:
            self.logging.logger.error("something went wrong sending notification e-mail: " + str(ex))
