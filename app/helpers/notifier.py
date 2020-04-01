import collections
import json
import smtplib
from email.mime.text import MIMEText

from configparser import NoOptionError


class Notifier:
    """
    Send e-mails containing outlier information
    """
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

        try:
            nr_max_que_len = settings.config.getint("notifier", "max_cache_ignore")
        except NoOptionError:
            nr_max_que_len = 1000

        self.last_seen_ignore_queue = collections.deque(maxlen=nr_max_que_len)

    def notify_on_outlier(self, outlier=None):
        """
        Checks if an outlier should be reported by e-mail or not, based on the queue content
        :param outlier: the outlier object to send by e-mail
        """
        if outlier.outlier_dict["summary"] in self.last_seen_ignore_queue:
            self.logging.logger.debug("not notifying on outlier because it is still in the ignore queue - " +
                                      outlier.outlier_dict["summary"])
        else:
            self.last_seen_ignore_queue.append(outlier.outlier_dict["summary"])
            email_dict = dict()
            email_dict["subject"] = "Eagle Eye - outlier alert: " + outlier.outlier_dict["summary"]
            email_dict["body"] = str(outlier) + "\n\n\n========RAW EVENT========\n\n\n\n" + json.dumps(outlier.doc,
                                                                                                       sort_keys=True,
                                                                                                       indent=4)

            self.send_email(email_dict)

    def send_email(self, email_dict):
        """
        Send a notification e-mail with the content of the dictionary provided as argument
        :param email_dict: contains all information for the e-mail including body, subject, from, recipient
        """
        try:
            msg = MIMEText(email_dict["body"])
            msg['Subject'] = email_dict["subject"]
            msg['From'] = self.smtp_user
            msg['To'] = self.notification_email

            s = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            s.login(self.smtp_user, self.smtp_pass)
            s.send_message(msg)
            s.quit()
        except Exception:
            self.logging.logger.error("something went wrong sending notification e-mail", exc_info=True)
