from helpers.singletons import es, settings, logging
from helpers.notifier import Notifier


class TestStubNotifier:

    def __init__(self):
        self.list_email = list()
        self.default_notifier_methods = self._get_default_notifier_methods()
        self.apply_new_notifier()

    def apply_new_notifier(self):
        if es.notifier is None:
            es.notifier = Notifier(settings, logging)
        es.notifier.send_email = self.send_email

    def restore_notifier(self):
        if self.default_notifier_methods is None:
            es.notifier = None
        else:
            es.notifier.send_email = self.default_notifier_methods['send_email']
        self.list_email = list()

    def _get_default_notifier_methods(self):
        if es.notifier is not None:
            return {
                    'send_email': es.notifier.send_email
                }
        else:
            return None

    def send_email(self, email_dict):
        self.list_email.append(email_dict)

    def get_list_email(self):
        return self.list_email
