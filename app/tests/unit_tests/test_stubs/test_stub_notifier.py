from helpers.singletons import es


class TestStubNotifier:

    def __init__(self):
        self.list_email = list()
        self.default_notifier_methods = self._get_default_notifier_methods()
        self.apply_new_notifier()

    def apply_new_notifier(self):
        es.notifier.send_email = self.send_email

    def restore_notifier(self):
        es.notifier.send_email = self.default_notifier_methods['send_email']

    def _get_default_notifier_methods(self):
        return {
                'send_email': es.notifier.send_email
            }

    def send_email(self, email_dict):
        self.list_email.append(email_dict)

    def get_list_email(self):
        return self.list_email
