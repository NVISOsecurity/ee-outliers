import unittest

from tests.unit_tests.test_stubs.test_stub_notifier import TestStubNotifier
from tests.unit_tests.utils.test_settings import TestSettings




class TestNotifier(unittest.TestCase):

    def setUp(self):
        self.test_settings = TestSettings()

    def tearDown(self):
        self.test_settings.restore_default_configuration_path()

    def test_notify_on_outlier_correctly_create_email(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/notifications_test.conf")
        self.test_notifier = TestStubNotifier()

        # TODO

        self.test_notifier.restore_notifier()
        self.assertEqual(len(self.test_notifier.get_list_email()), 1)
