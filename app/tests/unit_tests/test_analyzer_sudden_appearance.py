import unittest

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from helpers.singletons import logging
from tests.unit_tests.utils.update_settings import UpdateSettings


class TestMetricsAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 1 # TODO

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

