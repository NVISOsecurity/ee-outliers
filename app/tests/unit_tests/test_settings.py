import unittest

from tests.unit_tests.utils.update_settings import UpdateSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

from helpers.outlier import Outlier

test_whitelist_single_literal_file = "/app/tests/unit_tests/files/whitelist_tests_03_with_general.conf"
test_whitelist_multiple_literal_file = "/app/tests/unit_tests/files/whitelist_tests_01_with_general.conf"


class TestSettings(unittest.TestCase):

    def setUp(self):
        self.test_settings = UpdateSettings()

    def tearDown(self):
        self.test_settings.restore_default_configuration_path()

    def test_whitelist_correctly_reload_after_update_config(self):
        self.test_settings.change_configuration_path(test_whitelist_single_literal_file)

        dummy_doc_gen = DummyDocumentsGenerate()
        doc = dummy_doc_gen.generate_document(create_outlier=True, outlier_observation="dummy observation",
                                              filename="osquery_get_all_processes_with_listening_conns.log")

        # With this configuration, outlier is not whitlisted
        self.assertFalse(Outlier.is_whitelisted_doc(doc))

        # Update configuration
        self.test_settings.change_configuration_path(test_whitelist_multiple_literal_file)
        # Now outlier is whitelisted
        self.assertTrue(Outlier.is_whitelisted_doc(doc))
