import unittest
from configparser import NoSectionError, DuplicateOptionError, DuplicateSectionError

from tests.unit_tests.utils.update_settings import UpdateSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

from helpers.outlier import Outlier
from helpers.singletons import settings

from helpers.settings import print_failed_configs_and_exit as print_failed_configs_and_exit

test_whitelist_single_literal_file = "/app/tests/unit_tests/files/whitelist_tests_03_with_general.conf"
test_whitelist_multiple_literal_file = "/app/tests/unit_tests/files/whitelist_tests_01_with_general.conf"
test_whitelist_duplicate_option_file = "/app/tests/unit_tests/files/whitelist_tests_duplicate_keys.conf"
test_whitelist_duplicate_section_file = "/app/tests/unit_tests/files/whitelist_tests_duplicate_section.conf"
test_config_without_whitelist_file = "/app/tests/unit_tests/files/config_without_whitelist_tests.conf"
test_config_that_does_not_exist = "/app/tests/unit_tests/files/this_file_does_not_exist.conf"
test_config_that_is_a_directory = "/app/tests/unit_tests/files/"


class TestSettings(unittest.TestCase):

    def setUp(self):
        import logging as base_logging
        base_logging.disable(base_logging.CRITICAL)

        self.test_settings = UpdateSettings()

    def tearDown(self):
        import logging as base_logging
        base_logging.disable(base_logging.NOTSET)

        self.test_settings.restore_default_configuration_path()

    def test_whitelist_correctly_reload_after_update_config(self):
        self.test_settings.change_configuration_path(test_whitelist_single_literal_file)

        dummy_doc_gen = DummyDocumentsGenerate()
        doc = dummy_doc_gen.generate_document({"create_outlier": True, "outlier_observation": "dummy observation",
                                               "filename": "osquery_get_all_processes_with_listening_conns.log"})

        # With this configuration, outlier is not whitlisted
        self.assertFalse(Outlier.is_whitelisted_doc(doc))

        # Update configuration
        self.test_settings.change_configuration_path(test_whitelist_multiple_literal_file)
        # Now outlier is whitelisted
        self.assertTrue(Outlier.is_whitelisted_doc(doc))

    def test_duplicate_whitelist_keys_not_crash(self):
        self.test_settings.change_configuration_path(test_whitelist_duplicate_option_file)
        self.assertEqual(settings.config.get("whitelist_literals", "single_key"), "dummy_whitelist_item_two")

    def test_error_when_forgot_whitelist_config(self):
        with self.assertRaises(NoSectionError):
            self.test_settings.change_configuration_path(test_config_without_whitelist_file)

    def test_error_on_duplicate_key_check(self):
        self.test_settings.change_configuration_path(test_whitelist_duplicate_option_file)
        result = settings.check_no_duplicate_key()
        self.assertIsInstance(result, DuplicateOptionError)

    def test_error_on_duplicate_section_check(self):
        self.test_settings.change_configuration_path(test_whitelist_duplicate_section_file)
        result = settings.check_no_duplicate_key()
        self.assertIsInstance(result, DuplicateSectionError)

    # Test on process_configuration_files function
    def test_error_when_config_file_does_not_exist(self):
        with self.assertRaises(SystemExit) as cm:
            self.test_settings.change_configuration_path(test_config_that_does_not_exist)
        self.assertEqual(cm.exception.code, 2)

    # Test on process_configuration_files function
    def test_error_when_config_file_is_a_directory(self):
        with self.assertRaises(SystemExit) as cm:
            self.test_settings.change_configuration_path(test_config_that_is_a_directory)
        self.assertEqual(cm.exception.code, 2)

    # Test on check_no_failed_config_paths function
    def test_error_when_failed_config_file_exists_on_interactive_mode(self):
        with self.assertRaises(SystemExit) as cm:
            print_failed_configs_and_exit({test_config_that_does_not_exist})
        self.assertEqual(cm.exception.code, 2)

    def test_error_when_multiple_failed_config_files_exist(self):
        failed_config_files = {test_config_that_does_not_exist, test_config_that_is_a_directory}
        with self.assertRaises(SystemExit) as cm:
            print_failed_configs_and_exit(failed_config_files)
        self.assertEqual(cm.exception.code, 2)

    def test_no_exceptions_on_valid_config_file(self):
        try:
            self.test_settings.change_configuration_path(test_whitelist_multiple_literal_file)
        except Exception:
            self.fail("loading a valid configuration file raised an unexpected exception!")

    # Test on check_no_failed_config_paths function
    def test_error_when_no_failed_config_paths_exist(self):
        failed_config_files = {}
        raised = False
        try:
            print_failed_configs_and_exit(failed_config_files)
        except SystemExit:
            raised = True
        self.assertFalse(raised)

