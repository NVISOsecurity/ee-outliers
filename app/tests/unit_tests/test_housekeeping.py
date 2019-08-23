import unittest

import json
import copy

from helpers.housekeeping import HousekeepingJob
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.update_settings import UpdateSettings

test_file_no_whitelist_path_config = "/app/tests/unit_tests/files/housekeeping_no_whitelist.conf"
test_file_whitelist_path_config = "/app/tests/unit_tests/files/whitelist_tests_01.conf"
doc_without_outlier_test_file = json.load(open(
    "/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_outlier.json"))


class TestHousekeeping(unittest.TestCase):

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()
        self.config_backup = dict()

    def tearDown(self):
        self.test_es.restore_es()
        self.test_settings.restore_default_configuration_path()

    def _backup_config(self, file_path):
        with open(file_path, 'r') as content_file:
            self.config_backup[file_path] = content_file.read()

    def _restore_config(self, file_path):
        if file_path in self.config_backup.keys():
            with open(file_path, 'w') as file_object:
                file_object.write(self.config_backup[file_path])
        else:
            raise KeyError('The configuration ' + file_path + ' was never backup')

    def test_housekeeping_correctly_remove_whitelisted_outlier_when_file_modification(self):
        self.test_settings.change_configuration_path(test_file_no_whitelist_path_config)
        self._backup_config(test_file_no_whitelist_path_config)
        housekeeping = HousekeepingJob()

        # Add document to "Database"
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)
        self.test_es.add_doc(doc_with_outlier)

        filecontent = ""
        with open(test_file_no_whitelist_path_config, 'r') as test_file:
            for line in test_file:
                if "# WHITELIST" in line:
                    break
                filecontent += line

        # Update configuration (read new config and append to default)
        with open(test_file_whitelist_path_config, 'r') as test_file:
            filecontent += test_file.read()

        with open(test_file_no_whitelist_path_config, 'w') as test_file:
            test_file.write(filecontent)

        housekeeping.execute_housekeeping()

        # Fetch result
        result = [elem for elem in self.test_es._scan()][0]

        # Compute expected result:
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        self._restore_config(test_file_no_whitelist_path_config)
        self.assertEqual(result, doc_without_outlier)

    def test_housekeeping_not_execute_no_whitelist_parameter_change(self):
        self.test_settings.change_configuration_path(test_file_no_whitelist_path_config)
        self._backup_config(test_file_no_whitelist_path_config)
        housekeeping = HousekeepingJob()

        # Add document to "Database"
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)
        self.test_es.add_doc(doc_with_outlier)

        # Update configuration (create new section and append to default)
        filecontent = "\n\n[dummy_section]\nparam=1"

        with open(test_file_no_whitelist_path_config, 'a') as test_file:
            test_file.write(filecontent)

        housekeeping.execute_housekeeping()

        # Fetch result
        result = [elem for elem in self.test_es._scan()][0]

        self._restore_config(test_file_no_whitelist_path_config)
        self.assertEqual(result, doc_with_outlier)
