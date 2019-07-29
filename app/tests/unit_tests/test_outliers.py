import json
import unittest

import copy
import re

import helpers.es
from helpers.outlier import Outlier
from helpers.singletons import es
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.test_settings import TestSettings

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_outlier.json"))
doc_with_two_outliers_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_two_outliers.json"))
doc_with_three_outliers_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_three_outliers.json"))
doc_for_whitelist_testing_file = json.load(open("/app/tests/unit_tests/files/doc_for_whitelist_testing.json"))

nested_doc_for_whitelist_test = {'169.254.184.188', 'fe80::491a:881a:b1bf:b539', 2, 1, '1535026336',
                                 '1535017696_osquery_get_all_scheduled_tasks.log',
                                 'User_Feed_Synchronization-{9CD0CFAD-350E-46BA-8338-932284EF7332}', None,
                                 'OsqueryFilter', 'get_all_scheduled_tasks', 'Dummy Workstations',
                                 'osquery_get_all_scheduled_tasks.log', "['user:jvanderzweep', 'host:DUMMY-WIN10-JVZ']",
                                 ' C:\\Windows\\system32\\msfeedssync.exe sync',
                                 '\\User_Feed_Synchronization-{9CD0CFAD-350E-46BA-8338-932284EF7332}'}


class TestOutlierOperations(unittest.TestCase):
    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = TestSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def test_add_outlier_to_doc(self):
        doc = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=doc)
        test_outlier.outlier_dict["observation"] = "dummy observation"

        doc_with_outlier = helpers.es.add_outlier_to_document(test_outlier)
        self.assertDictEqual(doc_with_outlier_test_file, doc_with_outlier)

    def test_remove_outlier_from_doc(self):
        doc = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=doc)
        test_outlier.outlier_dict["observation"] = "dummy observation"

        doc_with_outlier = helpers.es.add_outlier_to_document(test_outlier)

        doc_without_outlier = helpers.es.remove_outliers_from_document(doc_with_outlier)
        self.assertDictEqual(doc_without_outlier, doc_without_outlier_test_file)

    def test_add_duplicate_outlier_to_doc(self):
        doc = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=doc)

        doc_with_outlier = helpers.es.add_outlier_to_document(test_outlier)
        doc_with_outlier = helpers.es.add_outlier_to_document(test_outlier)

        self.assertDictEqual(doc, doc_with_outlier)

    def test_add_two_outliers_to_doc(self):
        doc = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type",  outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=doc)
        test_outlier.outlier_dict["observation"] = "dummy observation"

        test_outlier_2 = Outlier(outlier_type="dummy type 2", outlier_reason="dummy reason 2",
                                 outlier_summary="dummy summary 2", doc=doc)
        test_outlier_2.outlier_dict["observation_2"] = "dummy observation 2"

        helpers.es.add_outlier_to_document(test_outlier)
        doc_with_two_outliers = helpers.es.add_outlier_to_document(test_outlier_2)

        self.assertDictEqual(doc_with_two_outliers, doc_with_two_outliers_test_file)

    def test_add_three_outliers_to_doc(self):
        doc = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=doc)
        test_outlier.outlier_dict["observation"] = "dummy observation"

        test_outlier_2 = Outlier(outlier_type="dummy type 2", outlier_reason="dummy reason 2",
                                 outlier_summary="dummy summary 2", doc=doc)
        test_outlier_2.outlier_dict["observation_2"] = "dummy observation 2"

        test_outlier_3 = Outlier(outlier_type="dummy type 3", outlier_reason="dummy reason 3",
                                 outlier_summary="dummy summary 3", doc=doc)
        test_outlier_3.outlier_dict["observation_3"] = "dummy observation 3"

        helpers.es.add_outlier_to_document(test_outlier)
        helpers.es.add_outlier_to_document(test_outlier_2)
        doc_with_three_outliers = helpers.es.add_outlier_to_document(test_outlier_3)

        self.assertDictEqual(doc_with_three_outliers, doc_with_three_outliers_test_file)

    def test_add_remove_tag_from_doc(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)

        # Remove non-existing tag
        doc = helpers.es.remove_tag_from_document(orig_doc, "tag_does_not_exist")
        self.assertDictEqual(doc, orig_doc)

        # Remove existing tag
        doc = helpers.es.remove_tag_from_document(orig_doc, "outlier")

        if "outlier" in doc["_source"]["tags"]:
            raise AssertionError("Tag still present in document, even after removal!")

    def test_whitelist_literal_match(self):
        whitelist_item = r"C:\Windows\system32\msfeedssync.exe sync"
        result = Outlier.dictionary_matches_specific_whitelist_item_literally(whitelist_item,
                                                                              nested_doc_for_whitelist_test)
        self.assertTrue(result)

    def test_whitelist_literal_mismatch(self):
        whitelist_item = r"C:\Windows\system32\msfeedssync.exe syncWRONG"
        result = Outlier.dictionary_matches_specific_whitelist_item_literally(whitelist_item,
                                                                              nested_doc_for_whitelist_test)
        self.assertFalse(result)

    def test_whitelist_regexp_match(self):
        whitelist_item = r"^.*.exe sync$"
        p = re.compile(whitelist_item.strip(), re.IGNORECASE)
        result = Outlier.dictionary_matches_specific_whitelist_item_regexp(p, nested_doc_for_whitelist_test)
        self.assertTrue(result)

    def test_whitelist_regexp_mismatch(self):
        whitelist_item = r"^.*.exeZZZZZ sync$"
        p = re.compile(whitelist_item.strip(), re.IGNORECASE)
        result = Outlier.dictionary_matches_specific_whitelist_item_regexp(p, nested_doc_for_whitelist_test)
        self.assertFalse(result)

    def test_whitelist_config_file_multi_item_match(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=orig_doc)

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_01.conf")
        self.assertTrue(test_outlier.is_whitelisted())

    def test_single_literal_to_match_in_doc_with_outlier(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=orig_doc)

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_02.conf")
        self.assertTrue(test_outlier.is_whitelisted())

    def test_single_literal_not_to_match_in_doc_with_outlier(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=orig_doc)

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_03.conf")
        self.assertFalse(test_outlier.is_whitelisted())

    def test_whitelist_config_file_multi_item_match_with_three_fields_and_whitespace(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=orig_doc)

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_04.conf")
        self.assertTrue(test_outlier.is_whitelisted())

    def test_whitelist_config_file_multi_item_mismatch_with_three_fields_and_whitespace(self):
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary", doc=orig_doc)

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_05.conf")
        self.assertFalse(test_outlier.is_whitelisted())

    def test_whitelist_config_change_remove_multi_item_literal(self):
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        self.test_es.add_doc(doc_with_outlier)
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_01_with_general.conf")
        es.remove_all_whitelisted_outliers()
        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, doc_without_outlier)

    def test_whitelist_config_change_single_literal_not_to_match_in_doc_with_outlier(self):
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)
        self.test_es.add_doc(doc_with_outlier)
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/whitelist_tests_03_with_general.conf")
        es.remove_all_whitelisted_outliers()
        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, doc_with_outlier)
