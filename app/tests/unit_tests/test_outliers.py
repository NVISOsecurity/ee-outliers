import json
import unittest

import copy

import helpers.es
import helpers.logging
from helpers.outlier import Outlier

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_outlier.json"))
doc_with_two_outliers_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_two_outliers.json"))
doc_with_three_outliers_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_three_outliers.json"))
doc_for_whitelist_testing_file = json.load(open("/app/tests/unit_tests/files/doc_for_whitelist_testing.json"))


class TestOutlierOperations(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_outlier_to_doc(self):
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")
        test_outlier.outlier_dict["observation"] = "dummy observation"

        doc_with_outlier = helpers.es.add_outlier_to_document(doc_without_outlier_test_file, test_outlier)
        self.assertDictEqual(doc_with_outlier_test_file, doc_with_outlier)

    def test_remove_outlier_from_doc(self):
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")
        test_outlier.outlier_dict["observation"] = "dummy observation"

        doc_with_outlier = helpers.es.add_outlier_to_document(doc_without_outlier_test_file, test_outlier)

        doc_without_outlier = helpers.es.remove_outliers_from_document(doc_with_outlier)
        self.assertDictEqual(doc_without_outlier, doc_without_outlier_test_file)

    def test_add_duplicate_outlier_to_doc(self):
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")

        doc = copy.deepcopy(doc_without_outlier_test_file)

        doc_with_outlier = helpers.es.add_outlier_to_document(doc, test_outlier)
        doc_with_outlier = helpers.es.add_outlier_to_document(doc_with_outlier, test_outlier)

        self.assertDictEqual(doc, doc_with_outlier)

    def test_add_two_outliers_to_doc(self):
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")
        test_outlier.outlier_dict["observation"] = "dummy observation"

        test_outlier_2 = Outlier(outlier_type="dummy type 2", outlier_reason="dummy reason 2", outlier_summary="dummy summary 2")
        test_outlier_2.outlier_dict["observation_2"] = "dummy observation 2"

        doc = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = helpers.es.add_outlier_to_document(doc, test_outlier)
        doc_with_two_outliers = helpers.es.add_outlier_to_document(doc_with_outlier, test_outlier_2)

        self.assertDictEqual(doc_with_two_outliers, doc_with_two_outliers_test_file)

    def test_add_three_outliers_to_doc(self):
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")
        test_outlier.outlier_dict["observation"] = "dummy observation"

        test_outlier_2 = Outlier(outlier_type="dummy type 2", outlier_reason="dummy reason 2", outlier_summary="dummy summary 2")
        test_outlier_2.outlier_dict["observation_2"] = "dummy observation 2"

        test_outlier_3 = Outlier(outlier_type="dummy type 3", outlier_reason="dummy reason 3", outlier_summary="dummy summary 3")
        test_outlier_3.outlier_dict["observation_3"] = "dummy observation 3"

        doc = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = helpers.es.add_outlier_to_document(doc, test_outlier)
        doc_with_two_outliers = helpers.es.add_outlier_to_document(doc_with_outlier, test_outlier_2)
        doc_with_three_outliers = helpers.es.add_outlier_to_document(doc_with_two_outliers, test_outlier_3)

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
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")

        result = test_outlier.matches_specific_whitelist_item(whitelist_item, "literal", additional_dict_values_to_check=doc_for_whitelist_testing_file)
        self.assertTrue(result)

    def test_whitelist_literal_mismatch(self):
        whitelist_item = r"C:\Windows\system32\msfeedssync.exe syncWRONG"
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")

        result = test_outlier.matches_specific_whitelist_item(whitelist_item, "literal", additional_dict_values_to_check=doc_for_whitelist_testing_file)
        self.assertFalse(result)

    def test_whitelist_regexp_match(self):
        whitelist_item = r"^.*.exe sync$"
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")

        result = test_outlier.matches_specific_whitelist_item(whitelist_item, "regexp", additional_dict_values_to_check=doc_for_whitelist_testing_file)
        self.assertTrue(result)

    def test_whitelist_regexp_mismatch(self):
        whitelist_item = r"^.*.exeZZZZZ sync$"
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason", outlier_summary="dummy summary")

        result = test_outlier.matches_specific_whitelist_item(whitelist_item, "regexp", additional_dict_values_to_check=doc_for_whitelist_testing_file)
        self.assertFalse(result)
