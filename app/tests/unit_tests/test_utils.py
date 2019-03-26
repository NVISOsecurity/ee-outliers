import unittest
import helpers.utils
from millify import millify
import numpy as np
import copy
import json

doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_outlier.json"))
doc_with_asset_edgecases = json.load(open("/app/tests/unit_tests/files/doc_with_asset_edgecases.json"))


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_ip_ranges(self):
        if not helpers.utils.match_ip_ranges("127.0.0.1", ["127.0.0.0/24"]):
            raise AssertionError("Error matching IP ranges!")

        if not helpers.utils.match_ip_ranges("127.0.0.1", ["127.0.0.1/32"]):
            raise AssertionError("Error matching IP ranges!")

        if helpers.utils.match_ip_ranges("127.0.0.1", ["192.0.0.1/16"]):
            raise AssertionError("Error matching IP ranges!")

    def test_shannon_entropy_dummy(self):
        _str = "dummy"
        entropy = helpers.utils.shannon_entropy(_str)
        self.assertAlmostEqual(entropy, 1.921928094887)

    def test_shannon_entropy_empty_string(self):
        _str = ""
        entropy = helpers.utils.shannon_entropy(_str)
        self.assertAlmostEqual(entropy, 0)

    def test_millify_25k(self):
        mill_str = millify(25000)
        self.assertEqual(mill_str, "25k")

    def test_millify_1000k(self):
        # This is a bug in the library!!
        mill_str = millify(999999)
        self.assertEqual(mill_str, "1000k")

    def test_millify_1M(self):
        mill_str = millify(1000000)
        self.assertEqual(mill_str, "1M")

    def test_flatten_sentence_two_elements(self):
        test_sentence = ["test", "123"]
        self.assertEqual("test - 123", helpers.utils.flatten_sentence(test_sentence))

    def test_flatten_sentence_complex_nested_list(self):
        test_sentence = ["test", ["123", "234"]]  # Too complex, we don't flatten this, there is no reasonable way
        self.assertEqual(None, helpers.utils.flatten_sentence(test_sentence))

    def test_flatten_single_number(self):
        test_sentence = 1
        self.assertEqual("1", helpers.utils.flatten_sentence(test_sentence))

    def test_flatten_list_of_two_numbers(self):
        test_sentence = [1, 2]
        self.assertEqual("1 - 2", helpers.utils.flatten_sentence(test_sentence))

    def test_flatten_none(self):
        test_sentence = None
        self.assertEqual(None, helpers.utils.flatten_sentence(test_sentence))

    def test_flatten_test_string(self):
        test_sentence = "test"
        self.assertEqual("test", helpers.utils.flatten_sentence(test_sentence))

    # fields: {hostname: [WIN-DRA, WIN-EVB], draman}
    # output: [[WIN-DRA, draman], [WIN-EVB, draman]]
    def test_flatten_fields_into_sentences_1(self):
        sentence_format = ["hostname", "username"]
        fields = dict({"hostname": ["WIN-A", "WIN-B"], "username": "draman"})

        res = helpers.utils.flatten_fields_into_sentences(fields, sentence_format)

        expected_res = [['WIN-A', 'draman'], ['WIN-B', 'draman']]
        self.assertEqual(res, expected_res)

    def test_flatten_fields_into_sentences_2(self):
        sentence_format = ["hostname", "username"]
        fields = dict({"hostname": ["WIN-A", "WIN-B"], "username": ["evb", "draman"]})
        res = helpers.utils.flatten_fields_into_sentences(fields, sentence_format)

        expected_res = [['WIN-A', 'evb'], ['WIN-B', 'evb'], ['WIN-A', 'draman'], ['WIN-B', 'draman']]
        self.assertEqual(res, expected_res)

    def test_flatten_fields_into_sentences_3(self):
        sentence_format = ["hostname", "username"]
        fields = dict({"hostname": ["WIN-A", "WIN-A"], "username": ["evb", "draman"]})
        res = helpers.utils.flatten_fields_into_sentences(fields, sentence_format)

        # In our implementation, duplicates are allowed!
        expected_res = [['WIN-A', 'evb'], ['WIN-A', 'evb'], ['WIN-A', 'draman'], ['WIN-A', 'draman']]
        self.assertEqual(res, expected_res)

    def test_flatten_fields_into_sentences_complex(self):
        # More complex example
        sentence_format = ["intro", "event_type", "source_ip", "ip_summary_legacy", "info"]
        fields = {'intro': ['Intro 1', 'Intro 2'], 'event_type': 'test_event', 'source_ip': '8.8.8.8', 'ip_summary_legacy': ['Summary 1', 'Summary 2'], 'info': ['Info 1', 'Info 2']}

        sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=sentence_format)

        expected_sentences_length = 1
        for k, v in fields.items():
            if type(v) is list:
                expected_sentences_length = expected_sentences_length * len(v)

        self.assertEqual(len(sentences), expected_sentences_length)

    def test_replace_placeholder_fields_with_values_no_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has no placeholders", fields=None)
        self.assertEqual(res, "this one has no placeholders")

    def test_replace_placeholder_fields_with_values_single_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has {one} placeholders", fields=dict({"one": "hello"}))
        self.assertEqual(res, "this one has hello placeholders")

    def test_replace_placeholder_fields_with_values_two_matches(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="{one} {two}!", fields=dict({"one": "hello", "two": "world"}))
        self.assertEqual(res, "hello world!")

    def test_replace_placeholder_fields_with_values_case_insensitive_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has {OnE} case insensitive placeholders", fields=dict({"one": "hello", "two": "world"}))

        self.assertEqual(res, "this one has hello case insensitive placeholders")

    def test_is_base64_encoded_none(self):
        test_str = None
        res = helpers.utils.is_base64_encoded(test_str)
        self.assertEqual(res, False)

    def test_is_base64_encoded_hello_world(self):
        test_str = "hello world"
        res = helpers.utils.is_base64_encoded(test_str)
        self.assertEqual(res, False)

    def test_is_base64_encoded_actual_encoded_string(self):
        test_str = "QVlCQUJUVQ=="
        res = helpers.utils.is_base64_encoded(test_str)
        self.assertEqual(res, "AYBABTU")

    def test_is_base64_encoded_empty_string(self):
        test_str = ""
        res = helpers.utils.is_base64_encoded(test_str)
        self.assertEqual(res, "")

    def test_decision_frontier_percentile(self):
        with self.assertRaises(ValueError):
            helpers.utils.get_decision_frontier("does_not_exist", [0, 1, 2], 2, "high")

        # Test percentile - IMPORTANT - the values array is converted into a set before calculating the percentile!
        res = helpers.utils.get_decision_frontier("percentile", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10)
        self.assertEqual(res, 1)

    def test_decision_frontier_mad_1(self):
        # Test MAD
        res = helpers.utils.get_decision_frontier("mad", [1, 1, 2, 2, 4, 6, 9], 1, "high")  # MAD should be 2

        median = np.nanmedian([1, 1, 2, 2, 4, 6, 9])
        sensitivity = 1
        mad = 1
        self.assertEqual(median + sensitivity * mad, res)  # 1 = sensitivity, 1 = MAD, median = 2

    def test_decision_mad_2(self):
        res = helpers.utils.get_decision_frontier("mad", [1, 1, 2, 2, 4, 6, 9], 2, "high")  # MAD should be 4

        median = np.nanmedian([1, 1, 2, 2, 4, 6, 9])
        sensitivity = 2
        mad = 1
        self.assertEqual(median + sensitivity * mad, res)  # 2 = sensitivity, 1 = MAD, median = 2

    def test_extract_outlier_asset_information_simple_matching(self):
        from helpers.singletons import settings, es

        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        fields = es.extract_fields_from_document(orig_doc)

        # test case for simple asset matching
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        self.assertIn("user: dummyuser", outlier_assets)
        self.assertIn("host: DUMMY-PC", outlier_assets)

    def test_extract_outlier_asset_information_list_values(self):
        from helpers.singletons import settings, es

        orig_doc = copy.deepcopy(doc_with_asset_edgecases)
        fields = es.extract_fields_from_document(orig_doc)

        # test case for asset fields containing multiple values in an array
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        self.assertIn("user: dummyuser1, dummyuser2", outlier_assets)  # test case for array assets
        self.assertEqual(len(outlier_assets), 2)  # blank asset fields, such as the PC name in the JSON file, should NOT be added as assets. Both IP and user should match, so 2 matches.

    def test_extract_outlier_asset_information_case_insensitive_value(self):
        from helpers.singletons import settings, es

        # test case for case insensitive asset matching
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        fields = es.extract_fields_from_document(orig_doc)
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        self.assertIn("ip: 192.168.67.175", outlier_assets)

    def test_dict_contains_dotkey_case_sensitive_matches_full_dictionary_match(self):
        # case sensitive key matching - match
        test_key = "_source.OsqueryFilter.total_size"
        self.assertTrue(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=True))

    def test_dict_contains_dotkey_case_sensitive_matches_partial_dictionary_match(self):
        # case sensitive key matching - match
        test_key = "_source.OsqueryFilter"
        self.assertTrue(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=True))

    def test_dict_contains_dotkey_case_sensitive_mismatches(self):
        # case sensitive key matching - mismatch
        test_key = "_source.Osqueryfilter.total_size"
        self.assertFalse(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=True))

    def test_dict_contains_dotkey_case_insensitive_matches_full_match(self):
        # case insensitive key matching - match
        test_key = "_source.OsqueryFilter.total_size"
        self.assertTrue(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=False))

    def test_dict_contains_dotkey_case_insensitive_matches_lots_of_case_changes_match(self):
        # case insensitive key matching - match
        test_key = "_sOurCe.OsqueryfIltEr.TotAl_Size"
        self.assertTrue(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=False))

    def test_dict_contains_dotkey_case_insensitive_mismatches_first_element(self):
        # case insensitive key matching - mismatch
        test_key = "_sourceS.OsqueryFilter.total_size"
        self.assertFalse(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=False))

    def test_dict_contains_dotkey_case_insensitive_mismatches_second_element(self):
        # case insensitive key matching - mismatch
        test_key = "_source.OsqueryFilterZ.total_size"
        self.assertFalse(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=False))

    def test_dict_contains_dotkey_case_insensitive_mismatches_first_and_only_element(self):
        # case insensitive key matching - mismatch
        test_key = "_sOurCez"
        self.assertFalse(helpers.utils.dict_contains_dotkey(doc_with_asset_edgecases, test_key, case_sensitive=False))
