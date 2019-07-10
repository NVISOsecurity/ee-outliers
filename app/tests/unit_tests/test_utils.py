import unittest
from _ast import expr, excepthandler

import helpers.utils
from helpers.singletons import logging
from millify import millify
import numpy as np
import copy
import json
from statistics import mean, median

doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_outlier.json"))
doc_with_asset_edgecases = json.load(open("/app/tests/unit_tests/files/doc_with_asset_edgecases.json"))

test_file_modification_path = "/app/tests/unit_tests/files/file_modification_test.conf"

list_values_array = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 6, 7, 8, 9, 10],
    [3, 4, 2, 3, 4, 5, 6],
    [2, 4, 2, 3, 4, 5, 6, 7, 8, 9, 8],
    [1, 1, 2, 2, 4, 6, 9],
    [0, 1, 2, 3, 4, 7, 8, 9],
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
]
list_sensitivity = [10, 5, 1, 2]

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

    def test_flatten_dict_separator_in_field_name(self):
        test_dict = {'i.': {'j': 0}}
        test_dict_2 = {'i': {'.j': 0}}

        self.assertEqual(helpers.utils.flatten_dict(test_dict), helpers.utils.flatten_dict(test_dict_2))

    def test_flatten_dict_simple_case_1(self):
        test_dict = {'i.': {'j': 0}}
        test_dict_res = {'i..j': 0}

        self.assertEqual(helpers.utils.flatten_dict(test_dict), test_dict_res)

    def test_flatten_dict_simple_case_2(self):
        test_dict_res = {'i.j': 0}
        self.assertEqual(helpers.utils.flatten_dict(test_dict_res), test_dict_res)

    def test_flatten_dict_simple_case_3(self):
        test_dict_res = {'i': 'testing'}
        self.assertEqual(helpers.utils.flatten_dict(test_dict_res), test_dict_res)

    def test_flatten_dict_complex_case_1(self):
        test_dict = {'i': {'j': {'k': 'test'}}}
        test_dict_res = {'i.j.k': 'test'}
        self.assertEqual(helpers.utils.flatten_dict(test_dict), test_dict_res)

    def test_flatten_dict_complex_case_custom_separator(self):
        test_dict = {'i': {'j': {'k': 'test'}}}
        test_dict_res = {'i+j+k': 'test'}
        self.assertEqual(helpers.utils.flatten_dict(test_dict, sep="+"), test_dict_res)

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
        fields = {'intro': ['Intro 1', 'Intro 2'], 'event_type': 'test_event', 'source_ip': '8.8.8.8',
                  'ip_summary_legacy': ['Summary 1', 'Summary 2'], 'info': ['Info 1', 'Info 2']}

        sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=sentence_format)

        expected_sentences_length = 1
        for k, v in fields.items():
            if type(v) is list:
                expected_sentences_length = expected_sentences_length * len(v)

        self.assertEqual(len(sentences), expected_sentences_length)

    def test_replace_placeholder_fields_with_values_no_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has no placeholders",
                                                                   fields=None)
        self.assertEqual(res, "this one has no placeholders")

    def test_replace_placeholder_fields_with_values_single_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has {one} placeholders",
                                                                   fields=dict({"one": "hello"}))
        self.assertEqual(res, "this one has hello placeholders")

    def test_replace_placeholder_fields_with_values_two_matches(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="{one} {two}!",
                                                                   fields=dict({"one": "hello", "two": "world"}))
        self.assertEqual(res, "hello world!")

    def test_replace_placeholder_fields_with_values_case_insensitive_match(self):
        res = helpers.utils.replace_placeholder_fields_with_values(placeholder="this one has {OnE} case insensitive " +\
                                                                               "placeholders",
                                                                   fields=dict({"one": "hello", "two": "world"}))
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

    def test_is_hex_encoded_none(self):
        test_hex = None
        res = helpers.utils.is_hex_encoded(test_hex)
        self.assertEqual(res, False)

    def test_is_hex_encoded_valid_int(self):
        test_hex = "20"
        res = helpers.utils.is_hex_encoded(test_hex)
        self.assertEqual(res, "32")

    def test_is_hex_encoded_zero(self):
        test_hex = "0"
        res = helpers.utils.is_hex_encoded(test_hex)
        self.assertEqual(res, "0")

    def test_is_hex_encoded_invalid_str(self):
        test_hex = "test"
        res = helpers.utils.is_hex_encoded(test_hex)
        self.assertEqual(res, False)

    def test_is_url_valid_with_http(self):
        str_url = "http://nviso.be"
        self.assertTrue(helpers.utils.is_url(str_url))

    def test_is_url_valid_with_https(self):
        str_url = "https://nviso.be"
        self.assertTrue(helpers.utils.is_url(str_url))

    def test_is_url_valid_with_https_and_www(self):
        str_url = "https://www.nviso.be"
        self.assertTrue(helpers.utils.is_url(str_url))

    def test_is_url_valid_with_http_and_www(self):
        str_url = "http://www.nviso.be"
        self.assertTrue(helpers.utils.is_url(str_url))

    def test_is_url_not_valid_with_www(self):
        str_url = "www.nviso.be"
        self.assertFalse(helpers.utils.is_url(str_url))

    def test_is_url_not_valid_without_www_and_http(self):
        str_url = "nviso.be"
        self.assertFalse(helpers.utils.is_url(str_url))

    def test_is_url_not_valid_str(self):
        str_url = "test"
        self.assertFalse(helpers.utils.is_url(str_url))

    def test_is_url_not_valid_wit_http_and_port(self):
        str_url = "http://nviso.be:80"
        self.assertTrue(helpers.utils.is_url(str_url))


    def test_decision_frontier_not_exist(self):
        with self.assertRaises(ValueError):
            helpers.utils.get_decision_frontier("does_not_exist", [0, 1, 2], 2, "high")

    def test_decision_frontier_percentile(self):

        for values_array in list_values_array:
            # Test percentile - IMPORTANT - the values array is converted into a set before calculating the percentile!
            set_values_array = list(set(values_array))
            for sensitivity in list_sensitivity:

                expectedRes = np.percentile(set_values_array, sensitivity)

                if(expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("percentile", values_array, sensitivity)
                else:
                    res = helpers.utils.get_decision_frontier("percentile", values_array, sensitivity)
                self.assertEqual(res, expectedRes)


    def test_decision_frontier_pct_of_max_value(self):
        for values_array in list_values_array:
            max_values_array = max(values_array)
            for sensitivity in list_sensitivity:

                expectedRes = np.float64(max_values_array * (sensitivity / 100))
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("pct_of_max_value", values_array, sensitivity)
                else:
                    res = helpers.utils.get_decision_frontier("pct_of_max_value", values_array, sensitivity)
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_pct_of_median_value(self):
        for values_array in list_values_array:
            median_values_array = median(values_array)
            for sensitivity in list_sensitivity:

                expectedRes = np.float64(median_values_array * (sensitivity / 100))
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("pct_of_median_value", values_array, sensitivity)
                else:
                    res = helpers.utils.get_decision_frontier("pct_of_median_value", values_array, sensitivity)
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_pct_of_avg_value(self):
        for values_array in list_values_array:
            mean_values_array = mean(values_array)
            for sensitivity in list_sensitivity:

                expectedRes = np.float64(mean_values_array * (sensitivity / 100))
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("pct_of_avg_value", values_array, sensitivity)
                else:
                    res = helpers.utils.get_decision_frontier("pct_of_avg_value", values_array, sensitivity)
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_mad_low(self):
        for values_array in list_values_array:
            mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)
            for sensitivity in list_sensitivity:

                expectedRes = np.nanmedian(values_array) - sensitivity * mad
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "low")
                else:
                    res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "low")
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_mad_high(self):
        for values_array in list_values_array:
            mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)
            for sensitivity in list_sensitivity:

                expectedRes = np.nanmedian(values_array) + sensitivity * mad
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "high")
                else:
                    res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "high")
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_mad_zero(self):
        values_array = [1, 1]
        sensitivity = 10
        # Here mad = 0
        # mad formula:
        # mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)

        res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "low")
        # So use std:
        expected_value = np.nanmean(values_array) - sensitivity * np.std(values_array)
        self.assertEqual(res, expected_value)


        res = helpers.utils.get_decision_frontier("mad", values_array, sensitivity, "high")
        # So use std:
        expected_value = np.nanmean(values_array) + sensitivity * np.std(values_array)
        self.assertEqual(res, expected_value)

    def test_decision_frontier_madpos_low(self):
        for values_array in list_values_array:
            mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)
            for sensitivity in list_sensitivity:

                expectedResult = np.nanmedian(values_array) - sensitivity * mad
                expectedRes = np.float64(max([expectedResult, 0]))
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "low")
                else:
                    res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "low")
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_madpos_high(self):
        for values_array in list_values_array:
            mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)
            for sensitivity in list_sensitivity:

                expectedResult = np.nanmedian(values_array) + sensitivity * mad
                expectedRes = np.float64(max([expectedResult, 0]))
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "high")
                else:
                    res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "high")

                self.assertEqual(res, expectedRes)

    def test_decision_frontier_madpos_zero(self):
        values_array = [1, 1]
        sensitivity = 10
        # Here mad = 0
        # mad formula:
        # mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)

        res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "low")
        # So use std:
        expected_value = np.nanmean(values_array) - sensitivity * np.std(values_array)
        expected_value = np.float64(max([expected_value, 0]))
        self.assertEqual(res, expected_value)


        res = helpers.utils.get_decision_frontier("madpos", values_array, sensitivity, "high")
        # So use std:
        expected_value = np.nanmean(values_array) + sensitivity * np.std(values_array)
        expected_value = np.float64(max([expected_value, 0]))
        self.assertEqual(res, expected_value)


    def test_decision_frontier_stdev_low(self):
        for values_array in list_values_array:
            nanmean_values_array = np.nanmean(values_array)
            std_values_array = np.std(values_array)
            for sensitivity in list_sensitivity:

                expectedRes = nanmean_values_array - sensitivity * std_values_array
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("stdev", values_array, sensitivity, "low")
                else:
                    res = helpers.utils.get_decision_frontier("stdev", values_array, sensitivity, "low")

                self.assertEqual(res, expectedRes)

    def test_decision_frontier_stdev_high(self):
        for values_array in list_values_array:
            nanmean_values_array = np.nanmean(values_array)
            std_values_array = np.std(values_array)
            for sensitivity in list_sensitivity:
                expectedRes = nanmean_values_array + sensitivity * std_values_array
                if (expectedRes < 0):
                    with self.assertLogs(logging.logger, level='WARNING') as cm:
                        res = helpers.utils.get_decision_frontier("stdev", values_array, sensitivity, "high")
                else:
                    res = helpers.utils.get_decision_frontier("stdev", values_array, sensitivity, "high")
                self.assertEqual(res, expectedRes)

    def test_decision_frontier_stdev_not_valid(self):
        with self.assertRaises(ValueError):
            helpers.utils.get_decision_frontier("stdev", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], 10)

        with self.assertRaises(ValueError):
            helpers.utils.get_decision_frontier("stdev", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], 10, "test_not_valid")

    def test_decision_frontier_float(self):
        for sensitivity in list_sensitivity:
            res = helpers.utils.get_decision_frontier("float", [], sensitivity)
            self.assertEqual(res, np.float64(sensitivity))


    def test_is_outlier_high_invalid(self):
        term_value_count = 0
        decision_frontier = 1
        trigger_on = "high"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertFalse(res)

    def test_is_outlier_high_valid(self):
        term_value_count = 1
        decision_frontier = 0
        trigger_on = "high"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertTrue(res)

    def test_is_outlier_high_invalid_equals_high(self):
        term_value_count = 0
        decision_frontier = 0
        trigger_on = "high"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertFalse(res)

    def test_is_outlier_low_invalid(self):
        term_value_count = 1
        decision_frontier = 0
        trigger_on = "low"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertFalse(res)

    def test_is_outlier_low_valid(self):
        term_value_count = 0
        decision_frontier = 1
        trigger_on = "low"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertTrue(res)

    def test_is_outlier_high_invalid_equals_low(self):
        term_value_count = 0
        decision_frontier = 0
        trigger_on = "low"
        res = helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)
        self.assertFalse(res)

    def test_is_outlier_invalid_trigger(self):
        term_value_count = 1
        decision_frontier = 0
        trigger_on = "test"
        with self.assertRaises(ValueError):
            helpers.utils.is_outlier(term_value_count, decision_frontier, trigger_on)

    def test_extract_outlier_asset_information_simple_matching(self):
        from helpers.singletons import settings, es

        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        fields = es.extract_fields_from_document(orig_doc, extract_derived_fields=False)

        # test case for simple asset matching
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        self.assertIn("user: dummyuser", outlier_assets)
        self.assertIn("host: DUMMY-PC", outlier_assets)

    def test_extract_outlier_asset_information_list_values(self):
        from helpers.singletons import settings, es

        orig_doc = copy.deepcopy(doc_with_asset_edgecases)
        fields = es.extract_fields_from_document(orig_doc, extract_derived_fields=False)

        # test case for asset fields containing multiple values in an array
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        self.assertIn("user: dummyuser1", outlier_assets)  # test case for array assets
        self.assertIn("user: dummyuser2", outlier_assets)  # test case for array assets
        self.assertEqual(len(outlier_assets), 3)  # blank asset fields, such as the PC name in the JSON file, should
        # NOT be added as assets. Both IP and user should match, so 2 matches.

    def test_extract_outlier_asset_information_case_insensitive_value(self):
        from helpers.singletons import settings, es

        # test case for case insensitive asset matching
        orig_doc = copy.deepcopy(doc_with_outlier_test_file)
        fields = es.extract_fields_from_document(orig_doc, extract_derived_fields=False)
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

    def test_file_modification_not_modified(self):
        file_mod_watcher = helpers.watchers.FileModificationWatcher()
        file_mod_watcher.add_files([test_file_modification_path])

        self.assertFalse(file_mod_watcher.files_changed())

    def test_file_modification_modified(self):
        file_mod_watcher = helpers.watchers.FileModificationWatcher()
        file_mod_watcher.add_files([test_file_modification_path])

        with open(test_file_modification_path, 'r') as test_file:
            filecontent = test_file.read()

        filecontent.replace('schedule=0 0 * * *', 'schedule=1 0 * * *')

        with open(test_file_modification_path, 'w') as test_file:
            test_file.write(filecontent)

        self.assertTrue(file_mod_watcher.files_changed())

    def test_nested_dict_values_one_dict(self):
        test_dict ={'a': 1, 'b': 2}
        expectedList = [1, 2]
        for elem in helpers.utils.nested_dict_values(test_dict):
            expectedList.remove(elem)

        self.assertEqual(expectedList, [])

    def test_nested_dict_values_two_nested_dict(self):
        test_dict ={'a': 1, 'b': {'c': [2, 'x'], 'd': '3'}}
        expectedList = [1, [2, 'x'], '3']
        for elem in helpers.utils.nested_dict_values(test_dict):
            expectedList.remove(elem)

        self.assertEqual(expectedList, [])

    def test_nested_dict_values_three_nested_dict(self):
        test_dict ={'a': 1, 'b': {'c': 2, 'd': '3'}, 'e': {'f': [4, 'x'], 'g': 5, 'h': {'i': 6, 'j': 7}}}
        expectedList = [1, 2, '3', [4, 'x'], 5, 6, 7]
        for elem in helpers.utils.nested_dict_values(test_dict):
            expectedList.remove(elem)

        self.assertEqual(expectedList, [])

    def test_nested_dict_values_empty_dict(self):
        test_dict ={}
        for elem in helpers.utils.nested_dict_values(test_dict):
            raise AssertionError("Detect nested value which does not exist")
