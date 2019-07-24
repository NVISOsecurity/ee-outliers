import json
import unittest

import copy
import random

from collections import defaultdict
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.terms import TermsAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))
doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_beaconing_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_beaconing_outlier.json"))
doc_with_beaconing_outlier_without_score_sort_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_with_beaconing_outlier_without_score_sort.json"))

LIST_AGGREGATOR_VALUE = ["agg-WIN-EVB-draman", "agg-WIN-DRA-draman"]
LIST_TARGET_VALUE = ["WIN-DRA-draman", "WIN-EVB-draman", "LINUX-DRA-draman"]
LIST_DOC = [doc_without_outlier_test_file]


class TestTermsAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = TestSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def _test_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_with_whitelist.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

        # Whitelisted (ignored)
        doc1_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_01_test_file)
        self.test_es.add_doc(doc1_without_outlier)
        # Not whitelisted (add)
        doc2_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_02_test_file)
        self.test_es.add_doc(doc2_without_outlier)
        # Not whitelisted
        doc3_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_03_test_file)
        self.test_es.add_doc(doc3_without_outlier)

        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 2)

    def test_evaluate_batch_for_outliers_not_enough_target_buckets_one_doc_max_two(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
        self.assertEqual(result, [])

    def test_evaluate_batch_for_outliers_limit_target_buckets_two_doc_max_two(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

        # Create one document with one aggregator
        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
        # Create a second document with another aggregator
        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value2 = random.choice(LIST_TARGET_VALUE)
        observations2 = {}
        doc2 = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value2, observations2,
                                                      doc2)

        # Expect to get nothing due to "min_target_buckets" set to 2
        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
        self.assertEqual(result, [])

    # coeff_of_variation
    def test_terms_evaluate_coeff_of_variation_like_expected_document(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = TermsAnalyzer("terms_beaconing_dummy_test")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        expected_doc = copy.deepcopy(doc_with_beaconing_outlier_without_score_sort_test_file)
        # Add doc to the database
        self.test_es.add_doc(doc_without_outlier)

        # Make test (suppose that all doc match with the query)
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, expected_doc)
