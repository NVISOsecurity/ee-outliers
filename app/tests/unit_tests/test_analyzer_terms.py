import json
import unittest

import copy
import random

from collections import defaultdict
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.terms import TermsAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))
doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_terms_outlier_coeff_of_variation_no_score_sort = json.load(
    open("/app/tests/unit_tests/files/doc_with_terms_outlier_coeff_of_variation_no_score_sort.json"))

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

    @staticmethod
    def _preperate_data_terms():
        eval_terms_array = defaultdict()
        # "random" value
        aggregator_value = "key"
        target_value = "test"
        observations = {'a': 1, 'test': 'ok'}
        doc = {'source': 'this', 'target': 12}
        return eval_terms_array, aggregator_value, target_value, observations, doc

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
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_no_bucket")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        expected_doc = copy.deepcopy(doc_with_terms_outlier_coeff_of_variation_no_score_sort)
        # Add doc to the database
        self.test_es.add_doc(doc_without_outlier)

        # Make test (suppose that all doc match with the query)
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, expected_doc)

    def test_terms_generated_document_coeff_of_variation_not_respect_min(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_no_bucket")

        doc_generator = DummyDocumentsGenerate()
        nbr_val = 24  # Like 24 hours
        min_trigger_sensitivity = analyzer.model_settings["trigger_sensitivity"]
        default_value = 5  # Per default, 5 documents create per hour (arbitrarily)
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        all_doc = doc_generator.create_doc_uniq_target_variable_at_least_specific_coef_variation(
            nbr_val, min_trigger_sensitivity, max_difference, default_value)
        self.test_es.add_multiple_docs(all_doc)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for doc in es.scan():
            if "outliers" in doc['_source']:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 0)

    def test_terms_generated_document_coeff_of_variation_respect_min(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_no_bucket")

        doc_generator = DummyDocumentsGenerate()
        nbr_val = 24  # Like 24 hours
        max_trigger_sensitivity = analyzer.model_settings["trigger_sensitivity"]
        default_value = 5  # Per default, 5 documents create per hour (arbitrarily)
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        all_doc = doc_generator.create_doc_uniq_target_variable_at_most_specific_coef_variation(
            nbr_val, max_trigger_sensitivity, max_difference, default_value)
        self.test_es.add_multiple_docs(all_doc)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for doc in es.scan():
            if "outliers" in doc['_source']:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, len(all_doc))

    def test_add_term_to_batch_empty(self):
        eval_terms_array = defaultdict()
        aggregator_value = ""
        target_value = ""
        observations = {}
        doc = {}
        # Create expected result
        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["targets"] = [target_value]
        expected_eval_terms[aggregator_value]["observations"] = [{}]
        expected_eval_terms[aggregator_value]["raw_docs"] = [{}]

        self.assertEqual(TermsAnalyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                         doc), expected_eval_terms)

    def test_add_term_to_batch_no_modification(self):
        eval_terms_array, aggregator_value, target_value, observations, doc = self._preperate_data_terms()
        # Create expected result
        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["targets"] = [target_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]

        self.assertEqual(TermsAnalyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                         doc), expected_eval_terms)

    def test_add_term_to_batch_concerv_extra_value(self):
        eval_terms_array, aggregator_value, target_value, observations, doc = self._preperate_data_terms()
        # Add extra value:
        eval_terms_array["newKey"] = defaultdict(list)
        eval_terms_array["newKey2"] = "empty"
        eval_terms_array[aggregator_value] = defaultdict(list)
        eval_terms_array[aggregator_value]["targets"] = [target_value]
        eval_terms_array[aggregator_value]["test"] = 12
        # Create expected result
        expected_eval_terms = defaultdict()
        expected_eval_terms["newKey"] = defaultdict(list)
        expected_eval_terms["newKey2"] = "empty"
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["targets"] = [target_value, target_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]
        expected_eval_terms[aggregator_value]["test"] = 12

        self.assertEqual(TermsAnalyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                         doc), expected_eval_terms)
