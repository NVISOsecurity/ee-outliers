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

DEFAULT_OUTLIERS_KEY_FIELDS = ["type", "reason", "summary", "model_name", "model_type", "total_outliers"]
EXTRA_OUTLIERS_KEY_FIELDS = ["term_count", "non_outlier_values_sample", "aggregator", "term", "decision_frontier",
                             "trigger_method"]


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

    def test_terms_whitelist_work_test_es_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        command_query = "SELECT * FROM dummy_table"  # must be bigger than the trigger value (here 3)
        nbr_generated_documents = 5

        # Generate document that match outlier
        command_name = "default_name_"
        for i in range(nbr_generated_documents):
            self.test_es.add_doc(dummy_doc_generate.generate_document(command_query=command_query,
                                                                      command_name=command_name + str(i)))
        # Generate whitelist document
        self.test_es.add_doc(dummy_doc_generate.generate_document(hostname="whitelist_hostname",
                                                                  command_query=command_query,
                                                                  command_name=command_name + str(
                                                                      nbr_generated_documents)))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_with_whitelist.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es.scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, nbr_generated_documents)

    def test_terms_detect_one_outlier_es_check(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        nbr_doc_generated_per_hours = [5, 3, 1, 2]

        # Generate documents
        self.test_es.add_multiple_docs(dummy_doc_generate.generate_doc_time_variable_sensitivity(
            nbr_doc_generated_per_hours))
        # Only the first groupe of document must be detected like an Outlier because the limit is on 3

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float")
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es.scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 5)

    def test_terms_detect_one_outlier_batch_check(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        nbr_doc_generated_per_hours = [5, 3, 1, 2]

        # Generate documents
        self.test_es.add_multiple_docs(dummy_doc_generate.generate_doc_time_variable_sensitivity(
            nbr_doc_generated_per_hours))
        # Only the first groupe of document must be detected like an Outlier because the limit is on 3

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 5)

    def test_terms_small_batch_treat_all(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Init the list of user
        nbr_doc_per_hours = 5
        nbr_hours = 10
        nbr_doc_generated_per_hours = [nbr_doc_per_hours for _ in range(nbr_hours)]
        # If the number of document per hours is not a divisor of the batch limit, all document will not be detected

        # Generate documents
        self.test_es.add_multiple_docs(dummy_doc_generate.generate_doc_time_variable_sensitivity(
            nbr_doc_generated_per_hours))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_small_batch_eval.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), nbr_doc_per_hours*nbr_hours)

    def test_terms_small_batch_last_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Init the list of user with 18 values of 2
        nbr_doc_generated_per_hours = [2 for _ in range(18)]
        # Add a value at the end that must be detected like outlier (limit on 3)
        nbr_doc_generated_per_hours.append(4)

        # Generate documents
        self.test_es.add_multiple_docs(dummy_doc_generate.generate_doc_time_variable_sensitivity(
            nbr_doc_generated_per_hours))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_small_batch_eval.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 4)

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

    def test_terms_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertTrue("timestamp_year" in result['_source'])

    def test_terms_use_derived_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_terms_not_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertFalse("timestamp_year" in result['_source'])

    def test_terms_not_use_derived_fields_but_present_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        # The parameter use_derived_fields haven't any impact on outliers keys
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_terms_default_outlier_infos(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float_low")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in result['_source']['outliers'] for elem in DEFAULT_OUTLIERS_KEY_FIELDS]
        self.assertTrue(all(all_fields_exists))

    def test_terms_extra_outlier_infos_all_present(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_02.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float_low")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in result['_source']['outliers'] for elem in EXTRA_OUTLIERS_KEY_FIELDS]
        self.assertTrue(all(all_fields_exists))

    def test_terms_extra_outlier_infos_new_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_02.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_float_low")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in EXTRA_OUTLIERS_KEY_FIELDS + DEFAULT_OUTLIERS_KEY_FIELDS
                             for elem in result['_source']['outliers']]
        self.assertTrue(all(all_fields_exists))
