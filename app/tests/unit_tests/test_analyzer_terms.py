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

doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))
doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_terms_outlier_coeff_of_variation_no_score_sort = json.load(
    open("/app/tests/unit_tests/files/doc_with_terms_outlier_coeff_of_variation.json"))

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

    # This test work only if we try to detect whitelist element on non outliers elements
    # Here the count is not lower than three, so documents aren't outliers, and we never see that the first one is
    # whitelisted
    #
    # def test_whitelist_batch_document_not_process_all(self):
    #     self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_with_whitelist.conf")
    #     analyzer = TermsAnalyzer("terms_dummy_test")
    #
    #     # Whitelisted (ignored)
    #     doc1_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_01_test_file)
    #     self.test_es.add_doc(doc1_without_outlier)
    #     # Not whitelisted (add)
    #     doc2_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_02_test_file)
    #     self.test_es.add_doc(doc2_without_outlier)
    #     # Not whitelisted
    #     doc3_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_03_test_file)
    #     self.test_es.add_doc(doc3_without_outlier)
    #
    #     analyzer.evaluate_model()
    #
    #     self.assertEqual(len(analyzer.outliers), 2)

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
        for elem in es._scan():
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
        for elem in es._scan():
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

        self.assertEqual(analyzer.total_outliers, 5)

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

        self.assertEqual(analyzer.total_outliers, nbr_doc_per_hours*nbr_hours)

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

        self.assertEqual(analyzer.total_outliers, 4)

    def test_evaluate_batch_for_outliers_not_enough_target_buckets_one_doc_max_two(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        doc = copy.deepcopy(random.choice(LIST_DOC))
        current_batch = analyzer._add_document_to_batch(defaultdict(), [target_value], [aggregator_value], doc)

        result, remaining_terms = analyzer._evaluate_batch_for_outliers(batch=current_batch)
        self.assertEqual(result, [])

    def test_evaluate_batch_for_outliers_limit_target_buckets_two_doc_max_two(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

        # Create one document with one aggregator
        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        doc = copy.deepcopy(random.choice(LIST_DOC))
        current_batch = analyzer._add_document_to_batch(defaultdict(), [target_value], [aggregator_value], doc)
        # Create a second document with another aggregator
        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value2 = random.choice(LIST_TARGET_VALUE)
        doc2 = copy.deepcopy(random.choice(LIST_DOC))
        current_batch = analyzer._add_document_to_batch(current_batch, [target_value2], [aggregator_value2], doc2)

        # Expect to get nothing due to "min_target_buckets" set to 2
        result, remaining_terms = analyzer._evaluate_batch_for_outliers(batch=current_batch)
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

        result = [elem for elem in es._scan()][0]
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
        for doc in es._scan():
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
        for doc in es._scan():
            if "outliers" in doc['_source']:
                nbr_outliers += 1

        self.assertEqual(nbr_outliers, len(all_doc))

    def test_terms_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("timestamp_year" in result['_source'])

    def test_terms_use_derived_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_terms_not_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertFalse("timestamp_year" in result['_source'])

    def test_terms_not_use_derived_fields_but_present_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
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

        result = [elem for elem in es._scan()][0]
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

        result = [elem for elem in es._scan()][0]
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

        result = [elem for elem in es._scan()][0]
        all_fields_exists = [elem in EXTRA_OUTLIERS_KEY_FIELDS + DEFAULT_OUTLIERS_KEY_FIELDS
                             for elem in result['_source']['outliers']]
        self.assertTrue(all(all_fields_exists))

    def test_add_document_to_batch_empty_target(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        dummy_doc = dummy_doc_generate.generate_document()

        current_batch = {"dummy_key": "dummy_value"}
        result = TermsAnalyzer._add_document_to_batch(current_batch, list(), ["dummy_aggregator"], dummy_doc)
        self.assertEqual(result, current_batch)

    def test_add_document_to_batch_empty_aggergator(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        dummy_doc = dummy_doc_generate.generate_document()

        current_batch = {"dummy_key": "dummy_value"}
        result = TermsAnalyzer._add_document_to_batch(current_batch, ["dummy_target"], list(), dummy_doc)
        self.assertEqual(result, current_batch)

    def test_add_document_to_batch_one_aggregator_and_one_target(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        dummy_doc = dummy_doc_generate.generate_document()
        target_value = "dummy_target"
        aggregator_value = "dummy_aggregator"

        current_batch = {"dummy_key": "dummy_value"}
        result = TermsAnalyzer._add_document_to_batch(current_batch, [target_value], [aggregator_value], dummy_doc)

        expected_batch = current_batch.copy()
        expected_batch[aggregator_value] = defaultdict(list)
        expected_batch[aggregator_value]["targets"].append(target_value)
        expected_batch[aggregator_value]["observations"].append(dict())
        expected_batch[aggregator_value]["raw_docs"].append(dummy_doc)

        self.assertEqual(result, expected_batch)

    def test_min_target_buckets_detect_outlier(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_whitelist_batch.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_batch_whitelist_within_float")
        # Recap:
        # min_target_buckets=4
        # trigger_sensitivity=5
        # trigger_on=high
        # trigger_method=float

        # Dont encode with a matrix to keep order of document
        doc_to_generate = [
            # New batch:
            #       0  1  2
            # agg1 [5, 1, 1]
            # agg2 [1, 1, 1]
            ("agg1", 0),
            ("agg2", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg2", 1),
            ("agg1", 1),
            ("agg2", 2),
            ("agg1", 2),
            # New batch
            #       2  3
            # agg1 [1, 1]
            # agg2 [5, 1]
            ("agg2", 2),
            ("agg2", 2),
            ("agg2", 2),
            ("agg2", 2),
            ("agg1", 2),
            ("agg2", 2),
            ("agg1", 3),
            ("agg2", 3)]

        # At the end:
        #       0  1  2  3
        # agg1 [5, 0, 2, 1]
        # agg2 [1, 1, 6, 1]
        # So only agg2 - 2 (6 documents) need to be flagged

        dummy_doc_gen = DummyDocumentsGenerate()
        for aggregator, target_value in doc_to_generate:
            user_id = target_value
            hostname = aggregator
            doc_generated = dummy_doc_gen.generate_document(user_id=user_id, hostname=hostname)
            self.test_es.add_doc(doc_generated)

        analyzer.evaluate_model()

        list_outliers = []
        for doc in es._scan():
            if "outliers" in doc["_source"]:
                list_outliers.append((doc["_source"]["outliers"]["aggregator"][0],
                                      doc["_source"]["outliers"]["term"][0]))

        self.assertEqual(list_outliers, [("agg2", "2") for _ in range(6)])

    def test_min_target_buckets_dont_detect_outlier(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_whitelist_batch.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_batch_whitelist_within_float")
        # Recap:
        # min_target_buckets=4
        # trigger_sensitivity=5
        # trigger_on=high
        # trigger_method=float

        # Dont encode with a matrix to keep order of document
        doc_to_generate = [
            # New batch:
            #       0  1
            # agg1 [6, 1]
            # agg2 [1, 2]
            ("agg1", 0),
            ("agg2", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg1", 0),
            ("agg2", 1),
            ("agg2", 1),
            ("agg1", 1),
            # New Batch
            #       2
            # agg1 [0]
            # agg2 [1]
            ("agg1", 2)]

        # At the end:
        #       0  1  2
        # agg1 [6, 1, 1]
        # agg2 [1, 2]
        # Normally agg1 - 0 must be flagged, but here they doesn't have enough buckets values

        dummy_doc_gen = DummyDocumentsGenerate()
        for aggregator, target_value in doc_to_generate:
            user_id = target_value
            hostname = aggregator
            doc_generated = dummy_doc_gen.generate_document(user_id=user_id, hostname=hostname)
            self.test_es.add_doc(doc_generated)

        analyzer.evaluate_model()
        self.assertEqual(analyzer.total_outliers, 0)

    def test_batch_whitelist_work_with_min_target_bucket(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_whitelist_batch.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_batch_whitelist_within_float")
        # Recap:
        # min_target_buckets=4
        # trigger_sensitivity=5
        # trigger_on=high
        # trigger_method=float

        doc_to_generate = [
            # New batch:
            #       0  1  2
            # agg1 [3, 0, 1]
            # agg2 [1, 3, 2]
            ("agg1", 0, False),
            ("agg2", 0, False),
            ("agg1", 0, True),
            ("agg1", 0, False),
            ("agg2", 1, False),
            ("agg2", 1, False),
            ("agg2", 1, False),
            ("agg2", 2, False),
            ("agg2", 2, False),
            ("agg1", 2, False),
            # New batch
            #       2  3  4
            # agg1 [1, 0, 2]
            # agg2 [4, 3]
            ("agg2", 2, False),
            ("agg2", 2, False),
            ("agg2", 2, True),
            ("agg2", 2, False),
            ("agg1", 2, False),
            ("agg2", 3, False),
            ("agg2", 3, False),
            ("agg2", 3, False),
            ("agg1", 4, False),
            ("agg1", 4, False),
            # New batch
            #       4  5
            # agg1 [4, 1]
            ("agg1", 4, False),
            ("agg1", 4, False),
            ("agg1", 4, False),
            ("agg1", 4, False),
            ("agg1", 5, False)]

        # At the end:
        #       0  1  2  3  4
        # agg1 [3, 2, 2, 2, 6]
        # agg2 [1, 1, 6, 1]
        # So two outlier: agg1 - 4 and agg2 - 2.  But one of agg2 - 2 is whitelisted. So only 5 occurrences

        dummy_doc_gen = DummyDocumentsGenerate()
        for aggregator, target_value, is_whitelist in doc_to_generate:
            deployment_name = None
            if is_whitelist:
                deployment_name = "whitelist-deployment"
            user_id = target_value
            hostname = aggregator

            doc_generated = dummy_doc_gen.generate_document(deployment_name=deployment_name, user_id=user_id,
                                                            hostname=hostname)
            self.test_es.add_doc(doc_generated)
        analyzer.evaluate_model()

        list_outliers = []
        for doc in es._scan():
            if "outliers" in doc["_source"]:
                list_outliers.append((doc["_source"]["outliers"]["aggregator"][0],
                                      doc["_source"]["outliers"]["term"][0]))

        self.assertEqual(list_outliers, [("agg1", "4") for _ in range(6)])

    def test_batch_whitelist_work_doent_match_outlier_in_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_whitelist_batch.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_batch_whitelist_across_float")

        doc_to_generate = [
            # agg1 (0, 1, 2) -> 3 but with whitelist: (0, 2) -> 2
            # agg2 (0, 3, 4) -> 3
            ("agg1", 0, False),
            ("agg1", 1, True),
            ("agg2", 0, False),
            ("agg2", 0, False),
            ("agg1", 2, False),
            ("agg2", 3, False),
            ("agg2", 4, False)]

        dummy_doc_gen = DummyDocumentsGenerate()
        for aggregator, target_value, is_whitelist in doc_to_generate:
            deployment_name = None
            if is_whitelist:
                deployment_name = "whitelist-deployment"
            user_id = target_value
            hostname = aggregator

            doc_generated = dummy_doc_gen.generate_document(deployment_name=deployment_name, user_id=user_id,
                                                            hostname=hostname)
            self.test_es.add_doc(doc_generated)

        analyzer.evaluate_model()

        list_outliers = []
        for doc in es._scan():
            if "outliers" in doc["_source"]:
                list_outliers.append((doc["_source"]["outliers"]["aggregator"][0],
                                      doc["_source"]["outliers"]["term"][0]))

        # We detect agg2 but not agg1
        self.assertEqual(list_outliers, [("agg2", "0"), ("agg2", "0"), ("agg2", "3"), ("agg2", "4")])
