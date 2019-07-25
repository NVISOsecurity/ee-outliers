import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.metrics import MetricsAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))

DEFAULT_OUTLIERS_KEY_FIELDS = ["type", "reason", "summary", "model_name", "model_type", "total_outliers"]
EXTRA_OUTLIERS_KEY_FIELDS = ["target", "aggregator", "metric", "decision_frontier", "confidence"]


class TestMetricsAnalyzer(unittest.TestCase):
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

    def test_metrics_whitelist_work_test_es_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        command_query = "SELECT * FROM dummy_table"  # must be bigger than the trigger value (here 3)
        nbr_generated_documents = 5

        # Generate document that match outlier
        for _ in range(nbr_generated_documents):
            self.test_es.add_doc(dummy_doc_generate.generate_document(command_query=command_query))
        # Generate whitelist document
        self.test_es.add_doc(dummy_doc_generate.generate_document(hostname="whitelist_hostname",
                                                                  command_query=command_query))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_with_whitelist.conf")
        analyzer = MetricsAnalyzer("metrics_length_dummy_test")
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es.scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, nbr_generated_documents)

    def test_metrics_detect_one_outlier_es_check(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        list_user_id = [11, 10, 8, 0, 0, 0]

        # Generate document
        for user_id in list_user_id:
            self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=user_id))
        # Only the fist one must be detected like outlier, because user_id need to be bigger than 10

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_numerical_value_dummy_test")
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es.scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 1)

    def test_metrics_detect_one_outlier_batch_check(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        list_user_id = [11, 10, 8, 0, 0, 0]

        # Generate document
        for user_id in list_user_id:
            self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=user_id))
        # Only the fist one must be detected like outlier, because user_id need to be bigger than 10

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_numerical_value_dummy_test")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 1)

    def test_metrics_small_batch_treat_all(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Init the list of user
        default_user_id = 11
        number_of_user = 20
        list_user_id = [default_user_id for _ in range(number_of_user)]

        # Generate document
        for user_id in list_user_id:
            self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=user_id))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_small_batch_eval.conf")
        analyzer = MetricsAnalyzer("metrics_numerical_value_dummy_test")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), number_of_user)

    def test_metrics_small_batch_last_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Init the list of user
        default_user_id = 0
        number_of_user = 19
        list_user_id = [default_user_id for _ in range(number_of_user)]
        # Add a value at the end that must be detected like outlier (limit on 10)
        list_user_id.append(11)

        # Generate document
        for user_id in list_user_id:
            self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=user_id))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_small_batch_eval.conf")
        analyzer = MetricsAnalyzer("metrics_numerical_value_dummy_test")
        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 1)

    def test_metrics_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertTrue("timestamp_year" in result['_source'])

    def test_metrics_use_derived_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_metrics_not_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertFalse("timestamp_year" in result['_source'])

    def test_metrics_not_use_derived_fields_but_present_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_not_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        # The parameter use_derived_fields haven't any impact on outliers keys
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def _test_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_with_whitelist.conf")
        analyzer = MetricsAnalyzer("metrics_length_dummy_test")

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

    def test_simplequery_default_outlier_infos(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_02.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_no_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in result['_source']['outliers'] for elem in DEFAULT_OUTLIERS_KEY_FIELDS]
        self.assertTrue(all(all_fields_exists))

    def test_metrics_extra_outlier_infos_all_present(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_02.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_no_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in result['_source']['outliers'] for elem in EXTRA_OUTLIERS_KEY_FIELDS]
        self.assertTrue(all(all_fields_exists))

    def test_metrics_extra_outlier_infos_new_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document(user_id=11))

        # Run analyzer
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_02.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_no_derived")
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        all_fields_exists = [elem in EXTRA_OUTLIERS_KEY_FIELDS + DEFAULT_OUTLIERS_KEY_FIELDS
                             for elem in result['_source']['outliers']]
        self.assertTrue(all(all_fields_exists))
