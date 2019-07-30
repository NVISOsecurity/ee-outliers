import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.metrics import MetricsAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate
import helpers.utils

from collections import defaultdict

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_metrics_outlier.json"))


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

    def _preperate_dummy_data_terms(self):
        eval_metrics_array = defaultdict()
        # "random" value
        aggregator_value = "key"
        target_value = "test"
        metrics_value = 12
        observations = {'a': 1, 'test': 'ok'}
        doc = {'source': 'this', 'target': 12}
        return eval_metrics_array, aggregator_value, target_value, metrics_value, observations, doc

    def _preperate_data_terms_with_doc(self, metrics_value=1):
        eval_metrics_array = defaultdict()
        # "random" value
        aggregator_value = "key"
        target_value = "test"
        observations = {}
        return eval_metrics_array, aggregator_value, target_value, metrics_value, observations

    def test_add_metric_to_batch_empty(self):
        eval_metrics_array = defaultdict()
        aggregator_value = ""
        target_value = ""
        metrics_value = ""
        observations = {}
        doc = {}
        # Create expected result
        observations["target"] = [target_value]
        observations["aggregator"] = [aggregator_value]
        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["metrics"] = [metrics_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]

        result = MetricsAnalyzer.add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value,
                                                     observations, doc)
        self.assertEqual(result, expected_eval_terms)

    def test_add_metric_to_batch_no_modification(self):
        eval_metrics_array, aggregator_value, target_value, metrics_value, observations, doc = \
            self._preperate_dummy_data_terms()

        # Create expected result
        observations["target"] = [target_value]
        observations["aggregator"] = [aggregator_value]
        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["metrics"] = [metrics_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]

        result = MetricsAnalyzer.add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value,
                                                     observations, doc)
        self.assertEqual(result, expected_eval_terms)

    def test_calculate_metric_numerical_value(self):
        self.assertEqual(MetricsAnalyzer.calculate_metric("numerical_value", "12"), (float(12), dict()))

    def test_calculate_metric_length(self):
        self.assertEqual(MetricsAnalyzer.calculate_metric("length", "test"), (len("test"), dict()))

    def test_calculate_metric_entropy(self):
        self.assertEqual(MetricsAnalyzer.calculate_metric("entropy", "test"),
                         (helpers.utils.shannon_entropy("test"), dict()))

    def test_calculate_metric_hex_encoded_length(self):
        result = MetricsAnalyzer.calculate_metric("hex_encoded_length", "12c322adc020 12322029620")
        expected_observation = {
            'max_hex_encoded_length': 12,
            'max_hex_encoded_word': '12c322adc020'
        }
        self.assertEqual(result, (12, expected_observation))

    def test_calculate_metric_base64_encoded_length(self):
        result = MetricsAnalyzer.calculate_metric("base64_encoded_length", "houston we have a cHJvYmxlbQ==")
        expected_observation = {
            'max_base64_decoded_length': 7,
            'max_base64_decoded_word': 'problem'
        }

        self.assertEqual(result, (7, expected_observation))

    def test_calculate_metric_url_length(self):
        result = MetricsAnalyzer.calculate_metric("url_length", "why don't we go http://www.nviso.com")
        expected_observation = {
            'extracted_urls_length': 20,
            'extracted_urls': 'http://www.nviso.com'
        }

        self.assertEqual(result, (20, expected_observation))

    def test_calculate_metric_unexist_operation(self):
        self.assertEqual(MetricsAnalyzer.calculate_metric("dummy operation", ""), (None, dict()))

    def test_evaluate_batch_for_outliers_fetch_remain_metrics(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test")

        eval_metrics_array, aggregator_value, target_value, metrics_value, observations = \
            self._preperate_data_terms_with_doc()
        doc = DummyDocumentsGenerate().generate_document()
        metrics = MetricsAnalyzer.add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value,
                                                      observations, doc)

        result = analyzer.evaluate_batch_for_outliers(metrics, analyzer.model_settings, False)
        self.assertEqual(result, ([], metrics))

    def test_evaluate_batch_for_outliers_add_outlier(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test")

        eval_metrics_array, aggregator_value, target_value, metrics_value, observations = \
            self._preperate_data_terms_with_doc(metrics_value=12)
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        metrics = MetricsAnalyzer.add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value,
                                                      observations, doc_without_outlier)

        analyzer.evaluate_batch_for_outliers(metrics, analyzer.model_settings, True)
        result = [elem for elem in es.scan()][0]
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)

        self.assertEqual(result, doc_with_outlier)
