import unittest

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from helpers.singletons import settings
from analyzers.metrics import MetricsAnalyzer
import helpers.utils

from collections import defaultdict


class TestMetricsAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_es = TestStubEs()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        settings.process_configuration_files("/defaults/outliers.conf")
        settings.process_arguments()
        self.test_es.restore_es()

    def _preperate_data_terms(self):
        eval_metrics_array = defaultdict()
        # "random" value
        aggregator_value = "key"
        target_value = "test"
        metrics_value = "metric_value"
        observations = {'a': 1, 'test': 'ok'}
        doc = {'source': 'this', 'target': 12}
        return eval_metrics_array, aggregator_value, target_value, metrics_value, observations, doc

    def test_add_metric_to_batch_empty(self):
        eval_metrics_array = defaultdict()
        aggregator_value = ""
        target_value = ""
        metrics_value = ""
        observations = {}
        doc = {}
        # # Create expected result
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
            self._preperate_data_terms()

        # # Create expected result
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
