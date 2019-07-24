import json
import unittest

import copy
from collections import defaultdict

from helpers.singletons import es
from helpers.analyzer import Analyzer
from tests.unit_tests.test_stubs.test_stub_analyzer import TestStubAnalyzer
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.test_settings import TestSettings
from helpers.outlier import Outlier

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(
                            open("/app/tests/unit_tests/files/doc_with_analyzer_outlier_without_score_and_sort.json"))


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        # "es" use in Analyzer construction and in the method "process_outlier"
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

    def test_simple_process_outlier_return_good_outlier(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = TestStubAnalyzer("analyzer_dummy_test")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_fields = doc_without_outlier["_source"]
        outlier = analyzer.process_outlier(doc_fields, doc_without_outlier)
        expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
                                   outlier_summary='dummy summary', doc=doc_without_outlier)
        expected_outlier.outlier_dict['model_name'] = 'dummy_test'
        expected_outlier.outlier_dict['model_type'] = 'analyzer'
        self.assertEqual(outlier, expected_outlier)

    def test_simple_process_outlier_save_es(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = TestStubAnalyzer("analyzer_dummy_test")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)

        doc_fields = doc_without_outlier["_source"]
        analyzer.process_outlier(doc_fields, doc_without_outlier)

        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, doc_with_outlier)

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
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                    doc), expected_eval_terms)

    def test_add_term_to_batch_no_modification(self):
        eval_terms_array, aggregator_value, target_value, observations, doc = self._preperate_data_terms()
        # Create expected result
        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["targets"] = [target_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
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

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                    doc), expected_eval_terms)
