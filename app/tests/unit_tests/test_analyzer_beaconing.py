import unittest

import json
import random

from collections import defaultdict

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from helpers.singletons import settings, es
from analyzers.beaconing import BeaconingAnalyzer
from helpers.outlier import Outlier

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))

LIST_AGGREGATOR_VALUE = ["agg-WIN-EVB-draman", "agg-WIN-DRA-draman"]
LIST_TARGET_VALUE = ["WIN-DRA-draman", "WIN-EVB-draman", "LINUX-DRA-draman"]
LIST_DOC = [doc_without_outlier_test_file]


class TestBeaconingAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_es = TestStubEs()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        settings.process_configuration_files("/defaults/outliers.conf")
        settings.process_arguments()
        self.test_es.restore_es()

    def _create_eval_terms(self, aggregator_value, target_value, observation, doc):
        return {
                aggregator_value: {
                    "targets": [target_value],
                    "observations": [observation],
                    "raw_docs": [doc]
                }
            }

    def _create_outliers(self, outlier_type, outlier_reason, outlier_summary, model_type, model_name, term, aggregator,
                         confidence, decision_frontier, term_count):
        outlier = Outlier(outlier_type=outlier_type, outlier_reason=outlier_reason, outlier_summary=outlier_summary)
        outlier.outlier_dict["model_type"] = model_type
        outlier.outlier_dict["model_name"] = model_name
        outlier.outlier_dict["term"] = term
        outlier.outlier_dict["aggregator"] = aggregator
        outlier.outlier_dict["confidence"] = confidence
        outlier.outlier_dict["decision_frontier"] = decision_frontier
        outlier.outlier_dict["term_count"] = term_count
        return outlier

    def test_add_term_to_batch_one_fix_value(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        aggregator_value = random.choice(LIST_AGGREGATOR_VALUE)
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = random.choice(LIST_DOC)
        expected_result = self._create_eval_terms(aggregator_value, target_value, observations, doc)

        result = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
        self.assertEqual(expected_result, result)

    def test_add_term_to_batch_existing_data_in_eval_terms(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        eval_terms_array = {"test": "ok"}
        aggregator_value = random.choice(LIST_AGGREGATOR_VALUE)
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = random.choice(LIST_DOC)
        expected_result = self._create_eval_terms(aggregator_value, target_value, observations, doc)
        expected_result.update(eval_terms_array)

        result = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations, doc)
        self.assertEqual(expected_result, result)

    def test_evaluate_batch_for_outliers_not_enough_target_buckets(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        analyzer.extract_additional_model_settings()

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value2 = random.choice(LIST_TARGET_VALUE)
        observations2 = {}
        doc2 = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value2, observations2,
                                                      doc2)

        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
        self.assertEqual(result, [])

    def test_evaluate_batch_for_outliers_dqsdfq(self):  # TODO update name
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        analyzer.extract_additional_model_settings()

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = LIST_TARGET_VALUE[0]
        observations = {}
        doc = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        target_value2 = LIST_TARGET_VALUE[1]
        observations2 = {}
        doc2 = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value2, observations2,
                                                      doc2)
        observations3 = {}
        doc3 = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations3,
                                                      doc3)

        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value3 = random.choice(LIST_TARGET_VALUE)
        observations4 = {}
        doc4 = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value3, observations4,
                                                      doc4)

        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)

        # Create outlier
        test_outlier_linux = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
                                                   "dummy_test", target_value, aggregator_value, confidence=1.5,
                                                   decision_frontier=0.5, term_count=2)

        test_outlier_win = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
                                                 "dummy_test", target_value2, aggregator_value, confidence=0.5,
                                                 decision_frontier=0.5, term_count=1)

        # Add outlier to a list
        list_test_outliers = []
        list_test_outliers.append(test_outlier_linux)
        list_test_outliers.append(test_outlier_win)
        list_test_outliers.append(test_outlier_linux)

        self.assertEqual(result, list_test_outliers)

    # def test_prepare_and_process_outlier(self):  # TODO adapt name
    #     settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
    #     analyzer = BeaconingAnalyzer(self.es, "beaconing_dummy_test")
    #
    #     doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
    #     doc_fields = doc_without_outlier["_source"]
    #     # decision_frontier, term_value_count, terms, aggregator_value, term_counter):
    #     outlier = analyzer.prepare_and_process_outlier(doc_fields, doc_without_outlier)
    #
    #
    #     expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
    #                                outlier_summary='dummy summary')
    #     expected_outlier.outlier_dict['model_name'] = 'dummy_test'
    #     expected_outlier.outlier_dict['model_type'] = 'analyzer'
    #     pass
