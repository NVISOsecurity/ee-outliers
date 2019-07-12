import unittest

import copy
import json
import random

from collections import defaultdict

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from helpers.singletons import settings, es, logging
from analyzers.beaconing import BeaconingAnalyzer
from helpers.outlier import Outlier

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_beaconing_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_beaconing_outlier.json"))
doc_with_beaconing_outlier_without_score_sort_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_with_beaconing_outlier_without_score_sort.json"))

LIST_AGGREGATOR_VALUE = ["agg-WIN-EVB-draman", "agg-WIN-DRA-draman"]
LIST_TARGET_VALUE = ["WIN-DRA-draman", "WIN-EVB-draman", "LINUX-DRA-draman"]
LIST_DOC = [doc_without_outlier_test_file]


class TestBeaconingAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        settings.process_configuration_files("/defaults/outliers.conf")
        settings.process_arguments()
        self.test_es.restore_es()

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

    def test_evaluate_batch_for_outliers_not_enough_target_buckets_one_doc_max_two(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        analyzer.extract_additional_model_settings()

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
        self.assertEqual(result, [])

    def test_evaluate_batch_for_outliers_limit_target_buckets_two_doc_max_two(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        analyzer.extract_additional_model_settings()

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value2 = random.choice(LIST_TARGET_VALUE)
        observations2 = {}
        doc2 = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value2, observations2,
                                                      doc2)

        result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
        self.assertEqual(result, [])

    def test_evaluate_batch_for_outliers_detect_two_outliers(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")
        analyzer.extract_additional_model_settings()

        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = LIST_TARGET_VALUE[0]
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        target_value2 = LIST_TARGET_VALUE[1]
        observations2 = {}
        doc2 = random.choice(LIST_DOC)
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value2, observations2,
                                                      doc2)
        observations3 = {}
        doc3 = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations3,
                                                      doc3)

        aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
        target_value3 = random.choice(LIST_TARGET_VALUE)
        observations4 = {}
        doc4 = copy.deepcopy(random.choice(LIST_DOC))
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
        expected_outliers = []
        expected_outliers.append(test_outlier_linux)
        expected_outliers.append(test_outlier_win)
        expected_outliers.append(test_outlier_linux)

        self.assertEqual(result, expected_outliers)

    def test_prepare_and_process_outlier_one_outlier(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")

        # decision_frontier, term_value_count, terms, aggregator_value, term_counter):
        decision_frontier = 1
        term_value_count = 2
        aggregator_value = LIST_AGGREGATOR_VALUE[0]
        target_value = random.choice(LIST_TARGET_VALUE)
        observations = {}
        doc = copy.deepcopy(random.choice(LIST_DOC))
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        outlier = analyzer.prepare_and_process_outlier(decision_frontier, term_value_count, eval_terms_array,
                                                       aggregator_value, 0)

        expected_outlier = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
                                                 "dummy_test", target_value, aggregator_value, confidence=1,
                                                 decision_frontier=1, term_count=2)
        self.assertEqual(outlier, expected_outlier)

    def test_prepare_and_process_outlier_check_es_have_request(self):  # TODO adapt name
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")

        # decision_frontier, term_value_count, terms, aggregator_value, term_counter):
        decision_frontier = 1
        term_value_count = 2
        aggregator_value = "agg-WIN-EVB-draman"
        target_value = "WIN-DRA-draman"
        observations = {}
        doc = copy.deepcopy(doc_without_outlier_test_file)
        eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)

        analyzer.prepare_and_process_outlier(decision_frontier, term_value_count, eval_terms_array,
                                             aggregator_value, 0)

        expected_doc = copy.deepcopy(doc_with_beaconing_outlier_without_score_sort_test_file)

        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, expected_doc)

    def test_evaluate_model_beaconing_simple_case(self):
        settings.process_configuration_files("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = BeaconingAnalyzer("beaconing_dummy_test")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        expected_doc = copy.deepcopy(doc_with_beaconing_outlier_test_file)
        # Add doc to the database
        self.test_es.add_doc(doc_without_outlier)

        # Make test (suppose that all doc match with the query)
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, expected_doc)
