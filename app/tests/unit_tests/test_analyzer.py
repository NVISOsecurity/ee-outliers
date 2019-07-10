import unittest

from helpers.analyzer import Analyzer
from collections import defaultdict
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_es = TestStubEs()

    def tearDown(self):
        self.test_es.restore_es()

    def test_each_batch_was_processed(self):
        pass
        # simple_query_analyzer = SimplequeryAnalyzer(config_section_name=config_section_name)
        # mock_es = es.copy()

        # mock_es.count_documents = es_count_documents_dummy
        # mokch_es.scan = es_scan_dummy

        # simple_quer_analyzer.es = mock_es_obect
        # total_events = self.es_count_documents_dummy(search_query=search_query)
        # for doc in self.es_scan(search_query=search_query):

    def test_add_term_to_batch_empty(self):
        eval_terms_array = defaultdict()
        aggregator_value = "key"
        target_value = "test"
        observations = {}
        doc = {}

        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms[aggregator_value]["targets"] = [target_value]
        expected_eval_terms[aggregator_value]["observations"] = [{}]
        expected_eval_terms[aggregator_value]["raw_docs"] = [{}]

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                    doc), expected_eval_terms)

    def test_add_term_to_batch_no_modification(self):
        eval_terms_array = defaultdict()
        aggregator_value = "key"
        target_value = "test"
        observations = {}
        doc = {}

        result = defaultdict()
        result[aggregator_value] = defaultdict(list)
        result[aggregator_value]["targets"] = [target_value]
        result[aggregator_value]["observations"] = [observations]
        result[aggregator_value]["raw_docs"] = [doc]

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                    doc), result)

    def test_add_term_to_batch_case1(self):
        eval_terms_array = defaultdict()
        aggregator_value = "key"
        target_value = "test"
        observations = {'a': 1, 'test': 'ok'}
        doc = {'source': 'this', 'target': 12}
        eval_terms_array[aggregator_value] = defaultdict(list)
        eval_terms_array["newKey"] = defaultdict(list)
        eval_terms_array["newKey2"] = "empty"
        eval_terms_array[aggregator_value]["targets"] = [target_value]
        eval_terms_array[aggregator_value]["test"] = 12

        expected_eval_terms = defaultdict()
        expected_eval_terms[aggregator_value] = defaultdict(list)
        expected_eval_terms["newKey"] = defaultdict(list)
        expected_eval_terms["newKey2"] = "empty"
        expected_eval_terms[aggregator_value]["targets"] = [target_value, target_value]
        expected_eval_terms[aggregator_value]["observations"] = [observations]
        expected_eval_terms[aggregator_value]["raw_docs"] = [doc]
        expected_eval_terms[aggregator_value]["test"] = 12

        self.assertEqual(Analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations,
                                                    doc), expected_eval_terms)
