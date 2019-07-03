import unittest

from tests.unit_tests.mokup.mokup_es import mokup_es
from helpers.singletons import settings, logging

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.es = mokup_es(settings, logging)

    def test_each_batch_was_processed(self):
        pass
        # simple_query_analyzer = SimplequeryAnalyzer(config_section_name=config_section_name)
        # mock_es = es.copy()

        # mock_es.count_documents = es_count_documents_dummy
        # mokch_es.scan = es_scan_dummy

        # simple_quer_analyzer.es = mock_es_obect
        # total_events = self.es_count_documents_dummy(search_query=search_query)
        # for doc in self.es_scan(search_query=search_query):


"""
- process_outlier

"""
