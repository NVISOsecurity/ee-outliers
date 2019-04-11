import unittest
from helpers.singletons import settings, es, logging


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        pass
        #es.scan = scan_dummy(es_query = ..., )

    def test_each_batch_was_processed(self):
        pass
        #simple_query_analyzer = SimplequeryAnalyzer(config_section_name=config_section_name)
        #mock_es = es.copy()

        #mock_es.count_documents = es_count_documents_dummy
        #mokch_es.scan = es_scan_dummy

        #simple_quer_analyzer.es = mock_es_obect
        #total_events = self.es_count_documents_dummy(lucene_query=lucene_query)
        #for doc in self.es_scan(lucene_query=lucene_query):

    def es_count_documents_dummy(self):
        pass

    def es_scan_dummy(self):
        pass





