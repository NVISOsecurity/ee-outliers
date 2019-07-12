import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.simplequery import SimplequeryAnalyzer
from helpers.singletons import settings, logging, es

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(
                        open("/app/tests/unit_tests/files/doc_with_simple_query_outlier_without_score_and_sort.json"))


class TestSimplequeryAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        settings._restore_default_configuration_path()
        self.test_es.restore_es()

    def _get_simplequery_analyzer(self, config_file, config_section):
        settings._change_configuration_path(config_file)
        return SimplequeryAnalyzer(config_section_name=config_section)

    def test_one_doc_outlier_correctly_add(self):
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self._get_simplequery_analyzer("/app/tests/unit_tests/files/simplequery_test_01.conf",
                                       "simplequery_dummy_test").evaluate_model()
        # Fetch result to check if it is correct
        result = [elem for elem in es.scan()][0]
        self.assertEqual(result, doc_with_outlier)
