import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.simplequery import SimplequeryAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(
                        open("/app/tests/unit_tests/files/doc_with_simple_query_outlier_without_score_and_sort.json"))


class TestSimplequeryAnalyzer(unittest.TestCase):
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

    def _get_simplequery_analyzer(self, config_file, config_section):
        self.test_settings.change_configuration_path(config_file)
        return SimplequeryAnalyzer(config_section_name=config_section)

    def test_simplequery_whitelist_work_test_es_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        nbr_generated_documents = 5
        all_doc = dummy_doc_generate.create_documents(nbr_generated_documents)
        whitelisted_document = dummy_doc_generate.generate_document(hostname="whitelist_hostname")
        all_doc.append(whitelisted_document)
        self.test_es.add_multiple_docs(all_doc)

        # Run analyzer
        self._get_simplequery_analyzer("/app/tests/unit_tests/files/simplequery_test_whitelist.conf",
                                       "simplequery_dummy_test").evaluate_model()

        nbr_outliers = 0
        for elem in es.scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, nbr_generated_documents)

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
