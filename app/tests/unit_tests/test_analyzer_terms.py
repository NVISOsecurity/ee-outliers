import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.terms import TermsAnalyzer
from helpers.singletons import settings, logging, es
from tests.unit_tests.utils.test_settings import TestSettings

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))
doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_beaconing_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_beaconing_outlier.json"))


class TestTermsAnalyzer(unittest.TestCase):
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
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_with_whitelist.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

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

    def test_evaluate_model_beaconing_simple_case(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
        analyzer = TermsAnalyzer("terms_beaconing_dummy_test")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        expected_doc = copy.deepcopy(doc_with_beaconing_outlier_test_file)
        # Add doc to the database
        self.test_es.add_doc(doc_without_outlier)

        # Make test (suppose that all doc match with the query)
        analyzer.evaluate_model()

        result = [elem for elem in es.scan()][0]
        print(result["_source"])
        self.assertEqual(result, expected_doc)
