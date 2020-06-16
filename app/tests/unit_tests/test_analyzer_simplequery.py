import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.simplequery import SimplequeryAnalyzer
from helpers.singletons import logging, es
from helpers.analyzerfactory import AnalyzerFactory
from tests.unit_tests.utils.update_settings import UpdateSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

doc_without_outlier_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_without_outlier_with_highlight_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outlier_with_highlight.json"))
doc_with_outlier_with_highlight_match_test_file_01 = json.load(
    open("/app/tests/unit_tests/files/doc_with_simple_query_and_highlight_match_01.json"))
doc_with_outlier_with_highlight_match_test_file_02 = json.load(
    open("/app/tests/unit_tests/files/doc_with_simple_query_and_highlight_match_02.json"))
doc_with_outlier_test_file_01 = json.load(
    open("/app/tests/unit_tests/files/doc_with_simple_query_outlier_01.json"))
doc_with_outlier_test_file_02 = json.load(
    open("/app/tests/unit_tests/files/doc_with_simple_query_outlier_02.json"))

DEFAULT_OUTLIERS_KEY_FIELDS = ["type", "reason", "summary", "model_name", "model_type", "total_outliers",
                               "elasticsearch_filter"]

use_case_path = "/app/tests/unit_tests/files/use_cases/simplequery/"
use_case_simplequery_raw_configparser_test_percent_signs = use_case_path + \
                                                           "simplequery_raw_configparser_test_percent_signs.conf"
use_case_simplequery_dummy_test = use_case_path + "simplequery_dummy_test.conf"
use_case_simplequery_dummy_test_highlight_match_activated = use_case_path + \
                                                            "simplequery_dummy_test_highlight_match_activated.conf"
use_case_simplequery_dummy_test_highlight_match_unactivated = use_case_path + \
                                                            "simplequery_dummy_test_highlight_match_unactivated.conf"
use_case_simplequery_dummy_test_derived = use_case_path + "simplequery_dummy_test_derived.conf"
use_case_simplequery_dummy_test_not_derived = use_case_path + "simplequery_dummy_test_not_derived.conf"
use_case_whitelist_tests_model_whitelist_01 = use_case_path + "whitelist_tests_model_whitelist_01.conf"
use_case_whitelist_tests_model_whitelist_02 = use_case_path + "whitelist_tests_model_whitelist_02.conf"
use_case_simplequery_arbitrary_dummy_test = use_case_path + "simplequery_arbitrary_dummy_test.conf"

config_file_path = "/app/tests/unit_tests/files/"
config_file_simplequery_test_whitelist = config_file_path + "simplequery_test_whitelist.conf"
config_file_simplequery_test_01 = config_file_path + "simplequery_test_01.conf"
config_file_simplequery_test_02 = config_file_path + "simplequery_test_02.conf"
config_file_simplequery_test_highlight_match_activated = config_file_path + \
                                                         "simplequery_test_highlight_match_activated.conf"
config_file_simplequery_test_highlight_match_unactivated = config_file_path + \
                                                           "simplequery_test_highlight_match_unactivated.conf"


class TestSimplequeryAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def _get_simplequery_analyzer(self, config_file, config_section):
        self.test_settings.change_configuration_path(config_file)
        return SimplequeryAnalyzer(config_section_name=config_section)

    # Simply test if use cases containing a % sign also work correctly and don't generate an expcetion when being
    # parsed by the ConfigParser. This is the reason we use the RawConfigParser.
    # https://docs.python.org/2/library/configparser.html
    def test_simplequery_raw_configparser_test_percent_signs_in_query(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_whitelist)
        analyzer = AnalyzerFactory.create(use_case_simplequery_raw_configparser_test_percent_signs)
        analyzer.evaluate_model()

    def test_simplequery_whitelist_work_test_es_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        nbr_generated_documents = 5
        all_doc = dummy_doc_generate.create_documents(nbr_generated_documents)
        whitelisted_document = dummy_doc_generate.generate_document({"hostname": "whitelist_hostname"})
        all_doc.append(whitelisted_document)
        self.test_es.add_multiple_docs(all_doc)

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_whitelist)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, nbr_generated_documents)

    def test_one_doc_outlier_with_highlight_01(self):
        """
        Test if a doc is correctly generated with the highlight fields. The use case is with the field
        highlight_match=1 and there is nothing in related to highlight in configuration file.
        """
        doc_without_outlier = copy.deepcopy(doc_without_outlier_with_highlight_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_with_highlight_match_test_file_01)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_activated)
        analyzer.evaluate_model()

        # Fetch result to check if it is correct
        result = [elem for elem in es._scan()][0]
        self.assertEqual(result, doc_with_outlier)

    def test_one_doc_outlier_with_highlight_02(self):
        """
        Test if a doc is correctly generated with the highlight fields. The use case is with no fields related to
        highlight and the configuration file has highlight_match=1.
        """
        doc_without_outlier = copy.deepcopy(doc_without_outlier_with_highlight_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_with_highlight_match_test_file_02)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self.test_settings.change_configuration_path(config_file_simplequery_test_highlight_match_activated)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)
        analyzer.evaluate_model()

        # Fetch result to check if it is correct
        result = [elem for elem in es._scan()][0]
        self.assertEqual(result, doc_with_outlier)

    def test_one_doc_outlier_with_highlight_03(self):
        """
        Test if a doc is correctly generated with the highlight fields. The use case is with the field
        highlight_match=1 and the configuration file has highlight_match=0.
        """
        doc_without_outlier = copy.deepcopy(doc_without_outlier_with_highlight_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_with_highlight_match_test_file_01)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self.test_settings.change_configuration_path(config_file_simplequery_test_highlight_match_unactivated)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_activated)
        analyzer.evaluate_model()

        # Fetch result to check if it is correct
        result = [elem for elem in es._scan()][0]
        self.assertEqual(result, doc_with_outlier)

    def test_one_doc_outlier_with_highlight_04(self):
        """
        Test if a doc is correctly generated with the highlight fields. The use case is with the field
        highlight_match=0 and the configuration file has highlight_match=1.
        """
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file_02)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self.test_settings.change_configuration_path(config_file_simplequery_test_highlight_match_activated)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_unactivated)
        analyzer.evaluate_model()

        # Fetch result to check if it is correct
        result = [elem for elem in es._scan()][0]
        self.assertEqual(result, doc_with_outlier)

    def test_one_doc_outlier_correctly_add(self):
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file_01)

        # Insert value
        self.test_es.add_doc(doc_without_outlier)
        # Make test (supposed all doc work)
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)
        analyzer.evaluate_model()

        # Fetch result to check if it is correct
        result = [elem for elem in es._scan()][0]
        self.assertEqual(result, doc_with_outlier)

    def test_simplequery_use_highlight_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_activated)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("highlight" in result)

    def test_simplequery_use_matched_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_activated)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("matched_fields" in result['_source']['outliers'])

    def test_simplequry_use_matched_values_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_highlight_match_activated)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("matched_values" in result['_source']['outliers'])

    def test_simplequery_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_derived)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("timestamp_year" in result['_source'])

    def test_simplequery_use_derived_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_derived)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_simplequery_not_use_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_not_derived)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertFalse("timestamp_year" in result['_source'])

    def test_simplequery_not_use_derived_fields_but_present_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(config_file_simplequery_test_02)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test_not_derived)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_simplequery_default_outlier_infos(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        all_fields_exists = [elem in result['_source']['outliers'] for elem in DEFAULT_OUTLIERS_KEY_FIELDS]
        self.assertTrue(all(all_fields_exists))

    def test_simplequery_no_extra_outlier_infos(self):
        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        all_fields_exists = [elem in DEFAULT_OUTLIERS_KEY_FIELDS for elem in result['_source']['outliers']]
        self.assertTrue(all(all_fields_exists))

    def test_whitelist_literal_per_model_match_whitelist(self):
        doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(doc_generate.generate_document({"hostname": "HOSTNAME-WHITELISTED"}))

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_whitelist_tests_model_whitelist_01)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertFalse("outliers" in result["_source"])

    def test_whitelist_literal_per_model_not_match_whitelist(self):
        doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(doc_generate.generate_document({"hostname": "not_whitelist_hostname"}))

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_whitelist_tests_model_whitelist_01)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("outliers" in result["_source"])

    def test_whitelist_regex_per_model_match_whitelist(self):
        doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(doc_generate.generate_document({"hostname": "AAA-WHITELISTED"}))

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_whitelist_tests_model_whitelist_02)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertFalse("outliers" in result["_source"])

    def test_whitelist_regex_per_model_not_match_whitelist(self):
        doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(doc_generate.generate_document({"hostname": "Not-work-WHITELISTED"}))

        # Run analyzer
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_whitelist_tests_model_whitelist_02)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("outliers" in result["_source"])

    def test_arbitrary_key_config_present_in_outlier(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_arbitrary_dummy_test)

        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertEquals(result["_source"]["outliers"]["test_arbitrary_key"], ["arbitrary_value"])

    def test_arbitrary_key_config_not_present_int_other_model(self):
        # Dictionary and list could be share between different instance. This test check that a residual value is not
        # present in the dictionary
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        analyzer = AnalyzerFactory.create(use_case_simplequery_dummy_test)

        dummy_doc_generate = DummyDocumentsGenerate()

        # Generate document
        self.test_es.add_doc(dummy_doc_generate.generate_document())
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertFalse("test_arbitrary_key" in result["_source"]["outliers"])
