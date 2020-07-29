import json
import unittest

import copy

from helpers.singletons import es
from tests.unit_tests.test_stubs.test_stub_analyzer import TestStubAnalyzer
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.update_settings import UpdateSettings
from helpers.outlier import Outlier
from helpers.analyzerfactory import AnalyzerFactory
import helpers.analyzerfactory
import configparser

# Monkey patch the test stub analyzer mapping in the AnalyzerFactory
helpers.analyzerfactory.CLASS_MAPPING["analyzer"] = TestStubAnalyzer

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_analyzer_outlier.json"))

config_file_path = "/app/tests/unit_tests/files/"
config_file_analyzer_test_01 = config_file_path + "analyzer_test_01.conf"
config_file_analyzer_test_with_custom_timestamp_field = config_file_path + \
                                                        "analyzer_test_with_custom_timestamp_field.conf"

use_case_analyzer_files_path = "/app/tests/unit_tests/files/use_cases/analyzer/"
use_case_analyzer_dummy_test = use_case_analyzer_files_path + "analyzer_dummy_test.conf"
use_case_analyzer_arbitrary_dummy_test = use_case_analyzer_files_path + "analyzer_arbitrary_dummy_test.conf"
use_case_analyzer_multi_malformed_duplicate_option = use_case_analyzer_files_path + \
                                                     "analyzer_multi_malformed_duplicate_option.conf"
use_case_analyzer_multi_malformed_duplicate_section = use_case_analyzer_files_path + \
                                                      "analyzer_multi_malformed_duplicate_section.conf"


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        # "es" use in Analyzer construction and in the method "process_outlier"
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def test_simple_process_outlier_return_good_outlier(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzer = AnalyzerFactory.create(use_case_analyzer_dummy_test)

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        doc_fields = doc_without_outlier["_source"]
        outlier = analyzer.create_outlier(doc_fields, doc_without_outlier)
        expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
                                   outlier_summary='dummy summary',
                                   doc=doc_without_outlier)
        expected_outlier.outlier_dict['model_name'] = 'dummy_test'
        expected_outlier.outlier_dict['model_type'] = 'analyzer'
        expected_outlier.outlier_dict['elasticsearch_filter'] = 'es_valid_query'

        self.assertTrue(outlier.outlier_dict == expected_outlier.outlier_dict)

    def test_simple_process_outlier_save_es(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzer = AnalyzerFactory.create(use_case_analyzer_dummy_test)

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        self.test_es.add_doc(doc_without_outlier)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)

        doc_fields = doc_without_outlier["_source"]
        outlier = analyzer.create_outlier(doc_fields, doc_without_outlier)

        es.save_outlier(outlier)

        result = [elem for elem in es._scan()][0]

        self.assertEqual(result, doc_with_outlier)

    def test_arbitrary_key_config_present_in_analyzer(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzer = AnalyzerFactory.create(use_case_analyzer_arbitrary_dummy_test)

        self.assertDictEqual(analyzer.extra_model_settings, {"test_arbitrary_key": "arbitrary_value",
                                                             "elasticsearch_filter": "es_valid_query"})

    def test_create_multi_with_empty_config(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzers = AnalyzerFactory.create_multi(config_file_analyzer_test_01)

        self.assertTrue(len(analyzers) == 0)
    
    def test_create_multi_with_single(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzers = AnalyzerFactory.create_multi(use_case_analyzer_arbitrary_dummy_test)

        self.assertTrue(len(analyzers) == 1)
    
    def test_create_multi_with_malformed_duplicate_option(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzers = AnalyzerFactory.create_multi(use_case_analyzer_multi_malformed_duplicate_option, {'strict': False})

        self.assertTrue(len(analyzers) == 3)

    def test_create_multi_with_malformed_duplicate_section(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzers = AnalyzerFactory.create_multi(use_case_analyzer_multi_malformed_duplicate_section, {'strict': False})

        self.assertTrue(len(analyzers) == 2)
    
    def test_create_multi_with_malformed_duplicate_option_strict(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)

        with self.assertRaises(configparser.DuplicateOptionError):
            AnalyzerFactory.create_multi(use_case_analyzer_multi_malformed_duplicate_option)

    def test_create_multi_with_malformed_duplicate_section_strict(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)

        with self.assertRaises(configparser.DuplicateSectionError):
            AnalyzerFactory.create_multi(use_case_analyzer_multi_malformed_duplicate_section)

    def test_create_multi_mixed_types(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzers = AnalyzerFactory.create_multi(use_case_analyzer_files_path + "analyzer_multi_mixed_types.conf")

        simplequery_analyzer = analyzers[0]
        metrics_analyzer = analyzers[1]
        terms_analyzer = analyzers[2]

        self.assertTrue(simplequery_analyzer.model_type == 'simplequery')
        self.assertTrue(metrics_analyzer.model_type == 'metrics')
        self.assertTrue(terms_analyzer.model_type == 'terms')

    def test_default_timestamp_field(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_01)
        analyzer = AnalyzerFactory.create(use_case_analyzer_dummy_test)
        timestamp_field = analyzer.model_settings["timestamp_field"]
        default_timestamp_field = "@timestamp"
        self.assertEquals(timestamp_field, default_timestamp_field)

    def test_non_default_timestamp_field(self):
        self.test_settings.change_configuration_path(config_file_analyzer_test_with_custom_timestamp_field)
        analyzer = AnalyzerFactory.create(use_case_analyzer_dummy_test)
        timestamp_field = analyzer.model_settings["timestamp_field"]
        non_default_timestamp_field = "timestamp"
        self.assertEquals(timestamp_field, non_default_timestamp_field)
