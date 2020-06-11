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

# Monkey patch the test stub analyzer mapping in the AnalyzerFactory
helpers.analyzerfactory.CLASS_MAPPING["analyzer"] = TestStubAnalyzer

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_analyzer_outlier.json"))


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
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_dummy_test.conf")

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
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_dummy_test.conf")

        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        self.test_es.add_doc(doc_without_outlier)
        doc_with_outlier = copy.deepcopy(doc_with_outlier_test_file)

        doc_fields = doc_without_outlier["_source"]
        outlier = analyzer.create_outlier(doc_fields, doc_without_outlier)

        es.save_outlier(outlier)

        result = [elem for elem in es._scan()][0]

        self.assertEqual(result, doc_with_outlier)

    def test_arbitrary_key_config_present_in_analyzer(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_arbitrary_dummy_test.conf")

        self.assertDictEqual(analyzer.extra_model_settings, {"test_arbitrary_key": "arbitrary_value",
                                                             "elasticsearch_filter": "es_valid_query"})

    def test_create_multi_with_empty_config(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzers = AnalyzerFactory.create_multi("/app/tests/unit_tests/files/analyzer_test_01.conf")

        self.assertTrue(len(analyzers) == 0)
    
    def test_create_multi_with_single(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzers = AnalyzerFactory.create_multi("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_arbitrary_dummy_test.conf")

        self.assertTrue(len(analyzers) == 1)
    
    def test_create_multi_with_malformed(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzers = AnalyzerFactory.create_multi("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_multi_malformed.conf")

        self.assertTrue(len(analyzers) == 2)

    def test_create_multi_mixed_types(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzers = AnalyzerFactory.create_multi("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_multi_mixed_types.conf")

        simplequery_doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        simplequery_doc_fields = simplequery_doc_without_outlier["_source"]
        simplequery_outlier = analyzers[0].create_outlier(simplequery_doc_fields, simplequery_doc_without_outlier)
        simplequery_expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
                                   outlier_summary='dummy summary',
                                   doc=simplequery_doc_without_outlier)
        simplequery_expected_outlier.outlier_dict['model_type'] = 'simplequery'

        metrics_doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        metrics_doc_fields = simplequery_doc_without_outlier["_source"]
        metrics_outlier = analyzers[1].create_outlier(metrics_doc_fields, metrics_doc_without_outlier)
        metrics_expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
                                   outlier_summary='dummy summary',
                                   doc=simplequery_doc_without_outlier)
        metrics_expected_outlier.outlier_dict['model_type'] = 'metrics'

        terms_doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        terms_doc_fields = simplequery_doc_without_outlier["_source"]
        terms_outlier = analyzers[2].create_outlier(terms_doc_fields, terms_doc_without_outlier)
        terms_expected_outlier = Outlier(outlier_type=["dummy type"], outlier_reason=['dummy reason'],
                                   outlier_summary='dummy summary',
                                   doc=simplequery_doc_without_outlier)
        terms_expected_outlier.outlier_dict['model_type'] = 'terms'

        self.assertTrue(simplequery_outlier.outlier_dict == simplequery_expected_outlier.outlier_dict)
        self.assertTrue(metrics_outlier.outlier_dict == metrics_expected_outlier.outlier_dict)
        self.assertTrue(terms_outlier.outlier_dict == terms_expected_outlier.outlier_dict)
