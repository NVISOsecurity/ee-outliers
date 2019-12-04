import json
import unittest

import copy

from helpers.singletons import es
from tests.unit_tests.test_stubs.test_stub_analyzer import TestStubAnalyzer
from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.update_settings import UpdateSettings
from helpers.outlier import Outlier

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
        analyzer = TestStubAnalyzer("analyzer_dummy_test")

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
        analyzer = TestStubAnalyzer("analyzer_dummy_test")

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
        analyzer = TestStubAnalyzer("analyzer_arbitrary_dummy_test")

        self.assertDictEqual(analyzer.extra_model_settings, {"test_arbitrary_key": "arbitrary_value",
                                                             "elasticsearch_filter": "es_valid_query"})

    def test_arbitrary_key_config_with_whitelist_prefix_not_present_in_analyzer(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/analyzer_test_01.conf")
        analyzer = TestStubAnalyzer("analyzer_arbitrary_whitelist_prefix_dummy_test")

        self.assertDictEqual(analyzer.extra_model_settings, {"elasticsearch_filter": "es_valid_query"})
