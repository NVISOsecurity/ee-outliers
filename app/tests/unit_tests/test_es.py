import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.test_stubs.test_stub_analyzer import TestStubAnalyzer
from tests.unit_tests.utils.update_settings import UpdateSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate
from helpers.singletons import es
import helpers.es
from helpers.es import build_search_query
import helpers.analyzerfactory
from helpers.analyzerfactory import AnalyzerFactory

test_file_whitelist_path_config = "/app/tests/unit_tests/files/whitelist_tests_01_with_general.conf"
config_file_path = "/app/tests/unit_tests/files/"
config_file_simplequery_test_01 = config_file_path + "simplequery_test_01.conf"

helpers.analyzerfactory.CLASS_MAPPING["analyzer"] = TestStubAnalyzer


class TestEs(unittest.TestCase):

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()

    def tearDown(self):
        self.test_es.restore_es()
        self.test_settings.restore_default_configuration_path()

    def test_add_tag_to_document_no_tag(self):
        elem = {
            "_source": {
                "key": {
                    "test": 1
                }
            }
        }
        expected_result = copy.deepcopy(elem)
        expected_result["_source"]["tags"] = ["new_tag"]

        new_doc_result = helpers.es.add_tag_to_document(elem, "new_tag")
        self.assertEqual(new_doc_result, expected_result)

    def test_add_tag_to_document_already_a_tag(self):
        elem = {
                "_source": {
                    "key": {
                        "test": 1
                    },
                    "tags": ["ok"]
                }
            }
        expected_result = copy.deepcopy(elem)
        expected_result["_source"]["tags"].append("new_tag")

        new_doc_result = helpers.es.add_tag_to_document(elem, "new_tag")
        self.assertEqual(new_doc_result, expected_result)

    def test_remove_all_whitelisted_outliers(self):
        self.test_settings.change_configuration_path(test_file_whitelist_path_config)

        doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(doc_generate.generate_document({
            "create_outlier": True, "outlier_observation": "dummy observation",
            "outlier.model_name": "dummy_test", "outlier.model_type": "analyzer",
            "command_query": "osquery_get_all_processes_with_listening_conns.log"}))

        # Check that outlier correctly generated
        result = [doc for doc in es._scan()][0]
        self.assertTrue("outliers" in result["_source"])

        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/analyzer/analyzer_dummy_test.conf")

        # Remove whitelisted outlier
        es.remove_all_whitelisted_outliers({"analyzer_dummy_test": analyzer})

        # Check that outlier is correctly remove
        result = [doc for doc in es._scan()][0]
        self.assertFalse("outliers" in result["_source"])

    def test_get_highlight_settings_with_metrics_analyzer(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/metrics/metrics_dummy_test.conf")
        highlight_settings = es._get_highlight_settings(analyzer.model_settings)
        self.assertTrue(highlight_settings is None)

    def test_get_highlight_settings_with_terms_analyzer(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = AnalyzerFactory.create("/app/tests/unit_tests/files/use_cases/terms/terms_dummy_test.conf")
        highlight_settings = es._get_highlight_settings(analyzer.model_settings)
        self.assertTrue(highlight_settings is None)

    def test_get_highlight_settings_with_simplequery_analyzer_and_highlight_match_activated(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        use_case_file = "/app/tests/unit_tests/files/use_cases/simplequery/" \
                        "simplequery_dummy_test_highlight_match_activated.conf"
        analyzer = AnalyzerFactory.create(use_case_file)
        highlight_settings = es._get_highlight_settings(analyzer.model_settings)
        highlight_settings_test = dict()

        highlight_settings_test["pre_tags"] = ["<value>"]
        highlight_settings_test["post_tags"] = ["</value>"]
        highlight_settings_test["fields"] = dict()
        highlight_settings_test["fields"]["*"] = dict()

        self.assertTrue(highlight_settings == highlight_settings_test)

    def test_get_highlight_settings_with_simplequery_analyzer_and_highlight_match_unactivated(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        use_case_file = "/app/tests/unit_tests/files/use_cases/simplequery/" \
                        "simplequery_dummy_test_highlight_match_unactivated.conf"
        analyzer = AnalyzerFactory.create(use_case_file)
        highlight_settings = es._get_highlight_settings(analyzer.model_settings)

        self.assertTrue(highlight_settings is None)

    def test_get_highlight_settings_with_simplequery_analyzer_without_highlight_parameter(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        use_case_file = "/app/tests/unit_tests/files/use_cases/simplequery/simplequery_dummy_test.conf"
        analyzer = AnalyzerFactory.create(use_case_file)
        highlight_settings = es._get_highlight_settings(analyzer.model_settings)

        self.assertTrue(highlight_settings is None)

    def test_build_search_query(self):
        self.test_settings.change_configuration_path(config_file_simplequery_test_01)
        use_case_file = "/app/tests/unit_tests/files/use_cases/simplequery/simplequery_dummy_test.conf"
        analyzer = AnalyzerFactory.create(use_case_file)
        timestamp_field, history_window_days, history_window_hours = es._get_history_window(analyzer.model_settings)
        search_range = es.get_time_filter(days=history_window_days, hours=history_window_hours,
                                          timestamp_field=timestamp_field)
        dsl_search_query_1 = build_search_query(search_range=search_range,
                                              search_query=analyzer.search_query)
        dsl_search_query_2 = dict()
        dsl_search_query_2['query'] = dict()
        dsl_search_query_2['query']['bool'] = dict()
        dsl_search_query_2['query']['bool']['filter'] = list()
        dsl_search_query_2['query']['bool']['filter'].append(search_range)
        dsl_search_query_2['query']['bool']['filter'].extend(analyzer.search_query["filter"].copy())

        self.assertEquals(dsl_search_query_1, dsl_search_query_2)
