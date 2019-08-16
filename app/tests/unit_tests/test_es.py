import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate
from helpers.singletons import es
import helpers.es

test_file_whitelist_path_config = "/app/tests/unit_tests/files/whitelist_tests_01_with_general.conf"


class TestEs(unittest.TestCase):

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = TestSettings()

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
        self.test_es.add_doc(doc_generate.generate_document(
            create_outlier=True, outlier_observation="dummy observation",
            command_query="osquery_get_all_processes_with_listening_conns.log"))

        # Check that outlier correctly generated
        result = [doc for doc in es._scan()][0]
        self.assertTrue("outliers" in result["_source"])

        # Remove whitelisted outlier
        es.remove_all_whitelisted_outliers()

        # Check that outlier is correctly remove
        result = [doc for doc in es._scan()][0]
        self.assertFalse("outliers" in result["_source"])
