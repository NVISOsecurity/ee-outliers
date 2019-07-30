import unittest

import copy

import helpers.es


class TestEs(unittest.TestCase):

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
