import unittest

import helpers.es

class TestEs(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_tag_to_document_no_tag(self):
        elem = {
            "_source": {
                "key": {
                    "test": 1
                }
            }
        }
        newElem = elem.copy()
        newdoc = helpers.es.add_tag_to_document(elem, "new_tag")
        newElem["_source"]["tags"] = ["new_tag"]
        self.assertEqual(newdoc, newElem)

    def test_add_tag_to_document_already_a_tag(self):
        elem = {
                "_source": {
                    "key": {
                        "test": 1
                    },
                    "tags": ["ok"]
                }
            }
        newElem = elem.copy()
        newdoc = helpers.es.add_tag_to_document(elem, "new_tag")
        newElem["_source"]["tags"].append("new_tag")
        self.assertEqual(newdoc, newElem)

    def test_add_outlier_to_document(self):
        pass
