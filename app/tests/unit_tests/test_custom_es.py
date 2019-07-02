import unittest

from helpers.custom_es import custom_es
from helpers.singletons import settings, logging

class TestCustomEs(unittest.TestCase):
    def setUp(self):
        self.es = custom_es(settings, logging)

    def test_add_data(self):
        dictionary_value = {
            "key.test": 1,
            "key.key2.ok": "test",
            "new.test": [12, "ok"]
        }
        expectedResult = [{
            "_source": {
                'key': {
                    'test': 1,
                    'key2': {
                        'ok': 'test'
                    }
                },
                'new': {
                    'test': [12, 'ok']
                }
            },
            "_id": 0
        }]

        self.es.add_data(dictionary_value)
        self.assertEqual(self.es.count_documents(), 1)
        self.assertEqual([elem for elem in self.es.scan()], expectedResult)

    def test_default_empty(self):
        self.assertEqual(self.es.count_documents(), 0)
        self.assertEqual([elem for elem in self.es.scan()], [])

    def test_generate_data(self):
        nbrGenerate = 5
        self.es.generate_data(nbrGenerate)
        result = [elem for elem in self.es.scan()]
        self.assertEqual(len(result), nbrGenerate)

    def test_remove_outliers(self):
        nbrGenerate = 5
        self.es.generate_data(nbrGenerate)
        self.es.remove_all_outliers()
        result = [elem for elem in self.es.scan()]
        self.assertEqual(len(result), 0)

    def test_update_es(self):
        dictionary_value = {
            "key.test": "test"
        }
        self.es.add_data(dictionary_value)
        result = [elem for elem in self.es.scan()][0]
        result["_source"]["key"]["test"] = "update_value"
        self.es._update_es(result)
        new_result = [elem for elem in self.es.scan()][0]
        self.assertEqual(new_result, result)

