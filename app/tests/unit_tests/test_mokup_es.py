import unittest

from tests.unit_tests.mokup.mokup_es import mokup_es
from helpers.singletons import settings, logging


class TestMokupEs(unittest.TestCase):
    def setUp(self):
        self.es = mokup_es(settings, logging)

    def _get_example_dictionary_key_value_and_expected(self):
        dictionary_value = {
            "key.test": 1,
            "key.key2.ok": "test",
            "new.test": [12, "ok"]
        }
        expected_result = [{
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
        return dictionary_value, expected_result

    def _get_example_doc(self):
        return {
            '_source': {
                'key': {
                    'test': 'value'
                }
            },
            '_id': 3
        }

    def test_add_one_data_correctly_encode(self):
        dictionary_value, expected_result = self._get_example_dictionary_key_value_and_expected()
        self.es.add_data(dictionary_value)
        self.assertEqual([elem for elem in self.es.scan()], expected_result)

    def test_no_data_count_zero_document(self):
        self.assertEqual(self.es.count_documents(), 0)

    def test_no_data_scan_return_empty_list(self):
        self.assertEqual([elem for elem in self.es.scan()], [])

    def test_generate_data_count_number_results_of_scan(self):
        nbr_generate = 5
        self.es.generate_data(nbr_generate)
        result = [elem for elem in self.es.scan()]
        self.assertEqual(len(result), nbr_generate)

    def test_generate_data_check_result_count_documents(self):
        nbr_generate = 5
        self.es.generate_data(nbr_generate)
        self.assertEqual(self.es.count_documents(), nbr_generate)

    def test_remove_outliers_give_empty_list(self):
        nbr_generate = 5
        self.es.generate_data(nbr_generate)
        self.es.remove_all_outliers()
        result = [elem for elem in self.es.scan()]
        self.assertEqual(len(result), 0)

    def test_remove_outliers_give_zero_count_documents(self):
        nbr_generate = 5
        self.es.generate_data(nbr_generate)
        self.es.remove_all_outliers()
        self.assertEqual(self.es.count_documents(), 0)

    def test_update_es_correcly_work(self):
        dictionary_value = self._get_example_dictionary_key_value_and_expected()[0]
        self.es.add_data(dictionary_value)
        result = [elem for elem in self.es.scan()][0]
        result["_source"]["key"]["test"] = "update_value"
        self.es._update_es(result)
        new_result = [elem for elem in self.es.scan()][0]
        self.assertEqual(new_result, result)

    def test_add_doc_same_id_raise_error(self):
        data = self._get_example_doc()
        self.es.add_doc(data)
        with self.assertRaises(KeyError):
            self.es.add_doc(data)
