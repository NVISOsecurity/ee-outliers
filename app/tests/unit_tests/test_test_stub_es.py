import json
import unittest

import copy

from tests.unit_tests.test_stub.test_stub_es import testStubEs
from helpers.singletons import settings, logging
from helpers.outlier import Outlier

doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
doc_with_outlier_with_derived_timestamp_test_file = json.load(
                                    open("/app/tests/unit_tests/files/doc_with_outlier_with_derived_timestamp.json"))


class TestTestStubEs(unittest.TestCase):
    def setUp(self):
        self.es = testStubEs(settings, logging)

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

    def test_flush_bulk_actions_using_one_save_outlier(self):
        doc_with_outlier_with_derived_timestamp = copy.deepcopy(doc_with_outlier_with_derived_timestamp_test_file)
        doc_with_outlier_with_derived_timestamp.pop('sort')  # field add by es
        doc_with_outlier_with_derived_timestamp.pop('_score')  # field add by es
        doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
        test_outlier = Outlier(outlier_type="dummy type", outlier_reason="dummy reason",
                               outlier_summary="dummy summary")
        test_outlier.outlier_dict["observation"] = "dummy observation"

        self.es.save_outlier(doc_without_outlier, test_outlier)
        result = [elem for elem in self.es.scan()][0]
        self.assertEqual(result, doc_with_outlier_with_derived_timestamp)
