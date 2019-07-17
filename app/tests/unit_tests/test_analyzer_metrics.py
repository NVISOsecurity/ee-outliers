import json
import unittest

import copy
import re
import numpy as np
from collections import defaultdict

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.metrics import MetricsAnalyzer
from helpers.singletons import logging, es
import helpers.utils
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.generate_dummy_documents import GenerateDummyDocuments

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))


class TestMetricsAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = TestSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def test_metrics_whitelist_batch_document_not_process_all(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_with_whitelist.conf")
        analyzer = MetricsAnalyzer("metrics_length_dummy_test")

        # Whitelisted (ignored)
        doc1_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_01_test_file)
        self.test_es.add_doc(doc1_without_outlier)
        # Not whitelisted (add)
        doc2_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_02_test_file)
        self.test_es.add_doc(doc2_without_outlier)
        # Not whitelisted
        doc3_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_03_test_file)
        self.test_es.add_doc(doc3_without_outlier)

        analyzer.evaluate_model()

        self.assertEqual(len(analyzer.outliers), 2)

    ########################
    # Begin test for float #
    def test_metrics_generated_document_numerical_value_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < 2, "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > 2, "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(len(hostname) < 14, "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(len(hostname) > 14, "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(helpers.utils.shannon_entropy(hostname) < 3, "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_float")

        doc_generator = GenerateDummyDocuments()
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(helpers.utils.shannon_entropy(hostname) > 3, "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 12
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hex_value = doc["_source"]["test"]["hex_value"]

            target_value_words = re.split("[^a-fA-F0-9+]", str(hex_value))

            index_word = 0
            max_len = 0
            while index_word < len(target_value_words) and max_len < trigger_sensitivity:
                word = target_value_words[index_word]

                # let's match at least 5 characters, meaning 10 hex digits
                if len(word) > 10 and helpers.utils.is_hex_encoded(word):
                    max_len = len(word)
                index_word += 1

            self.assertEqual(max_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 12
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hex_value = doc["_source"]["test"]["hex_value"]

            target_value_words = re.split("[^a-fA-F0-9+]", str(hex_value))

            index_word = 0
            max_len = 0
            while index_word < len(target_value_words) and max_len <= trigger_sensitivity:
                word = target_value_words[index_word]

                # let's match at least 5 characters, meaning 10 hex digits
                if len(word) > 10 and helpers.utils.is_hex_encoded(word):
                    max_len = len(word)
                index_word += 1

            self.assertEqual(max_len > trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 8
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            base64_value = doc["_source"]["test"]["base64_value"]

            target_value_words = re.split("[^A-Za-z0-9+/=]", str(base64_value))

            index_word = 0
            max_len = 0
            while index_word < len(target_value_words) and max_len < trigger_sensitivity:
                decoded_word = helpers.utils.is_base64_encoded(target_value_words[index_word])

                # let's match at least 5 characters, meaning 10 hex digits
                if decoded_word and len(decoded_word) >= 5:
                    max_len = len(decoded_word)
                index_word += 1

            self.assertEqual(max_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 8
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            base64_value = doc["_source"]["test"]["base64_value"]

            target_value_words = re.split("[^A-Za-z0-9+/=]", str(base64_value))

            index_word = 0
            max_len = 0
            while index_word < len(target_value_words) and max_len <= trigger_sensitivity:
                decoded_word = helpers.utils.is_base64_encoded(target_value_words[index_word])

                # let's match at least 5 characters, meaning 10 hex digits
                if decoded_word and len(decoded_word) >= 5:
                    max_len = len(decoded_word)
                index_word += 1

            self.assertEqual(max_len > trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 18
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            url_value = doc["_source"]["test"]["url_value"]
            target_value_words = url_value.replace('"', ' ').split()

            index_word = 0
            total_len = 0
            while index_word < len(target_value_words) and total_len <= trigger_sensitivity:
                word = target_value_words[index_word]
                if helpers.utils.is_url(word):
                    total_len += len(word)
                index_word += 1

            self.assertEqual(total_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_float")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 18
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            url_value = doc["_source"]["test"]["url_value"]
            target_value_words = url_value.replace('"', ' ').split()

            index_word = 0
            total_len = 0
            while index_word < len(target_value_words) and total_len <= trigger_sensitivity:
                word = target_value_words[index_word]
                if helpers.utils.is_url(word):
                    total_len += len(word)
                index_word += 1

            self.assertEqual(total_len > trigger_sensitivity, "outliers" in doc["_source"])

    #############################
    # Begin test for percentile #
    def test_metrics_generated_document_numerical_value_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        list_user_id_per_deployment = {}
        for generate_doc in all_doc:
            deployment_name = generate_doc["_source"]["meta"]["deployment_name"]
            id_user = int(generate_doc["_source"]["meta"]["user_id"])
            if deployment_name not in list_user_id_per_deployment:
                list_user_id_per_deployment[deployment_name] = []
            list_user_id_per_deployment[deployment_name].append(id_user)

        frontiere_list = {}
        for deployment_name in list_user_id_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(list_user_id_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        list_user_id_per_deployment = {}
        for generate_doc in all_doc:
            deployment_name = generate_doc["_source"]["meta"]["deployment_name"]
            id_user = int(generate_doc["_source"]["meta"]["user_id"])
            if deployment_name not in list_user_id_per_deployment:
                list_user_id_per_deployment[deployment_name] = []
            list_user_id_per_deployment[deployment_name].append(id_user)

        frontiere_list = {}
        for deployment_name in list_user_id_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(list_user_id_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def _compute_list_hostname_per_deployment(self, documents):
        hostname_per_deployment = defaultdict(list)
        for generate_doc in documents:
            deployment_name = generate_doc["_source"]["meta"]["deployment_name"]
            hostname = generate_doc["_source"]["meta"]["hostname"]
            hostname_per_deployment[deployment_name].append(hostname)
        return hostname_per_deployment

    def test_metrics_generated_document_length_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = {}
        hostname_per_deployment = self._compute_list_hostname_per_deployment(all_doc)
        for deployment_name in hostname_per_deployment:
            hostname_len_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_len_per_deployment[deployment_name].append(len(hostname))

        frontiere_list = {}
        for deployment_name in hostname_len_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(hostname_len_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = {}
        hostname_per_deployment = self._compute_list_hostname_per_deployment(all_doc)
        for deployment_name in hostname_per_deployment:
            hostname_len_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_len_per_deployment[deployment_name].append(len(hostname))

        frontiere_list = {}
        for deployment_name in hostname_len_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(hostname_len_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_enropy_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = {}
        hostname_per_deployment = self._compute_list_hostname_per_deployment(all_doc)
        for deployment_name in hostname_per_deployment:
            hostname_entropy_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_entropy_per_deployment[deployment_name].append(helpers.utils.shannon_entropy(hostname))

        frontiere_list = {}
        for deployment_name in hostname_entropy_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(hostname_entropy_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_enropy_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_percentile")

        doc_generator = GenerateDummyDocuments()
        trigger_sensitivity = 25
        all_doc = doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = {}
        hostname_per_deployment = self._compute_list_hostname_per_deployment(all_doc)
        for deployment_name in hostname_per_deployment:
            hostname_entropy_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_entropy_per_deployment[deployment_name].append(helpers.utils.shannon_entropy(hostname))

        frontiere_list = {}
        for deployment_name in hostname_entropy_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(hostname_entropy_per_deployment[deployment_name])),
                                                            trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_enropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_enropy > frontiere_list[deployment_name], "outliers" in doc["_source"])
