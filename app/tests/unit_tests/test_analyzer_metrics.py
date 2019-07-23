import json
import unittest

import base64
import copy
import re
import numpy as np
from statistics import median, mean
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

    def _test_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
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

    ################
    # Util methods #
    def _compute_max_hex_encoded_length(self, hex_value, trigger_sensitivity=None):
        target_value_words = re.split("[^a-fA-F0-9+]", str(hex_value))

        index_word = 0
        max_len = 0
        while index_word < len(target_value_words) and (trigger_sensitivity is None or max_len <= trigger_sensitivity):
            word = target_value_words[index_word]

            # let's match at least 5 characters, meaning 10 hex digits
            if len(word) > 10 and helpers.utils.is_hex_encoded(word):
                max_len = len(word)
            index_word += 1
        return max_len

    def _compute_max_base64_encoded_length(self, base64_value, trigger_sensitivity=None):
        target_value_words = re.split("[^A-Za-z0-9+/=]", str(base64_value))

        index_word = 0
        max_len = 0
        while index_word < len(target_value_words) and (trigger_sensitivity is None or max_len <= trigger_sensitivity):
            decoded_word = helpers.utils.is_base64_encoded(target_value_words[index_word])

            # let's match at least 5 characters, meaning 10 hex digits
            if decoded_word and len(decoded_word) >= 5:
                max_len = len(decoded_word)
            index_word += 1
        return max_len

    def _compute_max_url_length(self, url_value, trigger_sensitivity=None):
        target_value_words = url_value.replace('"', ' ').split()

        index_word = 0
        total_len = 0
        while index_word < len(target_value_words) and \
                (trigger_sensitivity is None or total_len <= trigger_sensitivity):
            word = target_value_words[index_word]
            if helpers.utils.is_url(word):
                total_len += len(word)
            index_word += 1
        return total_len

    def _compute_list_target_per_deployment(self, documents, target_parent_key, target_key):
        target_per_deployment = defaultdict(list)
        for generate_doc in documents:
            deployment_name = generate_doc["_source"]["meta"]["deployment_name"]
            target_value = generate_doc["_source"][target_parent_key][target_key]
            target_per_deployment[deployment_name].append(target_value)
        return target_per_deployment

    def _generate_random_documents(self):
        self.doc_generator = GenerateDummyDocuments()
        all_doc = self.doc_generator.create_documents(20)
        self.test_es.add_multiple_docs(all_doc)
        return all_doc

    def _compute_list_val_user_id_per_deployment(self, all_doc):
        list_val_user_id_per_deployment = {}
        list_user_id_per_deployment = self._compute_list_target_per_deployment(all_doc, "meta", "user_id")
        for deployment in list_user_id_per_deployment:
            list_val_user_id_per_deployment[deployment] = []
            for user_id in list_user_id_per_deployment[deployment]:
                list_val_user_id_per_deployment[deployment].append(int(user_id))
        return list_val_user_id_per_deployment

    def _compute_hostname_len_per_deployment(self, all_doc):
        hostname_len_per_deployment = {}
        hostname_per_deployment = self._compute_list_target_per_deployment(all_doc, "meta", "hostname")
        for deployment_name in hostname_per_deployment:
            hostname_len_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_len_per_deployment[deployment_name].append(len(hostname))
        return hostname_len_per_deployment

    def _compute_hostname_entropy_per_deployment(self, all_doc):
        hostname_entropy_per_deployment = {}
        hostname_per_deployment = self._compute_list_target_per_deployment(all_doc, "meta", "hostname")
        for deployment_name in hostname_per_deployment:
            hostname_entropy_per_deployment[deployment_name] = []
            for hostname in hostname_per_deployment[deployment_name]:
                hostname_entropy_per_deployment[deployment_name].append(helpers.utils.shannon_entropy(hostname))
        return hostname_entropy_per_deployment

    def _compute_hex_val_length_per_deployment(self, all_doc):
        hex_val_length_per_deployment = {}
        hex_val_per_deployment = self._compute_list_target_per_deployment(all_doc, "test", "hex_value")
        for deployment_name in hex_val_per_deployment:
            hex_val_length_per_deployment[deployment_name] = []
            for hex_value in hex_val_per_deployment[deployment_name]:
                value = self._compute_max_hex_encoded_length(hex_value)
                hex_val_length_per_deployment[deployment_name].append(value)
        return hex_val_length_per_deployment

    def _compute_base64_length_per_deployment(self, all_doc):
        base64_length_per_deployment = {}
        base64_val_per_deployment = self._compute_list_target_per_deployment(all_doc, "test", "base64_value")
        for deployment_name in base64_val_per_deployment:
            base64_length_per_deployment[deployment_name] = []
            for base64_value in base64_val_per_deployment[deployment_name]:
                value = self._compute_max_base64_encoded_length(base64_value)
                base64_length_per_deployment[deployment_name].append(value)
        return base64_length_per_deployment

    def _compute_url_length_per_deployment(self, all_doc):
        url_length_per_deployment = {}
        url_val_per_deployment = self._compute_list_target_per_deployment(all_doc, "test", "url_value")
        for deployment_name in url_val_per_deployment:
            url_length_per_deployment[deployment_name] = []
            for url_value in url_val_per_deployment[deployment_name]:
                value = self._compute_max_url_length(url_value)
                url_length_per_deployment[deployment_name].append(value)
        return url_length_per_deployment

    def _compute_frontiere_list_percentile(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        for deployment_name in trigger_per_deployment:
            frontiere_list[deployment_name] = np.percentile(list(set(trigger_per_deployment[deployment_name])),
                                                            trigger_sensitivity)
        return frontiere_list

    def _compute_frontiere_list_max(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        for deployment_name in trigger_per_deployment:
            frontiere_list[deployment_name] = np.float64(max(trigger_per_deployment[deployment_name]) *
                                                         (trigger_sensitivity / 100))
        return frontiere_list

    def _compute_frontiere_list_mean(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        for deployment_name in trigger_per_deployment:
            frontiere_list[deployment_name] = np.float64(mean(trigger_per_deployment[deployment_name]) *
                                                         (trigger_sensitivity / 100))
        return frontiere_list

    def _compute_frontiere_list_median(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        for deployment_name in trigger_per_deployment:
            frontiere_list[deployment_name] = np.float64(median(trigger_per_deployment[deployment_name]) *
                                                         (trigger_sensitivity / 100))
        return frontiere_list

    def _compute_frontiere_list_mad_low(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        new_document_that_must_be_generate = defaultdict(list)
        for deployment_name in trigger_per_deployment:
            values_array = trigger_per_deployment[deployment_name]

            new_value = max(values_array)
            while helpers.utils.get_mad_decision_frontier(values_array, trigger_sensitivity, "low") < 0:
                values_array.append(new_value)
                new_document_that_must_be_generate[deployment_name].append(new_value)

            frontiere_list[deployment_name] = helpers.utils.get_mad_decision_frontier(values_array,
                                                                                      trigger_sensitivity, "low")
            if frontiere_list[deployment_name] == np.nanmedian(values_array):
                frontiere_list[deployment_name] = helpers.utils.get_stdev_decision_frontier(values_array, 1, "low")

        return frontiere_list, new_document_that_must_be_generate

    def _compute_frontiere_list_mad_high(self, trigger_per_deployment, trigger_sensitivity):
        frontiere_list = {}
        for deployment_name in trigger_per_deployment:
            values_array = trigger_per_deployment[deployment_name]
            # median absolute deviation
            mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)
            frontiere_list[deployment_name] = np.nanmedian(values_array) + trigger_sensitivity * mad
            if frontiere_list[deployment_name] == np.nanmedian(values_array):
                frontiere_list[deployment_name] = helpers.utils.get_stdev_decision_frontier(values_array, 1, "high")

        return frontiere_list

    def _compute_frontiere_list_stdev(self, trigger_per_deployment, trigger_sensitivity, trigger_on):
        frontiere_list = {}
        new_document_that_must_be_generate = defaultdict(list)
        for deployment_name in trigger_per_deployment:
            values_array = trigger_per_deployment[deployment_name]

            new_value = max(values_array)
            while helpers.utils.get_stdev_decision_frontier(values_array, trigger_sensitivity, trigger_on) < 0:
                values_array.append(new_value)
                new_document_that_must_be_generate[deployment_name].append(new_value)

            frontiere_list[deployment_name] = helpers.utils.get_stdev_decision_frontier(values_array,
                                                                                        trigger_sensitivity, trigger_on)

        return frontiere_list, new_document_that_must_be_generate

    #############################
    # Begin test for percentile #
    def test_metrics_generated_document_numerical_value_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_percentile_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_percentile")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_percentile(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ###################################
    # Begin test for pct_of_max_value #
    def test_metrics_generated_document_numerical_value_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_pct_of_max_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_pct_of_max_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_max(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ######################################
    # Begin test for pct_of_median_value #
    def test_metrics_generated_document_numerical_value_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_pct_of_median_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_pct_of_median_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_median(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ###################################
    # Begin test for pct_of_avg_value #
    def test_metrics_generated_document_numerical_value_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_pct_of_avg_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_pct_of_avg_value")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 25
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mean(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ######################
    # Begin test for mad #
    def test_metrics_generated_document_numerical_value_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_mad")
        all_doc = self._generate_random_documents()
        # Force positive value
        for doc in all_doc[:]:
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname = doc["_source"]["meta"]["hostname"]
            extra_doc = self.doc_generator.generate_document(hostname=hostname, deployment_name=deployment_name,
                                                             user_id=100)
            self.test_es.add_doc(extra_doc)
            all_doc.append(extra_doc)

        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(list_val_user_id_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            hostname_len_per_deployment, trigger_sensitivity)

        # Add document to avoir bugs
        for deployment_name, list_len_hostname in new_document_that_must_be_generate.items():
            for len_hostname in list_len_hostname:
                extra_doc = self.doc_generator.generate_document(deployment_name=deployment_name,
                                                                 hostname="a"*len_hostname)
                self.test_es.add_doc(extra_doc)
        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(hostname_len_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1

        # Fetch hostname with max entropy
        hostname_max_entropy = ""
        max_entropy = 0
        for doc in all_doc:
            hostname = doc["_source"]["meta"]["hostname"]
            hostname_entropy = helpers.utils.shannon_entropy(hostname)
            if hostname_entropy > max_entropy:
                hostname_max_entropy = hostname
                max_entropy = hostname_entropy

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            hostname_entropy_per_deployment, trigger_sensitivity)

        for deployment_name, list_entropy in new_document_that_must_be_generate.items():
            for _ in range(len(list_entropy)):
                extra_doc = self.doc_generator.generate_document(deployment_name=deployment_name,
                                                                 hostname=hostname_max_entropy)
                self.test_es.add_doc(extra_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(hostname_entropy_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_mad")
        all_doc = self._generate_random_documents()
        # Force positive value
        for doc in all_doc[:]:
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname = doc["_source"]["meta"]["hostname"]
            extra_doc = self.doc_generator.generate_document(hostname=hostname, deployment_name=deployment_name,
                                                             test_hex_value="5468697320697320612076657279206c6f6e6720" +
                                                                            "7465737420746f2061766f6964206e756c6c2076" +
                                                                            "616c7565")
            self.test_es.add_doc(extra_doc)
            all_doc.append(extra_doc)

        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(hex_val_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_mad")
        all_doc = self._generate_random_documents()
        # Force positive value
        for doc in all_doc[:]:
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname = doc["_source"]["meta"]["hostname"]
            extra_doc = self.doc_generator.generate_document(hostname=hostname, deployment_name=deployment_name,
                                                             test_base64_value="VGhpcyBpcyBhIHZlcnkgYmlnIHRleHQgdG8gY" +
                                                                               "mUgYWJsZSB0byBkbyBzb21lIHRlc3Rz")
            self.test_es.add_doc(extra_doc)
            all_doc.append(extra_doc)

        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(base64_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_mad")
        all_doc = self._generate_random_documents()
        # Force positive value
        for doc in all_doc[:]:
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname = doc["_source"]["meta"]["hostname"]
            extra_doc = self.doc_generator.generate_document(hostname=hostname, deployment_name=deployment_name,
                                                             test_url_value="http://long-url-example-to-avoid-" +
                                                                            "negative-mad.test")
            self.test_es.add_doc(extra_doc)
            all_doc.append(extra_doc)

        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_mad_low(
            url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_mad_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_mad")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list = self._compute_frontiere_list_mad_high(url_length_per_deployment, trigger_sensitivity)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ########################
    # Begin test for stdev #
    def test_metrics_generated_document_numerical_value_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1

        # Pre Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            list_val_user_id_per_deployment, trigger_sensitivity, "low")

        # Force positive value
        for deployment_name, list_user_id in new_document_that_must_be_generate.items():
            for user_id in list_user_id:
                extra_doc = self.doc_generator.generate_document(deployment_name=deployment_name, user_id=user_id)
                self.test_es.add_doc(extra_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        list_val_user_id_per_deployment = self._compute_list_val_user_id_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            list_val_user_id_per_deployment, trigger_sensitivity, "high")
        self.assertEquals(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hostname_len_per_deployment, trigger_sensitivity, "low")

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_len_per_deployment = self._compute_hostname_len_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hostname_len_per_deployment, trigger_sensitivity, "high")
        self.assertEqual(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_len = len(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_len > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hostname_entropy_per_deployment, trigger_sensitivity, "low")

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hostname_entropy_per_deployment = self._compute_hostname_entropy_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hostname_entropy_per_deployment, trigger_sensitivity, "high")
        self.assertEqual(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hostname_entropy = helpers.utils.shannon_entropy(doc["_source"]["meta"]["hostname"])
            self.assertEqual(hostname_entropy > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hex_val_length_per_deployment, trigger_sensitivity, "low")

        # Force positive value
        for deployment_name, list_hex_val_length in new_document_that_must_be_generate.items():
            for hex_val_length in list_hex_val_length:
                extra_doc = self.doc_generator.generate_document(deployment_name=deployment_name,
                                                                 test_hex_value="0"*hex_val_length)
                self.test_es.add_doc(extra_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        hex_val_length_per_deployment = self._compute_hex_val_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            hex_val_length_per_deployment, trigger_sensitivity, "high")
        self.assertEqual(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            hex_value_encoded = self._compute_max_hex_encoded_length(doc["_source"]["test"]["hex_value"])
            self.assertEqual(hex_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            base64_length_per_deployment, trigger_sensitivity, "low")

        # Force positive value
        for deployment_name, list_base64_val_length in new_document_that_must_be_generate.items():
            for base64_val_length in list_base64_val_length:
                extra_doc = self.doc_generator.generate_document(deployment_name=deployment_name,
                                                                 test_hex_value=base64.b64encode(
                                                                     ("0" * base64_val_length).encode()))
                self.test_es.add_doc(extra_doc)
        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        base64_length_per_deployment = self._compute_base64_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            base64_length_per_deployment, trigger_sensitivity, "high")
        self.assertEqual(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            base64_value_encoded = self._compute_max_base64_encoded_length(doc["_source"]["test"]["base64_value"])
            self.assertEqual(base64_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_stdev")
        all_doc = self._generate_random_documents()

        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            url_length_per_deployment, trigger_sensitivity, "low")

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded < frontiere_list[deployment_name], "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_stdev_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_stdev")
        all_doc = self._generate_random_documents()
        trigger_sensitivity = 1
        analyzer.evaluate_model()

        # Compute expected result
        url_length_per_deployment = self._compute_url_length_per_deployment(all_doc)
        frontiere_list, new_document_that_must_be_generate = self._compute_frontiere_list_stdev(
            url_length_per_deployment, trigger_sensitivity, "high")
        self.assertEqual(len(new_document_that_must_be_generate), 0)

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            url_value_encoded = self._compute_max_url_length(doc["_source"]["test"]["url_value"])
            self.assertEqual(url_value_encoded > frontiere_list[deployment_name], "outliers" in doc["_source"])

    ########################
    # Begin test for float #
    def test_metrics_generated_document_numerical_value_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_low_float")
        all_doc = self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id < 2, "outliers" in doc["_source"])

    def test_metrics_generated_document_numerical_value_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_numerical_value_high_float")
        self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            user_id = int(doc["_source"]["meta"]["user_id"])
            self.assertEqual(user_id > 2, "outliers" in doc["_source"])

    def test_metrics_generated_document_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_low_float")
        self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(len(hostname) < 14, "outliers" in doc["_source"])

    def test_metrics_generated_document_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_length_high_float")
        self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(len(hostname) > 14, "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_low_float")
        self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(helpers.utils.shannon_entropy(hostname) < 3, "outliers" in doc["_source"])

    def test_metrics_generated_document_entropy_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_entropy_high_float")
        self._generate_random_documents()
        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(helpers.utils.shannon_entropy(hostname) > 3, "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_low_float")
        self._generate_random_documents()
        trigger_sensitivity = 12
        analyzer.evaluate_model()

        for doc in es.scan():
            hex_value = doc["_source"]["test"]["hex_value"]
            max_len = self._compute_max_hex_encoded_length(hex_value, trigger_sensitivity)

            self.assertEqual(max_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_hex_encoded_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_hex_encoded_length_high_float")
        self._generate_random_documents()
        trigger_sensitivity = 12
        analyzer.evaluate_model()

        for doc in es.scan():
            hex_value = doc["_source"]["test"]["hex_value"]
            max_len = self._compute_max_hex_encoded_length(hex_value, trigger_sensitivity)

            self.assertEqual(max_len > trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_low_float")
        self._generate_random_documents()
        trigger_sensitivity = 8
        analyzer.evaluate_model()

        for doc in es.scan():
            base64_value = doc["_source"]["test"]["base64_value"]
            max_len = self._compute_max_base64_encoded_length(base64_value, trigger_sensitivity)

            self.assertEqual(max_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_base64_encoded_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_base64_encoded_length_high_float")
        self._generate_random_documents()
        trigger_sensitivity = 8
        analyzer.evaluate_model()

        for doc in es.scan():
            base64_value = doc["_source"]["test"]["base64_value"]
            max_len = self._compute_max_base64_encoded_length(base64_value, trigger_sensitivity)

            self.assertEqual(max_len > trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_low_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_low_float")
        self._generate_random_documents()
        trigger_sensitivity = 18
        analyzer.evaluate_model()

        for doc in es.scan():
            url_value = doc["_source"]["test"]["url_value"]
            total_len = self._compute_max_url_length(url_value, trigger_sensitivity)

            self.assertEqual(total_len < trigger_sensitivity, "outliers" in doc["_source"])

    def test_metrics_generated_document_url_length_high_float_value(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_01.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_url_length_high_float")
        self._generate_random_documents()
        trigger_sensitivity = 18
        analyzer.evaluate_model()

        for doc in es.scan():
            url_value = doc["_source"]["test"]["url_value"]
            total_len = self._compute_max_url_length(url_value, trigger_sensitivity)

            self.assertEqual(total_len > trigger_sensitivity, "outliers" in doc["_source"])
