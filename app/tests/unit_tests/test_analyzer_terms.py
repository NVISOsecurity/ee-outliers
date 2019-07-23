import json
import unittest

import copy
import numpy as np

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.terms import TermsAnalyzer
from helpers.singletons import settings, logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.generate_dummy_documents import GenerateDummyDocuments

doc_without_outliers_test_whitelist_01_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
doc_without_outliers_test_whitelist_02_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
doc_without_outliers_test_whitelist_03_test_file = json.load(
    open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))


class TestTermsAnalyzer(unittest.TestCase):
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

    def _test_terms_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_with_whitelist.conf")
        analyzer = TermsAnalyzer("terms_dummy_test")

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

    #############################
    # Begin test for percentile #
    def test_terms_generated_document_low_percentile_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_percentile_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.percentile([i for i in range(min_val, max_val + 1)], 25)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_percentile_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_percentile_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.percentile([i for i in range(min_val, max_val + 1)], 25)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_percentile_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_percentile_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.percentile([i for i in range(min_val, max_val + 1)], 25)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_percentile_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_percentile_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.percentile([i for i in range(min_val, max_val + 1)], 25)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    #############################
    # Begin test for pct of max #
    def test_terms_generated_document_low_pct_of_max_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_max_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(max_val * (80 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_max_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_max_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(max_val * (80 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_pct_of_max_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_max_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(max_val * (80 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_max_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_max_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(max_val * (80 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    #############################
    # Begin test for pct of med #
    def test_terms_generated_document_low_pct_of_med_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_med_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(5 * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_med_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_med_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(5 * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_pct_of_med_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_med_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(5 * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_med_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_med_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(5 * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    #############################
    # Begin test for pct of avg #
    def test_terms_generated_document_low_pct_of_avg_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_avg_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(np.median([i for i in range(min_val, max_val+1)]) * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_avg_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_avg_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = np.float64(np.median([i for i in range(min_val, max_val+1)]) * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_pct_of_avg_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_pct_of_avg_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(np.median([i for i in range(min_val, max_val+1)]) * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_pct_of_avg_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_pct_of_avg_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = np.float64(np.median([i for i in range(min_val, max_val + 1)]) * (90 / 100))
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    ######################
    # Begin test for mad #
    def test_terms_generated_document_low_mad_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_mad_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        list_values = [i for i in range(min_val, max_val+1)]
        frontiere = np.nanmedian(list_values) - 1 * np.nanmedian(np.absolute(list_values -
                                                                             np.nanmedian(list_values, 0)), 0)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_mad_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_mad_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        list_values = [i for i in range(min_val, max_val + 1)]
        frontiere = np.nanmedian(list_values) + 1 * np.nanmedian(np.absolute(list_values -
                                                                             np.nanmedian(list_values, 0)), 0)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_mad_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_mad_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        list_values = [i for i in range(min_val, max_val + 1)]
        frontiere = np.nanmedian(list_values) - 1 * np.nanmedian(np.absolute(list_values -
                                                                             np.nanmedian(list_values, 0)), 0)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_mad_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_mad_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 3
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        list_values = [i for i in range(min_val, max_val + 1)]
        frontiere = np.nanmedian(list_values) + 1 * np.nanmedian(np.absolute(list_values -
                                                                             np.nanmedian(list_values, 0)), 0)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    # TODO test: special case - if MAD is zero, then we use stdev instead of MAD

    #########################
    # Begin test for madpos #
    def test_terms_generated_document_low_madpos_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_madpos_value_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        frontiere = 0
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_madpos_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_madpos_value_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        frontiere = 0
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    ########################
    # Begin test for stdev #
    def test_terms_generated_document_low_stdev_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_stdev_value_within")

        doc_generator = GenerateDummyDocuments()
        nbr_val = 4
        min_trigger_sensitivity = 1
        default_value = 5  # Per default, 5 documents
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_max_sensitivity(
            nbr_val, min_trigger_sensitivity, max_difference, default_value)
        values = list(deployment_name_number.values())
        frontiere = np.nanmean(values) - min_trigger_sensitivity * np.std(values)

        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_stdev_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_stdev_value_within")

        doc_generator = GenerateDummyDocuments()
        nbr_val = 4
        min_trigger_sensitivity = 1
        default_value = 5  # Per default, 5 documents
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_min_sensitivity(
            nbr_val, min_trigger_sensitivity, max_difference, default_value)
        values = list(deployment_name_number.values())
        frontiere = np.nanmean(values) + min_trigger_sensitivity * np.std(values)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_stdev_value_across(self):
        # TODO possibility to have an error in this method (depending to the random)
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_stdev_value_across")

        doc_generator = GenerateDummyDocuments()
        nbr_val = 7
        min_trigger_sensitivity = 1
        default_value = 3  # Per default, 2 documents
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable_max_sensitivity(
            nbr_val, min_trigger_sensitivity, max_difference, default_value)
        values = list(hostname_name_number.values())
        frontiere = np.nanmean(values) - min_trigger_sensitivity * np.std(values)
        if frontiere < 0:
            print("frontiere negative in test_terms_generated_document_low_stdev_value_across")
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_stdev_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_stdev_value_across")

        doc_generator = GenerateDummyDocuments()
        nbr_val = 7
        min_trigger_sensitivity = 1
        default_value = 3  # Per default, 2 documents
        max_difference = 3  # Maximum difference between the number of document (so between 2 and 8 (included))
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable_min_sensitivity(
            nbr_val, min_trigger_sensitivity, max_difference, default_value)
        values = [elem for elem in hostname_name_number.values() if elem != 0]
        frontiere = np.nanmean(values) + min_trigger_sensitivity * np.std(values)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])

    ########################
    # Begin test for float #
    def test_terms_generated_document_low_float_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_float_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        frontiere = 5
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_float_value_within(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_float_within")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        frontiere = 5
        deployment_name_number, all_doc = doc_generator.create_doc_target_variable_range(min_val, max_val)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            deployment_name = doc["_source"]["meta"]["deployment_name"]
            self.assertEqual(deployment_name_number[deployment_name] > frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_low_float_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_low_float_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        frontiere = 5
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] < frontiere, "outliers" in doc["_source"])

    def test_terms_generated_document_high_float_value_across(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/terms_test_01.conf")
        analyzer = TermsAnalyzer("terms_dummy_test_high_float_across")

        doc_generator = GenerateDummyDocuments()
        min_val = 4
        max_val = 6
        frontiere = 5
        hostname_name_number, all_doc = doc_generator.create_doc_uniq_target_variable(min_val, max_val)
        self.test_es.add_multiple_docs(all_doc)

        analyzer.evaluate_model()

        for doc in es.scan():
            hostname = doc["_source"]["meta"]["hostname"]
            self.assertEqual(hostname_name_number[hostname] > frontiere, "outliers" in doc["_source"])
