import json
import unittest

import copy

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from analyzers.metrics import MetricsAnalyzer
from helpers.singletons import logging, es
from tests.unit_tests.utils.test_settings import TestSettings
from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate

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

    def test_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
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

    def _generate_metrics_doc_with_whitelist(self, doc_to_generate):
        # Use list of tuple (and not dict) to keep order
        dummy_doc_gen = DummyDocumentsGenerate()
        for aggregator, target_value, is_whitelist in doc_to_generate:
            deployment_name = None
            if is_whitelist:
                deployment_name = "whitelist-deployment"
            user_id = target_value
            hostname = aggregator

            doc_generated = dummy_doc_gen.generate_document(deployment_name=deployment_name, user_id=user_id,
                                                            hostname=hostname)
            self.test_es.add_doc(doc_generated)

    def test_metrics_batch_whitelist_three_outliers_one_whitelist(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_whitelist_batch.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_batch_whitelist_float")
        backup_min_eval_batch = MetricsAnalyzer.MIN_EVALUATE_BATCH
        MetricsAnalyzer.MIN_EVALUATE_BATCH = 5

        #            aggregator, target, is_whitelist
        doc_to_generate = [("agg1", 5, False),
                           ("agg1", 3, True),
                           ("agg2", 4, False),
                           ("agg2", 5, True),
                           # Batch limit
                           ("agg2", 3, False),
                           ("agg1", 5, False),
                           ("agg1", 7, False),  # Outlier
                           ("agg2", 2, False),
                           # Batch limit
                           ("agg1", 4, True),
                           ("agg2", 6, True),  # Outlier (but whitelist)
                           ("agg1", 3, False),
                           ("agg1", 5, False),
                           # Batch limit
                           ("agg2", 1, False),
                           ("agg2", 6, False),  # Outlier
                           ("agg1", 3, False)]
        self._generate_metrics_doc_with_whitelist(doc_to_generate)

        analyzer.evaluate_model()
        list_outliers = []
        for outlier in analyzer.outliers:
            list_outliers.append((outlier.outlier_dict["aggregator"], outlier.outlier_dict["target"]))

        self.assertEqual(list_outliers, [("agg1", "7"), ("agg2", "6")])
        MetricsAnalyzer.MIN_EVALUATE_BATCH = backup_min_eval_batch

    def test_metrics_batch_whitelist_outlier_detect_after_process_all_and_remove_whitelist(self):
        self.test_settings.change_configuration_path("/app/tests/unit_tests/files/metrics_test_whitelist_batch.conf")
        analyzer = MetricsAnalyzer("metrics_dummy_test_batch_whitelist_avg")
        backup_min_eval_batch = MetricsAnalyzer.MIN_EVALUATE_BATCH
        MetricsAnalyzer.MIN_EVALUATE_BATCH = 5

        #            aggregator, target, is_whitelist
        doc_to_generate = [("agg1", 5, False),
                           ("agg2", 5, False),
                           ("agg1", 5, False),
                           ("agg1", 3, False),
                           # Batch limit
                           ("agg1", 6, False),
                           ("agg2", 5, False),
                           ("agg1", 5, False),
                           ("agg1", 7, True)]
        self._generate_metrics_doc_with_whitelist(doc_to_generate)
        # The avg for agg1 is 5.1 but if we remove the whitelisted element, the avg is on 4.8

        analyzer.evaluate_model()
        list_outliers = []
        for outlier in analyzer.outliers:
            list_outliers.append((outlier.outlier_dict["aggregator"], outlier.outlier_dict["target"]))

        # Without the batch whitelist, the only outlier will be ("agg1", 6) (the ("agg1", 7) is whitelist).
        # But with batch whitelist, the avg is update and all value of "agg1" (except 3) are detected outlier
        self.assertEqual(list_outliers, [("agg1", "5"), ("agg1", "5"), ("agg1", "6"), ("agg1", "5")])
        MetricsAnalyzer.MIN_EVALUATE_BATCH = backup_min_eval_batch
