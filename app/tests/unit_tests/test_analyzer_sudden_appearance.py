import unittest

from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
from helpers.singletons import logging, es
from tests.unit_tests.utils.update_settings import UpdateSettings

from tests.unit_tests.utils.dummy_documents_generate import DummyDocumentsGenerate
from helpers.analyzerfactory import AnalyzerFactory

import datetime as dt

root_test_conf_files = "/app/tests/unit_tests/files/"
test_conf_file_with_whitelist = root_test_conf_files + "sudden_appearance_test_with_whitelist.conf"
test_conf_file_01 = root_test_conf_files + "sudden_appearance_test_01.conf"

root_test_use_case_files = "/app/tests/unit_tests/files/use_cases/sudden_appearance/"

DEFAULT_OUTLIERS_KEY_FIELDS = ["type", "reason", "summary", "model_name", "model_type", "total_outliers",
                               "elasticsearch_filter"]
EXTRA_OUTLIERS_KEY_FIELDS = ["prop_first_appear_in_time_window", "trigger_slide_window_proportion", "size_time_window",
                             "start_time_window", "end_time_window", "aggregator", "aggregator_value", "target",
                             "target_value", "num_target_value_in_window", "resume"]


def set_new_current_date(analyzer):
    now = dt.datetime.today()
    now = now.replace(hour=0, minute=0, second=0)
    now -= dt.timedelta(weeks=2, days=6, hours=0, minutes=0)
    analyzer.end_time = now


class TestSuddenAppearanceAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.verbosity = 0

    def setUp(self):
        self.test_es = TestStubEs()
        self.test_settings = UpdateSettings()

    def tearDown(self):
        # restore the default configuration file so we don't influence other unit tests that use the settings singleton
        self.test_settings.restore_default_configuration_path()
        self.test_es.restore_es()

    def test_sudden_appearance_whitelist_work_test_es_result(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        command_query = "SELECT * FROM dummy_table"  # must be bigger than the trigger value (here 3)
        nbr_generated_documents = 5

        # Generate document that match outlier
        command_name = "default_name_"
        for i in range(nbr_generated_documents):
            dummy_doc_generated = dummy_doc_generate.generate_document({"command_query": command_query,
                                                                        "command_name": command_name + str(i)})
            self.test_es.add_doc(dummy_doc_generated)

        whitelist_doc_generated = dummy_doc_generate.generate_document({"hostname": "whitelist_hostname",
                                                                        "command_query": command_query,
                                                                        "command_name": command_name + str(
                                                                            nbr_generated_documents)})
        self.test_es.add_doc(whitelist_doc_generated)

        # Run analyzer
        self.test_settings.change_configuration_path(test_conf_file_with_whitelist)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_dummy_test_01.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, nbr_generated_documents)

    def test_sudden_appearance_detect_no_outlier_es_check(self):

        # Generate documents
        dummy_doc_generate = DummyDocumentsGenerate()
        list_delta_hour = [1, 1, 1, 3, 3, 3, 4, 5, 5, 5, 15, 15]
        field_1_name = "user_id"
        list_field_1_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        field_2_name = "hostname"
        list_field_2_value = []
        for _ in range(len(list_delta_hour)):
            list_field_2_value.append("host1")
        generated_docs = dummy_doc_generate.generate_doc_time_variable_witt_custom_fields(list_delta_hour,
                                                                                          field_1_name,
                                                                                          list_field_1_value,
                                                                                          field_2_name,
                                                                                          list_field_2_value)
        self.test_es.add_multiple_docs(generated_docs)

        # Run analyzer
        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_dummy_test_02.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 0)

    def test_sudden_appearance_detect_one_outlier_es_check_1(self):
        # Generate documents
        dummy_doc_generate = DummyDocumentsGenerate()
        list_delta_hour = [1, 1, 1, 3, 3, 3, 4, 5, 5, 5, 15, 15]
        field_1_name = "user_id"
        list_field_1_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        field_2_name = "hostname"
        list_field_2_value = []
        for _ in range(len(list_delta_hour) - 1):
            list_field_2_value.append("host1")
        list_field_2_value.append("host2")
        generated_docs = dummy_doc_generate.generate_doc_time_variable_witt_custom_fields(list_delta_hour,
                                                                                          field_1_name,
                                                                                          list_field_1_value,
                                                                                          field_2_name,
                                                                                          list_field_2_value)
        self.test_es.add_multiple_docs(generated_docs)

        # Run analyzer
        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_dummy_test_02.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 1)

    def test_sudden_appearance_detect_one_outlier_es_check_2(self):
        # Generate documents
        dummy_doc_generate = DummyDocumentsGenerate()
        list_delta_hour = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 15]
        field_1_name = "user_id"
        list_field_1_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        field_2_name = "hostname"
        list_field_2_value = []
        for _ in range(len(list_delta_hour)):
            list_field_2_value.append("host1")

        generated_docs = dummy_doc_generate.generate_doc_time_variable_witt_custom_fields(list_delta_hour,
                                                                                          field_1_name,
                                                                                          list_field_1_value,
                                                                                          field_2_name,
                                                                                          list_field_2_value)
        self.test_es.add_multiple_docs(generated_docs)

        # Run analyzer
        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_dummy_test_03.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        nbr_outliers = 0
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                nbr_outliers += 1
        self.assertEqual(nbr_outliers, 1)

    def test_sudden_appearance_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_derived_fields_01.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("timestamp_year" in result['_source'])

    def test_sudden_appearance_no_derived_fields_in_doc(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_no_derived_fields.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]

        self.assertFalse("timestamp_year" in result['_source'])

    def test_sudden_appearance_derived_fields_in_outlier(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_derived_fields_02.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]
        self.assertTrue("derived_timestamp_year" in result['_source']['outliers'])

    def test_sudden_appearance_no_derived_fields(self):
        dummy_doc_generate = DummyDocumentsGenerate()
        self.test_es.add_doc(dummy_doc_generate.generate_document())

        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_no_derived_fields.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        result = [elem for elem in es._scan()][0]

        self.assertFalse("derived_timestamp_year" in result['_source']['outliers'])

    def test_sudden_extra_outlier_infos_all_present(self):
        # Generate documents
        dummy_doc_generate = DummyDocumentsGenerate()
        list_delta_hour = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 15]
        field_1_name = "user_id"
        list_field_1_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        field_2_name = "hostname"
        list_field_2_value = []
        for _ in range(len(list_delta_hour)):
            list_field_2_value.append("host1")

        generated_docs = dummy_doc_generate.generate_doc_time_variable_witt_custom_fields(list_delta_hour,
                                                                                          field_1_name,
                                                                                          list_field_1_value,
                                                                                          field_2_name,
                                                                                          list_field_2_value)
        self.test_es.add_multiple_docs(generated_docs)

        self.test_settings.change_configuration_path(test_conf_file_01)
        analyzer = AnalyzerFactory.create(root_test_use_case_files + "sudden_appearance_dummy_test_03.conf")
        set_new_current_date(analyzer)
        analyzer.evaluate_model()

        list_outlier = list()
        for elem in es._scan():
            if "outliers" in elem["_source"]:
                list_outlier.append(elem)

        all_fields_exists = [elem in EXTRA_OUTLIERS_KEY_FIELDS + DEFAULT_OUTLIERS_KEY_FIELDS
                             for elem in list_outlier[0]['_source']['outliers']]
        self.assertTrue(all(all_fields_exists))
