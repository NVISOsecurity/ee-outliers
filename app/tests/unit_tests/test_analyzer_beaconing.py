# import unittest
#
# import copy
# import json
# import random
#
# from collections import defaultdict
#
# from tests.unit_tests.test_stubs.test_stub_es import TestStubEs
# from tests.unit_tests.utils.test_settings import TestSettings
# from helpers.singletons import settings, es, logging
# from analyzers.beaconing import BeaconingAnalyzer
# from helpers.outlier import Outlier
#
# doc_without_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))
# doc_without_outliers_test_whitelist_01_test_file = json.load(
#     open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_01.json"))
# doc_without_outliers_test_whitelist_02_test_file = json.load(
#     open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_02.json"))
# doc_without_outliers_test_whitelist_03_test_file = json.load(
#     open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_03.json"))
# doc_without_outliers_test_whitelist_04_test_file = json.load(
#     open("/app/tests/unit_tests/files/doc_without_outliers_test_whitelist_04.json"))
# doc_with_beaconing_outlier_test_file = json.load(open("/app/tests/unit_tests/files/doc_with_beaconing_outlier.json"))
# doc_with_beaconing_outlier_without_score_sort_test_file = json.load(
#     open("/app/tests/unit_tests/files/doc_with_beaconing_outlier_without_score_sort.json"))
#
# LIST_AGGREGATOR_VALUE = ["agg-WIN-EVB-draman", "agg-WIN-DRA-draman"]
# LIST_TARGET_VALUE = ["WIN-DRA-draman", "WIN-EVB-draman", "LINUX-DRA-draman"]
# LIST_DOC = [doc_without_outlier_test_file]
#
#
# class TestBeaconingAnalyzer(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         logging.verbosity = 0
#
#     def setUp(self):
#         self.test_es = TestStubEs()
#         self.test_settings = TestSettings()
#
#     def tearDown(self):
#         # restore the default configuration file so we don't influence other unit tests that use the settings singleton
#         self.test_settings.restore_default_configuration_path()
#         self.test_es.restore_es()
#
#     def _create_outliers(self, outlier_type, outlier_reason, outlier_summary, model_type, model_name, term, aggregator,
#                          confidence, decision_frontier, term_count, doc):
#         outlier = Outlier(outlier_type=outlier_type, outlier_reason=outlier_reason, outlier_summary=outlier_summary,
#                           doc=doc)
#         outlier.outlier_dict["model_type"] = model_type
#         outlier.outlier_dict["model_name"] = model_name
#         outlier.outlier_dict["term"] = term
#         outlier.outlier_dict["aggregator"] = aggregator
#         outlier.outlier_dict["confidence"] = confidence
#         outlier.outlier_dict["decision_frontier"] = decision_frontier
#         outlier.outlier_dict["term_count"] = term_count
#         return outlier
#
#     def test_evaluate_batch_for_outliers_not_enough_target_buckets_one_doc_max_two(self):
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#         analyzer.extract_additional_model_settings()
#
#         aggregator_value = LIST_AGGREGATOR_VALUE[0]
#         target_value = random.choice(LIST_TARGET_VALUE)
#         observations = {}
#         doc = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
#
#         result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
#         self.assertEqual(result, [])
#
#     def test_evaluate_batch_for_outliers_limit_target_buckets_two_doc_max_two(self):
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#         analyzer.extract_additional_model_settings()
#
#         # Create one document with one aggregator
#         aggregator_value = LIST_AGGREGATOR_VALUE[0]
#         target_value = random.choice(LIST_TARGET_VALUE)
#         observations = {}
#         doc = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
#         # Create a second document with another aggregator
#         aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
#         target_value2 = random.choice(LIST_TARGET_VALUE)
#         observations2 = {}
#         doc2 = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value2, observations2,
#                                                       doc2)
#
#         # Expect to get nothing due to "min_target_buckets" set to 2
#         result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
#         self.assertEqual(result, [])
#
#     def test_evaluate_batch_for_outliers_detect_two_outliers(self):
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#         analyzer.extract_additional_model_settings()
#
#         # Create one document with one aggregator [0] and one target [0]
#         aggregator_value = LIST_AGGREGATOR_VALUE[0]
#         target_value = LIST_TARGET_VALUE[0]
#         observations = {}
#         doc = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
#
#         # Create another document with same aggregator[0] and same target [0]
#         observations2 = {}
#         doc2 = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value, observations2,
#                                                       doc2)
#
#         # Create another document with same aggregator [0] but different target [1]
#         target_value2 = LIST_TARGET_VALUE[1]
#         observations3 = {}
#         doc3 = random.choice(LIST_DOC)
#         eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value, target_value2, observations3,
#                                                       doc3)
#
#         # Create another document with different aggregator [1] and different target [random]
#         aggregator_value2 = LIST_AGGREGATOR_VALUE[1]
#         target_value3 = random.choice(LIST_TARGET_VALUE)
#         observations4 = {}
#         doc4 = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(eval_terms_array, aggregator_value2, target_value3, observations4,
#                                                       doc4)
#
#         result = analyzer.evaluate_batch_for_outliers(terms=eval_terms_array)
#         # Create expected outlier
#         # aggregator [0] and target[0]
#         test_outlier_linux_1 = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
#                                                      "dummy_test", target_value, aggregator_value,
#                                                      confidence=0.6666666666666667, decision_frontier=0.3333333333333333,
#                                                      term_count=2, doc=doc)
#         # aggregator [0] and target[1]
#         test_outlier_linux_2 = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
#                                                      "dummy_test", target_value, aggregator_value,
#                                                      confidence=0.6666666666666667,
#                                                      decision_frontier=0.3333333333333333,
#                                                      term_count=2, doc=doc3)
#
#         test_outlier_win = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
#                                                  "dummy_test", target_value2, aggregator_value,
#                                                  confidence=0.6666666666666667, decision_frontier=0.3333333333333333,
#                                                  term_count=1, doc=doc2)
#         # No result for aggregator [1] due to "min_target_buckets"
#
#         # Add outlier to a list
#         expected_outliers = []
#         expected_outliers.append(test_outlier_linux)  # First detected document (target [0])
#         expected_outliers.append(test_outlier_linux)  # Second detected document (target [0])
#         expected_outliers.append(test_outlier_win)  # Third detected document (target [1])
#         self.assertEqual(result, expected_outliers)
#
#     def test_prepare_and_process_outlier_one_outlier(self):
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#         analyzer.extract_additional_model_settings()
#
#         # Just ask to analyser to create an outlier
#         decision_frontier = 1
#         term_value_count = 2
#         aggregator_value = LIST_AGGREGATOR_VALUE[0]
#         target_value = random.choice(LIST_TARGET_VALUE)
#         observations = {}
#         doc = copy.deepcopy(random.choice(LIST_DOC))
#         eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
#
#         outlier = analyzer.prepare_and_process_outlier(decision_frontier, term_value_count, eval_terms_array,
#                                                        aggregator_value, 0)
#         # Create the expected outlier
#         expected_outlier = self._create_outliers(["dummy type"], ["dummy reason"], "dummy summary", "beaconing",
#                                                  "dummy_test", target_value, aggregator_value, confidence=0.0,
#                                                  decision_frontier=1, term_count=2, doc=doc)
#         # Check that we have the good result
#         self.assertEqual(outlier, expected_outlier)
#
#     def test_prepare_and_process_outlier_check_es_have_request(self):  # TODO adapt name
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#         analyzer.extract_additional_model_settings()
#
#         # Create a document and ask beaconing to add outlier informations
#         decision_frontier = 1
#         term_value_count = 2
#         aggregator_value = "agg-WIN-EVB-draman"
#         target_value = "WIN-DRA-draman"
#         observations = {}
#         doc = copy.deepcopy(doc_without_outlier_test_file)
#         eval_terms_array = analyzer.add_term_to_batch(defaultdict(), aggregator_value, target_value, observations, doc)
#
#         analyzer.prepare_and_process_outlier(decision_frontier, term_value_count, eval_terms_array,
#                                              aggregator_value, 0)
#
#         # Get expected document (with outlier)
#         expected_doc = copy.deepcopy(doc_with_beaconing_outlier_without_score_sort_test_file)
#
#         result = [elem for elem in es.scan()][0]
#         self.assertEqual(result, expected_doc)
#
#     def test_evaluate_model_beaconing_simple_case(self):
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_01.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#
#         doc_without_outlier = copy.deepcopy(doc_without_outlier_test_file)
#         expected_doc = copy.deepcopy(doc_with_beaconing_outlier_without_score_sort_test_file)
#         # Add doc to the database
#         self.test_es.add_doc(doc_without_outlier)
#
#         # Make test (suppose that all doc match with the query)
#         analyzer.evaluate_model()
#
#         result = [elem for elem in es.scan()][0]
#         self.assertEqual(result, expected_doc)
#
#     def _test_whitelist_batch_document_not_process_all(self):  # TODO FIX with new whitelist system
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_with_whitelist.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#
#         # Whitelisted (ignored)
#         doc1_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_01_test_file)
#         self.test_es.add_doc(doc1_without_outlier)
#         # Not whitelisted (add)
#         doc2_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_02_test_file)
#         self.test_es.add_doc(doc2_without_outlier)
#         # Not whitelisted
#         doc3_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_03_test_file)
#         self.test_es.add_doc(doc3_without_outlier)
#
#         analyzer.evaluate_model()
#
#         self.assertEqual(len(analyzer.outliers), 2)
#
#     def _test_whitelist_batch_document_no_whitelist_document(self):  # TODO FIX with new whitelist system
#         self.test_settings.change_configuration_path("/app/tests/unit_tests/files/beaconing_test_with_whitelist.conf")
#         analyzer = BeaconingAnalyzer("beaconing_dummy_test")
#
#         # Not whitelisted
#         doc2_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_02_test_file)
#         self.test_es.add_doc(doc2_without_outlier)
#         # Not whitelisted and add
#         doc3_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_03_test_file)
#         self.test_es.add_doc(doc3_without_outlier)
#         # Not whitelisted and add (also add because it is the last one)
#         doc4_without_outlier = copy.deepcopy(doc_without_outliers_test_whitelist_04_test_file)
#         self.test_es.add_doc(doc4_without_outlier)
#
#         analyzer.evaluate_model()
#
#         self.assertEqual(len(analyzer.outliers), 3)
