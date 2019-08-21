from configparser import NoOptionError

import numpy as np
from helpers.singletons import settings, es, logging
from helpers.outlier import Outlier
from collections import defaultdict
import re
import random
import helpers.utils
from helpers.analyzer import Analyzer
from numpy import float64

from typing import Dict, Set, Any, Tuple, List, DefaultDict, cast, Optional, Union

SUPPORTED_METRICS: List[str] = ["length", "numerical_value", "entropy", "base64_encoded_length", "hex_encoded_length",
                                "url_length", "relative_english_entropy"]
SUPPORTED_TRIGGERS: List[str] = ["high", "low"]


class MetricsAnalyzer(Analyzer):

    # The miminum amount of documents that should be part of a single aggregator in order to process is.
    # This prevents outliers being flagged in cases where there is too little data to work with to make a useful
    # conclusion. However, in the very last batch of the use case, all the remaining aggregators are processed,
    # even the ones for which the number of documents is less than MIN_EVALUATE_BATCH.
    MIN_EVALUATE_BATCH = 100

    def evaluate_model(self) -> None:
        batch: DefaultDict = defaultdict()  # Contain the current batch information
        remaining_metrics: DefaultDict = defaultdict()
        total_metrics_in_batch: int = 0

        self.total_events: int
        documents: List[Dict[str, Any]]
        self.total_events, documents = es.count_and_scan_documents(index=self.es_index, search_query=self.search_query,
                                                                   model_settings=self.model_settings)

        self.print_analysis_intro(event_type="evaluating " + self.config_section_name, total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events,
                            desc=self.model_name + " - evaluating " + self.model_type + " model")
        if self.total_events > 0:
            for doc in documents:
                logging.tick()
                # Extract target and aggregator values
                target_value: Optional[str]
                aggregator_sentences: Optional[List[List]]
                target_value, aggregator_sentences = self._compute_aggregator_and_target_value(doc)

                # If target and aggregator values exist
                if target_value is not None and aggregator_sentences is not None:
                    # Add current document to eval_metrics
                    batch, metric_added = self._add_document_to_batch(doc, batch, target_value,
                                                                      aggregator_sentences)

                    # We can only have 1 target field for metrics (as opposed to terms), so the total number of targets
                    # added is the same as the total number of aggregator sentences that were processed for this
                    # document
                    if metric_added:
                        total_metrics_in_batch += len(aggregator_sentences)

                is_last_batch: bool = (logging.current_step == self.total_events)  # Check if it is the last batch
                # Run if it is the last batch OR if the batch size is large enough
                if is_last_batch or total_metrics_in_batch >= settings.config.getint("metrics",
                                                                                     "metrics_batch_eval_size"):

                    # Display log message
                    log_message: str = "evaluating batch of " + "{:,}".format(total_metrics_in_batch) + " metrics "
                    if len(remaining_metrics) > 0:
                        log_message += "(+ " + "{:,}".format(len(remaining_metrics)) + " metrics from last batch) "
                    log_message += "[" + "{:,}".format(logging.current_step) + " events processed]"
                    logging.logger.info(log_message)

                    outliers_in_batch: List[Outlier]
                    outliers_in_batch, remaining_metrics = self._evaluate_batch_for_outliers(
                        batch=batch, is_last_batch=is_last_batch)

                    # For each result, save it in batch and in ES
                    if outliers_in_batch:
                        unique_summaries_in_batch: int = len(set(o.outlier_dict["summary"] for o in outliers_in_batch))
                        logging.logger.info("processing " + "{:,}".format(len(outliers_in_batch)) +
                                            " outliers in batch [" + "{:,}".format(unique_summaries_in_batch) +
                                            " unique summaries]")

                        for outlier in outliers_in_batch:
                            self.process_outlier(outlier)
                    else:
                        logging.logger.info("no outliers detected in batch")

                    # Reset data structures for next batch
                    batch = remaining_metrics
                    total_metrics_in_batch = 0

        self.print_analysis_summary()

    def _add_document_to_batch(self, doc: Dict, batch: DefaultDict, target_value: str,
                               aggregator_sentences: List[List]) -> Tuple[DefaultDict, bool]:
        """
        Compute different metrics and observation to add them to the batch

        :param doc: document where fields need to be extracted
        :param batch: batch where values need to be added
        :param target_value: target value used to extract data
        :param aggregator_sentences: aggregator value (used to group data)
        :return: the new batch and the metrics added
        """
        metrics_added: bool = False
        metric: Union[None, float, int]
        observations: Dict[str, Any]
        metric, observations = self.calculate_metric(self.model_settings["metric"], target_value)

        if metric is not None:  # explicitly check for none, since "0" can be OK as a metric!
            metrics_added = True
            for aggregator_sentence in aggregator_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)
                batch = self.add_metric_to_batch(batch, flattened_aggregator_sentence, target_value, metric,
                                                 observations, doc)
        return batch, metrics_added

    def _compute_aggregator_and_target_value(self, doc: Dict) -> Tuple[Optional[str], Optional[List[List]]]:
        """
        Compute the target value and the aggregator sentence. Return the two value or two None if one of the two could
        not be computed

        :param doc: the document for which the calculations must be made
        :return: target_value (could be None), aggregator_sentences (could be None)
        """
        fields: Dict = es.extract_fields_from_document(doc,
                                                       extract_derived_fields=self.model_settings["use_derived_fields"])
        try:
            target_value: Optional[str] = helpers.utils.flatten_sentence(helpers.utils.get_dotkey_value(
                fields, self.model_settings["target"], case_sensitive=True))
            aggregator_sentences: List[List] = helpers.utils.flatten_fields_into_sentences(
                fields=fields, sentence_format=self.model_settings["aggregator"])
        except (KeyError, TypeError):
            logging.logger.debug("skipping event which does not contain the target and aggregator " +
                                 "fields we are processing. - [" + self.model_name + "]")
            return None, None

        return target_value, aggregator_sentences

    def _evaluate_batch_for_outliers(self, batch: DefaultDict,
                                     is_last_batch: bool = False) -> Tuple[List[Outlier], DefaultDict]:
        """
        Evaluate one batch to detect outliers

        :param batch: the batch to analyze
        :param is_last_batch: boolean to say if it is the last batch
        :return: The list of outliers and the list of batch that need to be process again (due to not enough values)
        """
        # Initialize
        outliers: List[Outlier] = list()  # List of all detected outliers
        # List of aggregator value that doesn't contain enough data
        unprocessed_batch_elements: DefaultDict = defaultdict()

        for _, aggregator_value in enumerate(batch):
            # Compute for each aggregator
            new_outliers, has_sufficient_data, metrics_aggregator_value = self._evaluate_aggregator_for_outliers(
                batch[aggregator_value], aggregator_value, is_last_batch)

            # If the process was stopped because of insufficient data for this aggregator (based on MIN_EVALUATE_BATCH)
            if not has_sufficient_data:
                unprocessed_batch_elements[aggregator_value] = metrics_aggregator_value

            # Save new outliers detected
            outliers += new_outliers

        return outliers, unprocessed_batch_elements

    def _evaluate_aggregator_for_outliers(self, metrics_aggregator_value: Dict[str, Any], aggregator_value: str,
                                          is_last_batch: bool) -> Tuple[List[Outlier], bool, Dict[str, Any]]:
        """
        Evaluate all documents for a specific aggregator value

        :param metrics_aggregator_value: metrics linked to this aggregator
        :param aggregator_value: aggregator value that is evaluate
        :param is_last_batch: boolean to know if it is the last batch
        :return: List of outliers, boolean to know if there are enough data (False if not), new
        "metrics_aggregator_value" (update if some document are detected like outlier and whitelisted)
        """
        has_sufficient_data: bool = True
        first_run: bool = True  # Force to run one time the loop
        nr_whitelisted_element_detected: int = 0

        outliers: List[Outlier] = []
        list_documents_need_to_be_removed: List[int] = []

        # Treat this aggregator a first time ("first_run") and continue if there are enough value
        # and that we have remove some documents from the metrics_aggregator_value
        while (first_run or (has_sufficient_data and list_documents_need_to_be_removed)) and \
                len(metrics_aggregator_value["metrics"]) > 0:
            if not first_run:
                logging.logger.info("evaluating the batch again after removing " +
                                    "{:,}".format(nr_whitelisted_element_detected) + " whitelisted elements")

            first_run = False

            # Check if we have sufficient data, meaning at least MIN_EVALUATE_BATCH metrics. If not, stop the loop.
            # Else, evaluate for outliers.
            if len(metrics_aggregator_value["metrics"]) < MetricsAnalyzer.MIN_EVALUATE_BATCH and \
                    is_last_batch is False:
                has_sufficient_data = False
                break

            # Calculate the decision frontier
            decision_frontier: Union[int, float, float64] = helpers.utils.get_decision_frontier(
                self.model_settings["trigger_method"], metrics_aggregator_value["metrics"],
                self.model_settings["trigger_sensitivity"], self.model_settings["trigger_on"])
            logging.logger.debug("using decision frontier " + str(decision_frontier) + " for aggregator " +
                                 str(aggregator_value) + " - " + self.model_settings["metric"])
            logging.logger.debug("example metric from batch for " +
                                 metrics_aggregator_value["observations"][0]["target"] + ": " +
                                 str(metrics_aggregator_value["metrics"][0]))

            # For this aggregator, compute outliers and document that have been detected like outliers
            # but are whitelisted
            list_outliers, list_documents_need_to_be_removed = self._evaluate_each_aggregator_value_for_outliers(
                metrics_aggregator_value, decision_frontier)

            # If some document need to be removed
            if list_documents_need_to_be_removed:
                # Save the number of element that need to be removed
                nr_whitelisted_element_detected += len(list_documents_need_to_be_removed)
                logging.logger.debug("removing " + "{:,}".format((len(list_documents_need_to_be_removed))) +
                                     " whitelisted documents from the batch for aggregator " + str(aggregator_value))

                # Remove whitelist document from the list that we need to compute
                # Note: Browse the list of documents that need to be removed in reverse order
                # To remove first the biggest index and avoid a shift (if we remove index 0, all values must be
                # decrease by one)
                for index in list_documents_need_to_be_removed[::-1]:
                    MetricsAnalyzer.remove_metric_from_batch(metrics_aggregator_value, index)

            # If this aggregator value don't need to be recompute, save outliers
            else:
                outliers += list_outliers

        return outliers, has_sufficient_data, metrics_aggregator_value

    def _evaluate_each_aggregator_value_for_outliers(
            self, metrics_aggregator_value: Dict[str, Any],
            decision_frontier: Union[int, float, float64]) -> Tuple[List[Outlier], List[int]]:
        """
        Evaluate all value in an aggregator to detect if it is outlier

        :param metrics_aggregator_value: list of metrics linked to this aggregator
        :param decision_frontier: the decision frontier
        :return: list of outliers and list of document that need to be remove (because they have been detected like
        outliers and are whitelisted)
        """
        list_outliers: List[Outlier] = []
        list_documents_need_to_be_removed: List[int] = []
        non_outlier_values: Set = set()

        # Calculate all outliers in array
        for ii, metric_value in enumerate(metrics_aggregator_value["metrics"]):
            is_outlier: bool = helpers.utils.is_outlier(metric_value, decision_frontier,
                                                        self.model_settings["trigger_on"])

            if is_outlier:
                outlier: Outlier = self._compute_fields_observation_and_create_outlier(non_outlier_values,
                                                                                       metrics_aggregator_value, ii,
                                                                                       decision_frontier, metric_value)
                if not outlier.is_whitelisted():
                    list_outliers.append(outlier)
                else:
                    self.nr_whitelisted_elements += 1
                    list_documents_need_to_be_removed.append(ii)
            else:
                non_outlier_values.add(str(metric_value))

        return list_outliers, list_documents_need_to_be_removed

    def _compute_fields_observation_and_create_outlier(self, non_outlier_values: Set,
                                                       metrics_aggregator_value: Dict[str, Any], ii: int,
                                                       decision_frontier: Union[int, float, float64],
                                                       metric_value: Union[float, int]) -> Outlier:
        """
        Extract field from document and compute different element that will be placed in the observation

        :param metrics_aggregator_value: value of the metrics aggregator
        :param ii: index of the document that have been detected like outlier
        :param decision_frontier: the value of the decision frontier
        :param metric_value: the metric value
        :return: the created outlier
        """
        observations: Dict[str, Any] = metrics_aggregator_value["observations"][ii]

        if non_outlier_values:
            non_outlier_values_sample = ",".join(random.sample(
                non_outlier_values, 3 if len(non_outlier_values) > 3 else len(non_outlier_values)))
            observations["non_outlier_values_sample"] = non_outlier_values_sample
        else:
            observations["non_outlier_values_sample"] = []

        observations["metric"] = metric_value
        observations["decision_frontier"] = decision_frontier

        confidence: int = np.abs(decision_frontier - metric_value)
        observations["confidence"] = confidence

        # Extract fields from raw document
        fields: Dict = es.extract_fields_from_document(
            metrics_aggregator_value["raw_docs"][ii],
            extract_derived_fields=self.model_settings["use_derived_fields"])

        outlier: Outlier = self.create_outlier(fields, metrics_aggregator_value["raw_docs"][ii],
                                               extra_outlier_information=observations)
        return outlier

    def _extract_additional_model_settings(self) -> None:
        """
        Override method from Analyzer
        """
        try:
            self.model_settings["process_documents_chronologically"] = settings.config.getboolean(
                self.config_section_name, "process_documents_chronologically")
        except NoOptionError:
            self.model_settings["process_documents_chronologically"] = False

        self.model_settings["target"] = settings.config.get(self.config_section_name, "target")
        self.model_settings["aggregator"] = settings.config.get(self.config_section_name, "aggregator")\
            .replace(' ', '').split(",")

        self.model_settings["metric"] = settings.config.get(self.config_section_name, "metric")
        self.model_settings["trigger_on"] = settings.config.get(self.config_section_name, "trigger_on")
        self.model_settings["trigger_method"] = settings.config.get(self.config_section_name, "trigger_method")
        self.model_settings["trigger_sensitivity"] = settings.config.getint(self.config_section_name,
                                                                            "trigger_sensitivity")

        if self.model_settings["metric"] not in SUPPORTED_METRICS:
            raise ValueError("Unsupported metric " + self.model_settings["metric"])

        if self.model_settings["trigger_on"] not in SUPPORTED_TRIGGERS:
            raise ValueError("Unexpected outlier trigger condition " + self.model_settings["trigger_on"])

    @staticmethod
    def add_metric_to_batch(eval_metrics_array: DefaultDict, aggregator_value: Optional[str],
                            target_value: Optional[str], metrics_value: Union[None, float, int],
                            observations: Dict[str, Any], doc: Dict) -> DefaultDict:
        """
        Add a document to the batch. This method will extract data and save it in batch

        :param eval_metrics_array: existing batch values
        :param aggregator_value: aggregator value of this document
        :param target_value: target value of the document
        :param metrics_value: metric value
        :param observations: observations
        :param doc: document that need to be added
        :return: new batch value
        """
        observations["target"] = target_value
        observations["aggregator"] = aggregator_value

        if aggregator_value not in eval_metrics_array.keys():
            eval_metrics_array[aggregator_value] = defaultdict(list)

        eval_metrics_array[aggregator_value]["metrics"].append(metrics_value)
        eval_metrics_array[aggregator_value]["observations"].append(observations)
        eval_metrics_array[aggregator_value]["raw_docs"].append(doc)

        return eval_metrics_array

    @staticmethod
    def remove_metric_from_batch(eval_metrics_aggregator_value: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Remove value from batch (does the opposite of add_metric_to_batch)

        :param eval_metrics_aggregator_value: batch value that need to be update
        :param index: index of document that need to be removed
        :return: new batch
        """
        eval_metrics_aggregator_value["metrics"].pop(index)
        eval_metrics_aggregator_value["observations"].pop(index)
        eval_metrics_aggregator_value["raw_docs"].pop(index)

        return eval_metrics_aggregator_value

    @staticmethod
    def calculate_metric(metric: str, value: str) -> Tuple[Union[None, float, int], Dict]:
        observations: Dict[str, Any] = dict()
        target_value_words: List[str]

        # ------------------------------------
        # METRIC: Calculate numerical value
        # ------------------------------------
        # Example: numerical_value("2") => 2
        if metric == "numerical_value":
            try:
                return float(value), dict()
            except ValueError:
                # number can not be casted to a Float, just continue
                return None, dict()

        # ------------------------------------
        # METRIC: Calculate length of a string
        # ------------------------------------
        # Example: length("outliers") => 8
        elif metric == "length":
            return len(value), dict()

        # -------------------------------------
        # METRIC: Calculate entropy of a string
        # -------------------------------------
        # Example: entropy("houston") => 2.5216406363433186
        elif metric == "entropy":
            return helpers.utils.shannon_entropy(value), dict()

        elif metric == "relative_english_entropy":
            english_character_frequencies = \
                {'g': 0.02706810814315049, 'o': 0.07421531631063037, 'l': 0.04660619075683699, 'e': 0.0938650686651803,
                 'f': 0.016087468884472687, 'a': 0.08965206537963542, 'c': 0.046178435692422186, 'b': 0.021492396761465096,
                 'k': 0.017011742091988323, 'y': 0.017683540507870608, 'u': 0.03113815167654972, 't': 0.05877603780957555,
                 'w': 0.012812697524051385, 'i': 0.07074249978897978, 'r': 0.06343497059722608, 'm': 0.033597415407595026,
                 's': 0.06260194430883878, 'n': 0.06262892491736954, 'd': 0.031030885021106236, 'p': 0.026214752715696614,
                 'v': 0.013545577039801925, 'h': 0.027979827873085842, 'z': 0.007096836870275642, '-': 0.010803953745868712,
                 '3': 0.0020435937308682425, 'q': 0.002948193577996864, 'x': 0.006551510056881306, 'j': 0.006711051641353142,
                 '0': 0.0027525841661488358, '1': 0.0029865097894172872, '2': 0.0027598914142925837, '6': 0.0017662930320798498,
                 '4': 0.001964712923983166, '5': 0.0017608594373062934, '8': 0.0021597602398201366, '9': 0.0017470880850353834,
                 '7': 0.0015831434151435972}

            # Calculate Kullback Leibler entropy of URL compared to majestic million distribution
            entropy_value = helpers.utils.kl_divergence(value, english_character_frequencies)

            return entropy_value, observations

        # ------------------------------------------------------------------------------------
        # METRIC: Calculate total length of hexadecimal encoded substrings embedded in string
        # ------------------------------------------------------------------------------------
        elif metric == "hex_encoded_length":
            hex_encoded_words: List[str] = list()
            # at least length 10 to have 5 encoded characters
            target_value_words = re.split("[^a-fA-F0-9+]", str(value))

            for word in target_value_words:
                # let's match at least 5 characters, meaning 10 hex digits
                if len(word) > 10 and helpers.utils.is_hex_encoded(word):
                    hex_encoded_words.append(word)

            if len(hex_encoded_words) > 0:
                sorted_hex_encoded_words: List[str] = sorted(hex_encoded_words, key=len)
                observations["max_hex_encoded_length"] = len(sorted_hex_encoded_words[-1])
                observations["max_hex_encoded_word"] = sorted_hex_encoded_words[-1]

                return len(sorted_hex_encoded_words[-1]), observations
            else:
                return 0, dict()

        # ------------------------------------------------------------------------------------
        # METRIC: Calculate total length of base64 encoded substrings embedded in string
        # ------------------------------------------------------------------------------------
        # Example: base64_encoded_length("houston we have a cHJvYmxlbQ==") => base64_decoded_string: problem,
        # base64_encoded_length: 7
        elif metric == "base64_encoded_length":
            base64_decoded_words: List[str] = list()

            # Split all non-Base64 characters, so we can try to convert them to Base64 decoded strings
            target_value_words = re.split("[^A-Za-z0-9+/=]", str(value))

            for word in target_value_words:
                decoded_word: Union[None, bool, str] = helpers.utils.is_base64_encoded(word)
                # let's match at least 5 characters, meaning 10 base64 digits
                if decoded_word and len(cast(str, decoded_word)) >= 5:
                    base64_decoded_words.append(cast(str, decoded_word))

            if len(base64_decoded_words) > 0:
                sorted_base64_decoded_words: List[str] = sorted(base64_decoded_words, key=len)
                observations["max_base64_decoded_length"] = len(sorted_base64_decoded_words[-1])
                observations["max_base64_decoded_word"] = sorted_base64_decoded_words[-1]

                return len(sorted_base64_decoded_words[-1]), observations
            else:
                return 0, dict()

        # ---------------------------------------------------------
        # METRIC: Calculate total length of URLs embedded in string
        # ---------------------------------------------------------
        # Example: url_length("why don't we go http://www.dance.com") => extracted_urls_length: 20,
        # extracted_urls: http://www.dance.com
        elif metric == "url_length":
            extracted_urls_length: int = 0
            extracted_urls: List[str] = []

            # if the target value is a list of strings, convert it into a single list of strings
            # splits on whitespace by default, and on quotes, since we most likely will apply this to parameter
            # arguments
            target_value_words = value.replace('"', ' ').split()

            for word in target_value_words:
                is_url = helpers.utils.is_url(word)
                if is_url:
                    extracted_urls_length += len(word)
                    extracted_urls.append(word)

            if extracted_urls_length > 0:
                observations["extracted_urls_length"] = extracted_urls_length
                observations["extracted_urls"] = ','.join(extracted_urls)

            return extracted_urls_length, observations

        else:
            # metric method does not exist, we don't return anything useful
            return None, dict()
