from configparser import NoOptionError

import numpy as np
from helpers.singletons import settings, es, logging
from collections import defaultdict
import re
import helpers.utils
from helpers.analyzer import Analyzer

SUPPORTED_METRICS = ["length", "numerical_value", "entropy", "base64_encoded_length", "hex_encoded_length",
                     "url_length"]
SUPPORTED_TRIGGERS = ["high", "low"]


class MetricsAnalyzer(Analyzer):

    MIN_EVALUATE_BATCH = 100

    def _extract_additional_model_settings(self):
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

    def evaluate_model(self):
        eval_metrics = defaultdict()  # Contain the current batch information
        total_metrics_added = 0

        self.total_events = es.count_documents(index=self.es_index, search_query=self.search_query,
                                               model_settings=self.model_settings)
        self.print_analysis_intro(event_type="evaluating " + self.config_section_name, total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events,
                            desc=self.model_name + " - evaluating " + self.model_type + " model")
        if self.total_events > 0:
            for doc in es.scan(index=self.es_index, search_query=self.search_query, model_settings=self.model_settings):
                logging.tick()

                # Extract target and aggregator values
                target_value, aggregator_sentences = self._compute_aggregator_and_target_value(doc)

                # If target and aggregator values exist
                if target_value is not None and aggregator_sentences is not None:
                    # Add current document to eval_metrics
                    eval_metrics, metric_added = self._compute_eval_metrics_for_one_doc(doc, eval_metrics, target_value,
                                                                                        aggregator_sentences)
                    # If the current document have been added to the metric
                    if metric_added:
                        total_metrics_added += 1

                    is_last_batch = (logging.current_step == self.total_events)  # Check if it is the last batch
                    # Run if it is the last batch OR if the batch size is large enough
                    if is_last_batch or total_metrics_added >= settings.config.getint("metrics",
                                                                                      "metrics_batch_eval_size"):

                        logging.logger.info("evaluating batch of " + "{:,}".format(total_metrics_added) + " metrics [" +
                                            "{:,}".format(logging.current_step) + " events processed]")
                        remaining_metrics = self._evaluate_batch_save_outliers_and_display_logs(eval_metrics,
                                                                                                is_last_batch)

                        # Reset data structures for next batch
                        eval_metrics = remaining_metrics
                        total_metrics_added = 0

        self.print_analysis_summary()

    def _compute_eval_metrics_for_one_doc(self, doc, eval_metrics, target_value, aggregator_sentences):
        metrics_added = False
        metric, observations = self.calculate_metric(self.model_settings["metric"], target_value)

        if metric is not None:  # explicitly check for none, since "0" can be OK as a metric!
            metrics_added = True
            for aggregator_sentence in aggregator_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)
                eval_metrics = self.add_metric_to_batch(eval_metrics, flattened_aggregator_sentence,
                                                        target_value, metric, observations, doc)
        return eval_metrics, metrics_added

    def _compute_aggregator_and_target_value(self, doc):
        '''
        Compute the target value and the aggregator sentence. Return the two value or two None if one of the two could
        not be computed

        :param doc: the document for which the calculations must be made
        :return: target_value (could be None), aggregator_sentences (could be None)
        '''
        fields = es.extract_fields_from_document(
            doc, extract_derived_fields=self.model_settings["use_derived_fields"])
        try:
            target_value = helpers.utils.flatten_sentence(helpers.utils.get_dotkey_value(
                fields, self.model_settings["target"], case_sensitive=True))
            aggregator_sentences = helpers.utils.flatten_fields_into_sentences(
                fields=fields, sentence_format=self.model_settings["aggregator"])
        except (KeyError, TypeError):
            logging.logger.debug("skipping event which does not contain the target and aggregator " +
                                 "fields we are processing. - [" + self.model_name + "]")
            return None, None

        return target_value, aggregator_sentences

    def _evaluate_batch_save_outliers_and_display_logs(self, eval_metrics, is_last_batch):
        outliers, remaining_metrics = self._evaluate_batch_for_outliers(metrics=eval_metrics,
                                                                        is_last_batch=is_last_batch)

        # For each result, save it in batch and in ES
        for outlier in outliers:
            self.save_outlier_to_es(outlier)

        # Print message
        if len(outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in outliers))
            logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" +
                                str(unique_summaries) + " unique summaries]")
        else:
            logging.logger.info("no outliers detected in batch")

        return remaining_metrics

    def _evaluate_batch_for_outliers(self, metrics=None, is_last_batch=False):
        # Initialize
        outliers = list()  # List of all detected outliers
        remaining_metrics = dict()  # List of aggregator value that doesn't contain enough data

        for _, aggregator_value in enumerate(metrics):
            # Compute for each aggregator
            new_outliers, enough_value, metrics_aggregator_value = self._evaluate_aggregator_for_outliers(
                metrics[aggregator_value], aggregator_value, is_last_batch)

            # If the process was stop because they aren't enough value
            if not enough_value:
                remaining_metrics[aggregator_value] = metrics_aggregator_value

            # Save new outliers detected
            outliers += new_outliers

        return outliers, remaining_metrics

    def _evaluate_aggregator_for_outliers(self, metrics_aggregator_value, aggregator_value, is_last_batch):
        enough_value = True
        outliers = []
        first_run = True  # Force to run one time the loop
        list_documents_need_to_be_removed = []

        # Treat this aggregator a first time ("first_run") and continue if there are enough value
        # and that we have remove some documents from the metrics_aggregator_value
        while first_run or (enough_value and len(list_documents_need_to_be_removed) > 0):
            first_run = False

            if not first_run:
                logging.logger.debug("run again computation for " + str(aggregator_value) + " because " +
                                     str(len(list_documents_need_to_be_removed)) + " documents have been removed")

            # Check if we have sufficient data, meaning at least MIN_EVALUATE_BATCH metrics. If not, stop the loop.
            # Else, evaluate for outliers.
            if len(metrics_aggregator_value["metrics"]) < MetricsAnalyzer.MIN_EVALUATE_BATCH and \
                    is_last_batch is False:
                enough_value = False
                continue

            # Calculate the decision frontier
            decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                    metrics_aggregator_value["metrics"],
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])
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
            if len(list_documents_need_to_be_removed) > 0:
                # Remove whitelist document from the list that we need to compute
                # Note: Browse the list of documents that need to be removed in reverse order
                # To remove first the biggest index and avoid a shift (if we remove index 0, all values must be
                # decrease by one)
                for index in list_documents_need_to_be_removed[::-1]:
                    MetricsAnalyzer.remove_metric_from_batch(metrics_aggregator_value, index)

            # If this aggregator value don't need to be recompute, save outliers
            else:
                outliers += list_outliers

        return outliers, enough_value, metrics_aggregator_value

    def _evaluate_each_aggregator_value_for_outliers(self, metrics_aggregator_value, decision_frontier):
        list_outliers = []
        list_documents_need_to_be_removed = []

        # Calculate all outliers in array
        for ii, metric_value in enumerate(metrics_aggregator_value["metrics"]):
            is_outlier = helpers.utils.is_outlier(metric_value, decision_frontier,
                                                  self.model_settings["trigger_on"])

            if is_outlier:
                outlier = self._compute_fields_observation_and_create_outlier(metrics_aggregator_value, ii,
                                                                              decision_frontier, metric_value)
                if not outlier.is_whitelisted():
                    list_outliers.append(outlier)
                else:
                    list_documents_need_to_be_removed.append(ii)
                    
        return list_outliers, list_documents_need_to_be_removed

    def _compute_fields_observation_and_create_outlier(self, metrics_aggregator_value, ii, decision_frontier,
                                                       metric_value):
        confidence = np.abs(decision_frontier - metric_value)

        # Extract fields from raw document
        fields = es.extract_fields_from_document(
            metrics_aggregator_value["raw_docs"][ii],
            extract_derived_fields=self.model_settings["use_derived_fields"])

        observations = metrics_aggregator_value["observations"][ii]
        observations["metric"] = metric_value
        observations["decision_frontier"] = decision_frontier
        observations["confidence"] = confidence

        outlier = self.create_outlier(fields, metrics_aggregator_value["raw_docs"][ii],
                                      extra_outlier_information=observations, es_process_outlier=False)
        return outlier

    @staticmethod
    def add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value, observations, doc):
        observations["target"] = target_value
        observations["aggregator"] = aggregator_value

        if aggregator_value not in eval_metrics_array.keys():
            eval_metrics_array[aggregator_value] = defaultdict(list)

        eval_metrics_array[aggregator_value]["metrics"].append(metrics_value)
        eval_metrics_array[aggregator_value]["observations"].append(observations)
        eval_metrics_array[aggregator_value]["raw_docs"].append(doc)

        return eval_metrics_array

    @staticmethod
    def remove_metric_from_batch(eval_metrics_aggregator_value, index):
        eval_metrics_aggregator_value["metrics"].pop(index)
        eval_metrics_aggregator_value["observations"].pop(index)
        eval_metrics_aggregator_value["raw_docs"].pop(index)

        return eval_metrics_aggregator_value

    @staticmethod
    def calculate_metric(metric, value):

        observations = dict()

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

        # ------------------------------------------------------------------------------------
        # METRIC: Calculate total length of hexadecimal encoded substrings embedded in string
        # ------------------------------------------------------------------------------------
        elif metric == "hex_encoded_length":
            hex_encoded_words = list()
            # at least length 10 to have 5 encoded characters
            target_value_words = re.split("[^a-fA-F0-9+]", str(value))

            for word in target_value_words:
                # let's match at least 5 characters, meaning 10 hex digits
                if len(word) > 10 and helpers.utils.is_hex_encoded(word):
                    hex_encoded_words.append(word)

            if len(hex_encoded_words) > 0:
                sorted_hex_encoded_words = sorted(hex_encoded_words, key=len)
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
            base64_decoded_words = list()

            # Split all non-Base64 characters, so we can try to convert them to Base64 decoded strings
            target_value_words = re.split("[^A-Za-z0-9+/=]", str(value))

            for word in target_value_words:
                decoded_word = helpers.utils.is_base64_encoded(word)
                # let's match at least 5 characters, meaning 10 base64 digits
                if decoded_word and len(decoded_word) >= 5:
                    base64_decoded_words.append(decoded_word)

            if len(base64_decoded_words) > 0:
                sorted_base64_decoded_words = sorted(base64_decoded_words, key=len)
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
            extracted_urls_length = 0
            extracted_urls = []

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
