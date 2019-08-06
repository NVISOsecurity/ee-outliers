import random
from configparser import NoOptionError

from helpers.singletons import settings, es, logging
from collections import defaultdict
from collections import Counter
import helpers.utils
from helpers.analyzer import Analyzer


class TermsAnalyzer(Analyzer):

    def evaluate_model(self):
        self.total_events, documents = es.count_and_scan_documents(index=self.es_index, search_query=self.search_query,
                                                                   model_settings=self.model_settings)

        self.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)
        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating terms model")

        if self.total_events > 0:
            current_batch = defaultdict()
            total_targets_in_batch = 0

            for doc in documents:
                logging.tick()
                target_sentences, aggregator_sentences = self._compute_aggregator_and_target_value(
                    doc, self.model_settings["target"])

                if target_sentences is not None and aggregator_sentences is not None:
                    # Add current document to current_batch
                    current_batch = self._add_document_to_batch(current_batch, target_sentences,
                                                                aggregator_sentences, doc)

                    total_targets_in_batch += len(target_sentences) * len(aggregator_sentences)

                # Evaluate batch of events against the model
                is_last_batch = (logging.current_step == self.total_events)  # Check if it is the last batch
                # Run if it is the last batch OR if the batch size is large enough

                if is_last_batch or total_targets_in_batch >= settings.config.getint("terms", "terms_batch_eval_size"):
                    logging.logger.info("evaluating batch of " + "{:,}".format(total_targets_in_batch) + " terms [" +
                                        "{:,}".format(logging.current_step) + " events processed]")

                    # evaluate the current batch
                    outliers_in_batch, targets_for_next_batch = self._evaluate_batch_for_outliers(batch=current_batch)

                    if outliers_in_batch:
                        unique_summaries_in_batch = len(set(o.outlier_dict["summary"] for o in outliers_in_batch))
                        logging.logger.info("processing " + "{:,}".format(len(outliers_in_batch)) +
                                            " outliers in batch [" + "{:,}".format(unique_summaries_in_batch) +
                                            " unique summaries]")

                        for outlier in outliers_in_batch:
                            self.process_outlier(outlier)

                    else:
                        logging.logger.info("no outliers processed in batch")

                    # Reset data structures for next batch
                    current_batch = targets_for_next_batch
                    total_targets_in_batch = 0

        self.print_analysis_summary()

    @staticmethod
    def _add_document_to_batch(current_batch, target_sentences, aggregator_sentences, doc):
        """
        Add a document to the current batch

        :param current_batch: existing batch (where doc need to be saved)
        :param target_sentences: list of targets
        :param aggregator_sentences: list of aggregator
        :param doc: document that need to be added
        :return: batch with document inside
        """
        observations = dict()

        for target_sentence in target_sentences:
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence)

            for aggregator_sentence in aggregator_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)

                if flattened_aggregator_sentence not in current_batch.keys():
                    current_batch[flattened_aggregator_sentence] = defaultdict(list)

                current_batch[flattened_aggregator_sentence]["targets"].append(flattened_target_sentence)
                current_batch[flattened_aggregator_sentence]["observations"].append(observations)
                current_batch[flattened_aggregator_sentence]["raw_docs"].append(doc)

        return current_batch

    def _compute_aggregator_and_target_value(self, doc, target):
        """
        Extract target and aggregator sentence from a document

        :param doc: document where data need to be extract
        :param target: target key name
        :return: list of target and list of aggregator
        """
        fields = es.extract_fields_from_document(doc, extract_derived_fields=self.model_settings["use_derived_fields"])
        try:
            target_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=target)
            aggregator_sentences = helpers.utils.flatten_fields_into_sentences(
                fields=fields, sentence_format=self.model_settings["aggregator"])
        except (KeyError, TypeError):
            logging.logger.debug("Skipping event which does not contain the target and aggregator " +
                                 "fields we are processing. - [" + self.model_name + "]")
            return None, None
        return target_sentences, aggregator_sentences

    def _evaluate_batch_for_outliers(self, batch):
        """
        Find outliers in batch

        :param batch: batch to analyze
        :return: list of outlier and list of element that need to be computed again
        """

        # In case we want to count terms across different aggregators, we need to first iterate over all aggregators
        # and calculate the total number of unique terms for each aggregated value.
        # For example:
        # terms["smsc.exe"][A, B, C, D, D, E]
        # terms["abc.exe"][A, A, B]
        # is converted into:
        # unique_target_counts_across_aggregators: [5, 2] (the first term contains 5 unique values, the second
        # one contains 2)
        if self.model_settings["target_count_method"] == "across_aggregators":
            outliers = list()  # List outliers

            # List of document (per aggregator) that aren't outlier (to help user to see non match results)
            # Notice that this dictionary will only be used if there is a loop (first loop fill the dict. Second loop
            # take the result if outlier is detected).
            non_outlier_values = defaultdict(list)
            first_run = True  # Force to run one time the loop
            nr_whitelisted_element_detected = 0  # Number of elements that have been removed (due to whitelist)

            # Run the loop the first time and still elements are removed (due to whitelist)
            while first_run or nr_whitelisted_element_detected > 0:
                if not first_run:
                    logging.logger.info("evaluating the batch again after removing " + str(
                        nr_whitelisted_element_detected) + " whitelisted elements")
                first_run = False

                # Compute decision frontier and loop on all aggregator
                # For each of them, evaluate if it is an outlier and remove terms that are whitelisted (no return
                # because it is a dictionary)
                nr_whitelisted_element_detected, outliers = self._evaluate_aggregator_for_outlier_accross(
                    batch, non_outlier_values)

            # All outliers and no remaining terms
            return outliers, {}

        # In case we want to count terms within an aggregator, it's a bit easier.
        # For example:
        # terms["smsc.exe"][A, B, C, D, D, E]
        # terms["abc.exe"][A, A, B]
        # is converted into:
        # First iteration: "smsc.exe" -> counted_target_values: {A: 1, B: 1, C: 1, D: 2, E: 1}
        # For each aggregator, we iterate over all terms within it:
        # term_value_count for a document with term "A" then becomes "1" in the example above.
        # we then flag an outlier if that "1" is an outlier in the array ["1 1 1 2 1"]
        elif self.model_settings["target_count_method"] == "within_aggregator":
            outliers = list()
            targets_for_next_batch = dict()
            batch_copy = batch.copy()

            for aggregator_value in batch_copy.keys():
                outliers_in_aggregator, has_min_target_buckets = self._evaluate_aggregator_for_outliers_within(
                    batch, aggregator_value)
                if not has_min_target_buckets:
                    targets_for_next_batch[aggregator_value] = batch[aggregator_value]
                outliers += outliers_in_aggregator

            return outliers, targets_for_next_batch

        else:
            raise ValueError("Unexpected target count method " + self.model_settings["target_count_method"])

    def _evaluate_aggregator_for_outlier_accross(self, batch, non_outlier_values):
        """
        Evaluate all aggregator to detect outlier (accross context)

        :param batch: batch use to made the analyze
        :param non_outlier_values: list of values that aren't outlier
        :return: Number of element whitelisted and list of outliers
        """
        nr_whitelisted_element_detected = 0
        unique_target_counts_across_aggregators, decision_frontier = \
            self._compute_count_across_aggregators_and_decision_frontier(batch)

        logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " +
                             str(decision_frontier) + " across all aggregators")
        outliers = list()

        # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
        for i, aggregator_value in enumerate(batch):
            unique_target_count_across_aggregators = unique_target_counts_across_aggregators[i]
            new_list_outliers, list_documents_need_to_be_removed = \
                self._evaluate_each_aggregator_is_outliers_and_mark_across(batch, aggregator_value,
                                                                           unique_target_count_across_aggregators,
                                                                           decision_frontier,
                                                                           non_outlier_values[aggregator_value])

            # If some documents need to be removed
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
                    TermsAnalyzer.remove_term_from_batch(batch, aggregator_value, index)
            else:
                outliers += new_list_outliers

        # If at least one element need to be computed again
        if nr_whitelisted_element_detected > 0:
            outliers = list()  # Ignore detected outliers

        return nr_whitelisted_element_detected, outliers

    def _compute_count_across_aggregators_and_decision_frontier(self, batch):
        """
        Compute the target counts and the decision frontier

        :param batch: batch that need to be used to fetch data
        :return: the unique target count and the decision frontier
        """
        unique_target_counts_across_aggregators = list()

        # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
        for i, aggregator_value in enumerate(batch):
            # unique_targets_in_aggregated_value = loop 0: [A, B, C, D, E], loop 1: [A, A, B]
            # unique_target_counts_across_aggregators = loop 0: [5], loop 1: [5, 2]
            unique_targets_in_aggregated_value = set(batch[aggregator_value]["targets"])
            unique_target_counts_across_aggregators.append(len(unique_targets_in_aggregated_value))

        # Calculate the decision frontier
        # unique_target_counts_across_aggregators = [5, 2]
        decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                unique_target_counts_across_aggregators,
                                                                self.model_settings["trigger_sensitivity"],
                                                                self.model_settings["trigger_on"])
        return unique_target_counts_across_aggregators, decision_frontier

    def _evaluate_each_aggregator_is_outliers_and_mark_across(self, batch, aggregator_value,
                                                              unique_target_count_across_aggregators, decision_frontier,
                                                              non_outlier_values):
        """
        Compute value for a specific aggregator and try to detect outlier

        :param batch: batch
        :param aggregator_value: aggregator value that must be computed
        :param unique_target_count_across_aggregators: number of element for this aggregator
        :param decision_frontier: value of the decision frontier
        :param non_outlier_values: list of document that aren't outliers
        :return: the list of outliers and the list of document that have been detected like outlier but that are
        whitelisted (and that must be removed)
        """
        # Initialise with default value (that will be return if nothing is found)
        list_outliers = list()
        list_documents_need_to_be_removed = list()

        logging.logger.debug("unique target count for aggregator " + str(aggregator_value) + ": " +
                             str(unique_target_count_across_aggregators) + " - decision frontier " +
                             str(decision_frontier))
        # Check if current aggregator is outlier
        is_outlier = helpers.utils.is_outlier(unique_target_count_across_aggregators, decision_frontier,
                                              self.model_settings["trigger_on"])

        if is_outlier:
            list_outliers, list_documents_need_to_be_removed = self._mark_across_aggregator_document_as_outliers(
                batch, aggregator_value, unique_target_count_across_aggregators, decision_frontier, non_outlier_values)
        else:
            # Save non outliers list (do not be return because it is a dictionary)
            non_outlier_values += batch[aggregator_value]["targets"]

        return list_outliers, list_documents_need_to_be_removed

    def _mark_across_aggregator_document_as_outliers(self, batch, aggregator_value,
                                                     unique_target_count_across_aggregators, decision_frontier,
                                                     non_outlier_values):
        """
        Mark all document of a specific aggregator like an outlier

        :param batch: batch
        :param aggregator_value: the aggregator value
        :param unique_target_count_across_aggregators: number of element for this aggregator
        :param decision_frontier: value of the decision frontier
        :param non_outlier_values: list of document that aren't outliers
        :return: the list of outliers and the list of document that have been detected like outlier but that are
        whitelisted (and that must be removed)
        """
        # Initialise
        list_outliers = list()
        list_documents_need_to_be_removed = list()

        for ii, term_value in enumerate(batch[aggregator_value]["targets"]):
            outlier = self._create_outlier(non_outlier_values, unique_target_count_across_aggregators,
                                           aggregator_value, term_value, decision_frontier, batch, ii)
            if not outlier.is_whitelisted():
                list_outliers.append(outlier)
            else:
                self.nr_whitelisted_elements += 1
                list_documents_need_to_be_removed.append(ii)

        return list_outliers, list_documents_need_to_be_removed

    def _evaluate_aggregator_for_outliers_within(self, batch, aggregator_value):
        """
        Compute value for a specific aggregator and try to detect outlier

        :param batch: batch that must be analyze
        :param aggregator_value: the value that need to be analyze
        :return: list of outliers and boolean to know if this aggregator need to be compute again in the next batch (
        because the number of value is lower than the "min_target_buckets")
        """
        list_outliers = list()
        list_documents_need_to_be_removed = list()
        first_run = True  # Force to run one time the loop
        has_min_target_buckets = True

        while first_run or (list_documents_need_to_be_removed and batch[aggregator_value]["targets"]):
            first_run = False

            # Count percentage of each target value occurring
            counted_targets = Counter(batch[aggregator_value]["targets"])
            counted_target_values = list(counted_targets.values())

            logging.logger.debug("terms count for aggregator value " + aggregator_value + " -> " +
                                 str(counted_targets))

            # If not enough bucket we stop the loop
            if self.model_settings["min_target_buckets"] is not None and \
                    len(counted_targets) < self.model_settings["min_target_buckets"]:
                has_min_target_buckets = False
                break

            decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                    counted_target_values,
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])

            logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " +
                                 str(decision_frontier) + " for aggregator " + str(aggregator_value))

            new_list_outliers, list_documents_need_to_be_removed = \
                self._evaluate_each_aggregator_for_outliers(decision_frontier, batch, aggregator_value, counted_targets)

            # Remove document detected like outliers and whitelisted
            if list_documents_need_to_be_removed:
                logging.logger.debug("removing {:,}".format((len(list_documents_need_to_be_removed))) +
                                     " whitelisted documents from the batch for aggregator " + str(aggregator_value))

                # browse the list in reverse order (to remove first biggest index)
                for index in list_documents_need_to_be_removed[::-1]:
                    batch = TermsAnalyzer.remove_term_from_batch(batch, aggregator_value, index)
            else:
                list_outliers += new_list_outliers

        return list_outliers, has_min_target_buckets

    def _evaluate_each_aggregator_for_outliers(self, decision_frontier, batch, aggregator_value, counted_targets):
        """
        Test each document in an aggregator to detect Outlier (using "within" method)

        :param decision_frontier: value of the decision frontier
        :param batch: all batch elements
        :param aggregator_value: the aggregator value that must be evaluate
        :param counted_targets: number of element in this batch
        :return: the list of outliers and the list of document that are detected like outlier but that are withielisted
        """
        list_documents_need_to_be_removed = list()
        list_outliers = list()
        non_outlier_values = set()

        if self.model_settings["trigger_method"] == "coeff_of_variation":
            is_outlier = helpers.utils.is_outlier(decision_frontier, self.model_settings["trigger_sensitivity"],
                                                  self.model_settings["trigger_on"])
            if is_outlier:
                for ii, term_value in enumerate(batch[aggregator_value]["targets"]):
                    term_value_count = counted_targets[term_value]
                    outlier = self._create_outlier(non_outlier_values, term_value_count, aggregator_value,
                                                   term_value, decision_frontier, batch, ii)
                    if not outlier.is_whitelisted():
                        list_outliers.append(outlier)
                    else:
                        self.nr_whitelisted_elements += 1
                        list_documents_need_to_be_removed.append(ii)

        else:
            for ii, term_value in enumerate(batch[aggregator_value]["targets"]):
                term_value_count = counted_targets[term_value]
                is_outlier = helpers.utils.is_outlier(term_value_count, decision_frontier,
                                                      self.model_settings["trigger_on"])

                if is_outlier:
                    outlier = self._create_outlier(non_outlier_values, term_value_count, aggregator_value,
                                                   term_value, decision_frontier, batch, ii)
                    if not outlier.is_whitelisted():
                        list_outliers.append(outlier)
                    else:
                        self.nr_whitelisted_elements += 1
                        list_documents_need_to_be_removed.append(ii)

                else:
                    non_outlier_values.add(term_value)

        return list_outliers, list_documents_need_to_be_removed

    def _create_outlier(self, non_outlier_values, term_value_count, aggregator_value, term_value, decision_frontier,
                        batch, ii):
        """
        Create outlier with given parameter

        :param non_outlier_values: list of document that aren't outliers
        :param term_value_count: number of term
        :param aggregator_value: aggregator value
        :param term_value: term value
        :param decision_frontier: value of the decision frontier
        :param batch: batch
        :param ii: index of the document linked to this outlier
        :return: the created outlier
        """
        non_outlier_values_sample = ",".join(random.sample(non_outlier_values, min(3, len(non_outlier_values))))

        observations = dict()
        observations["non_outlier_values_sample"] = non_outlier_values_sample
        observations["term_count"] = term_value_count
        observations["aggregator"] = aggregator_value
        observations["term"] = term_value
        observations["decision_frontier"] = decision_frontier
        observations["trigger_method"] = str(self.model_settings["trigger_method"])

        calculated_observations = batch[observations["aggregator"]]["observations"][ii]
        calculated_observations.update(observations)

        raw_doc = batch[observations["aggregator"]]["raw_docs"][ii]
        fields = es.extract_fields_from_document(raw_doc,
                                                 extract_derived_fields=self.model_settings["use_derived_fields"])
        return self.create_outlier(fields, raw_doc, extra_outlier_information=calculated_observations)

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        try:
            self.model_settings["process_documents_chronologically"] = settings.config.getboolean(
                self.config_section_name, "process_documents_chronologically")
        except NoOptionError:
            self.model_settings["process_documents_chronologically"] = True

        # remove unnecessary whitespace, split fields
        self.model_settings["target"] = settings.config.get(self.config_section_name,
                                                            "target").replace(' ', '').split(",")

        # remove unnecessary whitespace, split fields
        self.model_settings["aggregator"] = settings.config.get(self.config_section_name,
                                                                "aggregator").replace(' ', '').split(",")

        self.model_settings["trigger_on"] = settings.config.get(self.config_section_name, "trigger_on")
        self.model_settings["trigger_method"] = settings.config.get(self.config_section_name, "trigger_method")
        self.model_settings["trigger_sensitivity"] = settings.config.getfloat(self.config_section_name,
                                                                              "trigger_sensitivity")

        self.model_settings["target_count_method"] = settings.config.get(self.config_section_name,
                                                                         "target_count_method")

        try:
            self.model_settings["min_target_buckets"] = settings.config.getint(self.config_section_name,
                                                                               "min_target_buckets")
            if self.model_settings["target_count_method"] != "within_aggregator":
                logging.logger.warning("'min_target_buckets' is only useful when 'target_count_method' is set " +
                                       "to 'within_aggregator'")
        except NoOptionError:
            self.model_settings["min_target_buckets"] = None

        # Validate model settings
        if self.model_settings["target_count_method"] not in {"within_aggregator", "across_aggregators"}:
            raise ValueError("target count method " + self.model_settings["target_count_method"] + " not supported")

        if self.model_settings["trigger_on"] not in {"high", "low"}:
            raise ValueError("Unexpected outlier trigger condition " + str(self.model_settings["trigger_on"]))

        if self.model_settings["trigger_method"] not in {"percentile", "pct_of_max_value", "pct_of_median_value",
                                                         "pct_of_avg_value", "mad", "madpos", "stdev", "float",
                                                         "coeff_of_variation"}:
            raise ValueError("Unexpected outlier trigger method " + str(self.model_settings["trigger_method"]))

    @staticmethod
    def remove_term_from_batch(eval_terms_array, aggregator_value, term_counter):
        eval_terms_array[aggregator_value]["targets"].pop(term_counter)
        eval_terms_array[aggregator_value]["observations"].pop(term_counter)
        eval_terms_array[aggregator_value]["raw_docs"].pop(term_counter)
        return eval_terms_array
