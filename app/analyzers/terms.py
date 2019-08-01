import random
from configparser import NoOptionError

from helpers.singletons import settings, es, logging
from collections import defaultdict
from collections import Counter
import helpers.utils
from helpers.analyzer import Analyzer

from typing import DefaultDict, Optional, Dict


class TermsAnalyzer(Analyzer):

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

        self.model_settings["brute_force_target"] = "*" in self.model_settings["target"]

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

    def evaluate_model(self):
        if self.model_settings["brute_force_target"]:
            logging.logger.warning("running terms model in brute force mode, could take a long time!")
            target_fields_to_brute_force = self._calculate_target_fields_to_brute_force()

            for target_field in target_fields_to_brute_force:
                self.model_settings["brute_forced_field"] = target_field
                search_query = es.filter_by_query_string(self.model_settings["es_query_filter"] + " AND _exists_:" +
                                                         self.model_settings["brute_forced_field"])
                self._evaluate_target(target=[self.model_settings["brute_forced_field"]], search_query=search_query,
                                      brute_force=True)
        else:
            self._evaluate_target(target=self.model_settings["target"], search_query=self.search_query,
                                  brute_force=False)

    def _evaluate_target(self, target, search_query, brute_force=False):
        self.total_events = es.count_documents(index=self.es_index, search_query=search_query,
                                               model_settings=self.model_settings)

        self.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)
        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating terms model")

        if brute_force:
            logging.logger.info("brute forcing field %s", str(target[0]))

        if self.total_events > 0:
            eval_terms_array = defaultdict()
            total_terms_added = 0

            outlier_batches_trend = 0
            for doc in es.scan(index=self.es_index, search_query=search_query, model_settings=self.model_settings):
                logging.tick()
                target_sentences, aggregator_sentences = self._compute_aggregator_and_target_value(doc, target)

                if target_sentences is not None and aggregator_sentences is not None:
                    # Add current document to eval_terms_array
                    eval_terms_array = self._compute_eval_terms_for_one_doc(brute_force, target_sentences,
                                                                            eval_terms_array, aggregator_sentences, doc)
                    total_terms_added += len(target_sentences)

                    # Evaluate batch of events against the model
                    is_last_batch = (logging.current_step == self.total_events)  # Check if it is the last batch
                    # Run if it is the last batch OR if the batch size is large enough
                    if is_last_batch or total_terms_added >= settings.config.getint("terms", "terms_batch_eval_size"):
                        logging.logger.info("evaluating batch of " + "{:,}".format(total_terms_added) + " terms")

                        # evaluate the current batch. Check if we continue brut force and get the remaining terms
                        remaining_terms, stop_brut_force, outlier_batches_trend = \
                            self._evaluate_batch_and_test_brut_force(eval_terms_array, is_last_batch, brute_force,
                                                                     outlier_batches_trend)
                        if stop_brut_force:
                            break

                        # Reset data structures for next batch
                        eval_terms_array = remaining_terms
                        total_terms_added = 0

        self.print_analysis_summary()

    def _compute_eval_terms_for_one_doc(self, brute_force, target_sentences, eval_terms_array, aggregator_sentences,
                                        doc):
        observations = dict()
        if brute_force:
            observations["brute_forced_field"] = self.model_settings["brute_forced_field"]

        for target_sentence in target_sentences:
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence)

            for aggregator_sentence in aggregator_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)
                eval_terms_array = TermsAnalyzer.add_term_to_batch(eval_terms_array,
                                                                   flattened_aggregator_sentence,
                                                                   flattened_target_sentence, observations,
                                                                   doc)
        return eval_terms_array

    def _compute_aggregator_and_target_value(self, doc, target):
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

    def _evaluate_batch_and_test_brut_force(self, eval_terms_array, is_last_batch, brute_force, outlier_batches_trend):
        stop_brut_force = False
        new_outlier_batches_trend, remaining_terms = self._run_evaluate_documents(eval_terms_array=eval_terms_array,
                                                                                  is_last_batch=is_last_batch)
        outlier_batches_trend += new_outlier_batches_trend

        if brute_force:
            if outlier_batches_trend == -3:
                logging.logger.info("too many batches without outliers, we are not going to continue " +
                                    "brute forcing")
                stop_brut_force = True

            elif outlier_batches_trend == 3:
                logging.logger.info("too many batches with outliers, we are not going to continue " +
                                    "brute forcing")
                stop_brut_force = True

        return remaining_terms, stop_brut_force, outlier_batches_trend

    def _run_evaluate_documents(self, eval_terms_array, is_last_batch):
        outliers, remaining_terms = self._evaluate_batch_for_outliers(terms=eval_terms_array,
                                                                      is_last_batch=is_last_batch)

        # For each result, save it in batch and in ES
        for outlier in outliers:
            self.save_outlier_to_es(outlier)

        if len(outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in outliers))
            logging.logger.info("total outliers in batch processed: " + "{:,}".format(len(outliers)) + " [" +
                                "{:,}".format(unique_summaries) + " unique summaries]")
            outlier_batches_trend = 1
        else:
            logging.logger.info("no outliers detected in batch")
            outlier_batches_trend = -1

        return outlier_batches_trend, remaining_terms

    def _calculate_target_fields_to_brute_force(self):
        batch_size = settings.config.getint("terms", "terms_batch_eval_size")

        self.total_events = es.count_documents(index=self.es_index, search_query=self.search_query,
                                               model_settings=self.model_settings)
        logging.init_ticker(total_steps=min(self.total_events, batch_size),
                            desc=self.model_name + " - extracting brute force fields")

        field_names_to_brute_force = set()
        if self.total_events > 0:
            num_docs_processed = 0
            for doc in es.scan(index=self.es_index, search_query=self.search_query, model_settings=self.model_settings):
                logging.tick()
                fields = es.extract_fields_from_document(
                            doc, extract_derived_fields=self.model_settings["use_derived_fields"])
                fields = helpers.utils.flatten_dict(fields)

                # create list instead of iterator so we can mutate the dictionary when iterating
                for field_name in list(fields.keys()):
                    # skip all fields that are related to outliers, we don't want to brute force them
                    if field_name.startswith('outliers.'):
                        logging.logger.debug("not brute forcing outliers field " + str(field_name))
                        continue

                    # only brute force nested fields, so not the top level fields such as timestamp,
                    # deployment name, etc.
                    if "." in field_name:
                        field_names_to_brute_force.add(field_name)

                # only process a single batch of events in order to decide which fields to brute force
                if num_docs_processed == batch_size:
                    break
                else:
                    num_docs_processed += 1

        logging.logger.info("going to brute force " + str(len(field_names_to_brute_force)) + " fields")
        return field_names_to_brute_force

    def _evaluate_batch_for_outliers(self, is_last_batch, terms=None):
        # In case we want to count terms across different aggregators, we need to first iterate over all aggregators
        # and calculate the total number of unique terms for each aggregated value.
        # For example:
        # terms["smsc.exe"][A, B, C, D, D, E]
        # terms["abc.exe"][A, A, B]
        # is converted into:
        # unique_target_counts_across_aggregators: [5, 2] (the first term contains 5 unique values, the second
        # one contains 2)
        if self.model_settings["target_count_method"] == "across_aggregators":
            return self._evaluate_batch_for_outliers_across_aggregators(terms)

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
            return self._evaluate_batch_for_outliers_within_aggregator(terms, is_last_batch)

        return list(), dict()

    # ===== Across ===== #
    def _evaluate_batch_for_outliers_across_aggregators(self, terms):
        # Init
        list_outliers = list()  # List outliers
        # List of document (per aggregator) that aren't outlier (to help user to see non match results)
        # Notice that this dictionary will only be used if there is a loop (first loop fill the dict. Second loop take
        # the result if outlier is detected).
        non_outlier_values = defaultdict(list)
        first_run = True  # Force to run one time the loop
        nr_whitelisted_element_detected = 0  # Number of elements that have been removed (due to whitelist)

        # Run the loop the first time and still elements are removed (due to whitelist)
        while first_run or nr_whitelisted_element_detected > 0:
            if not first_run:
                logging.logger.debug("run again computation of batch because " + str(nr_whitelisted_element_detected) +
                                     " documents have been removed")
            first_run = False

            # Compute decision frontier and loop on all aggregator
            # For each of them, evaluate if it is an outlier and remove terms that are whitelisted (no return because
            # it is a dictionary)
            nr_whitelisted_element_detected, list_outliers = self._evaluate_aggregator_for_outlier_accross(
                terms, non_outlier_values)

        # All outliers and no remaining terms
        return list_outliers, {}

    def _evaluate_aggregator_for_outlier_accross(self, terms, non_outlier_values):
        nr_whitelisted_element_detected = 0
        unique_target_counts_across_aggregators, decision_frontier = \
            self._compute_count_across_aggregators_and_decision_frontier(terms)

        logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " +
                             str(decision_frontier) + " across all aggregators")
        list_outliers = list()

        # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
        for i, aggregator_value in enumerate(terms):
            unique_target_count_across_aggregators = unique_target_counts_across_aggregators[i]
            new_list_outliers, list_documents_need_to_be_removed = \
                self._evaluate_each_aggregator_is_outliers_and_mark_across(terms, aggregator_value,
                                                                           unique_target_count_across_aggregators,
                                                                           decision_frontier,
                                                                           non_outlier_values[aggregator_value])

            # If some documents need to be removed
            if len(list_documents_need_to_be_removed) > 0:
                # Save the number of element that need to be removed
                nr_whitelisted_element_detected = len(list_documents_need_to_be_removed)
                logging.logger.debug("removing " + "{:,}".format((len(list_documents_need_to_be_removed))) +
                                     " whitelisted documents from the batch for aggregator " + str(aggregator_value))

                # Remove whitelist document from the list that we need to compute
                # Note: Browse the list of documents that need to be removed in reverse order
                # To remove first the biggest index and avoid a shift (if we remove index 0, all values must be
                # decrease by one)
                for index in list_documents_need_to_be_removed[::-1]:
                    TermsAnalyzer.remove_term_from_batch(terms, aggregator_value, index)
            else:
                list_outliers += new_list_outliers

        # If at least one element need to be computed again
        if nr_whitelisted_element_detected > 0:
            list_outliers = list()  # Ignore detected outliers

        return nr_whitelisted_element_detected, list_outliers

    def _compute_count_across_aggregators_and_decision_frontier(self, terms):
        unique_target_counts_across_aggregators = list()

        # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
        for i, aggregator_value in enumerate(terms):
            # unique_targets_in_aggregated_value = loop 0: [A, B, C, D, E], loop 1: [A, A, B]
            # unique_target_counts_across_aggregators = loop 0: [5], loop 1: [5, 2]
            unique_targets_in_aggregated_value = set(terms[aggregator_value]["targets"])
            unique_target_counts_across_aggregators.append(len(unique_targets_in_aggregated_value))

        # Calculate the decision frontier
        # unique_target_counts_across_aggregators = [5, 2]
        decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                unique_target_counts_across_aggregators,
                                                                self.model_settings["trigger_sensitivity"],
                                                                self.model_settings["trigger_on"])
        return unique_target_counts_across_aggregators, decision_frontier

    def _evaluate_each_aggregator_is_outliers_and_mark_across(self, terms, aggregator_value,
                                                              unique_target_count_across_aggregators, decision_frontier,
                                                              non_outlier_values):
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
                terms, aggregator_value, unique_target_count_across_aggregators, decision_frontier, non_outlier_values)
        else:
            # Save non outliers list (do not be return because it is a dictionary)
            non_outlier_values += terms[aggregator_value]["targets"]

        return list_outliers, list_documents_need_to_be_removed

    def _mark_across_aggregator_document_as_outliers(self, terms, aggregator_value,
                                                     unique_target_count_across_aggregators, decision_frontier,
                                                     non_outlier_values):
        # Initialise
        list_outliers = list()
        list_documents_need_to_be_removed = list()

        for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
            outlier = self._create_outlier(non_outlier_values, unique_target_count_across_aggregators,
                                           aggregator_value, term_value, decision_frontier, terms, ii)
            if not outlier.is_whitelisted():
                list_outliers.append(outlier)
            else:
                self.nr_whitelisted_elements += 1
                list_documents_need_to_be_removed.append(ii)

        return list_outliers, list_documents_need_to_be_removed

    # ===== Within ===== #
    def _evaluate_batch_for_outliers_within_aggregator(self, terms, is_last_batch):
        # Initialize
        list_outliers = list()
        remaining_terms = dict()

        process_need_to_be_computed = terms.copy()
        for aggregator_value in process_need_to_be_computed.keys():
            outliers, enough_value = self._evaluate_aggregator_for_outliers_within(terms, is_last_batch,
                                                                                   aggregator_value)
            if not enough_value and not is_last_batch:
                remaining_terms[aggregator_value] = terms[aggregator_value]
            list_outliers += outliers

        return list_outliers, remaining_terms

    def _evaluate_aggregator_for_outliers_within(self, terms, is_last_batch, aggregator_value):
        enough_value = True
        list_outliers = list()
        list_documents_need_to_be_removed = list()
        first_run = True  # Force to run one time the loop

        while first_run or (enough_value and len(list_documents_need_to_be_removed) > 0 and
                            len(terms[aggregator_value]["targets"]) > 0):
            first_run = False

            # Count percentage of each target value occurring
            counted_targets = Counter(terms[aggregator_value]["targets"])
            counted_target_values = list(counted_targets.values())

            logging.logger.debug("terms count for aggregator value " + aggregator_value + " -> " +
                                 str(counted_targets))

            # If not enough bucket we stop the loop
            if self.model_settings["min_target_buckets"] is not None and \
                    len(counted_targets) < self.model_settings["min_target_buckets"]:
                enough_value = False

                # If last batch we remove data from remaining_terms to avoid infinite loop
                if is_last_batch:
                    logging.logger.debug("less than " + str(self.model_settings["min_target_buckets"]) +
                                         " time buckets, skipping analysis")
                    del terms[aggregator_value]
                break

            decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                    counted_target_values,
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])

            logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " +
                                 str(decision_frontier) + " for aggregator " + str(aggregator_value))

            if self.model_settings["trigger_method"] == "coeff_of_variation":
                new_list_outliers, list_documents_need_to_be_removed = \
                    self._evaluate_each_aggregator_coeff_for_outliers_within(decision_frontier, terms, aggregator_value,
                                                                             counted_targets)

            else:
                new_list_outliers, list_documents_need_to_be_removed = \
                    self._evaluate_each_aggregator_for_outliers_within(decision_frontier, terms, aggregator_value,
                                                                       counted_targets)

            # Remove document detected like outliers and whitelisted
            if len(list_documents_need_to_be_removed) > 0:
                logging.logger.debug("removing {:,}".format((len(list_documents_need_to_be_removed))) +
                                     " whitelisted documents from the batch for aggregator " + str(aggregator_value))

                # browse the list in reverse order (to remove first biggest index)
                for index in list_documents_need_to_be_removed[::-1]:
                    TermsAnalyzer.remove_term_from_batch(terms, aggregator_value, index)
            else:
                list_outliers += new_list_outliers

        return list_outliers, enough_value

    def _evaluate_each_aggregator_coeff_for_outliers_within(self, decision_frontier, terms, aggregator_value,
                                                            counted_targets):
        list_documents_need_to_be_removed = list()
        list_outliers = list()

        # decision_frontier = coeff_of_variation. So we need to check if coeff_of_variation is high or low
        # of the sensitivity
        if helpers.utils.is_outlier(decision_frontier, self.model_settings["trigger_sensitivity"],
                                    self.model_settings["trigger_on"]):
            non_outlier_values = set()
            for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
                term_value_count = counted_targets[term_value]
                outlier = self._create_outlier(non_outlier_values, term_value_count, aggregator_value,
                                               term_value, decision_frontier, terms, ii)
                if not outlier.is_whitelisted():
                    list_outliers.append(outlier)
                else:
                    self.nr_whitelisted_elements += 1
                    list_documents_need_to_be_removed.append(ii)

        return list_outliers, list_documents_need_to_be_removed

    def _evaluate_each_aggregator_for_outliers_within(self, decision_frontier, terms, aggregator_value,
                                                      counted_targets):
        list_documents_need_to_be_removed = list()
        list_outliers = list()
        non_outlier_values = set()

        for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
            term_value_count = counted_targets[term_value]
            is_outlier = helpers.utils.is_outlier(term_value_count, decision_frontier,
                                                  self.model_settings["trigger_on"])

            if is_outlier:
                outlier = self._create_outlier(non_outlier_values, term_value_count, aggregator_value,
                                               term_value, decision_frontier, terms, ii)
                if not outlier.is_whitelisted():
                    list_outliers.append(outlier)
                else:
                    self.nr_whitelisted_elements += 1
                    list_documents_need_to_be_removed.append(ii)

            else:
                non_outlier_values.add(term_value)
        return list_outliers, list_documents_need_to_be_removed

    def _create_outlier(self, non_outlier_values, term_value_count, aggregator_value, term_value, decision_frontier,
                        terms, ii):
        non_outlier_values_sample = ",".join(random.sample(non_outlier_values, min(3, len(non_outlier_values))))

        observations = dict()
        observations["non_outlier_values_sample"] = non_outlier_values_sample
        observations["term_count"] = term_value_count
        observations["aggregator"] = aggregator_value
        observations["term"] = term_value
        observations["decision_frontier"] = decision_frontier
        observations["trigger_method"] = str(self.model_settings["trigger_method"])

        calculated_observations = terms[observations["aggregator"]]["observations"][ii]
        calculated_observations.update(observations)

        raw_doc = terms[observations["aggregator"]]["raw_docs"][ii]
        fields = es.extract_fields_from_document(raw_doc,
                                                 extract_derived_fields=self.model_settings["use_derived_fields"])
        return self.create_outlier(fields, raw_doc, extra_outlier_information=calculated_observations,
                                   es_process_outlier=False)

    @staticmethod
    def add_term_to_batch(eval_terms_array: DefaultDict, aggregator_value: Optional[str], target_value: Optional[str],
                          observations: Dict, doc: Dict) -> DefaultDict:
        if aggregator_value not in eval_terms_array.keys():
            eval_terms_array[aggregator_value] = defaultdict(list)

        eval_terms_array[aggregator_value]["targets"].append(target_value)
        eval_terms_array[aggregator_value]["observations"].append(observations)
        eval_terms_array[aggregator_value]["raw_docs"].append(doc)

        return eval_terms_array

    @staticmethod
    def remove_term_from_batch(eval_terms_array, aggregator_value, term_counter):
        eval_terms_array[aggregator_value]["targets"].pop(term_counter)
        eval_terms_array[aggregator_value]["observations"].pop(term_counter)
        eval_terms_array[aggregator_value]["raw_docs"].pop(term_counter)
        return eval_terms_array
