import random

from helpers.singletons import settings, es, logging
from collections import defaultdict
from collections import Counter
import helpers.utils
from helpers.analyzer import Analyzer
from helpers.outlier import Outlier
from numpy import float64

from typing import Set, Dict, List, DefaultDict, Any, Union, Optional


class TermsAnalyzer(Analyzer):

    def evaluate_model(self) -> None:
        self.extract_additional_model_settings()

        if self.model_settings["brute_force_target"]:
            logging.logger.warning("running terms model in brute force mode, could take a long time!")
            target_fields_to_brute_force: Set = self.calculate_target_fields_to_brute_force()

            for target_field in target_fields_to_brute_force:
                self.model_settings["brute_forced_field"] = target_field
                search_query: Dict[str, List] = es.filter_by_query_string(self.model_settings["es_query_filter"] + \
                                                                          " AND _exists_:" + \
                                                                          self.model_settings["brute_forced_field"])
                self.evaluate_target(target=[self.model_settings["brute_forced_field"]], search_query=search_query,
                                     brute_force=True)
        else:
            self.evaluate_target(target=self.model_settings["target"],
                                 search_query=es.filter_by_query_string(self.model_settings["es_query_filter"]),
                                 brute_force=False)

    def evaluate_target(self, target: List[str], search_query: Dict[str, List], brute_force: bool=False) -> None:
        self.total_events: int = es.count_documents(index=self.es_index, search_query=search_query)

        logging.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)
        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating terms model")

        if brute_force:
            logging.logger.info("brute forcing field %s", str(target[0]))

        eval_terms_array: DefaultDict = defaultdict()
        total_terms_added: int = 0

        outlier_batches_trend: int = 0
        for doc in es.scan(index=self.es_index, search_query=search_query):
            logging.tick()
            fields: Dict = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])

            will_process_doc: bool
            try:
                target_sentences: List[List] = helpers.utils.flatten_fields_into_sentences(fields=fields,
                                                                                           sentence_format=target)
                aggregator_sentences: List[List] = helpers.utils.flatten_fields_into_sentences(fields=fields,
                                                                    sentence_format=self.model_settings["aggregator"])
                will_process_doc = True
            except (KeyError, TypeError):
                logging.logger.debug("Skipping event which does not contain the target and aggregator fields we " + \
                                     "are processing. - [" + self.model_name + "]")
                will_process_doc = False

            if will_process_doc:
                observations: Dict[str, Any] = dict()

                if brute_force:
                    observations["brute_forced_field"] = self.model_settings["brute_forced_field"]

                for target_sentence in target_sentences:
                    flattened_target_sentence: Optional[str] = helpers.utils.flatten_sentence(target_sentence)

                    for aggregator_sentence in aggregator_sentences:
                        flattened_aggregator_sentence: Optional[str]=helpers.utils.flatten_sentence(aggregator_sentence)
                        eval_terms_array = self.add_term_to_batch(eval_terms_array, flattened_aggregator_sentence,
                                                                  flattened_target_sentence, observations, doc)

                total_terms_added += len(target_sentences)

            # Evaluate batch of events against the model
            last_batch: bool = (logging.current_step == self.total_events)
            if last_batch or total_terms_added >= settings.config.getint("terms", "terms_batch_eval_size"):
                logging.logger.info("evaluating batch of " + "{:,}".format(total_terms_added) + " terms")
                outliers: List[Outlier] = self.evaluate_batch_for_outliers(terms=eval_terms_array)

                if len(outliers) > 0:
                    unique_summaries: int = len(set(o.outlier_dict["summary"] for o in outliers))
                    logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" + \
                                        str(unique_summaries) + " unique summaries]")
                    outlier_batches_trend += 1
                else:
                    logging.logger.info("no outliers detected in batch")
                    outlier_batches_trend -= 1


                if outlier_batches_trend == -3 and brute_force:
                    logging.logger.info("too many batches without outliers, we are not going to continue brute forcing")
                    break

                elif outlier_batches_trend == 3 and brute_force:
                    logging.logger.info("too many batches with outliers, we are not going to continue brute forcing")
                    break

                # Reset data structures for next batch
                eval_terms_array = defaultdict()
                total_terms_added = 0

        self.print_analysis_summary()

    def calculate_target_fields_to_brute_force(self) -> Set:
        search_query: Dict[str, List] = es.filter_by_query_string(self.model_settings["es_query_filter"])
        batch_size: int = settings.config.getint("terms", "terms_batch_eval_size")

        self.total_events = es.count_documents(index=self.es_index, search_query=search_query)
        logging.init_ticker(total_steps=min(self.total_events, batch_size), 
                            desc=self.model_name + " - extracting brute force fields")

        field_names_to_brute_force: Set = set()
        num_docs_processed: int = 0
        for doc in es.scan(index=self.es_index, search_query=search_query):
            logging.tick()
            fields: Dict = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
            fields = helpers.utils.flatten_dict(fields)

            for field_name in list(fields.keys()):  # create list instead of iterator so we can mutate the
                # dictionary when iterating
                # skip all fields that are related to outliers, we don't want to brute force them
                if field_name.startswith('outliers.'):
                    logging.logger.debug("not brute forcing outliers field " + str(field_name))
                    continue

                # only brute force nested fields, so not the top level fields such as timestamp, deployment name, etc.
                if "." in field_name:
                    field_names_to_brute_force.add(field_name)

            # only process a single batch of events in order to decide which fields to brute force
            if num_docs_processed == batch_size:
                break
            else:
                num_docs_processed += 1

        logging.logger.info("going to brute force " + str(len(field_names_to_brute_force)) + " fields")
        return field_names_to_brute_force

    def extract_additional_model_settings(self) -> None:
        self.model_settings["target"] = settings.config.get(self.config_section_name, "target")\
                                            .replace(' ', '').split(",")  # remove unnecessary whitespace, split fields

        self.model_settings["brute_force_target"] = "*" in self.model_settings["target"]

        self.model_settings["aggregator"] = settings.config.get(self.config_section_name, "aggregator")\
                                            .replace(' ', '').split(",")  # remove unnecessary whitespace, split fields

        self.model_settings["trigger_on"] = settings.config.get(self.config_section_name, "trigger_on")
        self.model_settings["trigger_method"] = settings.config.get(self.config_section_name, "trigger_method")
        self.model_settings["trigger_sensitivity"] = settings.config.getint(self.config_section_name,
                                                                            "trigger_sensitivity")

        self.model_settings["target_count_method"] = settings.config.get(self.config_section_name,
                                                                         "target_count_method")

        # Validate model settings
        if self.model_settings["target_count_method"] not in {"within_aggregator", "across_aggregators"}:
            raise ValueError("target count method " + self.model_settings["target_count_method"] + " not supported")

        if self.model_settings["trigger_on"] not in {"high", "low"}:
            raise ValueError("Unexpected outlier trigger condition " + self.model_settings["trigger_on"])

    @staticmethod
    def add_term_to_batch(eval_terms_array: DefaultDict, aggregator_value: Optional[str], target_value: Optional[str],
                          observations: Dict, doc: Dict) -> DefaultDict:
        if aggregator_value not in eval_terms_array.keys():
            eval_terms_array[aggregator_value] = defaultdict(list)

        eval_terms_array[aggregator_value]["targets"].append(target_value)
        eval_terms_array[aggregator_value]["observations"].append(observations)
        eval_terms_array[aggregator_value]["raw_docs"].append(doc)

        return eval_terms_array

    def evaluate_batch_for_outliers(self, terms: DefaultDict) -> List[Outlier]:
        # Initialize
        outliers: List[Outlier] = list()
        decision_frontier: Union[int, float, float64]
        non_outlier_values: Set = set()
        is_outlier: Union[int, float, float64]
        non_outlier_values_sample: str
        observations: Dict[str, Any]
        calculated_observations: Dict
        raw_doc: Dict[str, Dict]
        fields: Dict

        # In case we want to count terms across different aggregators, we need to first iterate over all aggregators
        # and calculate the total number of unique terms for each aggregated value.
        # For example:
        # terms["smsc.exe"][A, B, C, D, D, E]
        # terms["abc.exe"][A, A, B]
        # is converted into:
        # unique_target_counts_across_aggregators: [5, 2] (the first term contains 5 unique values, the second
        # one contains 2)
        if self.model_settings["target_count_method"] == "across_aggregators":
            unique_target_counts_across_aggregators: List[int] = list()

            # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
            for i, aggregator_value in enumerate(terms):
                # unique_targets_in_aggregated_value = loop 0: [A, B, C, D, E], loop 1: [A, A, B]
                # unique_target_counts_across_aggregators = loop 0: [5], loop 1: [5, 2]
                unique_targets_in_aggregated_value: Set = set(terms[aggregator_value]["targets"])
                unique_target_counts_across_aggregators.append(len(unique_targets_in_aggregated_value))

            # Calculate the decision frontier
            # unique_target_counts_across_aggregators = [5, 2]
            decision_frontier = helpers.utils.get_decision_frontier(
                                                                    self.model_settings["trigger_method"],
                                                                    unique_target_counts_across_aggregators,
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])
            logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " + \
                                 str(decision_frontier) + " across all aggregators")

            
            # loop 0: {i=0, aggregator_value = "smsc.exe"}, loop 1: {i=1, aggregator_value = "abc.exe"},
            for i, aggregator_value in enumerate(terms):
                unique_target_count_across_aggregators: int = unique_target_counts_across_aggregators[i]
                logging.logger.debug("unique target count for aggregator " + str(aggregator_value) + ": " + \
                                     str(unique_target_count_across_aggregators) + " - decision frontier " + \
                                     str(decision_frontier))
                is_outlier = helpers.utils.is_outlier(unique_target_count_across_aggregators, decision_frontier,
                                                        self.model_settings["trigger_on"])

                if is_outlier:
                    for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
                        non_outlier_values_sample = ",".join(random.sample(non_outlier_values,
                                                                           min(3, len(non_outlier_values))))

                        observations = dict()
                        observations["non_outlier_values_sample"] = non_outlier_values_sample
                        observations["term_count"] = unique_target_count_across_aggregators
                        observations["aggregator"] = aggregator_value
                        observations["term"] = term_value
                        observations["decision_frontier"] = decision_frontier
                        observations["trigger_method"] = str(self.model_settings["trigger_method"])

                        calculated_observations = terms[observations["aggregator"]]["observations"][ii]
                        calculated_observations.update(observations)

                        raw_doc = terms[observations["aggregator"]]["raw_docs"][ii]
                        fields = es.extract_fields_from_document(raw_doc,
                                                    extract_derived_fields=self.model_settings["use_derived_fields"])
                        outliers.append(self.process_outlier(fields, raw_doc,
                                                             extra_outlier_information=calculated_observations))
                else:
                    for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
                        non_outlier_values.add(term_value)

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
            for i, aggregator_value in enumerate(terms):
                # Count percentage of each target value occuring
                counted_targets: Counter = Counter(terms[aggregator_value]["targets"])
                counted_target_values: List = list(counted_targets.values())

                logging.logger.debug("terms count for aggregator value " + aggregator_value + " -> " + \
                                     str(counted_targets))
                decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                        counted_target_values,
                                                                        self.model_settings["trigger_sensitivity"],
                                                                        self.model_settings["trigger_on"])

                logging.logger.debug("using " + self.model_settings["trigger_method"] + " decision frontier " + \
                                     str(decision_frontier) + " for aggregator " + str(aggregator_value))

                for ii, term_value in enumerate(terms[aggregator_value]["targets"]):
                    term_value_count: int = counted_targets[term_value]
                    is_outlier = helpers.utils.is_outlier(term_value_count, decision_frontier,
                                                          self.model_settings["trigger_on"])

                    if is_outlier:
                        non_outlier_values_sample = ",".join(random.sample(non_outlier_values,
                                                                           min(3, len(non_outlier_values))))

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
                        outliers.append(self.process_outlier(fields, raw_doc,
                                                             extra_outlier_information=calculated_observations))
                    else:
                        non_outlier_values.add(term_value)
        return outliers
