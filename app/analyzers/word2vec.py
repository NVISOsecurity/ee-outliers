import helpers.utils

from helpers.singletons import es, logging
from helpers.analyzer import Analyzer
import analyzers.ml_models.word2vec as word2vec
from collections import defaultdict

import re
from configparser import SectionProxy
from tabulate import tabulate
import numpy as np
import operator

from typing import List, Tuple, Optional

PRINT_COLORS = {"red": "\033[91m",
                "green": "\033[92m",
                "blue": "\033[94m",
                "end": "\033[0m"}


class Word2VecAnalyzer(Analyzer):

    def __init__(self, model_name: str, config_section: SectionProxy):
        super(Word2VecAnalyzer, self).__init__("word2vec", model_name, config_section)

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        self.model_settings["target"] = self.config_section["target"].replace(' ', '').split(",")

        self.model_settings["aggregator"] = self.config_section["aggregator"].replace(' ', '').split(",")

        # word2vec_batch_eval_size parameter
        self.model_settings["word2vec_batch_eval_size"] = self.extract_parameter("word2vec_batch_eval_size",
                                                                                 param_type="int")

        # min_target_buckets parameter
        self.model_settings["min_target_buckets"] = self.extract_parameter("min_target_buckets", param_type="int")

        # drop_duplicates parameter
        self.model_settings["drop_duplicates"] = self.extract_parameter("drop_duplicates",
                                                                        param_type="boolean",
                                                                        default=False)

        # use_prob_model parameter
        self.model_settings["use_prob_model"] = self.extract_parameter("use_prob_model",
                                                                       param_type="boolean",
                                                                       default=False)

        # separators parameter
        self.model_settings["separators"] = self.extract_parameter("separators")
        self.model_settings["separators"] = self.model_settings["separators"].strip('"')

        # size_window parameter
        self.model_settings["size_window"] = self.extract_parameter("size_window", param_type="int")

        # min_uniq_word_occurrence parameter
        self.model_settings["min_uniq_word_occurrence"] = self.extract_parameter("min_uniq_word_occurrence",
                                                                                 param_type="int",
                                                                                 default=1)

        # output_prob parameters
        self.model_settings["output_prob"] = self.extract_parameter("output_prob",
                                                                    param_type="boolean",
                                                                    default=True)
        # use_geo_mean parameter
        self.model_settings["use_geo_mean"] = self.model_settings["output_prob"]

        # num_epochs parameter
        self.model_settings["num_epochs"] = self.extract_parameter("num_epochs",
                                                                   param_type="int",
                                                                   default=1)

        # learning_rate parameter
        self.model_settings["learning_rate"] = self.extract_parameter("learning_rate",
                                                                      param_type="float",
                                                                      default=0.001)

        # embedding_size parameter
        self.model_settings["embedding_size"] = self.extract_parameter("embedding_size",
                                                                       param_type="int",
                                                                       default=40)

        # seed parameter
        self.model_settings["seed"] = self.extract_parameter("seed",
                                                             param_type="int",
                                                             default=0)

        # document need to be read chronologically if random seed is activated
        if self.model_settings["seed"] != 0:
            self.model_settings["process_documents_chronologically"] = True
        else:
            self.model_settings["process_documents_chronologically"] = self.config_section.getboolean(
                "process_documents_chronologically", False)

        # print_score_table parameter
        self.model_settings["print_score_table"] = self.extract_parameter("print_score_table",
                                                                          param_type="boolean",
                                                                          default=0)

        # print_confusion_matrix parameter
        self.model_settings["print_confusion_matrix"] = self.extract_parameter("print_confusion_matrix",
                                                                               param_type="boolean",
                                                                               default=0)

        # trigger_focus parameter
        self.model_settings["trigger_focus"] = self.extract_parameter("trigger_focus", default="word")
        if self.model_settings["trigger_focus"] not in {"word", "text"}:
            raise ValueError("Unexpected outlier trigger focus " + str(self.model_settings["trigger_focus"]))

        # trigger_score parameter
        self.model_settings["trigger_score"] = self.extract_parameter("trigger_score")
        if self.model_settings["trigger_score"] not in {"center", "context", "total", "mean"}:
            raise ValueError("Unexpected outlier trigger score " + str(self.model_settings["trigger_score"]))
        if self.model_settings["trigger_score"] == "mean" and self.model_settings["trigger_focus"] == "word":
            raise ValueError("trigger_focus=word is not compatible with trigger_score=mean")

        # trigger_on parameter
        self.model_settings["trigger_on"] = self.extract_parameter("trigger_on", default="low")
        if self.model_settings["trigger_on"] not in {"high", "low"}:
            raise ValueError("Unexpected outlier trigger condition " + str(self.model_settings["trigger_on"]))

        # trigger_method parameter
        self.model_settings["trigger_method"] = self.extract_parameter("trigger_method")
        if self.model_settings["trigger_method"] not in {"percentile", "pct_of_max_value", "pct_of_median_value",
                                                         "pct_of_avg_value", "mad", "madpos", "stdev", "float",
                                                         "coeff_of_variation"}:
            raise ValueError("Unexpected outlier trigger method " + str(self.model_settings["trigger_method"]))

        # trigger_sensitivity parameter
        self.model_settings["trigger_sensitivity"] = self.extract_parameter("trigger_sensitivity", param_type="float")

        self.current_batch_num = 0

    def evaluate_model(self):

        target_fields = self.model_settings["target"]
        aggr_fields = self.model_settings["aggregator"]
        self.total_events, documents = es.count_and_scan_documents(index=self.model_settings["es_index"],
                                                                   search_query=self.search_query,
                                                                   model_settings=self.model_settings)
        self.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)
        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating word2vec model")
        if documents:
            batch = dict()
            total_docs_in_batch = 0
            total_duplicates = 0
            num_targets_not_processed = 0

            for doc in documents:
                target_sentences, aggr_sentences = self._extract_target_and_aggr_sentences(doc=doc,
                                                                                           target_fields=target_fields,
                                                                                           aggr_fields=aggr_fields)
                if target_sentences is not None and aggr_sentences is not None:
                    batch, num_doc_add, num_duplicates = self._add_doc_and_target_sentences_to_batch(
                        current_batch=batch,
                        target_sentences=target_sentences,
                        aggr_sentences=aggr_sentences,
                        doc=doc)
                    total_docs_in_batch += num_doc_add
                    total_duplicates += num_duplicates

                if total_docs_in_batch >= self.model_settings["word2vec_batch_eval_size"] or \
                        logging.current_step + 1 == self.total_events:
                    # Display log info message
                    self._display_batch_log_message(total_docs_in_batch,
                                                    num_targets_not_processed,
                                                    total_duplicates,
                                                    logging.current_step + 1)
                    # Evaluate the current batch
                    outliers_in_batch, total_docs_removed = self._evaluate_batch_for_outliers(batch=batch)

                    if total_docs_removed == 0:
                        logging.logger.warning("Unable to fill the aggregator buckets for this batch.")

                    # Processing the outliers found
                    self._processing_outliers_in_batch(outliers_in_batch)

                    total_docs_in_batch -= total_docs_removed
                    num_targets_not_processed = total_docs_in_batch
                    self.current_batch_num += 1

                logging.tick()

    def _extract_target_and_aggr_sentences(self,
                                           doc: dict,
                                           target_fields: List[str],
                                           aggr_fields: List[str]) -> Tuple[Optional[List[List[str]]],
                                                                            Optional[List[List[str]]]]:
        """
        Extract target and aggregator sentence from a document

        :param doc: document where data need to be extract.
        :param target_fields: list of target key names.
        :param aggr_fields: list of aggregator key names.
        :return: list of list of target and list of list of aggregator
        """
        fields = es.extract_fields_from_document(doc, extract_derived_fields=self.model_settings["use_derived_fields"])
        try:
            target_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=target_fields)
            aggr_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=aggr_fields)
        except (KeyError, TypeError):
            logging.logger.debug("Skipping event which does not contain the target and aggregator " +
                                 "fields we are processing. - [" + self.model_name + "]")
            return None, None
        return target_sentences, aggr_sentences

    def _add_doc_and_target_sentences_to_batch(self,
                                               current_batch: dict,
                                               target_sentences: List[List[str]],
                                               aggr_sentences: List[List[str]],
                                               doc: dict) -> Tuple[dict, int, int]:
        """
        Add a document to the current batch.
        If drop_duplicates is activated and the document already appear in batch, it is not added to the batch.

        :param current_batch: existing batch (where doc need to be saved)
        :param target_sentences: list of list of targets
        :param aggr_sentences: list of list of aggregator
        :param doc: document that need to be added
        :return: Tuple of three elements representing respectively;
            - batch with target sentence and document inside
            - number of documents added to batch
            - number of document not added to batch
        """
        for target_sentence in target_sentences:
            # Remove escape character '\' that escapes regex special characters.
            # This escape character may be present into the regex expression contained in separators.
            separators_without_special_char = re.sub(r'\\(.)', r'\1', self.model_settings["separators"])
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence,
                                                                       sep_str=separators_without_special_char)

            for aggregator_sentence in aggr_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)

                if flattened_aggregator_sentence not in current_batch.keys():
                    current_batch[flattened_aggregator_sentence] = defaultdict(list)
                if self.model_settings["drop_duplicates"] and \
                        flattened_target_sentence in current_batch[flattened_aggregator_sentence]["targets"]:
                    return current_batch, 0, len(target_sentences) * len(aggr_sentences)

                current_batch[flattened_aggregator_sentence]["targets"].append(flattened_target_sentence)
                current_batch[flattened_aggregator_sentence]["raw_docs"].append(doc)

        return current_batch, len(target_sentences) * len(aggr_sentences), 0

    def _display_batch_log_message(self,
                                   total_targets_in_batch: int,
                                   num_targets_not_processed: int,
                                   total_duplicates: int,
                                   current_step: int):
        """
        Print info log message to stdout with information about the current batch.

        :param total_targets_in_batch: Total number of events in current batch
        :param num_targets_not_processed: Total number of events no processed during the previous batches
        :param total_duplicates: Total number of duplicates target sentences
        :param current_step: Total number of events processed
        """
        if total_duplicates > 0:
            logging.logger.info("BATCH %i - Drop_duplicates activated: %i/%i events have been removed",
                                self.current_batch_num,
                                total_duplicates,
                                current_step)

        log_message = "BATCH " + str(self.current_batch_num)
        log_message += " - evaluating batch of " + "{:,}".format(total_targets_in_batch) + " sentences "
        if num_targets_not_processed > 0:
            log_message += "(with " + "{:,}".format(num_targets_not_processed) + " sentences from last batch) "
        log_message += "[" + "{:,}".format(logging.current_step + 1) + " events processed]"
        logging.logger.info(log_message)

    def _evaluate_batch_for_outliers(self, batch):
        """
        Evaluates if the batch contains outliers.
        Loop over each aggregation, if the aggregation contains enough events it evaluates if the aggregation contains
        outliers. When evaluation of the aggregation is finished it removes the event from batch.

        :param batch: Batch containing the events.
        :return: (outliers_in_batch, total_targets_removed)
            - outliers_in_batch: list of outliers found in current batch.
            - total_targets_removed: the total number of event removed from batch.
        """
        outliers_in_batch = list()
        total_targets_removed = 0
        for aggr_key, aggr_elem in batch.items():
            num_targets = len(aggr_elem["targets"])
            if num_targets >= self.model_settings["min_target_buckets"]:
                agrr_outliers = self._evaluate_aggr_for_outliers(aggr_elem)

                outliers_in_batch.extend(agrr_outliers)

                # Remove aggr_elem from batch
                aggr_elem["targets"] = list()
                aggr_elem["raw_docs"] = list()
                total_targets_removed += num_targets

        return outliers_in_batch, total_targets_removed

    def _evaluate_aggr_for_outliers(self, aggr_elem):
        """
        Evaluates if the aggregation contains outliers.
        Creates a word2vec model, prepares the model by creating the vocabulary, trains the model, evaluates the events
        and finally finds the outliers.

        :param aggr_elem: aggregator element
        :return: List of outliers
        """
        # Create word2vec model
        w2v_model = word2vec.Word2Vec(self.model_settings["separators"],
                                      self.model_settings["size_window"],
                                      self.model_settings["num_epochs"],
                                      self.model_settings["learning_rate"],
                                      self.model_settings["embedding_size"],
                                      self.model_settings["seed"])

        # Add and prepare vocabulary of word2vec
        w2v_model.update_vocabulary_counter(aggr_elem["targets"])
        w2v_model.prepare_voc(min_voc_occurrence=self.model_settings["min_uniq_word_occurrence"])

        if self.model_settings["use_prob_model"]:
            model_eval_outputs = w2v_model.prob_model(aggr_elem["targets"], self.model_settings["output_prob"])
        else:
            # Train word2vec model
            w2v_model.train_model(aggr_elem["targets"])

            output_raw = not self.model_settings["output_prob"]
            # Eval word2vec model
            model_eval_outputs = w2v_model.eval_model(aggr_elem["targets"], output_raw)

        # Find outliers from the word2vec model outputs
        outliers = self._find_outliers(model_eval_outputs, aggr_elem, w2v_model)

        return outliers

    def _find_outliers(self, model_eval_outputs, aggr_elem, w2v_model):
        """
        Finds and creates outliers from the word2vec output "model_eval_outputs" and print information on stdout.

        :param model_eval_outputs: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id.
            - context word index in text.
            - context word id.
            - text index in eval_data.
            - probability/raw output of the context word id given the center word id.
        :param aggr_elem: aggregator element
        :param w2v_model: word2vec model
        :return: list of outlier
        """
        confusion_matrix_val = {"TP": 0,
                                "TN": 0,
                                "FP": 0,
                                "FN": 0}
        outliers = list()

        # Find all type of metrics concerning the text and the words
        word_scores_info, text_scores_info, compo_scores_info = self._find_all_scores(model_eval_outputs)
        score_type_to_text_idx_to_word_key_to_score, score_type_to_word_id_to_scores_list = word_scores_info
        score_type_to_text_idx_to_score = text_scores_info
        score_type_to_word_id_to_compo_word_to_score, score_type_to_compo_word_to_word_id_to_score = compo_scores_info

        # Compute the decision frontier
        word_id_to_decision_frontier, text_decision_frontier = self._find_decision_frontier(
            score_type_to_word_id_to_scores_list, score_type_to_text_idx_to_score)

        for text_idx in score_type_to_text_idx_to_score[self.model_settings["trigger_score"]]:
            text_str = aggr_elem["targets"][text_idx]
            # tokenize the text
            word_list = helpers.utils.split_text_by_separator(text_str, self.model_settings["separators"])

            text_analyzer = TextAnalyzer(text_idx,
                                         word_list,
                                         self.model_settings["trigger_focus"],
                                         self.model_settings["trigger_score"],
                                         self.model_settings["trigger_on"],
                                         self.model_settings["size_window"])
            text_analyzer.extract_text_scores(score_type_to_text_idx_to_score)
            text_analyzer.text_find_outlier(text_decision_frontier)

            for word_idx, word in enumerate(word_list):
                word_id = text_analyzer.find_word_id_and_fill_occur_word_row(word_idx, word, w2v_model)
                word_key = (word_idx, word_id)
                text_analyzer.fill_score_row_and_find_word_outlier(word_key,
                                                                   w2v_model,
                                                                   score_type_to_text_idx_to_word_key_to_score,
                                                                   word_id_to_decision_frontier,
                                                                   score_type_to_word_id_to_compo_word_to_score,
                                                                   score_type_to_compo_word_to_word_id_to_score)

            raw_doc = aggr_elem["raw_docs"][text_idx]
            fields = es.extract_fields_from_document(raw_doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
            if "label" not in fields:
                self.model_settings["print_confusion_matrix"] = False
            if self.model_settings["print_confusion_matrix"]:
                confusion_matrix_val = self._update_confusion_matrix_val(text_analyzer.find_outlier,
                                                                         fields["label"],
                                                                         confusion_matrix_val)
            if text_analyzer.find_outlier:
                if self.model_settings["print_score_table"]:
                    text_analyzer.print_score_table()
                    text_analyzer.print_most_expected_words()
                    text_analyzer.print_most_expected_window_words()

                observations = text_analyzer.get_observations()

                outlier = self.create_outlier(fields,
                                              raw_doc,
                                              extra_outlier_information=observations)
                outliers.append(outlier)

        self._print_confusion_matrix_and_metrics(confusion_matrix_val, self.model_settings["print_confusion_matrix"])

        return outliers

    def _find_all_scores(self, model_outputs):
        """
        Find metric score of each word and text.

        :param model_outputs: List of tuple. Each element within the tuple represent respectively;
            - center word index in text.
            - center word id.
            - context word index in text.
            - context word id.
            - text index in eval_data.
            - probability/raw output of the context word id given the center word id.
        :return: Tuple of three elements:
            - word_scores_info: Tuple of two elements:
                - score_type_to_text_idx_to_word_key_to_score: Dict[score_type][text_idx][word_key] = word_score
                - score_type_to_word_id_to_scores_list: Dict[score_type][word_id] = List[word_score]
            - text_scores_info = score_type_to_text_idx_to_score: Dict[score_type][text_idx] = text_score
            - compo_scores_info: Tuple of two elements:
                - score_type_to_word_id_to_compo_word_to_score: Dict[score_type][word_id][compo_word] = word_score
                - score_type_to_compo_word_to_word_id_to_score: Dict[scoe_type][compo_word][word_id] = word_score
        """
        #
        score_type_to_text_idx_to_word_key_to_score = {"center": dict(),
                                                       "context": dict(),
                                                       "total": dict()}
        score_type_to_word_id_to_scores_list = {"center": dict(),
                                                "context": dict(),
                                                "total": dict()}
        score_type_to_text_idx_to_score = {"center": dict(),
                                           "context": dict(),
                                           "total": dict(),
                                           "mean": dict()}
        score_type_to_word_id_to_compo_word_to_score = {"center": dict(),
                                                        "context": dict(),
                                                        "total": dict()}
        score_type_to_compo_word_to_word_id_to_score = {"center": dict(),
                                                        "context": dict(),
                                                        "total": dict()}
        current_text_idx = 0

        tmp_word_key_to_compo_word = dict()

        tmp_center_key_to_val_list = dict()
        tmp_context_key_to_val_list = dict()

        for score_type in score_type_to_text_idx_to_word_key_to_score:
            score_type_to_text_idx_to_word_key_to_score[score_type][current_text_idx] = dict()

        for center_idx, center_id, context_idx, context_id, text_idx, output_val in model_outputs:
            center_key = (center_idx, center_id)
            context_key = (context_idx, context_id)
            if current_text_idx == text_idx:
                if center_key not in tmp_center_key_to_val_list:
                    tmp_center_key_to_val_list[center_key] = list()
                    tmp_word_key_to_compo_word[center_key] = ""
                tmp_center_key_to_val_list[center_key].append(output_val)
                tmp_word_key_to_compo_word[center_key] += str(context_id) + "|"

                if context_key not in tmp_context_key_to_val_list:
                    tmp_context_key_to_val_list[context_key] = list()
                tmp_context_key_to_val_list[context_key].append(output_val)

            else:
                tmp_inputs = (current_text_idx,
                              tmp_center_key_to_val_list,
                              tmp_context_key_to_val_list,
                              tmp_word_key_to_compo_word)
                self._update_all_score(tmp_inputs,
                                       score_type_to_text_idx_to_word_key_to_score,
                                       score_type_to_word_id_to_scores_list,
                                       score_type_to_text_idx_to_score,
                                       score_type_to_word_id_to_compo_word_to_score,
                                       score_type_to_compo_word_to_word_id_to_score)

                tmp_word_key_to_compo_word = dict()
                tmp_word_key_to_compo_word[center_key] = str(context_id) + "|"

                tmp_center_key_to_val_list = dict()
                tmp_center_key_to_val_list[center_key] = list()
                tmp_center_key_to_val_list[center_key].append(output_val)
                tmp_context_key_to_val_list = dict()
                tmp_context_key_to_val_list[context_key] = list()
                tmp_context_key_to_val_list[context_key].append(output_val)

                for score_type in score_type_to_text_idx_to_word_key_to_score:
                    score_type_to_text_idx_to_word_key_to_score[score_type][text_idx] = dict()

                current_text_idx = text_idx
        tmp_inputs = (current_text_idx,
                      tmp_center_key_to_val_list,
                      tmp_context_key_to_val_list,
                      tmp_word_key_to_compo_word)
        self._update_all_score(tmp_inputs,
                               score_type_to_text_idx_to_word_key_to_score,
                               score_type_to_word_id_to_scores_list,
                               score_type_to_text_idx_to_score,
                               score_type_to_word_id_to_compo_word_to_score,
                               score_type_to_compo_word_to_word_id_to_score)

        word_scores_info = score_type_to_text_idx_to_word_key_to_score, score_type_to_word_id_to_scores_list
        text_scores_info = score_type_to_text_idx_to_score
        compo_scores_info = score_type_to_word_id_to_compo_word_to_score, score_type_to_compo_word_to_word_id_to_score

        return word_scores_info, text_scores_info, compo_scores_info

    def _update_all_score(self,
                          tmp_inputs,
                          score_type_to_text_idx_to_word_key_to_score,
                          score_type_to_word_id_to_scores_list,
                          score_type_to_text_idx_to_score,
                          score_type_to_word_id_to_compo_word_to_score,
                          score_type_to_compo_word_to_word_id_to_score):
        """
        Update all scores variables (score_type_to_word_id_to_compo_word_to_score,
        score_type_to_compo_word_to_word_id_to_score, score_type_to_text_idx_to_word_key_to_score,
        score_type_to_text_idx_to_word_key_to_score, score_type_to_word_id_to_scores_list,
        score_type_to_text_idx_to_score) from temporal variable tmp_inputs.

        :param tmp_inputs: tuple of four elements:
            - current_text_idx: current text index
            - tmp_center_key_to_val_list: Dict[center_key] = List[val]; list of values linked to each center_key
            - tmp_context_key_to_val_list: Dict[center_key] = List[val]; list of values linked to each context_key
            - tmp_word_key_to_compo_word: Dict[word_key] =  compo_word; composition of word_id linked to each word_key
        :param score_type_to_text_idx_to_word_key_to_score: Dict[score_type][text_idx][word_key] = word_score
        :param score_type_to_word_id_to_scores_list: Dict[score_type][word_id] = List[word_score]
        :param score_type_to_text_idx_to_score: Dict[score_type][text_idx] = text_score
        :param score_type_to_word_id_to_compo_word_to_score: Dict[score_type][word_id][compo_word] = word_score
        :param score_type_to_compo_word_to_word_id_to_score: Dict[scoe_type][compo_word][word_id] = word_score
        """
        current_text_idx, tmp_cntr_key_to_val_list, tmp_cntxt_key_to_val_list, tmp_word_key_to_compo_word = tmp_inputs

        center_word_score_list = list()
        context_word_score_list = list()
        total_word_score_list = list()

        total_val_list = list()
        for tmp_word_key in tmp_cntr_key_to_val_list:
            total_val_list.extend(tmp_cntr_key_to_val_list[tmp_word_key])

            center_word_score = mean(tmp_cntr_key_to_val_list[tmp_word_key], self.model_settings["use_geo_mean"])
            context_word_score = mean(tmp_cntxt_key_to_val_list[tmp_word_key], self.model_settings["use_geo_mean"])
            total_word_score = mean([center_word_score, context_word_score], self.model_settings["use_geo_mean"])

            compo_key = tmp_word_key_to_compo_word[tmp_word_key]
            if compo_key not in score_type_to_compo_word_to_word_id_to_score["center"]:
                score_type_to_compo_word_to_word_id_to_score["center"][compo_key] = dict()
                score_type_to_compo_word_to_word_id_to_score["context"][compo_key] = dict()
                score_type_to_compo_word_to_word_id_to_score["total"][compo_key] = dict()

            _, tmp_word_id = tmp_word_key
            score_type_to_compo_word_to_word_id_to_score["center"][compo_key][tmp_word_id] = center_word_score
            score_type_to_compo_word_to_word_id_to_score["context"][compo_key][tmp_word_id] = context_word_score
            score_type_to_compo_word_to_word_id_to_score["total"][compo_key][tmp_word_id] = total_word_score

            if tmp_word_id not in score_type_to_word_id_to_scores_list["center"]:
                score_type_to_word_id_to_compo_word_to_score["center"][tmp_word_id] = dict()
                score_type_to_word_id_to_compo_word_to_score["context"][tmp_word_id] = dict()
                score_type_to_word_id_to_compo_word_to_score["total"][tmp_word_id] = dict()

                score_type_to_word_id_to_scores_list["center"][tmp_word_id] = list()
                score_type_to_word_id_to_scores_list["context"][tmp_word_id] = list()
                score_type_to_word_id_to_scores_list["total"][tmp_word_id] = list()

            score_type_to_word_id_to_compo_word_to_score["center"][tmp_word_id][compo_key] = center_word_score
            score_type_to_word_id_to_compo_word_to_score["context"][tmp_word_id][compo_key] = context_word_score
            score_type_to_word_id_to_compo_word_to_score["total"][tmp_word_id][compo_key] = total_word_score

            score_type_to_word_id_to_scores_list["center"][tmp_word_id].append(center_word_score)
            score_type_to_word_id_to_scores_list["context"][tmp_word_id].append(context_word_score)
            score_type_to_word_id_to_scores_list["total"][tmp_word_id].append(total_word_score)

            score_type_to_text_idx_to_word_key_to_score["center"][current_text_idx][tmp_word_key] = center_word_score
            score_type_to_text_idx_to_word_key_to_score["context"][current_text_idx][tmp_word_key] = context_word_score
            score_type_to_text_idx_to_word_key_to_score["total"][current_text_idx][tmp_word_key] = total_word_score

            center_word_score_list.append(center_word_score)
            context_word_score_list.append(context_word_score)
            total_word_score_list.append(total_word_score)

        score_type_to_text_idx_to_score["center"][current_text_idx] = mean(center_word_score_list,
                                                                           self.model_settings["use_geo_mean"])
        score_type_to_text_idx_to_score["context"][current_text_idx] = mean(context_word_score_list,
                                                                            self.model_settings["use_geo_mean"])
        score_type_to_text_idx_to_score["total"][current_text_idx] = mean(total_word_score_list,
                                                                          self.model_settings["use_geo_mean"])
        score_type_to_text_idx_to_score["mean"][current_text_idx] = mean(total_val_list,
                                                                         self.model_settings["use_geo_mean"])

    def _find_decision_frontier(self, score_type_to_word_id_to_scores_list, score_type_to_text_idx_to_score):
        """
        Find the decision frontier for each word_id and the texts.

        :param score_type_to_word_id_to_scores_list: Dict[score_type][word_id] = List[word_score]
        :param score_type_to_text_idx_to_score: Dict[score_type][text_idx] = text_score
        :return (word_id_to_decision_frontier, text_decision_frontier):
            - word_id_to_decision_frontier: Dict[word_id] = decision frontier value
            - text_decision_frontier: text decision frontier value
        """
        word_id_to_decision_frontier = None
        text_decision_frontier = None
        if self.model_settings["trigger_focus"] == "text":
            text_decision_frontier = helpers.utils.get_decision_frontier(
                self.model_settings["trigger_method"],
                list(score_type_to_text_idx_to_score[self.model_settings["trigger_score"]].values()),
                self.model_settings["trigger_sensitivity"],
                self.model_settings["trigger_on"])
        else:
            word_id_to_decision_frontier = dict()
            for word_id, list_score in \
                    score_type_to_word_id_to_scores_list[self.model_settings["trigger_score"]].items():
                word_id_to_decision_frontier[word_id] = helpers.utils.get_decision_frontier(
                    self.model_settings["trigger_method"],
                    list_score,
                    self.model_settings["trigger_sensitivity"],
                    self.model_settings["trigger_on"])
        return word_id_to_decision_frontier, text_decision_frontier

    def _print_confusion_matrix_and_metrics(self, confusion_matrix_val, label_field_exist):
        if label_field_exist:
            self._print_confusion_matrix(confusion_matrix_val)
            self._print_precision_recall_metrics(confusion_matrix_val)

    @staticmethod
    def _update_confusion_matrix_val(find_outlier, label_value, confusion_matrix_val):
        if find_outlier:
            if label_value is 1:
                confusion_matrix_val["TP"] += 1
            else:
                confusion_matrix_val["FP"] += 1
        else:
            if label_value is 1:
                confusion_matrix_val["FN"] += 1
            else:
                confusion_matrix_val["TN"] += 1
        return confusion_matrix_val

    @staticmethod
    def _print_confusion_matrix(confusion_matrix_val):
        """
        Print confusion matrix table on stdout.

        :param confusion_matrix_val:
        """
        table = list()
        title_row = ["", "Positive Prediction", "Negative Prediction"]
        num_tp = confusion_matrix_val["TP"]
        num_fn = confusion_matrix_val["FN"]
        num_fp = confusion_matrix_val["FP"]
        num_tn = confusion_matrix_val["TN"]
        positive_class_row = ["Positive Class", "TP = " + str(num_tp), "FN = " + str(num_fn)]
        negative_class_row = ["Negative Class", "FP = " + str(num_fp), "TN = " + str(num_tn)]
        table.append(title_row)
        table.append(positive_class_row)
        table.append(negative_class_row)
        tabulate_table = tabulate(table, headers="firstrow", tablefmt="grid")
        logging.logger.info("Confusion matrix:\n" + str(tabulate_table))

    @staticmethod
    def _print_precision_recall_metrics(confusion_matrix_val):
        """
        Print precision, recall and F-Measure metrics.

        :param confusion_matrix_val:
        """
        num_tp = confusion_matrix_val["TP"]
        num_fn = confusion_matrix_val["FN"]
        num_fp = confusion_matrix_val["FP"]
        num_pos_class = num_tp + num_fn
        if num_pos_class > 0:
            recall = num_tp / num_pos_class
        else:
            recall = None
        num_pos_pred = num_tp + num_fp
        if num_pos_pred > 0 and num_tp > 0:
            precision = num_tp / num_pos_pred
            f_measure = (2 * precision * recall) / (precision + recall)
        else:
            precision = None
            f_measure = None
        logging.logger.info("Precision: " + str(precision))
        logging.logger.info("Recall: " + str(recall))
        logging.logger.info("F-Score: " + str(f_measure))

    def _processing_outliers_in_batch(self, outliers_in_batch):
        """
        Print on stdout if log_level=INFO, information about the outliers processed in the current batch and save
        outliers (in statistic and) in ES database if not whitelisted.

        :param outliers_in_batch: list of outliers found in current batch.
        """
        if outliers_in_batch:
            unique_summaries_in_batch = len(set(o.outlier_dict["summary"] for o in outliers_in_batch))
            logging.logger.info("BATCH " + str(self.current_batch_num) +
                                " - processing " + "{:,}".format(len(outliers_in_batch)) +
                                " outliers in batch [" + "{:,}".format(unique_summaries_in_batch) +
                                " unique summaries]")

            for outlier in outliers_in_batch:
                self.process_outlier(outlier)
        else:
            logging.logger.info("no outliers processed in batch")


class TextAnalyzer:
    def __init__(self, text_idx, word_list, trigger_focus, trigger_score, trigger_on, size_window):
        self.text_idx = text_idx
        self.word_list = word_list
        self.trigger_focus = trigger_focus
        self.trigger_score = trigger_score
        self.trigger_on = trigger_on
        self.size_window = size_window

        self.find_outlier = False
        self.score_type_to_text_score = dict()
        self.score_type_to_word_score_list = {"center": list(),
                                              "context": list(),
                                              "total": list()}
        self.occur_word_list = list()

        self.text_observations = dict()
        self.text_observations["size_window"] = self.size_window
        self.text_observations["score_type"] = self.trigger_score
        self.word_observations = {"score": dict(),
                                  "decision_frontier": dict(),
                                  "confidence": dict(),
                                  "expected_words": dict(),
                                  "expected_window_words": dict(),
                                  "size_window": self.size_window,
                                  "score_type": self.trigger_score}

    def extract_text_scores(self, score_type_to_text_idx_to_score):
        for score_type, text_idx_to_score in score_type_to_text_idx_to_score.items():
            self.score_type_to_text_score[score_type] = text_idx_to_score[self.text_idx]

    def text_find_outlier(self, text_decision_frontier):
        """
        if self.trigger_focus == "text", check if the text is outlier and update outlier information with
        self.text_observations.

        :param text_decision_frontier: Decision frontier of the current text.
        """
        text_score = self.score_type_to_text_score[self.trigger_score]
        if self.trigger_focus == "text" and self.is_outlier(text_score, text_decision_frontier):
            self.find_outlier = True
            self.text_observations["score"] = text_score
            self.text_observations["decision_frontier"] = text_decision_frontier
            self.text_observations["confidence"] = np.abs(text_decision_frontier - text_score)

    def find_word_id_and_fill_occur_word_row(self, word_idx, word, w2v_model):
        """
        Find word_id from word and fill the list of word occurrence self.occur_word_list

        :param word_idx: word index in text
        :param word: string that represent a word
        :param w2v_model: word2vec model object
        :return: word_id
        """
        voc_counter_dict = dict(w2v_model.voc_counter)
        # if word is not in w2v_model.wor2id it means it's a unknown word.
        if word in w2v_model.word2id:
            word_id = w2v_model.word2id[word]

            self.occur_word_list.append(voc_counter_dict[word])
        else:
            unknown_token = w2v_model.unknown_token
            word_id = w2v_model.word2id[unknown_token]
            # color in blue unknown words
            self.word_list[word_idx] = PRINT_COLORS["blue"] + word + PRINT_COLORS["end"]
            occurrence_text = str(voc_counter_dict[word])
            occurrence_text += "(" + PRINT_COLORS["blue"] + str(w2v_model.num_unknown_occurrence)
            occurrence_text += PRINT_COLORS["end"] + ")"
            self.occur_word_list.append(occurrence_text)
        return word_id

    def fill_score_row_and_find_word_outlier(self,
                                             word_key,
                                             w2v_model,
                                             score_type_to_text_idx_to_word_key_to_score,
                                             word_id_to_decision_frontier,
                                             score_type_to_word_id_to_compo_word_to_score,
                                             score_type_to_compo_word_to_word_id_to_score):
        """
        Format and add the current word score/metric to all word score lists. Word score/metric is formatted in
        two-decimal scientific annotation and colored in red or green if he is considered as an outlier/anomaly or not.
        It also add outlier information to self.word_observations.

        :param word_key: Current word key. word_key = (word_idx, word_id).
        :param w2v_model: word2vec model object.
        :param score_type_to_text_idx_to_word_key_to_score: Dict[score_type][text_idx][word_key] = word_score
        :param word_id_to_decision_frontier: Dict[word_id] = decision frontier value
        :param score_type_to_word_id_to_compo_word_to_score: Dict[score_type][word_id][compo_word] = word_score
        :param score_type_to_compo_word_to_word_id_to_score: Dict[scoe_type][compo_word][word_id] = word_score
        """
        for score_row_type, score_list in self.score_type_to_word_score_list.items():
            word_score = score_type_to_text_idx_to_word_key_to_score[score_row_type][self.text_idx][word_key]

            if self.trigger_focus == "word" and score_row_type == self.trigger_score:
                _, word_id = word_key
                word_decision_frontier = word_id_to_decision_frontier[word_id]
                compo_word_to_score = score_type_to_word_id_to_compo_word_to_score[score_row_type][word_id]
                if self.is_outlier(word_score, word_decision_frontier) and len(compo_word_to_score) > 1:
                    self.find_outlier = True

                    word = w2v_model.id2word[word_id]
                    self.word_observations["score"][word] = word_score
                    self.word_observations["decision_frontier"][word] = word_decision_frontier
                    confidence = np.abs(word_decision_frontier - word_score)
                    self.word_observations["confidence"][word] = confidence

                    # color in red
                    score_list.append(score_to_table_format(word_score, "red"))

                    most_prob_compo = max(compo_word_to_score.items(), key=operator.itemgetter(1))[0]
                    list_word_id_most_prob_compo = re.split("\|", most_prob_compo)[:-1]
                    list_word_most_prob_compo = [w2v_model.id2word[int(word_id_str)] for word_id_str in
                                                 list_word_id_most_prob_compo]
                    self.word_observations["expected_window_words"][word] = str(list_word_most_prob_compo)

                    most_prob_word = self._find_most_prob_word(word_key,
                                                               score_type_to_compo_word_to_word_id_to_score,
                                                               w2v_model)
                    if most_prob_word is not None:
                        self.word_observations["expected_words"][word] = most_prob_word
                else:
                    # color in green
                    score_list.append(score_to_table_format(word_score, "green"))
            else:
                score_list.append(score_to_table_format(word_score))

    def _find_most_prob_word(self, outlier_key, score_type_to_compo_word_to_word_id_to_score, w2v_model):
        """
        Find the most expected words in place of the outlier word represented by outlier_key.

        :param outlier_key: Current word key. word_key = (word_idx, word_id).
        :param score_type_to_compo_word_to_word_id_to_score: Dict[scoe_type][compo_word][word_id] = word_score
        :param w2v_model: word2vec model object.
        :return most_prob_word: String representation of the most likely word in place of the word represented by
        outlier_key.
        """
        most_prob_word = None
        outlier_idx, outlier_id = outlier_key
        compo_word_win = ""
        range_down = max(0, outlier_idx - self.size_window)
        range_up = min(outlier_idx + 1 + self.size_window, len(self.word_list))
        for i in range(range_down, range_up):
            if self.word_list[i] in w2v_model.word2id:
                word_id = w2v_model.word2id[self.word_list[i]]
            else:
                unknown_token = w2v_model.unknown_token
                word_id = w2v_model.word2id[unknown_token]
            if i != outlier_idx:
                compo_word_win += str(word_id) + "|"

        most_prob_word_id = max(
            score_type_to_compo_word_to_word_id_to_score[self.trigger_score][compo_word_win].items(),
            key=operator.itemgetter(1))[0]
        if most_prob_word_id != outlier_id:
            most_prob_word = w2v_model.id2word[most_prob_word_id]
        return most_prob_word

    def print_score_table(self):
        """
        Print on stdout a table with all the metric scores. Color scores in red when they are
        considered as outlier/anomaly.
        """
        table = list()
        table_fields_name = [" "]
        table_fields_name.extend(self.word_list)
        table_fields_name.append("TOTAL")
        table.append(table_fields_name)
        self.occur_word_list.insert(0, "Word batch occurrence")
        self.occur_word_list.append("")
        table.append(self.occur_word_list)

        self.score_type_to_word_score_list["center"].insert(0, "<--Center score-->")
        self.score_type_to_word_score_list["context"].insert(0, "-->Context score<--")
        self.score_type_to_word_score_list["total"].insert(0, "Total score")

        for score_row_type, score_row in self.score_type_to_word_score_list.items():
            text_score = self.score_type_to_text_score[score_row_type]

            if self.trigger_focus == "text" and self.trigger_score == score_row_type:
                score_row.append(score_to_table_format(text_score, "red"))
            else:
                score_row.append(score_to_table_format(text_score))
            table.append(score_row)

        mean_row = [" " for _ in range(len(table_fields_name) - 2)]
        mean_row.insert(0, "MEAN")
        text_score = self.score_type_to_text_score["mean"]
        if self.trigger_score == "mean":
            mean_row.append(score_to_table_format(text_score, "red"))
        else:
            mean_row.append(score_to_table_format(text_score))
        table.append(mean_row)
        tabulate_table = tabulate(table, headers="firstrow", tablefmt="grid")
        logging.logger.info("Outlier info:\n" + str(tabulate_table))

    def print_most_expected_words(self):
        """
        Print on stdout the most expected word in place of the outlier words.
        """
        for word, expected_word in self.word_observations["expected_words"].items():
            logging.logger.info("The most likely word in place of " +
                                PRINT_COLORS["red"] +
                                word +
                                PRINT_COLORS["end"] +
                                " is: " +
                                PRINT_COLORS["green"] + expected_word + PRINT_COLORS["end"])

    def print_most_expected_window_words(self):
        """
        Print on stdout if the most expected composition of words within the window of the outlier words.
        """
        for word, expected_window_words in self.word_observations["expected_window_words"].items():
            logging.logger.info("The most probable context words within the window of " +
                                PRINT_COLORS["red"] + word + PRINT_COLORS["end"] +
                                " are: " + PRINT_COLORS["green"] + expected_window_words +
                                PRINT_COLORS["end"])

    def get_observations(self):
        if self.trigger_focus == "text":
            return self.text_observations
        else:
            return self.word_observations

    def is_outlier(self, score, decision_frontier):
        if self.trigger_on == "low":
            is_outlier = score < decision_frontier
        else:
            is_outlier = score > decision_frontier
        return is_outlier


def mean(iterable, use_geo_mean):
    a = np.array(iterable)
    if use_geo_mean:
        # return geometric mean
        return a.prod() ** (1.0 / len(a))
    else:
        # return arithmetic mean
        return np.mean(a)


def score_to_table_format(score, color=None):
    # Scientific notation with two number after the "."
    output = "{:.2e}".format(score)
    if color:
        output = PRINT_COLORS[color] + output + PRINT_COLORS["end"]
    return output
