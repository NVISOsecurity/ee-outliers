import helpers.utils

from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
import analyzers.ml_models.word2vec as word2vec
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import re
from configparser import SectionProxy
from tabulate import tabulate
import numpy as np
from scipy import spatial
import operator

from typing import List, Tuple, Dict, Optional

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
        # TODO something odd about this parameter!
        self.model_settings["process_documents_chronologically"] = self.config_section.getboolean(
            "process_documents_chronologically", True)

        self.model_settings["target"] = self.config_section["target"].replace(' ', '').split(",")

        self.model_settings["aggregator"] = self.config_section["aggregator"].replace(' ', '').split(",")

        self.model_settings["use_derived_fields"] = self.config_section.getboolean("use_derived_fields")

        # word2vec_batch_eval_size parameter
        self.model_settings["word2vec_batch_eval_size"] = self._extract_parameter("word2vec_batch_eval_size",
                                                                                  param_type="int")

        # min_target_buckets parameter
        self.model_settings["min_target_buckets"] = self._extract_parameter("min_target_buckets", param_type="int")

        # drop_duplicates parameter
        self.model_settings["drop_duplicates"] = self._extract_parameter("drop_duplicates",
                                                                         param_type="boolean")

        # use_prob_model parameter
        self.model_settings["use_prob_model"] = self._extract_parameter("use_prob_model",
                                                                        param_type="boolean")

        # separators parameter
        self.model_settings["separators"] = self._extract_parameter("separators")
        self.model_settings["separators"] = self.model_settings["separators"].strip('"')

        # size_window parameter
        self.model_settings["size_window"] = self._extract_parameter("size_window", param_type="int")

        # min_uniq_word_occurrence parameter
        self.model_settings["min_uniq_word_occurrence"] = self._extract_parameter("min_uniq_word_occurrence",
                                                                                  param_type="int")
        
        # output_prob parameters
        self.model_settings["output_prob"] = self._extract_parameter("output_prob", param_type="boolean")
        # use_geo_mean parameter
        self.model_settings["use_geo_mean"] = self.model_settings["output_prob"]

        # num_epochs parameter
        self.model_settings["num_epochs"] = self._extract_parameter("num_epochs", param_type="int")

        # learning_rate parameter
        self.model_settings["learning_rate"] = self._extract_parameter("learning_rate", param_type="float")

        # embedding_size parameter
        self.model_settings["embedding_size"] = self._extract_parameter("embedding_size", param_type="int")
        
        # seed parameter
        self.model_settings["seed"] = self._extract_parameter("seed", param_type="int")

        # print_score_table parameter TODO

        # tensorboard parameter TODO
        self.model_settings["tensorboard"] = self._extract_parameter("tensorboard", param_type="boolean")

        # trigger_focus parameter
        self.model_settings["trigger_focus"] = self._extract_parameter("trigger_focus")
        if self.model_settings["trigger_focus"] not in {"word", "text"}:
            raise ValueError("Unexpected outlier trigger focus " + str(self.model_settings["trigger_focus"]))

        # trigger_score parameter
        self.model_settings["trigger_score"] = self._extract_parameter("trigger_score")
        if self.model_settings["trigger_score"] not in {"center", "context", "total", "mean"}:
            raise ValueError("Unexpected outlier trigger score " + str(self.model_settings["trigger_score"]))
        if self.model_settings["trigger_score"] == "mean" and self.model_settings["trigger_focus"] == "word":
            raise ValueError("trigger_focus=word is not compatible with trigger_score=mean")

        # trigger_on parameter
        self.model_settings["trigger_on"] = self._extract_parameter("trigger_on")
        if self.model_settings["trigger_on"] not in {"high", "low"}:
            raise ValueError("Unexpected outlier trigger condition " + str(self.model_settings["trigger_on"]))

        # trigger_method parameter
        self.model_settings["trigger_method"] = self._extract_parameter("trigger_method")
        if self.model_settings["trigger_method"] not in {"percentile", "pct_of_max_value", "pct_of_median_value",
                                                         "pct_of_avg_value", "mad", "madpos", "stdev", "float",
                                                         "coeff_of_variation", "z_score"}:
            raise ValueError("Unexpected outlier trigger method " + str(self.model_settings["trigger_method"]))

        # trigger_sensitivity parameter
        self.model_settings["trigger_sensitivity"] = self._extract_parameter("trigger_sensitivity", param_type="float")

        self.current_batch_num = 0

    def _extract_parameter(self, param_name, param_type=None):
        param_value = None
        if param_type is None or param_type == "string":
            param_value = self.config_section.get(param_name)
            if param_value is None:
                param_value = settings.config.get("word2vec", param_name)
        elif param_type == "int":
            param_value = self.config_section.getint(param_name)
            if param_value is None:
                param_value = settings.config.getint("word2vec", param_name)
        elif param_type == "float":
            self.model_settings[param_name] = self.config_section.getfloat(param_name)
            if param_value is None:
                param_value = settings.config.getfloat("word2vec", param_name)
        elif param_type == "boolean":
            param_value = self.config_section.getboolean(param_name)
            if param_value is None:
                param_value = settings.config.getboolean("word2vec", param_name, fallback=False)
        else:
            raise ValueError("Unexpected outlier trigger focus " + str(self.model_settings["trigger_focus"]))

        return param_value

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
                        # TODO put better comment
                        logging.logger.warning("Unable to fill any of the aggregator buckets for this batch.")

                    # Processing the outliers found
                    self._processing_outliers_in_batch(outliers_in_batch)

                    total_docs_in_batch -= total_docs_removed
                    num_targets_not_processed = total_docs_in_batch
                    self.current_batch_num += 1

                logging.tick()

    # TODO function from terms.py --> should transfer it to utils?
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
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence, sep_str='')

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
                agrr_outliers = self._evaluate_aggr_for_outliers(aggr_elem, aggr_key)

                outliers_in_batch.extend(agrr_outliers)

                # Remove aggr_elem from batch
                aggr_elem["targets"] = list()
                aggr_elem["raw_docs"] = list()
                total_targets_removed += num_targets

        return outliers_in_batch, total_targets_removed

    def _evaluate_aggr_for_outliers(self, aggr_elem, aggr_key):
        """
        Evaluates if the aggregation contains outliers.
        Creates a word2vec model, prepares the model by creating the vocabulary, trains the model, evaluates the events
        and finally finds the outliers.

        :param aggr_elem: aggregator element
        :param aggr_key: aggregator key
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
            loss_values = w2v_model.train_model(aggr_elem["targets"])

            # TODO remove tensorboard?
            if self.model_settings["tensorboard"]:
                matplotlib_to_tensorboard(loss_values, aggr_key, self.current_batch_num)
                # list_to_tensorboard(loss_values, aggr_key, self.current_batch_num)

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
        num_tp = 0  # Number of True Positive
        num_tn = 0  # Number of False Positive
        num_fp = 0  # Number of False Positive
        num_fn = 0  # Number of False Negative
        print_precision_recall_metric = False
        outliers = list()

        # Find all type of metrics concerning the text and the words
        word_scores_info, text_scores_info, compo_scores_info = self._find_all_scores(model_eval_outputs)
        score_type_to_text_idx_to_word_key_to_score, score_type_to_word_id_to_scores_list = word_scores_info
        score_type_to_text_idx_to_score = text_scores_info
        score_type_to_word_id_to_compo_word_to_score, score_type_to_compo_word_to_word_id_to_score = compo_scores_info

        # Compute the decision frontier
        word_id_to_decision_frontier, text_decision_frontier = self._find_decision_frontier(
            score_type_to_word_id_to_scores_list, score_type_to_text_idx_to_score)

        score_type_to_word_id_to_mean_stdev, score_type_to_text_mean_stdev = self._find_mean_and_stdev(
            score_type_to_word_id_to_scores_list, score_type_to_text_idx_to_score)

        for text_idx, text_str in enumerate(aggr_elem["targets"]):

            list_info_outliers = list()
            find_outlier = False

            # tokenize the text
            word_list = re.split(self.model_settings["separators"], text_str)

            # rows for printing the table
            occur_word_list = list()
            score_type_to_word_score_list = {"center": list(),
                                             "context": list(),
                                             "total": list()}

            for word_idx, word in enumerate(word_list):
                # find word id and compute word occurrence
                word_id = self._find_word_id_and_fill_occur_word_row(word, occur_word_list, w2v_model, word_list,
                                                                     word_idx)

                # find outliers if "trigger_focus=text"
                text_score = score_type_to_text_idx_to_score[self.model_settings["trigger_score"]][text_idx]
                if self.model_settings["trigger_focus"] == "text" and text_score < text_decision_frontier:
                    find_outlier = True

                # fill score rows for table and find outliers if "trigger_focus=word"
                word_key = (word_idx, word_id)
                is_outlier, list_word_id_most_prob_compo = self._fill_score_row_and_find_outlier(
                    text_idx,
                    word_key,
                    score_type_to_word_score_list,
                    score_type_to_text_idx_to_word_key_to_score,
                    word_id_to_decision_frontier,
                    score_type_to_word_id_to_compo_word_to_score,
                    score_type_to_word_id_to_mean_stdev)
                if is_outlier:
                    find_outlier = True
                    list_info_outliers.append((word_idx, word_id, list_word_id_most_prob_compo))

            raw_doc = aggr_elem["raw_docs"][text_idx]
            fields = es.extract_fields_from_document(raw_doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
            if "label" in fields:
                print_precision_recall_metric = True
                if find_outlier:
                    if fields["label"] is 0:
                        num_tp += 1
                    else:
                        num_fp += 1
                else:
                    if fields["label"] is 0:
                        num_fn += 1
                    else:
                        num_tn += 1

            if find_outlier:
                score_info = score_type_to_word_score_list, occur_word_list
                self._print_score_table(word_list,
                                        score_info,
                                        score_type_to_text_idx_to_score,
                                        text_idx,
                                        score_type_to_text_mean_stdev)

                observations = dict()
                if self.model_settings["trigger_focus"] == "text":
                    text_idx_to_score = score_type_to_text_idx_to_score[self.model_settings["trigger_score"]]
                    observations["score"] = text_idx_to_score[text_idx]
                    observations["decision_frontier"] = text_decision_frontier
                    observations["confidence"] = np.abs(text_decision_frontier - text_idx_to_score[text_idx])
                else:
                    observations["score"] = dict()
                    observations["decision_frontier"] = dict()
                    observations["confidence"] = dict()
                    observations["expected_word"] = dict()
                    observations["expected_window_words"] = dict()
                    for outlier_idx, outlier_id, list_word_id_most_prob_compo in list_info_outliers:
                        most_prob_word = self._print_most_prob_word(word_list,
                                                                    outlier_idx,
                                                                    outlier_id,
                                                                    score_type_to_compo_word_to_word_id_to_score,
                                                                    w2v_model)
                        expected_window_words = self._print_most_compo_word(list_word_id_most_prob_compo,
                                                                            outlier_id,
                                                                            w2v_model)
                        text_idx_to_word_key_to_score = score_type_to_text_idx_to_word_key_to_score[
                            self.model_settings["trigger_score"]]
                        word_key_to_score = text_idx_to_word_key_to_score[text_idx]
                        outlier_key = (outlier_idx, outlier_id)

                        outlier_word = w2v_model.id2word[outlier_id]
                        observations["expected_word"][outlier_word] = most_prob_word
                        observations["expected_window_words"][outlier_word] = expected_window_words
                        observations["score"][outlier_word] = word_key_to_score[outlier_key]
                        observations["decision_frontier"][outlier_word] = word_id_to_decision_frontier[outlier_id]
                        confidence = np.abs(word_key_to_score[outlier_key] - word_id_to_decision_frontier[outlier_id])
                        observations["confidence"][outlier_word] = confidence

                # raw_doc = aggr_elem["raw_docs"][text_idx]
                # fields = es.extract_fields_from_document(
                #     raw_doc,
                #     extract_derived_fields=self.model_settings["use_derived_fields"])

                outlier = self.create_outlier(fields,
                                              raw_doc,
                                              extra_outlier_information=observations)
                outliers.append(outlier)
        if print_precision_recall_metric:
            self._print_confusion_matrix(num_tp, num_fn, num_fp, num_tn)
            self._print_precision_recall_metric(num_tp, num_fn, num_fp, num_tn)

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
        current_text_idx, tmp_center_key_to_val_list, tmp_context_key_to_val_list, tmp_word_key_to_compo_word = tmp_inputs

        center_word_score_list = list()
        context_word_score_list = list()
        total_word_score_list = list()

        total_val_list = list()
        for tmp_word_key in tmp_center_key_to_val_list:
            total_val_list.extend(tmp_center_key_to_val_list[tmp_word_key])

            center_word_score = mean(tmp_center_key_to_val_list[tmp_word_key], self.model_settings["use_geo_mean"])
            context_word_score = mean(tmp_context_key_to_val_list[tmp_word_key], self.model_settings["use_geo_mean"])
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

    def _find_mean_and_stdev(self,  score_type_to_word_id_to_scores_list, score_type_to_text_idx_to_score):
        """
        Find (mean, stdev) tuple for each word_id and all the text scores.

        :param score_type_to_word_id_to_scores_list: Dict[score_type][word_id] = List[word_score]
        :param score_type_to_text_idx_to_score: Dict[score_type][text_idx] = text_score
        :return (score_type_to_word_id_to_mean_stdev, score_type_to_text_mean_stdev):
            - score_type_to_word_id_to_mean_stdev: Dict[score_type][word_id] = (mean, stdev)
            - score_type_to_text_mean_stdev: Dict[score_type] = (mean, stdev)
        """
        score_type_to_word_id_to_mean_stdev = None
        score_type_to_text_mean_stdev = None
        if self.model_settings["trigger_method"] == "z_score":
            score_type_to_word_id_to_mean_stdev = dict()
            for score_type in score_type_to_word_id_to_scores_list:
                score_type_to_word_id_to_mean_stdev[score_type] = dict()
                for word_id, list_score in score_type_to_word_id_to_scores_list[score_type].items():
                    word_mean_stdev = helpers.utils.get_mean_and_stdev(list_score)
                    score_type_to_word_id_to_mean_stdev[score_type][word_id] = word_mean_stdev

            score_type_to_text_mean_stdev = dict()
            for score_type in score_type_to_text_idx_to_score:
                text_mean_stdev = helpers.utils.get_mean_and_stdev(
                    list(score_type_to_text_idx_to_score[score_type].values()))
                score_type_to_text_mean_stdev[score_type] = text_mean_stdev

        return score_type_to_word_id_to_mean_stdev, score_type_to_text_mean_stdev

    def _find_word_id_and_fill_occur_word_row(self, word, occur_word_list, w2v_model, word_list, word_idx):
        """
        Find word_id from word and fill the list of word occurrence occur_word_row

        :param word: string that represent a word
        :param occur_word_list: list of occurrence of each word
        :param w2v_model: word2vec model object
        :param word_list: list of words in current text
        :param word_idx: word index in text
        :return: word_id
        """
        voc_counter_dict = dict(w2v_model.voc_counter)
        # if word is not in w2v_model.wor2id it means it's a unknown word.
        if word in w2v_model.word2id:
            word_id = w2v_model.word2id[word]

            occur_word_list.append(voc_counter_dict[word])
        else:
            unknown_token = w2v_model.unknown_token
            word_id = w2v_model.word2id[unknown_token]
            # color in blue unknown words
            word_list[word_idx] = PRINT_COLORS["blue"] + word + PRINT_COLORS["end"]
            occurrence_text = str(voc_counter_dict[word]) + \
                              "(" + PRINT_COLORS["blue"] + \
                              str(w2v_model.num_unknown_occurrence) + \
                              PRINT_COLORS["end"] + ")"
            occur_word_list.append(occurrence_text)
        return word_id

    def _fill_score_row_and_find_outlier(self,
                                         text_idx,
                                         word_key,
                                         score_type_to_word_score_list,
                                         score_type_to_text_idx_to_word_key_to_score,
                                         word_id_to_decision_frontier,
                                         score_type_to_word_id_to_compo_word_to_score,
                                         score_type_to_word_id_to_mean_stdev):
        """
        Format and add the current word score/metric to all word score lists. Word score/metric is formatted in
         two-decimal scientific annotation and colored in red or green if he is considered as an outlier/anomaly or not.

        :param text_idx: Current text index
        :param word_key: Current word key. word_key = (word_idx, word_id).
        :param score_type_to_word_score_list: Dict[score_type] = List[word_score]; where word_score is formatted into
        String  two-decimal scientific annotation. Depending on the trigger_focus, the trigger_score parameters and if
         the word score is considered as anomaly, word_score will be colored in green or red.
        :param score_type_to_text_idx_to_word_key_to_score: Dict[score_type][text_idx][word_key] = word_score
        :param word_id_to_decision_frontier: Dict[word_id] = decision frontier value
        :param score_type_to_word_id_to_compo_word_to_score: Dict[score_type][word_id][compo_word] = word_score
        :param score_type_to_word_id_to_mean_stdev: TODO
        :return: (is_outlier, list_word_id_most_prob_compo):
            - is_outlier: Boolean set to true if at least one of the word score has a score smaller than the
            word decision frontier.
            - list_word_id_most_prob_compo: list of word_id representing the most probable combination of word within
            the window of the current word represented by word_key.
        """
        list_word_id_most_prob_compo = list()
        is_outlier = False
        _, word_id = word_key
        for score_row_type, score_list in score_type_to_word_score_list.items():
            elem_row_score = score_type_to_text_idx_to_word_key_to_score[score_row_type][text_idx][word_key]
            decision_frontier = word_id_to_decision_frontier[word_id]
            if self.model_settings["trigger_method"] == "z_score":
                word_mean_stdev = score_type_to_word_id_to_mean_stdev[score_row_type][word_id]
                elem_row_score = helpers.utils.get_z_score(elem_row_score, word_mean_stdev)
                decision_frontier = helpers.utils.get_z_score(decision_frontier, word_mean_stdev)

            if self.model_settings["trigger_focus"] == "word" and score_row_type == self.model_settings[
                "trigger_score"]:
                if elem_row_score < decision_frontier:
                    is_outlier = True
                    # color in red
                    score_list.append(score_to_table_format(elem_row_score, "red"))

                    most_prob_compo = max(score_type_to_word_id_to_compo_word_to_score[score_row_type][word_id].items(),
                                          key=operator.itemgetter(1))[0]
                    list_word_id_most_prob_compo = re.split("\|", most_prob_compo)[:-1]
                else:
                    # color in green
                    score_list.append(score_to_table_format(elem_row_score, "green"))
            else:
                score_list.append(score_to_table_format(elem_row_score))
        return is_outlier, list_word_id_most_prob_compo

    def _print_score_table(self,
                           word_list,
                           score_info,
                           score_type_to_text_idx_to_score,
                           current_text_idx,
                           score_type_to_text_mean_stdev):
        """
        Print on stdout if log_level=DEBUG a table with all the metric scores. Color scores in red when they are
        considered as outlier/anomaly.

        :param word_list: list of words of the current text
        :param score_info: (score_type_to_word_score_list, occur_word_list)
            - score_type_to_word_score_list: Dict[score_type] = List[word_score]; where word_score is formatted into
        String  two-decimal scientific annotation. Depending on the trigger_focus, the trigger_score parameters and if
         the word score is considered as anomaly, word_score will be colored in green or red.
            - occur_word_list: list of occurrence of each word.
        :param score_type_to_text_idx_to_score: Dict[score_type][text_idx] = text_score
        :param current_text_idx: Current text index
        :param score_type_to_text_mean_stdev: Dict[score_type] = (mean, stdev)
        """
        # words, score_rows, score_type_to_text_idx_to_score, text_idx
        score_type_to_word_score_list, occur_word_list = score_info
        table = list()
        table_fields_name = [" "]
        table_fields_name.extend(word_list)
        table_fields_name.append("TOTAL")
        table.append(table_fields_name)
        occur_word_list.insert(0, "Word batch occurrence")
        occur_word_list.append("")
        table.append(occur_word_list)

        score_type_to_word_score_list["center"].insert(0, "<--Center score-->")
        score_type_to_word_score_list["context"].insert(0, "-->Context score<--")
        score_type_to_word_score_list["total"].insert(0, "Total score")

        for score_row_type, score_row in score_type_to_word_score_list.items():
            text_score = score_type_to_text_idx_to_score[score_row_type][current_text_idx]
            if self.model_settings["trigger_method"] == "z_score":
                text_score = helpers.utils.get_z_score(text_score, score_type_to_text_mean_stdev[score_row_type])

            if self.model_settings["trigger_focus"] == "text" and \
                    self.model_settings["trigger_score"] == score_row_type:
                score_row.append(score_to_table_format(text_score, "red"))
            else:
                score_row.append(score_to_table_format(text_score))
            table.append(score_row)

        mean_row = [" " for _ in range(len(table_fields_name) - 2)]
        mean_row.insert(0, "MEAN")
        text_score = score_type_to_text_idx_to_score["mean"][current_text_idx]
        if self.model_settings["trigger_method"] == "z_score":
            text_score = helpers.utils.get_z_score(text_score, score_type_to_text_mean_stdev["mean"])
        if self.model_settings["trigger_score"] == "mean":
            mean_row.append(score_to_table_format(text_score, "red"))
        else:
            mean_row.append(score_to_table_format(text_score))
        table.append(mean_row)
        tabulate_table = tabulate(table, headers="firstrow", tablefmt="fancy_grid")
        logging.logger.debug("Outlier info:\n" + str(tabulate_table))

    def _print_most_prob_word(self,
                              word_list,
                              outlier_idx,
                              outlier_id,
                              score_type_to_compo_word_to_word_id_to_score,
                              w2v_model):
        """
        Print on stdout if log_level=DEBUG the most expected words in place of the current outlier word.

        :param word_list: list of words in the current text.
        :param outlier_idx: outlier word index in current text.
        :param outlier_id: outlier word id
        :param score_type_to_compo_word_to_word_id_to_score: Dict[scoe_type][compo_word][word_id] = word_score
        :param w2v_model: word2vec model object
        :return: the most likely word in place of outlier_id
        """
        most_prob_word = None
        compo_word_win = ""
        range_down = max(0, outlier_idx - self.model_settings["size_window"])
        range_up = min(outlier_idx + 1 + self.model_settings["size_window"], len(word_list))
        for i in range(range_down, range_up):
            if word_list[i] in w2v_model.word2id:
                word_id = w2v_model.word2id[word_list[i]]
            else:
                unknown_token = w2v_model.unknown_token
                word_id = w2v_model.word2id[unknown_token]
            if i != outlier_idx:
                compo_word_win += str(word_id) + "|"

        most_prob_word_id = max(
            score_type_to_compo_word_to_word_id_to_score[self.model_settings["trigger_score"]][compo_word_win].items(),
            key=operator.itemgetter(1))[0]
        if most_prob_word_id != outlier_id:
            most_prob_word = w2v_model.id2word[int(most_prob_word_id)]
            logging.logger.debug("The most likely word in place of " +
                                 PRINT_COLORS["red"] +
                                 w2v_model.id2word[outlier_id] +
                                 PRINT_COLORS["end"] +
                                 " is: " +
                                 PRINT_COLORS["green"] + most_prob_word + PRINT_COLORS["end"])

        return most_prob_word

    @staticmethod
    def _print_most_compo_word(list_word_id_most_prob_compo, outlier_id, w2v_model):
        """
        Print on stdout if log_level=DEBUG the most expected composition of word within the window of the outlier
        word represented by outlier_id.

        :param list_word_id_most_prob_compo: list of word_id representing the most probable combination of word within
         the window of the current word represented by word_key.
        :param outlier_id: current outlier word id
        :param w2v_model: word2vec model object
        :return: string format of the most probable composition of words within the window or outlier_id.
        """
        list_word_most_prob_compo = [w2v_model.id2word[int(word_id_str)] for word_id_str in
                                     list_word_id_most_prob_compo]
        logging.logger.debug("The most probable context words within the window of " +
                             PRINT_COLORS["red"] + w2v_model.id2word[outlier_id] + PRINT_COLORS["end"] +
                             " are: " + PRINT_COLORS["green"] + str(list_word_most_prob_compo) +
                             PRINT_COLORS["end"])
        return str(list_word_most_prob_compo)

    @staticmethod
    def _print_confusion_matrix(num_tp, num_fn, num_fp, num_tn):
        table = list()
        title_row = ["", "Positive Prediciton", "Negative Prediction"]
        positive_class_row = ["Positive Class", "TP = " + str(num_tp), "FN = " + str(num_fn)]
        negative_class_row = ["Negative Class", "FP = " + str(num_fp), "TN = " + str(num_tn)]
        table.append(title_row)
        table.append(positive_class_row)
        table.append(negative_class_row)
        tabulate_table = tabulate(table, headers="firstrow", tablefmt="fancy_grid")
        logging.logger.debug("Confusion matrix:\n" + str(tabulate_table))

    @staticmethod
    def _print_precision_recall_metric(num_tp, num_fn, num_fp, num_tn):
        num_pos_class = num_tp + num_fn
        recall = num_tp / num_pos_class
        num_pos_pred = num_tp + num_fp
        if num_pos_pred > 0 and num_tp > 0:
            precision = num_tp / num_pos_pred
            f_measure = (2 * precision * recall) / (precision + recall)
        else:
            precision = None
            f_measure = None
        logging.logger.debug("Precision: " + str(precision))
        logging.logger.debug("Recall: " + str(recall))
        logging.logger.debug("F-Measure: " + str(f_measure))

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


def matplotlib_to_tensorboard(list_elem, aggr_key, current_batch_num):
    """
    Plot graph

    :param list_elem: list of float
    :param aggr_key: Aggregator key name
    :param current_batch_num: Current batch number
    """
    writer = SummaryWriter('TensorBoard')
    tag = "BATCH " + str(current_batch_num) + " - Aggregator: " + aggr_key
    figure = plt.figure()
    plt.plot(list_elem)
    plt.title(tag)
    writer.add_figure(tag=tag, figure=figure)
    writer.close()


def list_to_tensorboard(list_elem, aggr_key, current_batch_num):
    writer = SummaryWriter('TensorBoard')
    tag = "BATCH " + str(current_batch_num) + " - Aggregator: " + aggr_key
    for i, elem in enumerate(list_elem):
        writer.add_scalar(tag, elem, i)
    writer.close()


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


def w2v_and_prob_comparison(w2v_center_context_text_prob_list, prob_center_context_text_prob_list):
    _, _, _, prob_w2v = zip(*w2v_center_context_text_prob_list)
    _, _, _, prob_prob = zip(*prob_center_context_text_prob_list)
    cos_sim = 1 - spatial.distance.cosine(prob_w2v, prob_prob)

    logging.logger.debug("Cos similarity:" + str(cos_sim))
