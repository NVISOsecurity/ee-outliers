import helpers.utils

from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
import analyzers.ml_models.word2vec_dev as word2vec
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


class Word2VecDevAnalyzer(Analyzer):

    def __init__(self, model_name: str, config_section: SectionProxy):
        super(Word2VecDevAnalyzer, self).__init__("word2vec_dev", model_name, config_section)

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        # TODO train_model unnecessary because we train and evaluate it anyway
        self.model_settings["train_model"] = self.config_section.getboolean("train_model")

        self.model_settings["target_fields"] = self.config_section["target_fields"].replace(' ', '').split(",")

        self.model_settings["aggr_fields"] = self.config_section["aggregator_fields"].replace(' ', '').split(",")
        self.model_settings["min_target_buckets"] = self.config_section.getint("min_target_buckets")

        self.model_settings["separators"] = re.sub("\'", '', self.config_section["separators"])

        self.model_settings["size_window"] = int(self.config_section["size_window"])
        self.model_settings["drop_duplicates"] = self.config_section.getboolean("drop_duplicates")
        self.model_settings["use_derived_fields"] = self.config_section.getboolean("use_derived_fields")

        self.model_settings["tensorboard"] = self.config_section.getboolean("tensorboard")

        self.model_settings["num_epochs"] = self.config_section.getint("num_epochs")
        self.model_settings["learning_rate"] = self.config_section.getfloat("learning_rate")
        self.model_settings["embedding_size"] = self.config_section.getint("embedding_size")
        self.model_settings["seed"] = self.config_section.getint("seed")
        self.model_settings["min_uniq_word_occurrence"] = self.config_section.getint("min_uniq_word_occurrence")

        self.model_settings["use_geo_mean"] = self.config_section.getboolean("use_geo_mean")
        self.model_settings["use_prob_model"] = self.config_section.getboolean("use_prob_model")

        self.model_settings["trigger_focus"] = self.config_section["trigger_focus"]
        self.model_settings["trigger_score"] = self.config_section["trigger_score"]
        self.model_settings["trigger_on"] = self.config_section["trigger_on"]
        self.model_settings["trigger_method"] = self.config_section["trigger_method"]
        self.model_settings["trigger_sensitivity"] = self.config_section.getfloat("trigger_sensitivity")
        if self.model_settings["trigger_focus"] not in {"word", "text"}:
            raise ValueError("Unexpected outlier trigger focus " + str(self.model_settings["trigger_focus"]))
        if self.model_settings["trigger_score"] not in {"center", "context", "total", "mean"}:
            raise ValueError("Unexpected outlier trigger score " + str(self.model_settings["trigger_score"]))
        if self.model_settings["trigger_score"] == "mean" and self.model_settings["trigger_focus"] == "word":
            raise ValueError("trigger_focus=word is not compatible with trigger_score=mean")
        if self.model_settings["trigger_on"] not in {"high", "low"}:
            raise ValueError("Unexpected outlier trigger condition " + str(self.model_settings["trigger_on"]))

        if self.model_settings["trigger_method"] not in {"percentile", "pct_of_max_value", "pct_of_median_value",
                                                         "pct_of_avg_value", "mad", "madpos", "stdev", "float",
                                                         "coeff_of_variation"}:
            raise ValueError("Unexpected outlier trigger method " + str(self.model_settings["trigger_method"]))

        self.batch_train_size = settings.config.getint("machine_learning", "word2vec_batch_train_size")
        self.current_batch_num = 0

    def evaluate_model(self):

        target_fields = self.model_settings["target_fields"]
        aggr_fields = self.model_settings["aggr_fields"]
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
                        batch=batch,
                        target_sentences=target_sentences,
                        aggr_sentences=aggr_sentences,
                        doc=doc)
                    total_docs_in_batch += num_doc_add
                    total_duplicates += num_duplicates

                if total_docs_in_batch >= self.batch_train_size or logging.current_step + 1 == self.total_events:
                    if total_duplicates > 0:
                        logging.logger.info("Drop_duplicates activated: %i/%i events have been removed",
                                            total_duplicates,
                                            logging.current_step + 1)
                    # Display log message
                    self._display_batch_log_message(total_docs_in_batch, num_targets_not_processed)

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

            if num_targets_not_processed > 0:
                log_message = "" + "{:,}".format(num_targets_not_processed) + " sentences not processed in last batch"
                logging.logger.info(log_message)

    # TODO function from terms.py --> should transfer it to utils?
    def _extract_target_and_aggr_sentences(self,
                                           doc: dict,
                                           target_fields: List[str],
                                           aggr_fields: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Extract target and aggregator sentence from a document

        :param doc: document where data need to be extract
        :param target: target key name
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
                                               batch: dict,
                                               target_sentences: List[List[str]],
                                               aggr_sentences: List[List[str]],
                                               doc: dict) -> Tuple[dict, int, int]:
        """
        Add a document to the current batch.
        If drop_duplicates is activated and the document already appear in batch, it is not added to the batch.

        :param current_batch: existing batch (where doc need to be saved)
        :param target_sentences: list of targets
        :param aggregator_sentences: list of aggregator
        :param doc: document that need to be added
        :return: batch with document inside, number of documents added to batch, number of document not added to batch
        """
        for target_sentence in target_sentences:
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence, sep_str='')

            for aggregator_sentence in aggr_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)

                if flattened_aggregator_sentence not in batch.keys():
                    batch[flattened_aggregator_sentence] = defaultdict(list)
                if self.model_settings["drop_duplicates"] and \
                        flattened_target_sentence in batch[flattened_aggregator_sentence]["targets"]:
                    return batch, 0, len(target_sentences) * len(aggr_sentences)

                batch[flattened_aggregator_sentence]["targets"].append(flattened_target_sentence)
                batch[flattened_aggregator_sentence]["raw_docs"].append(doc)

        return batch, len(target_sentences) * len(aggr_sentences), 0

    def _display_batch_log_message(self, total_targets_in_batch, num_targets_not_processed):
        log_message = "BATCH " + str(self.current_batch_num)
        log_message += " - evaluating batch of " + "{:,}".format(total_targets_in_batch) + " sentences "
        if num_targets_not_processed > 0:
            log_message += "(with " + "{:,}".format(num_targets_not_processed) + " sentences from last batch) "
        log_message += "[" + "{:,}".format(logging.current_step + 1) + " events processed]"
        logging.logger.info(log_message)

    def _evaluate_batch_for_outliers(self, batch):
        outliers = list()
        total_targets_removed = 0
        for aggr_key, aggr_elem in batch.items():
            num_targets = len(aggr_elem["targets"])
            if num_targets >= self.model_settings["min_target_buckets"]:
                agrr_outliers = self._evaluate_aggr_for_outliers(aggr_elem, aggr_key)
                outliers.extend(agrr_outliers)
                total_targets_removed += num_targets

        return outliers, total_targets_removed

    def _evaluate_aggr_for_outliers(self, aggr_elem, aggr_key):
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
            center_context_text_prob_list = w2v_model.prob_model(aggr_elem["targets"],
                                                                 self.model_settings["use_geo_mean"])
        else:
            # Train word2vec model
            loss_values = w2v_model.train_model(aggr_elem["targets"])

            if self.model_settings["tensorboard"]:
                matplotlib_to_tensorboard(loss_values, aggr_key, self.current_batch_num)
                # list_to_tensorboard(loss_values, aggr_key, self.current_batch_num)

            # Eval word2vec model
            center_context_text_prob_list = w2v_model.eval_model(aggr_elem["targets"],
                                                                 self.model_settings["use_geo_mean"])

        # Find outliers from the word2vec model outputs
        outliers = self._find_outliers(center_context_text_prob_list, aggr_elem, w2v_model)

        # Remove aggr_elem from batch
        aggr_elem["targets"] = list()
        aggr_elem["raw_docs"] = list()

        return outliers

    def _find_outliers(self, center_context_text_prob_list, aggr_elem, w2v_model):
        outliers = list()

        voc_counter_dict = dict(w2v_model.voc_counter)
        text_word_score_dict, word_score_dict, text_score_dict, word_idx_to_compo_word_in_win_score, compo_word_win_to_word_idx_context_score = self._find_all_scores(
            center_context_text_prob_list)

        word_decision_frontier, text_decision_frontier = self._find_decision_frontier(word_score_dict, text_score_dict)

        for text_idx, text_str in enumerate(aggr_elem["targets"]):
            dict_outlier_idx_to_list_word_idx_most_prob_compo = dict()
            find_outlier = False

            # tokenize the text
            words = re.split(self.model_settings["separators"], text_str)

            # rows for printing the table
            occur_word_row = list()
            score_rows = {"center": list(),
                          "context": list(),
                          "total": list()}

            for w, word in enumerate(words):
                # find word index and compute word occurrence
                if word in w2v_model.word2idx:
                    word_idx = w2v_model.word2idx[word]

                    occur_word_row.append(voc_counter_dict[word])
                else:
                    unknown_token = w2v_model.unknown_token
                    word_idx = w2v_model.word2idx[unknown_token]

                    words[w] = PRINT_COLORS["blue"] + word + PRINT_COLORS["end"]
                    occurrence_text = str(voc_counter_dict[word]) + \
                                      "(" + PRINT_COLORS["blue"] + \
                                      str(w2v_model.num_unknown_occurrence) + \
                                      PRINT_COLORS["end"] + ")"
                    occur_word_row.append(occurrence_text)

                if self.model_settings["trigger_focus"] == "text" and \
                        text_score_dict[self.model_settings["trigger_score"]][text_idx] < text_decision_frontier:
                    find_outlier = True
                # TODO check the think with find_outlier
                is_outlier, list_word_idx_most_prob_compo = self._fill_score_row_and_find_outlier(text_idx,
                                                                                                  word_idx,
                                                                                                  score_rows,
                                                                                                  text_word_score_dict,
                                                                                                  word_decision_frontier,
                                                                                                  word_idx_to_compo_word_in_win_score)
                if is_outlier:
                    find_outlier = True
                    # list_outlier_position.append(w)
                    dict_outlier_idx_to_list_word_idx_most_prob_compo[w] = (word_idx, list_word_idx_most_prob_compo)

            if find_outlier:
                self._print_score_table(words, occur_word_row, score_rows, text_score_dict, text_idx)

                for i, (outlier_idx, list_word_idx_most_prob_compo) in dict_outlier_idx_to_list_word_idx_most_prob_compo.items():
                    # outlier_position = list_outlier_position[i]
                    self._print_most_prob_word(words, i, outlier_idx, compo_word_win_to_word_idx_context_score, w2v_model)
                    self._print_most_compo_word(list_word_idx_most_prob_compo, outlier_idx, w2v_model)

                raw_doc = aggr_elem["raw_docs"][text_idx]
                fields = es.extract_fields_from_document(
                    raw_doc,
                    extract_derived_fields=self.model_settings["use_derived_fields"])
                outlier = self.create_outlier(fields,
                                              raw_doc,
                                              extra_outlier_information=None)
                outliers.append(outlier)

        return outliers

    def _find_all_scores(self, center_context_text_prob_list):
        word_idx_to_compo_word_in_win_score = {"center": dict(),
                                               "context": dict(),
                                               "total": dict()}
        compo_word_in_win_to_word_idx_score = {"center": dict(),
                                               "context": dict(),
                                               "total": dict()}

        text_word_score_dict = {"center": dict(),
                                "context": dict(),
                                "total": dict()}
        word_score_dict = {"center": dict(),
                           "context": dict(),
                           "total": dict()}
        text_score_dict = {"center": dict(),
                           "context": dict(),
                           "total": dict(),
                           "mean": dict()}

        current_text_idx = 0

        tmp_word_idx_to_comp_word_in_win_dict = dict()

        tmp_center_prob_list = dict()
        tmp_context_prob_list = dict()

        for score_type in text_word_score_dict:
            text_word_score_dict[score_type][current_text_idx] = dict()

        for center_idx, context_idx, text_idx, prob in center_context_text_prob_list:
            if current_text_idx == text_idx:
                if center_idx not in tmp_center_prob_list:
                    tmp_center_prob_list[center_idx] = list()
                    tmp_word_idx_to_comp_word_in_win_dict[center_idx] = ""
                tmp_center_prob_list[center_idx].append(prob)
                tmp_word_idx_to_comp_word_in_win_dict[center_idx] += str(context_idx) + "|"

                if context_idx not in tmp_context_prob_list:
                    tmp_context_prob_list[context_idx] = list()
                tmp_context_prob_list[context_idx].append(prob)
            else:
                self.update_all_score(tmp_word_idx_to_comp_word_in_win_dict,
                                      word_idx_to_compo_word_in_win_score,
                                      compo_word_in_win_to_word_idx_score,
                                      tmp_center_prob_list,
                                      tmp_context_prob_list,
                                      text_word_score_dict,
                                      word_score_dict,
                                      text_score_dict,
                                      current_text_idx)

                tmp_word_idx_to_comp_word_in_win_dict = dict()
                tmp_word_idx_to_comp_word_in_win_dict[center_idx] = str(context_idx) + "|"

                tmp_center_prob_list = dict()
                tmp_center_prob_list[center_idx] = list()
                tmp_center_prob_list[center_idx].append(prob)
                tmp_context_prob_list = dict()
                tmp_context_prob_list[context_idx] = list()
                tmp_context_prob_list[context_idx].append(prob)

                for score_type in text_word_score_dict:
                    text_word_score_dict[score_type][text_idx] = dict()
                current_text_idx = text_idx

        self.update_all_score(tmp_word_idx_to_comp_word_in_win_dict,
                              word_idx_to_compo_word_in_win_score,
                              compo_word_in_win_to_word_idx_score,
                              tmp_center_prob_list,
                              tmp_context_prob_list,
                              text_word_score_dict,
                              word_score_dict,
                              text_score_dict,
                              current_text_idx)

        return text_word_score_dict, word_score_dict, text_score_dict, word_idx_to_compo_word_in_win_score, compo_word_in_win_to_word_idx_score

    def update_all_score(self,
                         tmp_word_idx_to_comp_word_in_win_dict,
                         word_idx_to_compo_word_in_win_score,
                         compo_word_in_win_to_word_idx_score,
                         tmp_center_word_prob_list,
                         tmp_context_word_prob_list,
                         text_word_score_dict,
                         word_score_dict,
                         text_score_dict,
                         current_text_idx):
        '''
        TODO move to static
        '''
        center_word_score_list = list()
        context_word_score_list = list()
        total_word_score_list = list()

        total_prob_list = list()
        for tmp_word_idx in tmp_center_word_prob_list:
            total_prob_list.extend(tmp_center_word_prob_list[tmp_word_idx])

            center_word_score = mean(tmp_center_word_prob_list[tmp_word_idx], self.model_settings["use_geo_mean"])
            context_word_score = mean(tmp_context_word_prob_list[tmp_word_idx], self.model_settings["use_geo_mean"])
            total_word_score = mean([center_word_score, context_word_score], self.model_settings["use_geo_mean"])

            compo_key = tmp_word_idx_to_comp_word_in_win_dict[tmp_word_idx]
            if compo_key not in compo_word_in_win_to_word_idx_score["center"]:
                compo_word_in_win_to_word_idx_score["center"][compo_key] = dict()
                compo_word_in_win_to_word_idx_score["context"][compo_key] = dict()
                compo_word_in_win_to_word_idx_score["total"][compo_key] = dict()

            compo_word_in_win_to_word_idx_score["center"][compo_key][tmp_word_idx] = center_word_score
            compo_word_in_win_to_word_idx_score["context"][compo_key][tmp_word_idx] = context_word_score
            compo_word_in_win_to_word_idx_score["total"][compo_key][tmp_word_idx] = total_word_score

            if tmp_word_idx not in word_score_dict["center"]:
                word_idx_to_compo_word_in_win_score["center"][tmp_word_idx] = dict()
                word_idx_to_compo_word_in_win_score["context"][tmp_word_idx] = dict()
                word_idx_to_compo_word_in_win_score["total"][tmp_word_idx] = dict()

                word_score_dict["center"][tmp_word_idx] = list()
                word_score_dict["context"][tmp_word_idx] = list()
                word_score_dict["total"][tmp_word_idx] = list()

            word_idx_to_compo_word_in_win_score["center"][tmp_word_idx][compo_key] = center_word_score
            word_idx_to_compo_word_in_win_score["context"][tmp_word_idx][compo_key] = context_word_score
            word_idx_to_compo_word_in_win_score["total"][tmp_word_idx][compo_key] = total_word_score

            word_score_dict["center"][tmp_word_idx].append(center_word_score)
            word_score_dict["context"][tmp_word_idx].append(context_word_score)
            word_score_dict["total"][tmp_word_idx].append(total_word_score)

            text_word_score_dict["center"][current_text_idx][tmp_word_idx] = center_word_score
            text_word_score_dict["context"][current_text_idx][tmp_word_idx] = context_word_score
            text_word_score_dict["total"][current_text_idx][tmp_word_idx] = total_word_score

            center_word_score_list.append(center_word_score)
            context_word_score_list.append(context_word_score)
            total_word_score_list.append(total_word_score)

        text_score_dict["center"][current_text_idx] = mean(center_word_score_list, self.model_settings["use_geo_mean"])
        text_score_dict["context"][current_text_idx] = mean(context_word_score_list,
                                                            self.model_settings["use_geo_mean"])
        text_score_dict["total"][current_text_idx] = mean(total_word_score_list, self.model_settings["use_geo_mean"])

        text_score_dict["mean"][current_text_idx] = mean(total_prob_list, self.model_settings["use_geo_mean"])

    def _find_decision_frontier(self, word_score_dict, text_score_dict):
        word_decision_frontier = None
        text_decision_frontier = None
        if self.model_settings["trigger_focus"] == "text":
            # logging.logger.debug(text_score_dict["geo_mean"].values())
            text_decision_frontier = helpers.utils.get_decision_frontier(
                self.model_settings["trigger_method"],
                list(text_score_dict[self.model_settings["trigger_score"]].values()),
                self.model_settings["trigger_sensitivity"],
                self.model_settings["trigger_on"])
        else:
            word_decision_frontier = dict()
            for word_idx, list_score in word_score_dict[self.model_settings["trigger_score"]].items():
                word_decision_frontier[word_idx] = helpers.utils.get_decision_frontier(
                    self.model_settings["trigger_method"],
                    list_score,
                    self.model_settings["trigger_sensitivity"],
                    self.model_settings["trigger_on"])
        return word_decision_frontier, text_decision_frontier

    def _fill_score_row_and_find_outlier(self,
                                         text_idx,
                                         word_idx,
                                         score_rows,
                                         text_word_score_dict,
                                         word_decision_frontier,
                                         word_idx_to_compo_word_in_win_score):
        list_word_idx_most_prob_compo = list()
        is_outlier = False
        for score_row_type, score_row in score_rows.items():
            elem_row_score = text_word_score_dict[score_row_type][text_idx][word_idx]
            if self.model_settings["trigger_focus"] == "word" and score_row_type == self.model_settings[
                "trigger_score"]:
                if elem_row_score < word_decision_frontier[word_idx]:
                    is_outlier = True
                    # color in red
                    score_row.append(score_to_table_format(elem_row_score, "red"))

                    most_prob_compo = max(word_idx_to_compo_word_in_win_score[score_row_type][word_idx].items(),
                                          key=operator.itemgetter(1))[0]
                    list_word_idx_most_prob_compo = re.split("\|", most_prob_compo)[:-1]
                else:
                    # color in green
                    score_row.append(score_to_table_format(elem_row_score, "green"))
            else:
                score_row.append(score_to_table_format(elem_row_score))
        return is_outlier, list_word_idx_most_prob_compo

    def _print_score_table(self, words, occur_word_row, score_rows, text_score_dict, current_text_idx):
        table = list()
        table_fields_name = [" "]
        table_fields_name.extend(words)
        table_fields_name.append("TOTAL")
        table.append(table_fields_name)
        occur_word_row.insert(0, "Word batch occurrence")
        occur_word_row.append("")
        table.append(occur_word_row)

        score_rows["center"].insert(0, "<--Center score-->")
        score_rows["context"].insert(0, "-->Context score<--")
        score_rows["total"].insert(0, "Total score")

        for score_row_type, score_row in score_rows.items():
            if self.model_settings["trigger_focus"] == "text" and \
                    self.model_settings["trigger_score"] == score_row_type:
                score_row.append(score_to_table_format(text_score_dict[score_row_type][current_text_idx], "red"))
            else:
                score_row.append(score_to_table_format(text_score_dict[score_row_type][current_text_idx]))
            table.append(score_row)

        mean_row = [" " for i in range(len(table_fields_name) - 2)]
        mean_row.insert(0, "MEAN")
        if self.model_settings["trigger_score"] == "mean":
            mean_row.append(score_to_table_format(text_score_dict["mean"][current_text_idx], "red"))
        else:
            mean_row.append(score_to_table_format(text_score_dict["mean"][current_text_idx]))
        table.append(mean_row)
        tabulate_table = tabulate(table, headers="firstrow", tablefmt="fancy_grid")
        logging.logger.debug("Outlier info:\n" + str(tabulate_table))

    def _print_most_prob_word(self,
                              words,
                              outlier_position,
                              outlier_idx,
                              compo_word_win_to_word_idx_context_score,
                              w2v_model):
        compo_word_win = ""
        range_down = max(0, outlier_position - self.model_settings["size_window"])
        range_up = min(outlier_position + 1 + self.model_settings["size_window"], len(words))
        for i in range(range_down, range_up):
            if words[i] in w2v_model.word2idx:
                word_idx = w2v_model.word2idx[words[i]]
            else:
                unknown_token = w2v_model.unknown_token
                word_idx = w2v_model.word2idx[unknown_token]
            if i != outlier_position:
                compo_word_win += str(word_idx) + "|"
                logging.logger.debug(words[i])

        most_prob_context_word_idx = max(
            compo_word_win_to_word_idx_context_score[self.model_settings["trigger_score"]][compo_word_win].items(),
            key=operator.itemgetter(1))[0]
        if most_prob_context_word_idx != outlier_idx:
            most_prob_context_word = w2v_model.idx2word[int(most_prob_context_word_idx)]
            logging.logger.debug("The most probable word within the window of " +
                                 PRINT_COLORS["red"] +
                                 w2v_model.idx2word[outlier_idx] +
                                 PRINT_COLORS["end"] +
                                 " is: " +
                                 PRINT_COLORS["green"] + most_prob_context_word + PRINT_COLORS["end"])

    def _print_most_compo_word(self, list_word_idx_most_prob_compo, outlier_idx, w2v_model):
        list_word_most_prob_compo = [w2v_model.idx2word[int(word_idx_str)] for word_idx_str in
                                     list_word_idx_most_prob_compo]
        logging.logger.debug("The most probable composition of words within the window of " +
                             PRINT_COLORS["red"] + w2v_model.idx2word[outlier_idx] + PRINT_COLORS["end"] +
                             " is: " + PRINT_COLORS["green"] + str(list_word_most_prob_compo) +
                             PRINT_COLORS["end"])

    def _processing_outliers_in_batch(self, outliers_in_batch):
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
