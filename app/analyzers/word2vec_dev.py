import helpers.utils

from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
import analyzers.ml_models.word2vec_dev as word2vec
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import re
import numpy as np
from scipy import spatial

class Word2VecDevAnalyzer(Analyzer):

    def __init__(self, model_name, config_section):
        super(Word2VecDevAnalyzer, self).__init__("word2vec_dev", model_name, config_section)

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        # TODO train_model unnecessary because we train and evaluate it anyway
        self.model_settings["train_model"] = self.config_section.getboolean("train_model")

        self.model_settings["target_fields"] = self.config_section["target_fields"].replace(' ', '').split(",")
        # TODO put a limit of the combination of aggregator ? Or message when not enough sentences has been gathered in
        # a specific agregation
        self.model_settings["aggr_fields"] = self.config_section["aggregator_fields"].replace(' ', '').split(",")
        self.model_settings["min_target_buckets"] = self.config_section.getint("min_target_buckets")

        self.model_settings["separators"] = self.config_section["separators"]
        self.model_settings["size_window"] = int(self.config_section["size_window"])
        self.model_settings["drop_duplicates"] = self.config_section.getboolean("drop_duplicates")
        self.model_settings["use_derived_fields"] = self.config_section.getboolean("use_derived_fields")

        self.model_settings["tensorboard"] = self.config_section.getboolean("tensorboard")

        self.model_settings["num_epochs"] = self.config_section.getint("num_epochs")
        self.model_settings["learning_rate"] = self.config_section.getfloat("learning_rate")
        self.model_settings["embedding_size"] = self.config_section.getint("embedding_size")
        self.model_settings["seed"] = self.config_section.getint("seed")

        self.model_settings["use_prob_model"] = self.config_section.getboolean("use_prob_model")

        logging.logger.debug("Use prob model: " + str(self.model_settings["use_prob_model"]))

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
            batch = defaultdict()
            total_targets_in_batch = 0
            total_targets_rmv = 0
            num_targets_not_processed = 0

            for doc in documents:
                target_sentences, aggr_sentences = self._extract_target_and_aggr_sentences(doc=doc,
                                                                                           target_fields=target_fields,
                                                                                           aggr_fields=aggr_fields)

                if target_sentences is not None and aggr_sentences is not None:
                    batch, num_add_targets, num_rmv_targets = self._add_doc_and_target_sentences_to_batch(batch=batch,
                                                                        target_sentences=target_sentences,
                                                                        aggr_sentences=aggr_sentences,
                                                                        doc=doc)
                    total_targets_rmv += num_rmv_targets
                    total_targets_in_batch += num_add_targets

                if total_targets_in_batch >= self.batch_train_size or logging.current_step + 1 == self.total_events:
                    if total_targets_rmv > 0:
                        logging.logger.info("Drop_duplicates activated: %i/%i events have been removed",
                                            total_targets_rmv,
                                            logging.current_step + 1)
                    log_message = " BATCH " + str(self.current_batch_num) + " - num aggr:" + str(len(batch))
                    logging.logger.debug(log_message)
                    # Display log message
                    self._display_batch_log_message(total_targets_in_batch, num_targets_not_processed)

                    # Evaluate the current batch
                    outliers_in_batch, total_targets_removed = self._evaluate_batch_for_outliers(batch=batch)

                    if total_targets_removed == 0:
                        # TODO put better comment
                        raise ValueError("Unable to fill the aggregator buckets for this batch.")

                    # Processing the outliers found
                    self._processing_outliers_in_batch(outliers_in_batch)

                    total_targets_in_batch -= total_targets_removed
                    num_targets_not_processed = total_targets_in_batch
                    self.current_batch_num += 1

                logging.tick()

            if num_targets_not_processed > 0:
                log_message = "" + "{:,}".format(num_targets_not_processed) + " sentences not processed in last batch"
                logging.logger.info(log_message)

    # TODO function from terms.py --> should transfer it to utils?
    def _extract_target_and_aggr_sentences(self, doc, target_fields, aggr_fields):
        """
        Extract target and aggregator sentence from a document

        :param doc: document where data need to be extract
        :param target: target key name
        :return: list of target and list of aggregator
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


    def _add_doc_and_target_sentences_to_batch(self, batch, target_sentences, aggr_sentences, doc):
        """
        Add a document to the current batch

        :param current_batch: existing batch (where doc need to be saved)
        :param target_sentences: list of targets
        :param aggregator_sentences: list of aggregator
        :param doc: document that need to be added
        :return: batch with document inside
        """
        for target_sentence in target_sentences:
            flattened_target_sentence = helpers.utils.flatten_sentence(target_sentence, sep_str='')

            for aggregator_sentence in aggr_sentences:
                flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)

                if flattened_aggregator_sentence not in batch.keys():
                    batch[flattened_aggregator_sentence] = defaultdict(list)
                if self.model_settings["drop_duplicates"] and \
                        flattened_target_sentence in batch[flattened_aggregator_sentence]["targets"]:
                    return batch, 0, len(target_sentences)*len(aggr_sentences)

                batch[flattened_aggregator_sentence]["targets"].append(flattened_target_sentence)
                batch[flattened_aggregator_sentence]["raw_docs"].append(doc)

        return batch, len(target_sentences)*len(aggr_sentences), 0

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
        w2v_model.prepare_voc()

        if self.model_settings["use_prob_model"]:
            center_context_text_prob_list = w2v_model.stat_model(aggr_elem["targets"])
        else:
            # Train word2vec model
            loss_values = w2v_model.train_model(aggr_elem["targets"])

            if self.model_settings["tensorboard"]:
                test_matplotlib_to_tensorboard(loss_values, aggr_key, self.current_batch_num)
                # list_to_tensorboard(loss_values, aggr_key, self.current_batch_num)

            # Eval word2vec model
            center_context_text_prob_list = w2v_model.eval_model(aggr_elem["targets"])

            stat_comparison(center_context_text_prob_list, w2v_model, aggr_elem["targets"])

        # Find outliers from the word2vec model outputs
        # outliers = self._find_outliers(aggr_elem, center_context_text_prob_list, thresholds)
        outliers, word_context_center_score_dict = self._find_outliers3(aggr_elem, center_context_text_prob_list, w2v_model)

        if self.model_settings["tensorboard"]:
            for word, context_center_score in word_context_center_score_dict.items():
                logging.logger.debug(word)
                logging.logger.debug(context_center_score)

        # Remove aggr_elem from batch
        aggr_elem["targets"] = list()
        aggr_elem["raw_docs"] = list()

        return outliers

    def _find_outliers3(self, aggr_elem, center_context_text_prob_list, w2v_model):
        outliers = list()
        context_score = dict()
        center_score = dict()
        context_score_list = dict()
        center_score_list = dict()
        word_conext_center_score_dict = defaultdict()
        current_text_idx = 0
        thresholds = 0.001
        for center_idx, context_idx, text_idx, context_prob in center_context_text_prob_list:
            if current_text_idx == text_idx:
                if context_idx not in context_score_list:
                    context_score_list[context_idx] = list()
                context_score_list[context_idx].append(context_prob)

                if center_idx not in center_score_list:
                    center_score_list[center_idx] = list()
                center_score_list[center_idx].append(context_prob)

            else:

                for context_key in context_score_list:
                    context_score[context_key] = geo_mean(context_score_list[context_key])
                for center_key in center_score_list:
                    center_score[center_key] = geo_mean(center_score_list[center_key])
                text_str = aggr_elem["targets"][current_text_idx]
                words = re.split(self.model_settings["separators"], text_str)
                message_log = "| "
                is_outlier = False

                for w, word in enumerate(words):
                    if word in w2v_model.word2idx:
                        word_id = w2v_model.word2idx[word]
                    else:
                        unknown_token = w2v_model.unknown_token
                        word_id = w2v_model.word2idx[unknown_token]
                    voc_counter_dict = dict(w2v_model.voc_counter)
                    color = '\033[92m' # green
                    if center_score[word_id] < thresholds:
                        is_outlier = True
                        color = '\033[91m' # red
                    color_context = '\033[92m'
                    if context_score[word_id] < thresholds:
                        color_context = '\033[91m'# red

                    context_center_score = geo_mean([context_score[word_id], center_score[word_id]])
                    color_context_center = '\033[92m'# green
                    if context_center_score < thresholds:
                        color_context_center = '\033[91m'# red

                    message_log += word + \
                                   "[" + color_context + "{:.2e}".format(context_score[word_id]) + '\033[0m' + \
                                   "][" + color + "{:.2e}".format(center_score[word_id]) + '\033[0m' + \
                                   "][" + color_context_center + "{:.2e}".format(context_center_score) + '\033[0m' + \
                                   "][" + str(voc_counter_dict[word]) + "] | "

                    if word not in word_conext_center_score_dict:
                        word_conext_center_score_dict[word] = defaultdict(list)
                    word_conext_center_score_dict[word]["context_score"].append(context_score[word_id])
                    word_conext_center_score_dict[word]["center_score"].append(center_score[word_id])

                total_context_score = geo_mean(list(context_score.values()))
                total_center_score = geo_mean(list(center_score.values()))
                total_score = math.pow(total_context_score*total_center_score, 1/2)
                message_log += "TOTAL: [" + "{:.2e}".format(total_context_score) + \
                              "][" + "{:.2e}".format(total_center_score) + \
                              "][" + "{:.2e}".format(total_score) + "]"

                if is_outlier:
                    logging.logger.debug("\n" + message_log)
                    raw_doc = aggr_elem["raw_docs"][current_text_idx]
                    fields = es.extract_fields_from_document(
                        raw_doc,
                        extract_derived_fields=self.model_settings["use_derived_fields"])
                    outlier = self.create_outlier(fields,
                                                  raw_doc,
                                                  extra_outlier_information=None)
                    outliers.append(outlier)
                context_score = dict()
                center_score = dict()
                context_score_list = dict()
                center_score_list = dict()

                context_score_list[context_idx] = list()
                context_score_list[context_idx].append(context_prob)
                center_score_list[center_idx] = list()
                center_score_list[center_idx].append(context_prob)

                current_text_idx = text_idx

        return outliers, word_conext_center_score_dict

    def _find_outliers2(self, aggr_elem, center_context_text_prob_list, w2v_model):
        outliers = list()
        context_score = dict()
        center_score = dict()
        context_num_rel = dict() #Number of relations
        center_num_rel = dict()
        word_conext_center_score_dict = defaultdict()
        current_text_idx = 0
        thresholds = 0.001
        for center_idx, context_idx, text_idx, context_prob in center_context_text_prob_list:
            if current_text_idx == text_idx:
                if context_idx not in context_score:
                    context_score[context_idx] = context_prob
                    context_num_rel[context_idx] = 1
                else:
                    context_score[context_idx] *= context_prob
                    context_num_rel[context_idx] += 1

                if center_idx not in center_score:
                    center_score[center_idx] = context_prob
                    center_num_rel[center_idx] = 1
                else:
                    center_score[center_idx] *= context_prob
                    center_num_rel[center_idx] += 1
            else:

                for context_key in context_score:
                    context_score[context_key] = math.pow(context_score[context_key], 1/context_num_rel[context_key])
                for center_key in center_score:
                    center_score[center_key] = math.pow(center_score[center_key], 1/center_num_rel[center_key])
                text_str = aggr_elem["targets"][current_text_idx]
                words = re.split(self.model_settings["separators"], text_str)
                message_log = "| "
                is_outlier = False

                for w, word in enumerate(words):
                    if word in w2v_model.word2idx:
                        word_id = w2v_model.word2idx[word]
                    else:
                        unknown_token = w2v_model.unknown_token
                        word_id = w2v_model.word2idx[unknown_token]
                    voc_counter_dict = dict(w2v_model.voc_counter)
                    color = '\033[92m' # green
                    if center_score[word_id] < thresholds:
                        color = '\033[91m' # red
                    color_context = '\033[92m'
                    if context_score[word_id] < thresholds:
                        color_context = '\033[91m'# red

                    context_center_score = math.pow(context_score[word_id]*center_score[word_id], 1/2)
                    color_context_center = '\033[92m'# green
                    if context_center_score < thresholds:
                        is_outlier = True
                        color_context_center = '\033[91m'# red

                    message_log += word + \
                                   "[" + color_context + "{:.2e}".format(context_score[word_id]) + '\033[0m' + \
                                   "][" + color + "{:.2e}".format(center_score[word_id]) + '\033[0m' + \
                                   "][" + color_context_center + "{:.2e}".format(context_center_score) + '\033[0m' + \
                                   "][" + str(voc_counter_dict[word]) + "] | "

                    if word not in word_conext_center_score_dict:
                        word_conext_center_score_dict[word] = defaultdict(list)
                    word_conext_center_score_dict[word]["context_score"].append(context_score[word_id])
                    word_conext_center_score_dict[word]["center_score"].append(center_score[word_id])

                total_context_score = geo_mean_overflow(list(context_score.values()))
                total_center_score = geo_mean_overflow(list(center_score.values()))
                total_score = math.pow(total_context_score*total_center_score, 1/2)
                message_log += "TOTAL: [" + "{:.2e}".format(total_context_score) + \
                              "][" + "{:.2e}".format(total_center_score) + \
                              "][" + "{:.2e}".format(total_score) + "]"

                if is_outlier:
                    logging.logger.debug(message_log)
                    raw_doc = aggr_elem["raw_docs"][current_text_idx]
                    fields = es.extract_fields_from_document(
                        raw_doc,
                        extract_derived_fields=self.model_settings["use_derived_fields"])
                    outlier = self.create_outlier(fields,
                                                  raw_doc,
                                                  extra_outlier_information=None)
                    outliers.append(outlier)
                context_score = dict()
                center_score = dict()
                context_num_rel = dict()
                center_num_rel = dict()
                context_score[context_idx] = context_prob
                context_num_rel[context_idx] = 1
                center_score[center_idx] = context_prob
                center_num_rel[center_idx] = 1

                current_text_idx = text_idx

        return outliers, word_conext_center_score_dict

    def _find_outliers(self, aggr_elem, center_context_text_prob_list, thresholds):
        outliers = list()
        current_text_idx = 0
        current_text_probs = list()
        for center, context, text, probs in center_context_text_prob_list:
            for i in range(len(center)):
                context_idx = context[i].item()
                context_prob = probs[i][context_idx].item()
                text_idx = text[i].item()
                if current_text_idx == text_idx:
                    current_text_probs.append(context_prob)
                else:

                    if min(current_text_probs) < thresholds[context_idx]:
                        raw_doc = aggr_elem["raw_docs"][current_text_idx]
                        outlier_target_text = aggr_elem["targets"][current_text_idx]
                        logging.logger.debug("FIND OUTLIER!!!!")
                        logging.logger.debug(outlier_target_text)
                        fields = es.extract_fields_from_document(
                            raw_doc,
                            extract_derived_fields=self.model_settings["use_derived_fields"])
                        outlier = self.create_outlier(fields,
                                                      raw_doc,
                                                      extra_outlier_information=None)
                        outliers.append(outlier)

                    current_text_idx = text_idx
                    current_text_probs = [context_prob]

        return outliers

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


def test_matplotlib_to_tensorboard(list_elem, aggr_key, current_batch_num):
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


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))


def stat_comparison(center_context_text_prob_list, w2v_model, dataset):
    center_context_text_prob_list_stat = w2v_model.stat_model(dataset)

    _, _, _, prob_w2v = zip(*center_context_text_prob_list)
    _, _, _, prob_stat = zip(*center_context_text_prob_list_stat)
    cos_sim = 1 - spatial.distance.cosine(prob_w2v, prob_stat)

    logging.logger.debug("Cos similarity:" + str(cos_sim))

    # for i in range(len(center_context_text_prob_list)):
    #     center_idx1, context_idx1, text_idx1, prob1 = center_context_text_prob_list[i]
    #     center_idx2, context_idx2, text_idx2, prob2 = center_context_text_prob_list_stat[i]
    #     if center_idx1 != center_idx2 or context_idx1 != context_idx2 or text_idx1 != text_idx2:
    #         logging.logger.debug("NOT SAME COMBINATION FOR W2V AND STAT MODEL")
    #     logging.logger.debug(str(prob1) + " " + str(prob2))

