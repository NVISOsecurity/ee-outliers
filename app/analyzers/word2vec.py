import numpy as np
import helpers.utils
import analyzers.ml_models.word2vec as word2vec

from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
from helpers.outlier import Outlier

from typing import List, Dict


class Word2VecAnalyzer(Analyzer):

    def extract_extra_model_settings(self) -> None:
        self.model_settings["sentence_format"] = settings.config.get(self.config_section_name, "sentence_format")\
                                            .replace(' ', '').split(",")  # remove unnecessary whitespace, split fields
        logging.logger.debug("using word2vec sentence format " + ','.join(self.model_settings["sentence_format"]))

        self.model_settings["train_model"] = settings.config.getboolean(self.config_section_name, "train_model")

        self.model_settings["trigger_on"] = settings.config.get(self.config_section_name, "trigger_on")
        self.model_settings["trigger_method"] = settings.config.get(self.config_section_name, "trigger_method")
        self.model_settings["trigger_sensitivity"] = settings.config.getint(self.config_section_name,
                                                                            "trigger_sensitivity")

    def train_model(self) -> None:
        w2v_model: word2vec.Word2Vec = word2vec.Word2Vec(name=self.model_name)
        search_query: Dict[str, List] = es.filter_by_query_string(self.model_settings["es_query_filter"])

        sentences: List[tuple] = list()

        self.total_events = es.count_documents(search_query=search_query)
        training_data_size_pct = settings.config.getint("machine_learning", "training_data_size_pct")
        training_data_size = self.total_events / 100 * training_data_size_pct

        logging.print_analysis_intro(event_type="training " + self.model_name, total_events=self.total_events)
        total_training_events = int(min(training_data_size, self.total_events))

        logging.init_ticker(total_steps=total_training_events, 
                            desc=self.model_name + " - preparing word2vec training set")
        for doc in es.scan(index=self.es_index, search_query=search_query):
            if len(sentences) < total_training_events:
                logging.tick()
                fields = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
                if set(self.model_settings["sentence_format"]).issubset(fields.keys()):
                    new_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields,
                                                                sentence_format=self.model_settings["sentence_format"])
                    for sentence in new_sentences:
                        sentences.append(tuple(sentence))

                    # Remove all duplicates from sentences for training - REMOVED FOR TESTING
                    # sentences = list(sentences)
            else:
                # We have collected sufficient training data
                break

        # Now, train the model
        if len(sentences) > 0:
            w2v_model.train_model(sentences)
        else:
            logging.logger.warning("no sentences to train model on. Are you sure the sentence configuration is " + \
                                   "correctly defined?")

    def evaluate_model(self) -> None:
        self.extract_extra_model_settings()

        # Train the model
        if self.model_settings["train_model"]:
            self.train_model()
            return

        w2v_model: word2vec.Word2Vec = word2vec.Word2Vec(name=self.model_name)
        search_query = es.filter_by_query_string(self.model_settings["es_query_filter"])

        if not w2v_model.is_trained():
            logging.logger.warning("model was not trained! Skipping analysis.")
        else:
            # Check if we need to run the test data instead of real data
            if w2v_model.use_test_data:
                logging.print_generic_intro("using test data instead of live data to evaluate model " + self.model_name)
                self.evaluate_test_sentences(w2v_model=w2v_model)
                return

            self.total_events = es.count_documents(index=self.es_index, search_query=search_query)
            logging.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)

            logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating word2vec model")

            raw_docs: List[Dict] = list()
            eval_sentences: List = list()

            for doc in es.scan(index=self.es_index, search_query=search_query):
                logging.tick()
                fields = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])

                try:
                    new_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields,
                                                                sentence_format=self.model_settings["sentence_format"])
                    eval_sentences.extend(new_sentences)
                except KeyError:
                    logging.logger.debug("skipping event which does not contain the target and aggregator fields we " +\
                                         "are processing. - [" + self.model_name + "]")
                    continue

                for _ in new_sentences:
                    raw_docs.append(doc)

                # Evaluate batch of events against the model
                if logging.current_step == self.total_events or \
                        len(eval_sentences) >= settings.config.getint("machine_learning", "word2vec_batch_eval_size"):
                    logging.logger.info("evaluating batch of " + str(len(eval_sentences)) + " sentences")
                    outliers = self.evaluate_batch_for_outliers(w2v_model=w2v_model, eval_sentences=eval_sentences,
                                                                raw_docs=raw_docs)

                    if len(outliers) > 0:
                        unique_summaries = len(set(o.outlier_dict["summary"] for o in outliers))
                        logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" + \
                                            str(unique_summaries) + " unique summaries]")

                    # Reset data structures for next batch
                    raw_docs = list()
                    eval_sentences = list()

    def evaluate_batch_for_outliers(self, w2v_model: word2vec.Word2Vec, eval_sentences: List,
                                    raw_docs: List[Dict]) -> List[Outlier]:
        # Initialize
        outliers: List[Outlier] = list()

        # all_words_probs: contains an array of arrays. the nested arrays contain the probabilities of a word on that
        # index to have a certain probability, in the context of another word
        sentence_probs = w2v_model.evaluate_sentences(eval_sentences)

        for i, single_sentence_prob in enumerate(sentence_probs):
            # If the probability is nan, it means that the sentence could not be evaluated, and we can't reason about it.
            # This happens for example whenever the sentence is made up entirely of words that aren't known to the trained model.
            if single_sentence_prob is np.nan:
                continue

            unique_probs = list(set(sentence_probs))

            decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"], unique_probs,
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])
            is_outlier = helpers.utils.is_outlier(single_sentence_prob, decision_frontier,
                                                  self.model_settings["trigger_on"])
            if is_outlier:
                fields = es.extract_fields_from_document(raw_docs[i],
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
                outliers.append(self.process_outlier(fields, raw_docs[i], extra_outlier_information=dict()))
            else:
                if w2v_model.use_test_data:
                    logging.logger.info("Not an outlier: " + str(eval_sentences[i]) + " - " + str(single_sentence_prob))

        return outliers

    def evaluate_test_sentences(self, w2v_model: word2vec.Word2Vec) -> None:
        test_sentences = self.generate_test_sentences()
        sentence_probs = w2v_model.evaluate_sentences(test_sentences)

        for i, single_sentence_prob in enumerate(sentence_probs):
            if single_sentence_prob is np.nan:
                logging.logger.info("could not calculate probability, skipping evaluation of " + str(test_sentences[i]))
                continue

            unique_probs = list(set(sentence_probs))

            decision_frontier = helpers.utils.get_decision_frontier(self.model_settings["trigger_method"],
                                                                    unique_probs,
                                                                    self.model_settings["trigger_sensitivity"],
                                                                    self.model_settings["trigger_on"])
            is_outlier = helpers.utils.is_outlier(single_sentence_prob, decision_frontier,
                                                  self.model_settings["trigger_on"])
            if is_outlier:
                logging.logger.info("outlier: " + str(test_sentences[i]) + " - " + str(single_sentence_prob))
            else:
                logging.logger.info("not an outlier: " + str(test_sentences[i]) + " - " + str(single_sentence_prob))

    def generate_test_sentences(self) -> List[List[str]]:
        sentences: List[List[str]] = list()
        if self.model_name == "suspicious_user_login":
            sentences.append(['user1', 'dummy-pc-name-user1'])
            sentences.append(['user2', 'dummy-pc-name-user2'])
            sentences.append(['user2', 'dummy-pc-name-user1'])
            sentences.append(['user2', 'dummy-pc-unknown'])
        elif self.model_name == "suspicious_autoexec_names":
            sentences.append(['services', 'MALWARE.EXE'])
        else:
            logging.logger.warning("no test sentences found for model " + self.model_name)

        return sentences
