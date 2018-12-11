import numpy as np
from configparser import NoOptionError

from helpers.outlier import Outlier
from helpers.singletons import settings, es, logging
import analyzers.ml_models.word2vec as word2vec
import re
import helpers.utils
import time


def perform_analysis():
    for name in settings.config.sections():
        if name.startswith("word2vec_"):
                param, model_name = name.split("word2vec_", 1)

                should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(name, "run_model")
                should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(name, "test_model")
                should_train_model = settings.config.getboolean("general", "train_models") and settings.config.getboolean(name, "train_model")

                if should_test_model or should_run_model or should_train_model:
                    model_settings = extract_model_settings(name)

                    logging.logger.debug(model_name + " - using sentence format " + ','.join(model_settings["sentence_format"]))
                    run_generic_word2vec_model(section_name=name, model_name=model_name, model_settings=model_settings)


def extract_model_settings(section_name):
    model_settings = dict()
    model_settings["es_query_filter"] = settings.config.get(section_name, "es_query_filter")
    model_settings["sentence_format"] = settings.config.get(section_name, "sentence_format").replace(' ','').split(",")  # remove unnecessary whitespace, split fields

    model_settings["outlier_reason"] = settings.config.get(section_name, "outlier_reason")
    model_settings["outlier_type"] = settings.config.get(section_name, "outlier_type")
    model_settings["outlier_summary"] = settings.config.get(section_name, "outlier_summary")
    model_settings["outlier_sensitivity_stdevs"] = settings.config.getint(section_name, "outlier_sensitivity_stdevs")

    try:
        model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(section_name, "should_notify")
    except NoOptionError:
        model_settings["should_notify"] = False

    return model_settings


def train_model(model_name=None, model_settings=None):
    w2v_model = word2vec.Word2Vec(name=model_name)
    lucene_query = es.filter_by_query_string(model_settings["es_query_filter"])

    sentences = list()

    total_events = es.count_documents(lucene_query=lucene_query)
    training_data_size_pct = settings.config.getint("machine_learning", "training_data_size_pct")
    training_data_size = total_events / 100 * training_data_size_pct

    logging.print_analysis_intro(event_type="training " + model_name, total_events=total_events)
    total_training_events = int(min(training_data_size, total_events))

    logging.init_ticker(total_steps=total_training_events, desc=model_name + " - preparing word2vec training set")
    for doc in es.scan(lucene_query=lucene_query):
        if len(sentences) < total_training_events:
            logging.tick()
            fields = es.extract_fields_from_document(doc)
            if set(model_settings["sentence_format"]).issubset(fields.keys()):
                new_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=model_settings["sentence_format"])
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
        logging.logger.warning("no sentences to train model on. Are you sure the sentence configuration is correctly defined?")


def evaluate_model(model_name=None, model_settings=None):
    w2v_model = word2vec.Word2Vec(name=model_name)
    lucene_query = es.filter_by_query_string(model_settings["es_query_filter"])

    if not w2v_model.is_trained():
        logging.logger.warning("Model was not trained! Skipping analysis.")
    else:
        # Check if we need to run the test data instead of real data
        if w2v_model.use_test_data:
            logging.print_generic_intro("using test data instead of live data to evaluate model " + model_name)
            evaluate_test_sentences(w2v_model=w2v_model, model_settings=model_settings)
            return

        total_events = es.count_documents(lucene_query=lucene_query)
        logging.print_analysis_intro(event_type="evaluating " + model_name, total_events=total_events)

        logging.init_ticker(total_steps=total_events, desc=model_name + " - evaluating word2vec model")

        raw_docs = list()
        eval_sentences = list()

        for doc in es.scan(lucene_query=lucene_query):
            logging.tick()
            fields = es.extract_fields_from_document(doc)

            try:
                new_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=model_settings["sentence_format"])
                eval_sentences.extend(new_sentences)
            except KeyError:
                logging.logger.debug("skipping event which does not contain the target and aggregator fields we are processing. - [" + model_name + "]")
                continue

            for _ in new_sentences:
                raw_docs.append(doc)

            # Evaluate batch of events against the model
            if logging.current_step == total_events or len(eval_sentences) >= settings.config.getint("machine_learning", "word2vec_batch_eval_size"):
                logging.logger.info("evaluating batch of " + str(len(eval_sentences)) + " sentences")
                outliers = evaluate_batch_for_outliers(w2v_model=w2v_model, eval_sentences=eval_sentences, raw_docs=raw_docs, model_settings=model_settings)

                if len(outliers) > 0:
                    unique_summaries = len(set(o.get_observation("summary") for o in outliers))
                    logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" + str(unique_summaries) + " unique]")

                # Reset data structures for next batch
                raw_docs = list()
                eval_sentences = list()


def run_generic_word2vec_model(section_name=None, model_name=None, model_settings=None):
    # Train the model
    if settings.config.getboolean(section_name, "train_model"):
        train_model(model_name=model_name, model_settings=model_settings)

    # Evaluate against the model
    if settings.config.getboolean(section_name, "run_model"):
        evaluate_model(model_name=model_name, model_settings=model_settings)


def flatten_field(field=None):
    if type(field) is list:
        # Make sure the list does not contain nested lists, but only strings. If it's a nested list, we give up and return None
        if any(isinstance(i, list) or isinstance(i, dict) for i in field):
            return None
        else:
            # We convert a list value such as [1,2,3] into a single string, so the model can use it: 1-2-3
            field_value = "-".join(field)
            return field_value
    elif type(field) is dict:
        return None
    else:
        # We just cast to string in all other cases
        field_value = str(field)
        return field_value


def evaluate_batch_for_outliers(w2v_model=None, eval_sentences=None, raw_docs=None, model_settings=None):
    # Initialize
    outliers = list()

    # all_words_probs: contains an array of arrays. the nested arrays contain the probabilities of a word on that index to have a certain probability, in the context of another word
    sentence_probs = w2v_model.evaluate_sentences(eval_sentences)

    for i, single_sentence_prob in enumerate(sentence_probs):
        # If the probability is nan, it means that the sentence could not be evaluated, and we can't reason about it.
        # This happens for example whenever the sentence is made up entirely of words that aren't known to the trained model.
        if single_sentence_prob is np.nan:
            continue

        unique_probs = list(set(sentence_probs))

        # if is_outlier_cutoff_percentage(single_sentence_prob, cutoff=0.005):
        # if is_outlier_std(single_sentence_prob, unique_probs, model_settings):
        if is_outlier_mad(single_sentence_prob, unique_probs, model_settings):
            outlier_summary = model_settings["outlier_summary"]

            # Extract fields from raw document
            fields = es.extract_fields_from_document(raw_docs[i])
            outlier_summary = replace_placeholder_string_with_fields(outlier_summary, fields)

            outlier = Outlier(type=model_settings["outlier_type"], reason=model_settings["outlier_reason"], summary=outlier_summary)
            outlier.add_observation("probability", str(single_sentence_prob))

            outliers.append(outlier)
            es.process_outliers(doc=raw_docs[i], outliers=[outlier], should_notify=model_settings["should_notify"])
        else:
            if w2v_model.use_test_data:
                logging.logger.info("Not an outlier: " + str(eval_sentences[i]) + " - " + str(single_sentence_prob))
    return outliers


def replace_placeholder_string_with_fields(placeholder, fields):
    # Replace fields from fieldmappings in summary
    regex = re.compile(r'\{([^\}]*)\}')
    field_name_list = regex.findall(placeholder)  # ['source_ip','destination_ip'] for example

    for field_name in field_name_list:
        try:
            placeholder = placeholder.replace('{' + field_name + '}', flatten_field(helpers.utils.get_dotkey_value(fields, field_name)))
        except KeyError:
            placeholder = placeholder.replace('{' + field_name + '}', "{field " + field_name + " not found in event}")

    return placeholder


def is_outlier_cutoff_percentage(prob, cutoff):
    if prob < cutoff:
        return True
    else:
        return False


def is_outlier_top_n(prob, probs, w2v_model):
    if helpers.utils.is_in_top_x(probs, prob, int(0.1 * w2v_model.vocabulary_size)):
        return False
    else:
        return True


def is_outlier_std(prob, probs, model_settings):
    std = np.std(probs)
    if prob < np.nanmean(probs) - model_settings["outlier_sensitivity_stdevs"] * std:
        return True
    else:
        return False


def is_outlier_mad(prob, probs, model_settings):
    mad = np.nanmedian(np.absolute(probs - np.nanmedian(probs, 0)), 0)  # median absolute deviation
    decision_frontier = np.nanmedian(probs) - model_settings["outlier_sensitivity_stdevs"] * mad

    if prob < decision_frontier:
        return True
    else:
        return False


def evaluate_test_sentences(w2v_model=None, model_settings=None):
    test_sentences = generate_test_sentences(model_name=w2v_model.name)
    sentence_probs = w2v_model.evaluate_sentences(test_sentences)

    for i, single_sentence_prob in enumerate(sentence_probs):
        if single_sentence_prob is np.nan:
            logging.logger.info("could not calculate probability, skipping evaluation of " + str(test_sentences[i]))
            continue

        unique_probs = list(set(sentence_probs))
        if is_outlier_mad(single_sentence_prob, unique_probs, model_settings):
            logging.logger.info("outlier: " + str(test_sentences[i]) + " - " + str(single_sentence_prob))
        else:
            logging.logger.info("not an outlier: " + str(test_sentences[i]) + " - " + str(single_sentence_prob))


def generate_test_sentences(model_name=None):
    sentences = list()
    if model_name == "suspicious_user_login":
        sentences.append(['user1', 'dummy-pc-name-user1'])
        sentences.append(['user2', 'dummy-pc-name-user2'])
        sentences.append(['user2', 'dummy-pc-name-user1'])
        sentences.append(['user2', 'dummy-pc-unknown'])
    elif model_name == "suspicious_autoexec_names":
        sentences.append(['services', 'MALWARE.EXE'])
    else:
        logging.logger.warning("no test sentences found for model " + model_name)

    return sentences
