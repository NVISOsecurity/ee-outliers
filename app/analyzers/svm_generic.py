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
        if name.startswith("svm_"):
                param, model_name = name.split("svm_", 1)

                should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(name, "run_model")
                should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(name, "test_model")
                should_train_model = settings.config.getboolean("general", "train_models") and settings.config.getboolean(name, "train_model")

                if should_test_model or should_run_model or should_train_model:
                    model_settings = extract_model_settings(name)
                    run_generic_svm_model(section_name=name, model_name=model_name, model_settings=model_settings)


def extract_model_settings(section_name):
    model_settings = dict()
    model_settings["es_query_filter"] = settings.config.get(section_name, "es_query_filter")

    model_settings["outlier_reason"] = settings.config.get(section_name, "outlier_reason")
    model_settings["outlier_type"] = settings.config.get(section_name, "outlier_type")
    model_settings["outlier_summary"] = settings.config.get(section_name, "outlier_summary")

    try:
        model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(section_name, "should_notify")
    except NoOptionError:
        model_settings["should_notify"] = False

    return model_settings


def train_model(model_name=None, model_settings=None):
    lucene_query = es.filter_by_query_string(model_settings["es_query_filter"])

    train_data = list()

    total_events = es.count_documents(lucene_query=lucene_query)
    training_data_size_pct = settings.config.getint("machine_learning", "training_data_size_pct")
    training_data_size = total_events / 100 * training_data_size_pct

    logging.print_analysis_intro(event_type="training " + model_name, total_events=total_events)
    total_training_events = int(min(training_data_size, total_events))

    logging.init_ticker(total_steps=total_training_events, desc=model_name + " - preparing SVM training set")
    for doc in es.scan(lucene_query=lucene_query):
        if len(train_data) < total_training_events:
            logging.tick()
            fields = es.extract_fields_from_document(doc)
            train_data.append(fields)
        else:
            # We have collected sufficient training data
            break

    # Now, train the model
    if len(train_data) > 0:
        pass # Train!!
    else:
        logging.logger.warning("no sentences to train model on. Are you sure the sentence configuration is correctly defined?")


def evaluate_model(model_name=None, model_settings=None):
    pass


def run_generic_svm_model(section_name=None, model_name=None, model_settings=None):
    # Train the model
    if settings.config.getboolean(section_name, "train_model"):
        train_model(model_name=model_name, model_settings=model_settings)

    # Evaluate against the model
    if settings.config.getboolean(section_name, "run_model"):
        evaluate_model(model_name=model_name, model_settings=model_settings)
