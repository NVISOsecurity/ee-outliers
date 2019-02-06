import json
from configparser import NoOptionError
from helpers.singletons import settings, es, logging


def perform_analysis():
    for name in settings.config.sections():
        if name.startswith("test_"):
            param, model_name = name.split("test_", 1)

            should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(name, "run_model")
            should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(name, "test_model")

            if should_test_model or should_run_model:
                model_settings = extract_model_settings(name)
                run_test_model(section_name=name, model_name=model_name, model_settings=model_settings)


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


def evaluate_model(model_name=None, model_settings=None):
    lucene_query = es.filter_by_query_string(model_settings["es_query_filter"])
    total_events = es.count_documents(lucene_query=lucene_query)

    logging.print_analysis_intro(event_type="evaluating " + model_name, total_events=total_events)
    logging.init_ticker(total_steps=total_events, desc=model_name + " - evaluating simplequery model")

    for doc in es.scan(lucene_query=lucene_query):
        logging.tick()
        fields = es.extract_fields_from_document(doc)

        # Add your model logic here
        logging.logger.info(json.dumps(fields, indent=4))


def run_test_model(section_name=None, model_name=None, model_settings=None):
    # Evaluate against the model
    evaluate_model(model_name=model_name, model_settings=model_settings)
