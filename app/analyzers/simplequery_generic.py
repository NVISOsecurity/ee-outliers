from configparser import NoOptionError

from helpers.outlier import Outlier
from helpers.singletons import settings, es, logging
import re
import helpers.utils


def perform_analysis():
    for name in settings.config.sections():
        if name.startswith("simplequery_"):
            param, model_name = name.split("simplequery_", 1)

            should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(name, "run_model")
            should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(name, "test_model")

            if should_test_model or should_run_model:
                model_settings = extract_model_settings(name)
                run_simplequery_model(section_name=name, model_name=model_name, model_settings=model_settings)


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

    outliers = list()
    for doc in es.scan(lucene_query=lucene_query):
        logging.tick()
        fields = es.extract_fields_from_document(doc)

        outlier_summary = replace_placeholder_string_with_fields(model_settings["outlier_summary"], fields)
        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        outlier = Outlier(type=model_settings["outlier_type"], reason=model_settings["outlier_reason"], summary=outlier_summary)

        if len(outlier_assets) > 0:
            outlier.add_observation("assets", outlier_assets)

        outliers.append(outlier)

        es.process_outliers(doc=doc, outliers=[outlier], should_notify=model_settings["should_notify"])

    if len(outliers) > 0:
        unique_summaries = len(set(o.get_observation("summary") for o in outliers))
        logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" + str(unique_summaries) + " unique]")


def run_simplequery_model(section_name=None, model_name=None, model_settings=None):
    # Evaluate against the model
    evaluate_model(model_name=model_name, model_settings=model_settings)


def replace_placeholder_string_with_fields(placeholder, fields):
    # Replace fields from fieldmappings in summary
    regex = re.compile(r'\{([^\}]*)\}')
    field_name_list = regex.findall(placeholder)  # ['source_ip','destination_ip'] for example

    for field_name in field_name_list:
        try:
            flattened_sentence = helpers.utils.flatten_sentence(helpers.utils.get_dotkey_value(fields, field_name))
            placeholder = placeholder.replace('{' + field_name + '}', flattened_sentence)
        except (KeyError, TypeError):
            placeholder = placeholder.replace('{' + field_name + '}', "{field " + field_name + " not found or null in event}")

    return placeholder
