from configparser import NoOptionError
import numpy as np
from helpers.outlier import Outlier
from helpers.singletons import settings, es, logging
from collections import defaultdict
import re
import helpers.utils


def perform_analysis():
    for name in settings.config.sections():
        if name.startswith("metrics_"):
            param, model_name = name.split("metrics_", 1)

            should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(name, "run_model")
            should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(name, "test_model")

            if should_test_model or should_run_model:
                model_settings = extract_model_settings(name)
                run_generic_metrics_model(section_name=name, model_name=model_name, model_settings=model_settings)


def extract_model_settings(section_name):
    model_settings = dict()
    model_settings["es_query_filter"] = settings.config.get(section_name, "es_query_filter")
    model_settings["target"] = settings.config.get(section_name, "target")

    model_settings["aggregator"] = settings.config.get(section_name, "aggregator").replace(' ', '').split(",")
    model_settings["metric"] = settings.config.get(section_name, "metric")
    model_settings["trigger_on"] = settings.config.get(section_name, "trigger_on")
    model_settings["trigger_method"] = settings.config.get(section_name, "trigger_method")
    model_settings["trigger_sensitivity"] = settings.config.getint(section_name, "trigger_sensitivity")

    model_settings["outlier_reason"] = settings.config.get(section_name, "outlier_reason")
    model_settings["outlier_type"] = settings.config.get(section_name, "outlier_type")
    model_settings["outlier_summary"] = settings.config.get(section_name, "outlier_summary")

    try:
        model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(section_name, "should_notify")
    except NoOptionError:
        model_settings["should_notify"] = False

    if model_settings["metric"] not in {"length", "numerical_value", "entropy", "base64_encoded_length", "hex_encoded_length", "url_length"}:
        raise ValueError("Unexpected metric " + model_settings["metric"])

    if model_settings["trigger_on"] not in {"high", "low"}:
        raise ValueError("Unexpected outlier trigger condition " + model_settings["trigger_on"])

    return model_settings


def add_metric_to_batch(eval_metrics_array, aggregator_value, target_value, metrics_value, observations, doc):

    observations["target"] = target_value
    observations["aggregator"] = aggregator_value

    if aggregator_value not in eval_metrics_array.keys():
        eval_metrics_array[aggregator_value] = defaultdict(list)

    eval_metrics_array[aggregator_value]["metrics"].append(metrics_value)
    eval_metrics_array[aggregator_value]["observations"].append(observations)
    eval_metrics_array[aggregator_value]["raw_docs"].append(doc)

    return eval_metrics_array


def evaluate_model(model_name=None, model_settings=None):
    lucene_query = es.filter_by_query_string(model_settings["es_query_filter"])
    total_events = es.count_documents(lucene_query=lucene_query)

    logging.print_analysis_intro(event_type="evaluating " + model_name, total_events=total_events)
    logging.init_ticker(total_steps=total_events, desc=model_name + " - evaluating metrics model")

    eval_metrics = defaultdict()
    total_metrics_added = 0

    for doc in es.scan(lucene_query=lucene_query):
        logging.tick()
        fields = es.extract_fields_from_document(doc)

        will_process_doc = False

        try:
            target_value = helpers.utils.flatten_sentence(helpers.utils.get_dotkey_value(fields, model_settings["target"]))
            aggregator_sentences = helpers.utils.flatten_fields_into_sentences(fields=fields, sentence_format=model_settings["aggregator"])
            will_process_doc = True
        except (KeyError, TypeError):
            logging.logger.debug("skipping event which does not contain the target and aggregator fields we are processing. - [" + model_name + "]")

        if will_process_doc:
            observations = dict()
            metric = None

            # ------------------------------------
            # METRIC: Calculate numerical value
            # ------------------------------------
            # Example: numerical_value("2") => 2
            if model_settings["metric"] == "numerical_value":
                try:
                    metric = float(target_value)
                    total_metrics_added = total_metrics_added + 1
                except ValueError:
                    # number can not be casted to a Float, just continue
                    pass
            # ------------------------------------
            # METRIC: Calculate length of a string
            # ------------------------------------
            # Example: length("outliers") => 8
            if model_settings["metric"] == "length":
                metric = len(target_value)
                total_metrics_added = total_metrics_added + 1

            # -------------------------------------
            # METRIC: Calculate entropy of a string
            # -------------------------------------
            # Example: entropy("houston") => 2.5216406363433186
            if model_settings["metric"] == "entropy":
                metric = helpers.utils.shannon_entropy(target_value)
                total_metrics_added = total_metrics_added + 1

            # ------------------------------------------------------------------------------------
            # METRIC: Calculate total length of hexadecimal encoded substrings embedded in string
            # ------------------------------------------------------------------------------------
            if model_settings["metric"] == "hex_encoded_length":
                hex_encoded_words = list()
                target_value_words = re.split("[^a-fA-F0-9+]", str(target_value))  # at least length 10 to have 5 encoded characters

                for word in target_value_words:
                    if len(word) > 10 and helpers.utils.is_hex_encoded(word):  # let's match at least 5 characters, meaning 10 hex digits
                        hex_encoded_words.append(word)

                if len(hex_encoded_words) > 0:
                    sorted_hex_encoded_words = sorted(hex_encoded_words, key=len)
                    observations["max_hex_encoded_length"] = len(sorted_hex_encoded_words[-1])
                    observations["max_hex_encoded_word"] = sorted_hex_encoded_words[-1]

                    metric = len(sorted_hex_encoded_words[-1])
                else:
                    metric = 0

                total_metrics_added = total_metrics_added + 1

            # ------------------------------------------------------------------------------------
            # METRIC: Calculate total length of base64 encoded substrings embedded in string
            # ------------------------------------------------------------------------------------
            # Example: base64_encoded_length("houston we have a cHJvYmxlbQ==") => base64_decoded_string: problem, base64_encoded_length: 7
            if model_settings["metric"] == "base64_encoded_length":
                base64_decoded_words = list()

                # Split all non-Base64 characters, so we can try to convert them to Base64 decoded strings
                target_value_words = re.split("[^A-Za-z0-9+/=]", str(target_value))

                for word in target_value_words:
                    decoded_word = helpers.utils.is_base64_encoded(word)
                    if decoded_word and len(decoded_word) >= 5:  # let's match at least 5 characters, meaning 10 base64 digits
                        base64_decoded_words.append(decoded_word)

                if len(base64_decoded_words) > 0:
                    sorted_base64_decoded_words = sorted(base64_decoded_words, key=len)
                    observations["max_base64_decoded_length"] = len(sorted_base64_decoded_words[-1])
                    observations["max_base64_decoded_word"] = sorted_base64_decoded_words[-1]

                    metric = len(sorted_base64_decoded_words[-1])
                else:
                    metric = 0

                total_metrics_added = total_metrics_added + 1

            # ---------------------------------------------------------
            # METRIC: Calculate total length of URLs embedded in string
            # ---------------------------------------------------------
            # Example: url_length("why don't we go http://www.dance.com") => extracted_urls_length: 20, extracted_urls: http://www.dance.com
            if model_settings["metric"] == "url_length":
                extracted_urls_length = 0
                extracted_urls = []

                # if the target value is a list of strings, convert it into a single list of strings
                target_value_words = target_value.replace('"', ' ').split()  # splits on whitespace by default, and on quotes, since we most likely will apply this to parameter arguments

                for word in target_value_words:
                    is_url = helpers.utils.is_url(word)
                    if is_url:
                        extracted_urls_length += len(word)
                        extracted_urls.append(word)

                if extracted_urls_length > 0:
                    observations["extracted_urls_length"] = extracted_urls_length
                    observations["extracted_urls"] = ','.join(extracted_urls)

                metric = extracted_urls_length
                total_metrics_added = total_metrics_added + 1

            if metric is not None:  # explicitly check for none, since "0" can be OK as a metric!
                for aggregator_sentence in aggregator_sentences:
                    flattened_aggregator_sentence = helpers.utils.flatten_sentence(aggregator_sentence)
                    eval_metrics = add_metric_to_batch(eval_metrics, flattened_aggregator_sentence, target_value, metric, observations, doc)

        # Evaluate batch of events against the model
        last_batch = (logging.current_step == total_events)
        if last_batch or total_metrics_added >= settings.config.getint("metrics", "metrics_batch_eval_size"):
            logging.logger.info("evaluating batch of " + "{:,}".format(total_metrics_added) + " metrics")
            outliers, remaining_metrics = evaluate_batch_for_outliers(metrics=eval_metrics, model_settings=model_settings, last_batch=last_batch)

            if len(outliers) > 0:
                unique_summaries = len(set(o.get_observation("summary") for o in outliers))
                logging.logger.info("total outliers in batch processed: " + str(len(outliers)) + " [" + str(unique_summaries) + " unique summaries]")
            else:
                logging.logger.info("no outliers detected in batch")

            # Reset data structures for next batch
            eval_metrics = remaining_metrics.copy()
            total_metrics_added = 0


def run_generic_metrics_model(section_name=None, model_name=None, model_settings=None):
    # Evaluate against the model
    evaluate_model(model_name=model_name, model_settings=model_settings)


def evaluate_batch_for_outliers(metrics=None, model_settings=None, last_batch=False):
    # Initialize
    outliers = list()
    remaining_metrics = metrics.copy()

    for i, aggregator_value in enumerate(metrics):

        # Check if we have sufficient data. if not, continue. Else, evaluate for outliers.
        if len(metrics[aggregator_value]["metrics"]) < 100 and last_batch is False:
            continue
        else:
            # Remove from remaining metrics, as we will be handling it in a second
            del remaining_metrics[aggregator_value]

        # Calculate the decision frontier
        decision_frontier = helpers.utils.get_decision_frontier(model_settings["trigger_method"], metrics[aggregator_value]["metrics"], model_settings["trigger_sensitivity"], model_settings["trigger_on"])
        logging.logger.debug("using decision frontier " + str(decision_frontier) + " for aggregator " + str(aggregator_value) + " - " + model_settings["metric"])
        logging.logger.debug("example metric from batch for " + metrics[aggregator_value]["observations"][0]["target"] + ": " + str(metrics[aggregator_value]["metrics"][0]))

        # Calculate all outliers in array
        for ii, metric_value in enumerate(metrics[aggregator_value]["metrics"]):
            is_outlier = helpers.utils.is_outlier(metric_value, decision_frontier, model_settings["trigger_on"])

            if is_outlier:
                confidence = np.abs(decision_frontier - metric_value)

                # Extract fields from raw document
                fields = es.extract_fields_from_document(metrics[aggregator_value]["raw_docs"][ii])

                observations = metrics[aggregator_value]["observations"][ii]
                merged_fields_and_observations = helpers.utils.merge_two_dicts(fields, observations)
                outlier_summary = helpers.utils.replace_placeholder_fields_with_values(model_settings["outlier_summary"], merged_fields_and_observations)

                outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
                if len(outlier_assets) > 0:
                    observations["assets"] = outlier_assets

                outlier = Outlier(type=model_settings["outlier_type"], reason=model_settings["outlier_reason"], summary=outlier_summary)

                outlier.add_observation("metric", metric_value)
                outlier.add_observation("decision_frontier", decision_frontier)
                outlier.add_observation("confidence", confidence)

                for k, v in observations.items():
                    outlier.add_observation(k, v)

                outliers.append(outlier)
                es.process_outliers(doc=metrics[aggregator_value]["raw_docs"][ii], outliers=[outlier], should_notify=model_settings["should_notify"])

    return outliers, remaining_metrics
