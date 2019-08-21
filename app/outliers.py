import random
import time
import os
import sys
import unittest
import numpy as np

import elasticsearch.exceptions

from datetime import datetime
from croniter import croniter

import helpers.utils
from helpers.singletons import settings, logging, es
from helpers.watchers import FileModificationWatcher
from helpers.housekeeping import HousekeepingJob

from analyzers.metrics import MetricsAnalyzer
from analyzers.simplequery import SimplequeryAnalyzer
from analyzers.terms import TermsAnalyzer
from analyzers.word2vec import Word2VecAnalyzer


##############
# Entrypoint #
##############
def run_outliers():
    # Run modes

    # if running in test mode, we just want to run the tests and exit as quick as possible.
    # no need to set up other things like logging, which should happen afterwards.
    if settings.args.run_mode == "tests":
        test_filename = 'test_*.py'
        test_directory = '/app/tests/unit_tests'

        suite = unittest.TestLoader().discover(test_directory, pattern=test_filename)
        test_result = unittest.TextTestRunner(verbosity=settings.config.getint("general", "log_verbosity")).run(suite)
        sys.exit(not test_result.wasSuccessful())

    # at this point, we know we are not running tests, so we should set up logging,
    # parse the configuration files, etc.
    setup_logging()
    print_intro()

    # everything has been setup correctly, we can now start analysis in the correct run mode
    if settings.args.run_mode == "daemon":
        run_daemon_mode()

    elif settings.args.run_mode == "interactive":
        run_interactive_mode()


def setup_logging():
    if os.environ.get("SENTRY_SDK_URL"):
        import sentry_sdk
        sentry_sdk.init(os.environ.get("SENTRY_SDK_URL"))

    # Configuration for which we need access to both settings and logging singletons should happen here
    logging.verbosity = settings.config.getint("general", "log_verbosity")
    logging.logger.setLevel(settings.config.get("general", "log_level"))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = settings.config.get("machine_learning", "tensorflow_log_level")

    # Log Handlers
    log_file = settings.config.get("general", "log_file")

    if os.path.exists(os.path.dirname(log_file)):
        logging.add_file_handler(log_file)
    else:
        logging.logger.warning("log directory for log file %s does not exist, check your settings! Only logging " +
                               "to stdout.", log_file)


def print_intro():
    logging.logger.info("outliers.py - version 0.2.2 - contact: research@nviso.be")
    logging.logger.info("run mode: " + settings.args.run_mode)

    logging.print_generic_intro("initializing")
    logging.logger.info("loaded " + str(len(settings.loaded_config_paths)) + " configuration files")

    if settings.failed_config_paths:
        logging.logger.error("failed to load " + str(len(settings.failed_config_paths)) + " configuration files that " +
                             "will be ignored")

        for failed_config_path in settings.failed_config_paths:
            logging.logger.error("\t+ failed to load configuration file " + str(failed_config_path))

    if settings.failing_regular_expressions:
        logging.logger.error("failed to parse " + str(len(settings.failing_regular_expressions)) + " regular " +
                             "expressions in whitelist that will be ignored")

        for failed_regular_expression in settings.failing_regular_expressions:
            logging.logger.error("\t+ failed to parse regular expression " + str(failed_regular_expression))


def run_daemon_mode():
    # In daemon mode, we also want to monitor the configuration file for changes.
    # In case of a change, we need to make sure that we are using this new configuration file
    for config_file in settings.args.config:
        logging.logger.info("monitoring configuration file " + config_file + " for changes")

    file_mod_watcher = FileModificationWatcher()
    file_mod_watcher.add_files(settings.args.config)

    # Initialize Elasticsearch connection
    es.try_to_init_connection()

    # Create housekeeping job, don't start it yet
    housekeeping_job = HousekeepingJob()

    first_run = True
    run_succeeded_without_errors = None

    while True:
        next_run = None
        should_schedule_next_run = False

        while (next_run is None or datetime.now() < next_run) and first_run is False and \
                run_succeeded_without_errors is True:
            if next_run is None:
                should_schedule_next_run = True

            # Check for configuration file changes and load them in case it's needed
            if file_mod_watcher.files_changed():
                logging.logger.info("configuration file changed, reloading")
                settings.process_configuration_files()
                should_schedule_next_run = True

            if should_schedule_next_run:
                next_run = croniter(settings.config.get("daemon", "schedule"), datetime.now()).get_next(datetime)
                logging.logger.info("next run scheduled on {0:%Y-%m-%d %H:%M:%S}".format(next_run))
                should_schedule_next_run = False

            time.sleep(5)

        settings.process_configuration_files()  # Refresh settings

        if first_run:
            first_run = False
            logging.logger.info("first run, so we will start immediately - after this, we will respect the cron " +
                                "schedule defined in the configuration file")

            # Wipe all existing outliers if needed
            if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
                logging.logger.info("wiping all existing outliers on first run")
                es.remove_all_outliers()
        else:
            # Make sure we are still connected to Elasticsearch before analyzing, in case something went wrong with
            # the connection in between runs
            es.try_to_init_connection()

        # Make sure housekeeping is up and running
        if not housekeeping_job.is_alive():
            housekeeping_job.start()

        # Perform analysis
        logging.print_generic_intro("starting outlier detection")
        analyzed_models = perform_analysis()
        print_analysis_summary(analyzed_models)

        errored_models = [analyzer for analyzer in analyzed_models if analyzer.unknown_error_analysis]

        # Check the result of the analysis
        if errored_models:
            run_succeeded_without_errors = False
            logging.logger.warning("ran into errors while analyzing use cases - not going to wait for the cron " +
                                   "schedule, we just start analyzing again after sleeping for a minute first")
            time.sleep(60)
        else:
            run_succeeded_without_errors = True

        logging.print_generic_intro("finished performing outlier detection")


def run_interactive_mode():
    es.try_to_init_connection()

    if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
        es.remove_all_outliers()

    # Make sure housekeeping is up and running
    housekeeping_job = HousekeepingJob()
    housekeeping_job.start()

    try:
        analyzed_models = perform_analysis()
        print_analysis_summary(analyzed_models)
    except KeyboardInterrupt:
        logging.logger.info("keyboard interrupt received, stopping housekeeping thread")
    except Exception:
        logging.logger.error("error running outliers in interactive mode", exc_info=True)
    finally:
        logging.logger.info("asking housekeeping jobs to shutdown after finishing")
        housekeeping_job.shutdown_flag.set()
        housekeeping_job.join()

    logging.logger.info("finished performing outlier detection")


def perform_analysis():
    """ The entrypoint for analysis """
    analyzers = list()

    for config_section_name in settings.config.sections():
        _analyzer = None
        try:
            if config_section_name.startswith("simplequery_"):
                _analyzer = SimplequeryAnalyzer(config_section_name=config_section_name)
                analyzers.append(_analyzer)

            elif config_section_name.startswith("metrics_"):
                _analyzer = MetricsAnalyzer(config_section_name=config_section_name)
                analyzers.append(_analyzer)

            elif config_section_name.startswith("terms_"):
                _analyzer = TermsAnalyzer(config_section_name=config_section_name)
                analyzers.append(_analyzer)

            elif config_section_name.startswith("beaconing_"):
                logging.logger.error("use of the beaconing model is deprecated, please use the terms model using " +
                                     "coeff_of_variation trigger method to convert use case " + config_section_name)

            elif config_section_name.startswith("word2vec_"):
                _analyzer = Word2VecAnalyzer(config_section_name=config_section_name)
                analyzers.append(_analyzer)
        except Exception:
            logging.logger.error("error while initializing analyzer " + config_section_name, exc_info=True)

    analyzers_to_evaluate = list()

    for analyzer in analyzers:
        if analyzer.should_run_model or analyzer.should_test_model:
            analyzers_to_evaluate.append(analyzer)

    random.shuffle(analyzers_to_evaluate)

    for index, analyzer in enumerate(analyzers_to_evaluate):
        if analyzer.configuration_parsing_error:
            continue

        try:
            analyzer.analysis_start_time = datetime.today().timestamp()
            analyzer.evaluate_model()
            analyzer.analysis_end_time = datetime.today().timestamp()
            analyzer.completed_analysis = True

            logging.logger.info("finished processing use case - " + str(index + 1) + "/" +
                                str(len(analyzers_to_evaluate)) + " [" + '{:.2f}'
                                .format(round((index + 1) / float(len(analyzers_to_evaluate)) * 100, 2)) +
                                "% done" + "]")
        except elasticsearch.exceptions.NotFoundError:
            analyzer.index_not_found_analysis = True
            logging.logger.warning("index %s does not exist, skipping use case" % analyzer.es_index)
        except Exception:
            analyzer.unknown_error_analysis = True
            logging.logger.error("error while analyzing use case", exc_info=True)
        finally:
            es.flush_bulk_actions(refresh=True)

    return analyzers_to_evaluate


def print_analysis_summary(analyzed_models):
    logging.logger.info("")
    logging.logger.info("============================")
    logging.logger.info("===== analysis summary =====")
    logging.logger.info("============================")

    completed_models = [analyzer for analyzer in analyzed_models if analyzer.completed_analysis]
    completed_models_with_events = [analyzer for analyzer in analyzed_models
                                    if (analyzer.completed_analysis and analyzer.total_events > 0)]

    no_index_models = [analyzer for analyzer in analyzed_models if analyzer.index_not_found_analysis]
    unknown_error_models = [analyzer for analyzer in analyzed_models if analyzer.unknown_error_analysis]
    configuration_parsing_error_models = [analyzer for analyzer in analyzed_models
                                          if analyzer.configuration_parsing_error]

    total_models_processed = len(completed_models) + len(no_index_models) + len(unknown_error_models)
    total_outliers_detected = sum([analyzer.total_outliers for analyzer in analyzed_models])
    total_outliers_whitelisted = sum([analyzer.nr_whitelisted_elements for analyzer in analyzed_models])
    logging.logger.info("total use cases processed: %i", total_models_processed)
    logging.logger.info("total outliers detected: " + "{:,}".format(total_outliers_detected))
    logging.logger.info("total whitelisted outliers: " + "{:,}".format(total_outliers_whitelisted))
    logging.logger.info("")
    logging.logger.info("succesfully analyzed use cases: %i", len(completed_models))
    logging.logger.info("succesfully analyzed use cases without events: %i",
                        len(completed_models) - len(completed_models_with_events))
    logging.logger.info("succesfully analyzed use cases with events: %i", len(completed_models_with_events))
    logging.logger.info("")
    logging.logger.info("use cases skipped because of missing index: %i", len(no_index_models))
    logging.logger.info("use cases skipped because of incorrect configuration: %i",
                        len(configuration_parsing_error_models))
    logging.logger.info("use cases that caused an error: %i", len(unknown_error_models))
    logging.logger.info("")

    analysis_times = [_.analysis_time_seconds for _ in completed_models_with_events]
    completed_models_with_events.sort(key=lambda _: _.analysis_time_seconds, reverse=True)

    if completed_models_with_events:
        logging.logger.info("total analysis time: " +
                            helpers.utils.seconds_to_pretty_str(seconds=round(float(np.sum(analysis_times)))))
        logging.logger.info("average analysis time: " +
                            helpers.utils.seconds_to_pretty_str(seconds=round(np.average(analysis_times))))

        # print most time consuming use cases
        logging.logger.info("")
        logging.logger.info("most time consuming use cases (top 10):")
        completed_models_with_events_taking_most_time = completed_models_with_events[:10]

        for model in completed_models_with_events_taking_most_time:
            logging.logger.info("\t+ " + model.config_section_name + " - " +
                                "{:,}".format(model.total_events) + " events - " +
                                "{:,}".format(model.total_outliers) + " outliers - " +
                                helpers.utils.seconds_to_pretty_str(round(model.analysis_time_seconds)))

    if configuration_parsing_error_models:
        logging.logger.info("")
        logging.logger.info("models for which the configuration parsing failed:")

        for model in configuration_parsing_error_models:
            logging.logger.info("\t+ " + model.config_section_name)

    if unknown_error_models:
        logging.logger.info("")
        logging.logger.info("models for which an unexpected error was encountered:")

        for model in unknown_error_models:
            logging.logger.info("\t+ " + model.config_section_name)

    if not analyzed_models:
        logging.logger.warning("no use cases were analyzed. are you sure your configuration file contains use " +
                               "cases, which are enabled?")

    logging.logger.info("============================")


if __name__ == '__main__':
    run_outliers()
