from datetime import datetime

import random
import time
import os
import sys
import unittest
import glob

from croniter import croniter

import numpy as np
import elasticsearch.exceptions

import helpers.utils
from helpers.singletons import settings, logging, es
from helpers.watchers import FileModificationWatcher
from helpers.housekeeping import HousekeepingJob
from helpers.analyzerfactory import AnalyzerFactory


def run_outliers():
    """
    Entrypoint into ee-outliers.
    From here we start using the appropriate run mode.
    """
    # If running in test mode, we just want to run the tests and exit as quick as possible.
    # no need to set up other things like logging, which should happen afterwards.
    if settings.args.run_mode == "tests":
        test_filename = 'test_*.py'
        test_directory = '/app/tests/unit_tests'

        suite = unittest.TestLoader().discover(test_directory, pattern=test_filename)
        test_result = unittest.TextTestRunner(verbosity=settings.config.getint("general", "log_verbosity")).run(suite)
        sys.exit(not test_result.wasSuccessful())

    # At this point, we know we are not running tests, so we should set up logging,
    # parse the configuration files, etc.
    setup_logging()
    print_intro()

    # Check no duplicate in settings
    error = settings.check_no_duplicate_key()
    if error is not None:
        logging.logger.warning(
            'duplicate value detected in configuration file. Only the last specified value will be used: %s', error)

    # Everything has been setup correctly, we can now start analysis in the correct run mode
    if settings.args.run_mode == "daemon":
        run_daemon_mode()

    elif settings.args.run_mode == "interactive":
        run_interactive_mode()


def setup_logging():
    """
    Setup the correct logging verbosity and file handlers.
    We also add a logger for Sentry in case it has been set in the environment.
    Sentry allows us to centrally collect error logging during development.
    """
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
        logging.logger.warning(
            'log directory for log file %s does not exist, check your settings! Only logging to stdout', log_file)


def print_intro():
    """
    Print the banner information including version, loaded configuration files and any parsing errors
    that might have occurred when loading them.
    """
    logging.logger.info("outliers.py - version %s - contact: research@nviso.be", helpers.utils.get_version())
    logging.logger.info("run mode: %s", settings.args.run_mode)

    logging.print_generic_intro("initializing")
    logging.logger.info("loaded %d configuration files", len(settings.loaded_config_paths))

    if settings.failed_config_paths:
        logging.logger.error("failed to load %d configuration files that will be "
                             "ignored", len(settings.failed_config_paths))

        for failed_config_path in settings.failed_config_paths:
            logging.logger.error("\t+ failed to load configuration file %s", failed_config_path)

    if settings.failing_regular_expressions:
        logging.logger.error("failed to parse %d regular expressions in whitelist that "
                             "will be ignored", len(settings.failing_regular_expressions))

        for failed_regular_expression in settings.failing_regular_expressions:
            logging.logger.error("\t+ failed to parse regular expression %s", failed_regular_expression)


# pylint: disable=too-many-branches
def run_daemon_mode():
    """
    Run outliers in daemon mode.
    In this mode, outliers will continue running based on the cron scheduled defined in the configuration file.
    """

    # In daemon mode, we also want to monitor the configuration file for changes.
    # In case of a change, we need to make sure that we are using this new configuration file
    for config_file in settings.args.config:
        logging.logger.info("monitoring configuration file %s for changes", config_file)

    # Monitor configuration files for potential changes
    file_mod_watcher = FileModificationWatcher()
    file_mod_watcher.add_files(settings.args.config)

    # Initialize Elasticsearch connection
    while not es.init_connection():
        time.sleep(60)

    # Create housekeeping job, don't start it yet
    housekeeping_job = HousekeepingJob()

    first_run = True
    run_succeeded_without_errors = None

    # The daemon should run forever, until the user kills it
    while True:
        next_run = None
        should_schedule_next_run = False

        # This loop will run for as long we don't need to perform an analysis
        while (next_run is None or datetime.now() < next_run) and first_run is False and \
                run_succeeded_without_errors is True:

            # Check if we already know when to perform the analysis next; if not, we need to schedule it
            if next_run is None:
                should_schedule_next_run = True

            # Check for configuration file changes and load them in case it's needed
            if file_mod_watcher.files_changed():
                logging.logger.info("configuration file changed, reloading")
                settings.process_configuration_files()
                should_schedule_next_run = True

            # Schedule a next rune based on the cron schedule defined in the configuration file
            if should_schedule_next_run:
                next_run = croniter(settings.config.get("daemon", "schedule"), datetime.now()).get_next(datetime)
                logging.logger.info("next run scheduled on {0:%Y-%m-%d %H:%M:%S}".format(next_run))
                should_schedule_next_run = False

            # Wait 5 seconds before checking the cron schedule again
            time.sleep(5)

        # Refresh settings in case the cron has changed for example
        settings.process_configuration_files()

        # On the first run, we might have to wipe all the existing outliers if this is set in the configuration file
        if first_run:
            first_run = False
            logging.logger.info("first run, so we will start immediately - after this, we will respect the cron "
                                "schedule defined in the configuration file")

            # Wipe all existing outliers if needed
            if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
                logging.logger.info("wiping all existing outliers on first run")
                es.remove_all_outliers()

        # Make sure we are still connected to Elasticsearch before analyzing, in case something went wrong with
        # the connection in between runs
        while not es.init_connection():
            time.sleep(60)

        # Make sure housekeeping is up and running
        if not housekeeping_job.is_alive():
            housekeeping_job.start()

        # Perform analysis and print the analysis summary at the end
        logging.print_generic_intro("starting outlier detection")
        analyzed_models = perform_analysis(housekeeping_job)
        print_analysis_summary(analyzed_models)

        errored_models = [analyzer for analyzer in analyzed_models if analyzer.unknown_error_analysis]

        # Check the result of the analysis. In case an error occured, we want to re-run right away (after a minute)
        if errored_models:
            run_succeeded_without_errors = False
            logging.logger.warning("ran into errors while analyzing use cases - not going to wait for the cron "
                                   "schedule, we just start analyzing again after sleeping for a minute first")
            time.sleep(60)
        else:
            run_succeeded_without_errors = True

        logging.print_generic_intro("finished performing outlier detection")


def run_interactive_mode():
    """
    Run outliers in interactive mode.
    In this mode, outliers will run onces and then stop.
    """

    # Initialize Elasticsearch connection
    while not es.init_connection():
        time.sleep(60)

    if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
        es.remove_all_outliers()

    # Make sure housekeeping is up and running
    housekeeping_job = HousekeepingJob()
    housekeeping_job.start()

    # The difference with daemon mode is that in interactive mode, we want to allow the user to stop execution on the
    # command line, interactively.
    try:
        analyzed_models = perform_analysis(housekeeping_job)
        print_analysis_summary(analyzed_models)
    except KeyboardInterrupt:
        logging.logger.info("keyboard interrupt received, stopping housekeeping thread")
    finally:
        logging.logger.info("asking housekeeping jobs to shutdown after finishing")
        housekeeping_job.stop_housekeeping()

    logging.logger.info("finished performing outlier detection")


def load_analyzers():
    analyzers = list()

    for use_case_arg in settings.args.use_cases:
        for use_case_file in glob.glob(use_case_arg, recursive=True):
            if not os.path.isdir(use_case_file):
                logging.logger.debug("Loading use case %s" % use_case_file)
                try:
                    analyzers.append(AnalyzerFactory.create(use_case_file))
                except ValueError as e:
                    logging.logger.error("An error occured when loading %s: %s" % (use_case_file, str(e)))

    return analyzers


# pylint: disable=too-many-branches
def perform_analysis(housekeeping_job):
    """ The entrypoint for analysis
    :return: List of analyzers that have been processed and analyzed
    """
    analyzers = load_analyzers()
    housekeeping_job.update_analyzer_list(analyzers)

    # In case the created analyzer is activated in test or run mode, add it to the list of analyzers to evaluate
    analyzers_to_evaluate = list()
    for analyzer in analyzers:
        # Analyzers that produced an error during configuration parsing should not be processed
        if analyzer.configuration_parsing_error:
            continue

        if analyzer.model_settings["run_model"] or analyzer.model_settings["test_model"]:
            analyzers_to_evaluate.append(analyzer)

    # In case a single analyzer is causing issues (for example taking up too much time & resources), then a naive
    # shuffle will prevent this analyzer from blocking all the analyzers from running that come after it.
    random.shuffle(analyzers_to_evaluate)

    # Now it's time actually evaluate all the models. We also make sure to add some information that will be useful
    # in the summary presented to the user at the end of running all the models.
    for index, analyzer in enumerate(analyzers_to_evaluate):
        try:
            analyzer.analysis_start_time = datetime.today().timestamp()
            analyzer.evaluate_model()
            analyzer.analysis_end_time = datetime.today().timestamp()
            analyzer.completed_analysis = True
            es.flush_bulk_actions()

            logging.logger.info("finished processing use case - %d / %d [%s%% done]", index + 1,
                                len(analyzers_to_evaluate),
                                '{:.2f}'.format(round((index + 1) / float(len(analyzers_to_evaluate)) * 100, 2)))
        except elasticsearch.exceptions.NotFoundError:
            analyzer.index_not_found_analysis = True
            logging.logger.warning("index %s does not exist, skipping use case", analyzer.model_settings["es_index"])
        except elasticsearch.helpers.BulkIndexError as e:
            analyzer.unknown_error_analysis = True
            logging.logger.error(f"BulkIndexError while analyzing use case: {e.args[0]}", exc_info=False)
            logging.logger.debug("Full stack trace and error message of BulkIndexError", exc_info=True)
        except Exception:  # pylint: disable=broad-except
            analyzer.unknown_error_analysis = True
            logging.logger.error("error while analyzing use case", exc_info=True)

    return analyzers_to_evaluate


def print_analysis_summary(analyzed_models):
    """
    Print a summary of the analysis
    :param analyzed_models: processed analyzers that should be summarized
    """
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
    logging.logger.info("total outliers detected: %s", "{:,}".format(total_outliers_detected))
    logging.logger.info("total whitelisted outliers: %s", "{:,}".format(total_outliers_whitelisted))
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
        logging.logger.info("total analysis time: %s",
                            helpers.utils.seconds_to_pretty_str(seconds=round(float(np.sum(analysis_times)))))
        logging.logger.info("average analysis time: %s",
                            helpers.utils.seconds_to_pretty_str(seconds=round(np.average(analysis_times))))

        # print most time consuming use cases
        logging.logger.info("")
        logging.logger.info("most time consuming use cases (top 10):")
        completed_models_with_events_taking_most_time = completed_models_with_events[:10]

        for model in completed_models_with_events_taking_most_time:
            logging.logger.info("\t+ %s - %s events - %s outliers - %s", model.model_type + "_" + model.model_name,
                                "{:,}".format(model.total_events),
                                "{:,}".format(model.total_outliers),
                                helpers.utils.seconds_to_pretty_str(round(model.analysis_time_seconds)))

    if configuration_parsing_error_models:
        logging.logger.info("")
        logging.logger.info("models for which the configuration parsing failed:")

        for model in configuration_parsing_error_models:
            logging.logger.info("\t+ %s", model.model_type + "_" + model.model_name)

    if unknown_error_models:
        logging.logger.info("")
        logging.logger.info("models for which an unexpected error was encountered:")

        for model in unknown_error_models:
            logging.logger.info("\t+ %s", model.model_type + "_" + model.model_name)

    if not analyzed_models:
        logging.logger.warning("no use cases were analyzed. are you sure your configuration file contains use "
                               "cases, which are enabled?")

    logging.logger.info("============================")


if __name__ == '__main__':
    run_outliers()
