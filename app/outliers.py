import random
import time
import os
import sys
import unittest

import traceback
from datetime import datetime
from croniter import croniter

from helpers.singletons import settings, logging, es
from helpers.watchers import FileModificationWatcher
from helpers.housekeeping import HousekeepingJob

from analyzers.metrics import MetricsAnalyzer
from analyzers.simplequery import SimplequeryAnalyzer
from analyzers.terms import TermsAnalyzer
from analyzers.beaconing import BeaconingAnalyzer
from analyzers.word2vec import Word2VecAnalyzer
from analyzers.autoencoder import AutoencoderAnalyzer

##############
# Entrypoint #
##############
if os.environ.get("SENTRY_SDK_URL"):
    import sentry_sdk
    sentry_sdk.init(os.environ.get("SENTRY_SDK_URL"))

if settings.args.run_mode == "tests":

    test_filename = 'test_*.py'
    test_directory = '/app/tests/unit_tests'

    suite = unittest.TestLoader().discover(test_directory, pattern=test_filename)
    unittest.TextTestRunner(verbosity=settings.config.getint("general", "log_verbosity")).run(suite)
    sys.exit()

# Configuration for which we need access to both settings and logging singletons should happen here
logging.verbosity = settings.config.getint("general", "log_verbosity")
logging.logger.setLevel(settings.config.get("general", "log_level"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = settings.config.get("machine_learning", "tensorflow_log_level")

# Log Handlers
LOG_FILE = settings.config.get("general", "log_file")

if os.path.exists(os.path.dirname(LOG_FILE)):
    logging.add_file_handler(LOG_FILE)
else:
    logging.logger.warning("log directory for log file %s does not exist, check your settings! Only logging to stdout.", LOG_FILE)

logging.logger.info("outliers.py started - contact: research@nviso.be")
logging.logger.info("run mode: " + settings.args.run_mode)

logging.print_generic_intro("initializing")
logging.logger.info("loaded " + str(len(settings.loaded_config_paths)) + " configuration files")

if settings.failed_config_paths:
    logging.logger.warning("failed to load " + str(len(settings.failed_config_paths)) + " configuration files")

    for failed_config_path in settings.failed_config_paths:
        logging.logger.warning("failed to load " + str(failed_config_path))


def perform_analysis():
    """ The entrypoint for analysis """
    analyzers = list()

    for config_section_name in settings.config.sections():
        try:
            if config_section_name.startswith("simplequery_"):
                simplequery_analyzer = SimplequeryAnalyzer(config_section_name=config_section_name)
                analyzers.append(simplequery_analyzer)

            if config_section_name.startswith("metrics_"):
                metrics_analyzer = MetricsAnalyzer(config_section_name=config_section_name)
                analyzers.append(metrics_analyzer)

            if config_section_name.startswith("terms_"):
                terms_analyzer = TermsAnalyzer(config_section_name=config_section_name)
                analyzers.append(terms_analyzer)

            if config_section_name.startswith("beaconing_"):
                beaconing_analyzer = BeaconingAnalyzer(config_section_name=config_section_name)
                analyzers.append(beaconing_analyzer)

            if config_section_name.startswith("word2vec_"):
                word2vec_analyzer = Word2VecAnalyzer(config_section_name=config_section_name)
                analyzers.append(word2vec_analyzer)

            if config_section_name.startswith("autoencoder_"):
                autoencoder_analyzer = AutoencoderAnalyzer(config_section_name=config_section_name)
                analyzers.append(autoencoder_analyzer)

        except Exception:
            logging.logger.error(traceback.format_exc())

    analyzers_to_evaluate = list()

    for idx, analyzer in enumerate(analyzers):
        if analyzer.should_run_model or analyzer.should_test_model:
            analyzers_to_evaluate.append(analyzer)

    random.shuffle(analyzers_to_evaluate)
    analyzed_models = 0
    for analyzer in analyzers_to_evaluate:
        try:
            analyzer.evaluate_model()
            analyzed_models = analyzed_models + 1
            logging.logger.info("finished processing use case - " + str(analyzed_models) + "/" + str(len(analyzers_to_evaluate)) + " [" + '{:.2f}'.format(round(float(analyzed_models) / float(len(analyzers_to_evaluate)) * 100, 2)) + "% done" + "]")
        except Exception:
            logging.logger.error(traceback.format_exc())
        finally:
            es.flush_bulk_actions(refresh=True)

    if analyzed_models == 0:
        logging.logger.warning("no use cases were analyzed. are you sure your configuration file contains use cases, which are enabled?")

    return analyzed_models == len(analyzers_to_evaluate)


# Run modes
if settings.args.run_mode == "daemon":
    # In daemon mode, we also want to monitor the configuration file for changes.
    # In case of a change, we need to make sure that we are using this new configuration file
    for config_file in settings.args.config:
        logging.logger.info("monitoring configuration file " + config_file + " for changes")

    file_mod_watcher = FileModificationWatcher()
    file_mod_watcher.add_files(settings.args.config)

    # Initialize Elasticsearch connection
    es.init_connection()

    # Start housekeeping activities
    housekeeping_job = HousekeepingJob()
    housekeeping_job.start()

    num_runs = 0
    first_run = True
    run_succeeded_without_errors = None

    while True:
        num_runs += 1
        next_run = None
        should_schedule_next_run = False

        while (next_run is None or datetime.now() < next_run) and first_run is False and run_succeeded_without_errors is True:
            if next_run is None:
                should_schedule_next_run = True

            # Check for configuration file changes and load them in case it's needed
            if file_mod_watcher.files_changed():
                logging.logger.info("configuration file changed, reloading")
                settings.process_arguments()
                should_schedule_next_run = True

            if should_schedule_next_run:
                next_run = croniter(settings.config.get("daemon", "schedule"), datetime.now()).get_next(datetime)
                logging.logger.info("next run scheduled on {0:%Y-%m-%d %H:%M:%S}".format(next_run))
                should_schedule_next_run = False

            time.sleep(5)

        if first_run:
            first_run = False
            logging.logger.info("first run, so we will start immediately - after this, we will respect the cron schedule defined in the configuration file")

        settings.process_arguments()  # Refresh settings

        if settings.config.getboolean("general", "es_wipe_all_existing_outliers") and num_runs == 1:
            logging.logger.info("wiping all existing outliers on first run")
            es.remove_all_outliers()

        logging.logger.info(settings.get_time_window_info())

        # Make sure we are connected to Elasticsearch before analyzing, in case something went wrong with the connection in between runs
        es.init_connection()

        # Make sure housekeeping is still up and running
        if not housekeeping_job.is_alive():
            housekeeping_job = HousekeepingJob()
            housekeeping_job.start()

        # Perform analysis
        logging.print_generic_intro("starting outlier detection")
        run_succeeded_without_errors = perform_analysis()
        if not run_succeeded_without_errors:
            logging.logger.warning("ran into errors while analyzing use cases - not going to wait for the cron schedule, we just start analyzing again")
        else:
            logging.logger.info("no errors encountered while analyzing use cases")

        logging.print_generic_intro("finished performing outlier detection")


if settings.args.run_mode == "interactive":
    es.init_connection()

    if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
        es.remove_all_outliers()

    logging.logger.info(settings.get_time_window_info())

    housekeeping_job = HousekeepingJob()
    housekeeping_job.start()

    try:
        perform_analysis()
    except KeyboardInterrupt:
        logging.logger.info("keyboard interrupt received, stopping housekeeping thread")
    except Exception:
        logging.logger.error(traceback.format_exc())
    finally:
        logging.logger.info("asking housekeeping jobs to shutdown after finishing")
        housekeeping_job.shutdown_flag.set()
        housekeeping_job.join()

    logging.logger.info("finished performing outlier detection")
