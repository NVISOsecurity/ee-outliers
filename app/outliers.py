import time

import traceback
from datetime import datetime
from croniter import croniter

from helpers.singletons import settings
from helpers.singletons import logging, es
from helpers.utils import FileModificationWatcher
from helpers.housekeeping import HousekeepingJob

import os
import dateutil.parser

from analyzers import metrics_generic
from analyzers import simplequery_generic
from analyzers import terms_generic
from analyzers import svm_generic
from analyzers import word2vec_generic
from analyzers import test_generic
from analyzers import beaconing_generic

##############
# Entrypoint #
##############
# Configuration for which we need access to both settings and logging singletons should happen here
logging.verbosity = settings.config.getint("general", "log_verbosity")
logging.logger.setLevel(settings.config.get("general", "log_level"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = settings.config.get("machine_learning", "tensorflow_log_level")

# Log Handlers
log_file = settings.config.get("general", "log_file")

if os.path.exists(os.path.dirname(log_file)):
    logging.add_file_handler(log_file)
else:
    logging.logger.warning("log directory for log file " + log_file + " does not exist, check your settings! Only logging to stdout.")

logging.logger.info("outliers.py started - contact: research@nviso.be")
logging.logger.info("run mode: " + settings.args.run_mode)
logging.print_generic_intro("initializing")


def perform_analysis():
    test_generic.perform_analysis()
    beaconing_generic.perform_analysis()
    metrics_generic.perform_analysis()
    simplequery_generic.perform_analysis()
    terms_generic.perform_analysis()
    svm_generic.perform_analysis()
    word2vec_generic.perform_analysis()


# Prepare log messages
search_start_range_printable = dateutil.parser.parse(settings.search_range_start).strftime('%Y-%m-%d %H:%M:%S')
search_end_range_printable = dateutil.parser.parse(settings.search_range_end).strftime('%Y-%m-%d %H:%M:%S')
time_window_info = "processing events between " + search_start_range_printable + " and " + search_end_range_printable

# Run modes
if settings.args.run_mode == "daemon":
    # In daemon mode, we also want to monitor the configuration file for changes.
    # In case of a change, we need to make sure that we are using this new configuration file
    logging.logger.info("monitoring configuration file " + settings.args.config + " for changes")
    file_mod_watcher = FileModificationWatcher()
    file_mod_watcher.add_files([settings.args.config])

    num_runs = 0
    while True:
        num_runs += 1
        next_run = None
        should_schedule_next_run = False

        while next_run is None or datetime.now() < next_run:
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

        settings.process_arguments()  # Refresh settings
        es.init_connection()

        if settings.config.getboolean("general", "es_wipe_all_existing_outliers") and num_runs == 1:
            logging.logger.info("wiping all existing outliers on first run")
            es.remove_all_outliers()

        logging.logger.info(time_window_info)

        # We place all of this in a try catch-all, so that any errors thrown by the analysis (timeouts, errors) won't make the daemon loop stop
        housekeeping_job = HousekeepingJob()
        housekeeping_job.start()

        try:
            perform_analysis()
        except Exception:
            logging.logger.error(traceback.format_exc())
        finally:
            housekeeping_job.shutdown_flag.set()
            housekeeping_job.join()

        logging.logger.info("finished performing outlier detection")

if settings.args.run_mode == "interactive":
    es.init_connection()

    if settings.config.getboolean("general", "es_wipe_all_existing_outliers"):
        logging.logger.info("wiping all existing outliers")
        es.remove_all_outliers()

    logging.logger.info(time_window_info)

    housekeeping_job = HousekeepingJob()
    housekeeping_job.start()

    try:
        perform_analysis()
    except KeyboardInterrupt:
        logging.logger.info("keyboard interrupt received, stopping housekeeping thread")
    except Exception as e:
        logging.logger.error(traceback.format_exc())
    finally:
        housekeeping_job.shutdown_flag.set()
        housekeeping_job.join()

    logging.logger.info("finished performing outlier detection")

if settings.args.run_mode == "tests":
    import unittest

    test_filename = 'test_*.py'
    test_directory = '/app/tests/unit_tests'

    suite = unittest.TestLoader().discover(test_directory, pattern=test_filename)
    unittest.TextTestRunner(verbosity=settings.config.getint("general", "log_verbosity")).run(suite)
    logging.logger.info("finished running tests")

