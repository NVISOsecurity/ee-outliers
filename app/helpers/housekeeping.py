import threading

from helpers.singletons import logging, settings, es
from helpers.watchers import FileModificationWatcher


class HousekeepingJob(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

        self.file_mod_watcher = FileModificationWatcher()
        self.file_mod_watcher.add_files(settings.args.config)

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()

        self.dict_analyzer = dict()

    def run(self):
        """
        Task to launch the housekeeping
        """
        logging.logger.info('housekeeping thread #%s started' % self.ident)

        # Remove all existing whitelisted items if needed
        while not self.shutdown_flag.is_set():
            if not self.shutdown_flag.wait(5):  # Return True if flag was set by outliers
                self.execute_housekeeping()

        logging.logger.info('housekeeping thread #%s stopped' % self.ident)

    def execute_housekeeping(self):
        """
        Execute the housekeeping
        """
        if self.file_mod_watcher.files_changed():
            # reload configuration file, in case new whitelisted items were added by the analyst, they
            # should be processed!
            settings.process_configuration_files()

            logging.logger.info("housekeeping - changes detected, process again housekeeping")
            self.remove_all_whitelisted_outliers()

    def stop_housekeeping(self):
        self.shutdown_flag.set()
        self.join()

    def remove_all_whitelisted_outliers(self):
        """
        Try to remove all whitelist outliers that are already in Elasticsearch
        """
        if settings.config.getboolean("general", "es_wipe_all_whitelisted_outliers"):
            try:
                logging.logger.info("housekeeping - going to remove all whitelisted outliers")
                total_docs_whitelisted = es.remove_all_whitelisted_outliers(self.dict_analyzer)

                if total_docs_whitelisted > 0:
                    logging.logger.info(
                        "housekeeping - total whitelisted documents cleared from outliers: " +
                        "{:,}".format(total_docs_whitelisted))
                else:
                    logging.logger.info("housekeeping - whitelist did not remove any outliers")

            except Exception:
                logging.logger.error("housekeeping - something went wrong removing whitelisted outliers", exc_info=True)

            logging.logger.info("housekeeping - finished round of cleaning whitelisted items")
