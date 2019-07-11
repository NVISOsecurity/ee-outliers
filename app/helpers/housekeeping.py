import threading
import traceback

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

    def run(self):
        logging.logger.info('housekeeping thread #%s started' % self.ident)

        # Remove all existing whitelisted items if needed
        while not self.shutdown_flag.is_set():

            if len(self.file_mod_watcher.files_changed()) > 0:
                # reload configuration file, in case new whitelisted items were added by the analyst, they
                # should be processed!
                settings.reload_configuration_files()

                if settings.config.getboolean("general", "es_wipe_all_whitelisted_outliers"):
                    try:
                        logging.logger.info("housekeeping - going to remove all whitelisted outliers")
                        total_docs_whitelisted = es.remove_all_whitelisted_outliers()

                        if total_docs_whitelisted > 0:
                            logging.logger.info(
                                "housekeeping - total whitelisted documents cleared from outliers: " + str(
                                    total_docs_whitelisted))
                        else:
                            logging.logger.info("housekeeping - whitelist did not remove any outliers")

                    except Exception:
                        logging.logger.error("housekeeping - something went removing whitelisted outliers")
                        logging.logger.error(traceback.format_exc())

                    logging.logger.info("housekeeping - finished round of cleaning whitelisted items")

        logging.logger.info('housekeeping thread #%s stopped' % self.ident)
