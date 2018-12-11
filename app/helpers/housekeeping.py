import time
import threading
from helpers.singletons import logging, settings, es


class HousekeepingJob(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()

    def run(self):
        logging.logger.info('housekeeping thread #%s started' % self.ident)
        housekeeping_interval_seconds = settings.config.getint("general", "housekeeping_interval_seconds")

        # Remove all existing whitelisted items if needed
        while not self.shutdown_flag.is_set():
            if settings.config.getboolean("general", "es_wipe_all_whitelisted_outliers"):
                settings.reload_configuration_file()  # reload configuration file, in case new whitelisted items were added by the analyst, they should be processed!
                total_docs_whitelisted = es.remove_all_whitelisted_outliers()

                if total_docs_whitelisted > 0:
                    logging.logger.info("housekeeping - total whitelisted documents cleared from outliers: " + str(total_docs_whitelisted))
                else:
                    logging.logger.info("housekeeping - whitelist did not remove any outliers")

            for i in range(housekeeping_interval_seconds):
                if not self.shutdown_flag.is_set():
                    time.sleep(1)
                else:
                    break

        logging.logger.info('housekeeping thread #%s stopped' % self.ident)