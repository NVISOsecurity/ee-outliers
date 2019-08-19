import threading

from helpers.singletons import logging, settings, es
from helpers.watchers import FileModificationWatcher

from typing import Dict, Any


class HousekeepingJob(threading.Thread):

    def __init__(self) -> None:
        threading.Thread.__init__(self)

        self.file_mod_watcher: FileModificationWatcher = FileModificationWatcher()
        self.file_mod_watcher.add_files(settings.args.config)

        self.last_config_parameters: Dict[str, Any] = self._get_config_whitelist_parameters()

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag: threading.Event = threading.Event()

    @staticmethod
    def _get_config_whitelist_parameters() -> Dict[str, Any]:
        """
        Get parameters linked to the whitelist

        :return: dictionary with whitelist parameters
        """
        return {
            'whitelist_literals': settings.config.items("whitelist_literals"),
            'whitelist_regexps': settings.config.items("whitelist_regexps"),
            'es_wipe_all_whitelisted_outliers': settings.config.getboolean("general",
                                                                           "es_wipe_all_whitelisted_outliers")
        }

    def run(self) -> None:
        """
        Task to launch the housekeeping
        """
        logging.logger.info('housekeeping thread #%s started' % self.ident)
        self.remove_all_whitelisted_outliers()

        # Remove all existing whitelisted items if needed
        while not self.shutdown_flag.is_set():
            self.shutdown_flag.wait(5)
            self.execute_housekeeping()

        logging.logger.info('housekeeping thread #%s stopped' % self.ident)

    def execute_housekeeping(self) -> None:
        """
        Execute the housekeeping
        """
        if len(self.file_mod_watcher.files_changed()) > 0:
            # reload configuration file, in case new whitelisted items were added by the analyst, they
            # should be processed!
            settings.process_configuration_files()

            if self.last_config_parameters != self._get_config_whitelist_parameters():
                self.last_config_parameters = self._get_config_whitelist_parameters()
                logging.logger.info("housekeeping - changes detected in the whitelist configuration")
                self.remove_all_whitelisted_outliers()

    @staticmethod
    def remove_all_whitelisted_outliers() -> None:
        """
        Try to remove all whitelist outliers that are already in ElasticSearch
        """
        if settings.config.getboolean("general", "es_wipe_all_whitelisted_outliers"):
            try:
                logging.logger.info("housekeeping - going to remove all whitelisted outliers")
                total_docs_whitelisted: int = es.remove_all_whitelisted_outliers()

                if total_docs_whitelisted > 0:
                    logging.logger.info(
                        "housekeeping - total whitelisted documents cleared from outliers: " +
                        "{:,}".format(total_docs_whitelisted))
                else:
                    logging.logger.info("housekeeping - whitelist did not remove any outliers")

            except Exception:
                logging.logger.error("housekeeping - something went wrong removing whitelisted outliers", exc_info=True)

            logging.logger.info("housekeeping - finished round of cleaning whitelisted items")
