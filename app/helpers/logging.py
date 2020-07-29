import math
import logging
from logging.handlers import WatchedFileHandler

import datetime as dt
import urllib3

from helpers.singleton import singleton


@singleton
class Logging:
    """
    Creates the logger singleton to be used across the entire project
    """
    logger = None

    current_step = None
    start_time = None
    total_steps = None
    desc = None
    verbosity = 0

    def __init__(self, logger_name):
        # disable HTTPS warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False

    def add_stdout_handler(self):
        """
        Create, format & add the handler that will log to standard output
        """
        handler = logging.StreamHandler()
        handler.setLevel(self.logger.level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add_file_handler(self, log_file):
        """
        Create, format & add the handler that will log to the log file
        """
        handler = WatchedFileHandler(log_file)
        handler.setLevel(self.logger.level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def init_ticker(self, total_steps=None, desc=None):
        """
        Initialize a ticker.
        Warning: this method is not independant. Call it only one at a time

        :param total_steps: number of total step
        :param desc: description of the ticker
        """
        self.total_steps = total_steps
        self.start_time = dt.datetime.today().timestamp()
        self.desc = desc
        self.current_step = 0

    def tick(self, num_event_proc=None):
        """
        Increment the number of tick

        :param num_event_proc: number of event already processed
        """
        self.current_step += 1

        if self.verbosity >= 5:
            should_log = True
        else:
            should_log = self.current_step % max(1, int(math.pow(10, (6 - self.verbosity)))) == 0 or \
                            self.current_step == self.total_steps

        if should_log:
            # avoid a division by zero
            time_diff = max(float(1), float(dt.datetime.today().timestamp() - self.start_time))
            if num_event_proc is None:
                ticks_per_second = "{:,}".format(round(float(self.current_step) / time_diff))
            else:
                ticks_per_second = "{:,}".format(round(float(num_event_proc) / time_diff))

            self.logger.info(self.desc + " [" + ticks_per_second + " eps. - " + '{:.2f}'
                             .format(round(float(self.current_step) / float(self.total_steps) * 100, 2)) +
                             "% done" + "]")

    def print_generic_intro(self, title):
        """
        Display title in log

        :param title: title to display
        """
        self.logger.info("")
        self.logger.info("===== " + title + " =====")
