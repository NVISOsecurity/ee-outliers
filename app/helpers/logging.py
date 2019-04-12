import math
import urllib3
import logging
import datetime as dt

from helpers.singleton import singleton


@singleton
class Logging:
    logger = None

    current_step = None
    start_time = None
    total_steps = None
    desc = None
    verbosity = None

    def __init__(self, logger_name):
        # Disable HTTPS warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        self.logger = logging.getLogger(logger_name)

    def add_stdout_handler(self):
        ch = logging.StreamHandler()
        ch.setLevel(self.logger.level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def add_file_handler(self, log_file):
        ch = logging.FileHandler(log_file)
        ch.setLevel(self.logger.level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def init_ticker(self, total_steps=None, desc=None):
        self.total_steps = total_steps
        self.start_time = dt.datetime.today().timestamp()
        self.desc = desc
        self.current_step = 0

    def tick(self):
        should_log = False
        self.current_step += 1

        if self.verbosity >= 5:
            should_log = True
        else:
            if (self.current_step % max(1, int(math.pow(10, (5 - self.verbosity))))) == 0:
                should_log = True

            if self.current_step == self.total_steps:
                should_log = True

        if should_log:
            time_diff = max(float(1), float(dt.datetime.today().timestamp() - self.start_time))  # avoid a division by zero
            ticks_per_second = "{:,}".format(round(float(self.current_step) / time_diff))

            self.logger.info(self.desc + " [" + ticks_per_second + " eps. - " + '{:.2f}'.format(round(float(self.current_step) / float(self.total_steps) * 100, 2)) + "% done" + "]")

    def print_generic_intro(self, title):
        self.logger.info("")
        self.logger.info("===== " + title + " =====")

    def print_analysis_intro(self, event_type, total_events):
        self.logger.info("")
        self.logger.info("===== " + event_type + " outlier detection =====")
        self.logger.info("analyzing " + "{:,}".format(total_events) + " events")

        if total_events == 0:
            self.logger.warning("no events to analyze!")
