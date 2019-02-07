import math
import urllib3
import logging
import json

from helpers.singleton import singleton


@singleton
class Logging:
    logger_name = None
    logger = None

    current_step = None
    total_steps = None
    desc = None
    verbosity = None

    def __init__(self, logger_name):
        self.logger_name = logger_name

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
            self.logger.info(self.desc + " [" + '{:.2f}'.format(round(float(self.current_step) / float(self.total_steps) * 100, 2)) + "% done" + "]")

    def print_section_divider(self):
        self.logger.info("-" * 50)

    def print_generic_intro(self, title):
        self.logger.info("")
        self.logger.info("===== " + title + " =====")

    def print_analysis_intro(self, event_type, total_events):
        self.logger.info("")
        self.logger.info("===== " + event_type + " outlier detection =====")
        self.logger.info("analyzing " + "{:,}".format(total_events) + " events")

        if total_events == 0:
            self.logger.warning("no events to analyze! If you expected events, make sure the history_window_days and timestamp_field configuration options are correctly configured")

    def print_analysis_abort(self, event_type):
        self.logger.info("")
        self.logger.info("===== " + event_type + " outlier detection aborted =====")
        self.logger.info("No " + event_type + " events in the specified time window")

    def pretty_print_dict(self, _dict):
        self.logger.info(json.dumps(_dict, sort_keys=True, indent=4))


def print_readable_numbers(big_num):
    return "{:,}".format(big_num)
