import math
import urllib3
import logging
import datetime as dt

from typing import Optional, cast

from helpers.singleton import singleton


@singleton
class Logging:
    #start_time = None"""
    logger: logging.Logger

    total_steps: Optional[int]
    desc: Optional[str]
    current_step: int

    verbosity: int = 0

    def __init__(self, logger_name: str) -> None:
        # Disable HTTPS warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        self.logger: logging.Logger = logging.getLogger(logger_name)

    def add_stdout_handler(self) -> None:
        ch: logging.StreamHandler = logging.StreamHandler()
        ch.setLevel(self.logger.level)
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                                         "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def add_file_handler(self, log_file: str) -> None:
        ch: logging.FileHandler = logging.FileHandler(log_file)
        ch.setLevel(self.logger.level)
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                                         "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def init_ticker(self, total_steps: int, desc: str) -> None:
        self.total_steps: int = total_steps
        self.start_time: float = dt.datetime.today().timestamp()
        self.desc: str = desc
        self.current_step: int = 0

    def tick(self) -> None:
        should_log: bool = False
        self.current_step += 1

        if self.verbosity >= 5:
            should_log = True
        else:
            if (self.current_step % max(1, int(math.pow(10, (5 - self.verbosity))))) == 0:
                should_log = True

            if self.current_step == self.total_steps:
                should_log = True

        if should_log:
            # avoid a division by zero
            time_diff: float = max(float(1), float(dt.datetime.today().timestamp() - self.start_time))
            ticks_per_second: str = "{:,}".format(round(float(self.current_step) / time_diff))

            self.logger.info(str(self.desc) + " [" + ticks_per_second + " eps. - " + '{:.2f}'
                             .format(round(float(self.current_step) / float(cast(int, self.total_steps)) * 100, 2)) +
                             "% done" + "]")

    def print_generic_intro(self, title: str) -> None:
        self.logger.info("")
        self.logger.info("===== " + title + " =====")

    def print_analysis_intro(self, event_type: str, total_events: int) -> None:
        self.logger.info("")
        self.logger.info("===== " + event_type + " outlier detection =====")
        self.logger.info("analyzing " + "{:,}".format(total_events) + " events")

        if total_events == 0:
            self.logger.warning("no events to analyze!")
