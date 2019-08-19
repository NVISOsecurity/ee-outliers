import math
import urllib3
import logging
import datetime as dt

from helpers.singleton import singleton  # type: ignore


@singleton
class Logging:
    # start_time = None"""
    logger: logging.Logger

    current_step: int

    verbosity: int = 0

    def __init__(self, logger_name: str) -> None:
        # disable HTTPS warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)

        self.logger: logging.Logger = logging.getLogger(logger_name)
        self.logger.propagate = False

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
        """
        Initialize a ticker.
        Warning: this method is not independant. Call it only one at a time

        :param total_steps: number of total step
        :param desc: description of the ticker
        """
        self.total_steps: int = total_steps
        self.start_time: float = dt.datetime.today().timestamp()
        self.desc: str = desc
        self.current_step: int = 0

    def tick(self) -> None:
        """
        Increment the number of tick
        """
        self.current_step += 1
        should_log: bool
        if self.verbosity >= 5:
            should_log = True
        else:
            should_log = self.current_step % max(1, int(math.pow(10, (6 - self.verbosity)))) == 0 or \
                            self.current_step == self.total_steps

        if should_log:
            # avoid a division by zero
            time_diff: float = max(float(1), float(dt.datetime.today().timestamp() - self.start_time))
            ticks_per_second: str = "{:,}".format(round(float(self.current_step) / time_diff))

            self.logger.info(self.desc + " [" + ticks_per_second + " eps. - " + '{:.2f}'
                             .format(round(float(self.current_step) / float(self.total_steps) * 100, 2)) +
                             "% done" + "]")

    def print_generic_intro(self, title: str) -> None:
        """
        Display title in log

        :param title: title to display
        """
        self.logger.info("")
        self.logger.info("===== " + title + " =====")
