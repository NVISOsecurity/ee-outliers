import configparser
import argparse
from helpers.singleton import singleton
from . import es

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Run mode", dest="run_mode")
interactive_parser = subparsers.add_parser('interactive')
daemon_parser = subparsers.add_parser('daemon')
tests_parser = subparsers.add_parser('tests')

# Interactive mode - options
interactive_parser.add_argument("--config", action='append', help="Configuration file location", required=True)

# Daemon mode - options
daemon_parser.add_argument("--config", action='append', help="Configuration file location", required=True)

# Tests mode - options
tests_parser.add_argument("--config", action='append', help="Configuration file location", required=True)


@singleton
class Settings:

    def __init__(self):
        self.args = None
        self.config = None
        self.known_processes = None

        self.search_range_start = None
        self.search_range_end = None
        self.search_range = None

        self.process_arguments()

    def process_arguments(self):
        args = parser.parse_args()
        self.args = args

        self.process_configuration_files(args.config)

        search_range = es.get_time_filter(days=self.config.getint("general", "history_window_days"), hours=self.config.getint("general", "history_window_hours"), timestamp_field=self.config.get("general", "timestamp_field", fallback="timestamp"))

        self.search_range_start = search_range["range"][str(self.config.get("general", "timestamp_field", fallback="timestamp"))]["gte"]
        self.search_range_end = search_range["range"][str(self.config.get("general", "timestamp_field", fallback="timestamp"))]["lte"]
        self.search_range = search_range

        # Daemon mode settings
        if args.run_mode == "daemon":
            pass
        if args.run_mode == "interactive":
            pass

    def reload_configuration_files(self):
        self.process_configuration_files(self.args.config)

    def process_configuration_files(self, config_paths):
        # Read configuration files
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str  # preserve case sensitivity in config keys, important for derived field names

        config.read(config_paths)
        self.config = config

    def get_sample_size(self, total_events=None, threshold_name=None):
        sample_size = self.config.getint("thresholds", threshold_name)

        # Calculate number of iterations
        if not (sample_size <= 0):
            return min(sample_size, total_events)
        else:
            return total_events

    def get_items(self, section=None):
        return self.config.items(section)
