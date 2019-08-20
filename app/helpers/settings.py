import configparser
import argparse
import re

from configparser import NoOptionError, NoSectionError

from helpers.singleton import singleton
import helpers.singletons

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

        self.loaded_config_paths = None
        self.failed_config_paths = None

        self.whitelist_literals_config = None
        self.whitelist_regexps_config = None
        self.failing_regular_expressions = set()

        self.args = parser.parse_args()
        self.process_configuration_files()

    def process_configuration_files(self):
        """
        Parse configuration and save some value
        """
        config_paths = self.args.config

        # Read configuration files
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str  # preserve case sensitivity in config keys, important for derived field names

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config

        self.whitelist_literals_config = self.config.items("whitelist_literals")
        self.whitelist_regexps_config = self.config.items("whitelist_regexps")

        # Verify that all regular expressions in the whitelist are valid.
        # If this is not the case, log an error to the user, as these will be ignored.
        for (_, each_whitelist_configuration_file_value) in self.whitelist_regexps_config:
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            for whitelist_val_to_check in whitelist_values_to_check:
                try:
                    re.compile(whitelist_val_to_check.strip(), re.IGNORECASE)
                except Exception:
                    self.failing_regular_expressions.add(whitelist_val_to_check)

        try:
            self.print_outliers_to_console = self.config.getboolean("general", "print_outliers_to_console")
        except NoOptionError:
            self.print_outliers_to_console = 0

        # Could produce an error, but don't catch it. Crash program if not define
        self.es_save_results = self.config.getboolean("general", "es_save_results")

        try:
            self.list_derived_fields = self.config.items("derivedfields")
        except NoSectionError:
            self.list_derived_fields = dict()

        try:
            self.list_assets = self.config.items("assets")
        except NoSectionError:
            self.list_assets = dict()
