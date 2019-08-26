import configparser
import argparse
import re

from configparser import NoOptionError, NoSectionError, DuplicateOptionError, DuplicateSectionError

from helpers.singleton import singleton  # type: ignore

from typing import List, Set, Optional, Tuple, Union


parser: argparse.ArgumentParser = argparse.ArgumentParser()

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

    def __init__(self) -> None:
        self.config: configparser.ConfigParser

        self.loaded_config_paths: List[str]
        self.failed_config_paths: Set[str]

        self.whitelist_literals_config: Optional[List[Tuple[str, str]]] = None
        self.whitelist_regexps_config: Optional[List[Tuple[str, str]]] = None
        self.failing_regular_expressions: Set[str] = set()

        self.args: argparse.Namespace = parser.parse_args()
        self.process_configuration_files()

    def process_configuration_files(self) -> None:
        """
        Parse configuration and save some value
        """
        config_paths = self.args.config

        # Read configuration files
        config: configparser.ConfigParser = configparser.ConfigParser(interpolation=None, strict=False)
        # preserve case sensitivity in config keys, important for derived field names
        config.optionxform = str  # type: ignore

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config

        self.whitelist_literals_config = self.config.items("whitelist_literals")
        self.whitelist_regexps_config = self.config.items("whitelist_regexps")

        # Verify that all regular expressions in the whitelist are valid.
        # If this is not the case, log an error to the user, as these will be ignored.
        for (_, each_whitelist_configuration_file_value) in self.whitelist_regexps_config:
            whitelist_values_to_check: List[str] = each_whitelist_configuration_file_value.split(",")

            for whitelist_val_to_check in whitelist_values_to_check:
                try:
                    re.compile(whitelist_val_to_check.strip(), re.IGNORECASE)
                except Exception:
                    self.failing_regular_expressions.add(whitelist_val_to_check)

        self.print_outliers_to_console: bool
        try:
            self.print_outliers_to_console = self.config.getboolean("general", "print_outliers_to_console")
        except NoOptionError:
            self.print_outliers_to_console = False

        # Could produce an error, but don't catch it. Crash program if not define
        self.es_save_results: bool = self.config.getboolean("general", "es_save_results")

        self.list_derived_fields: List[Tuple[str, str]]
        try:
            self.list_derived_fields = self.config.items("derivedfields")
        except NoSectionError:
            self.list_derived_fields = list()

        self.list_assets: List[Tuple[str, str]]
        try:
            self.list_assets = self.config.items("assets")
        except NoSectionError:
            self.list_assets = list()

    def check_no_duplicate_key(self) -> Union[None, DuplicateSectionError, DuplicateOptionError]:
        """
        Method to check if some duplicates are present in the configuration

        :return: the error (that contain message with duplicate), None if no duplicate
        """
        try:
            config = configparser.ConfigParser(interpolation=None, strict=True)
            # preserve case sensitivity in config keys, important for derived field names
            config.optionxform = str  # type: ignore
            config.read(self.args.config)
        except (DuplicateOptionError, DuplicateSectionError) as err:
            return err
        return None
