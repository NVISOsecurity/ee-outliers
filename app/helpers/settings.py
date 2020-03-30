import configparser
import argparse
import re
import os

from configparser import NoOptionError, NoSectionError

from helpers.singleton import singleton

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Run mode", dest="run_mode")
interactive_parser = subparsers.add_parser('interactive')
daemon_parser = subparsers.add_parser('daemon')
tests_parser = subparsers.add_parser('tests')

# Interactive mode - options
interactive_parser.add_argument("--config",
                                action='append',
                                type=lambda x: file_exists(interactive_parser, x),
                                help="Configuration file location",
                                required=True)
interactive_parser.add_argument("--use-cases",
                                action='append',
                                help="Additional use cases location",
                                required=True)

# Daemon mode - options
daemon_parser.add_argument("--config",
                           action='append',
                           type=lambda x: file_exists(daemon_parser, x),
                           help="Configuration file location",
                           required=True)
daemon_parser.add_argument("--use-cases",
                           action='append',
                           help="Additional use cases location",
                           required=True)

# Tests mode - options
tests_parser.add_argument("--config",
                          action='append',
                          type=lambda x: file_exists(tests_parser, x),
                          help="Configuration file location",
                          required=True)
tests_parser.add_argument("--use-cases",
                          action='append',
                          help="Additional use cases location",
                          required=True)


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
        config = configparser.ConfigParser(interpolation=None, strict=False)
        config.optionxform = str  # preserve case sensitivity in config keys, important for derived field names

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config

        # Literal whitelist
        self.whitelist_literals_config = self._extract_whitelist_literals_from_settings_section("whitelist_literals")
        # Regex whitelist
        self.whitelist_regexps_config, self.failing_regular_expressions = \
            self._extract_whitelist_regex_from_settings_section("whitelist_regexps")

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

    def _extract_whitelist_literals_from_settings_section(self, settings_section):
        list_whitelist_literals = list()
        fetch_whitelist_literals_elements = list(dict(self.config.items(settings_section)).values())

        for each_whitelist_configuration_file_value in fetch_whitelist_literals_elements:
            list_whitelist_literals.append(self.extract_whitelist_literal_from_value(str(
                each_whitelist_configuration_file_value)))
        return list_whitelist_literals

    def extract_whitelist_literal_from_value(self, value):
        list_whitelist_element = set()
        for one_whitelist_config_file_value in value.split(','):
            list_whitelist_element.add(one_whitelist_config_file_value.strip())
        return list_whitelist_element

    def _extract_whitelist_regex_from_settings_section(self, settings_section):
        whitelist_regexps_config_items = list(dict(self.config.items(settings_section)).values())
        list_whitelist_regexps = list()
        failing_regular_expressions = set()

        # Verify that all regular expressions in the whitelist are valid.
        # If this is not the case, log an error to the user, as these will be ignored.
        for each_whitelist_configuration_file_value in whitelist_regexps_config_items:
            new_compile_regex_whitelist_value, value_failing_regular_expressions = \
                self.extract_whitelist_regex_from_value(each_whitelist_configuration_file_value)

            # Fixes bug #462
            if len(new_compile_regex_whitelist_value) > 0:
                list_whitelist_regexps.append(new_compile_regex_whitelist_value)

            if len(value_failing_regular_expressions) > 0:
                failing_regular_expressions.union(value_failing_regular_expressions)

        return list_whitelist_regexps, failing_regular_expressions

    def extract_whitelist_regex_from_value(self, value):
        list_compile_regex_whitelist_value = set()
        failing_regular_expressions = set()
        for whitelist_val_to_check in value.split(","):
            try:
                list_compile_regex_whitelist_value.add(re.compile(whitelist_val_to_check.strip(), re.IGNORECASE))
            except Exception:
                # something went wrong compiling the regular expression, probably because of a user error such as
                # unbalanced escape characters. We should just ignore the regular expression and continue (and let
                # the user know in the beginning that some could not be compiled).  Even if we check for errors
                # in the beginning of running outliers, we might still run into issues when the configuration
                # changes during running of ee-outlies. So this should catch any remaining errors in the
                # whitelist that could occur with regexps.
                failing_regular_expressions.add(whitelist_val_to_check)

        return list_compile_regex_whitelist_value, failing_regular_expressions

    def check_no_duplicate_key(self):
        """
        Method to check if some duplicates are present in the configuration

        :return: the error (that contain message with duplicate), None if no duplicate
        """
        try:
            config = configparser.ConfigParser(interpolation=None, strict=True)
            config.optionxform = str  # preserve case sensitivity in config keys, important for derived field names
            config.read(self.args.config)
        except (configparser.DuplicateOptionError, configparser.DuplicateSectionError) as err:
            return err
        return None


def file_exists(argument_parser, file_path):
    """
    Method to check if the file at file_path exist.
    If file does not exist, it print a message to the standard error and terminates the program.
    Otherwise, it returns the file_path.

    :param argument_parser: ArgumentParser that return the error.
    :param file_path: The file location.
    :return: If file in file_path exists, it return file_path. Otherwise, it prints a message to the standard error and
    terminates the program.
    """
    if not os.path.isfile(file_path):
        argument_parser.error("The file %s does not exist." % file_path)
    else:
        return file_path
