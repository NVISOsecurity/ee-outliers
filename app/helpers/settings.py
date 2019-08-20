import configparser
import argparse
import re

from helpers.singleton import singleton

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
        self.whitelist_literals_per_model = dict()
        self.whitelist_regexps_config = None
        self.whitelist_regexps_per_model = dict()
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

        # Literal whitelist
        self.whitelist_literals_config = self._extract_whitelist_literals_from_settings_section("whitelist_literals")
        self._load_model_whitelist_literals()
        # Regex whitelist
        self.whitelist_regexps_config, self.failing_regular_expressions = \
            self._extract_whitelist_regex_from_settings_section("whitelist_regexps")
        self._load_model_whitelist_regexps()

    def _extract_whitelist_literals_from_settings_section(self, settings_section):
        list_whitelist_literals = list()
        fetch_whitelist_literals_elements = list(dict(self.config.items(settings_section)).values())

        for each_whitelist_configuration_file_value in fetch_whitelist_literals_elements:
            list_whitelist_element = list()
            for one_whitelist_config_file_value in str(each_whitelist_configuration_file_value).split(','):
                list_whitelist_element.append(one_whitelist_config_file_value.strip())

            # Append to global whitelist (and remove duplicate (transform in set))
            list_whitelist_literals.append(set(list_whitelist_element))
        return list_whitelist_literals

    def _load_model_whitelist_literals(self):
        regex_match = "whitelist_literals_([^\\_]+)_(.*)"
        for config_section_name in self.config.sections():
            match_whitelist_section = re.search(regex_match, config_section_name)
            if match_whitelist_section is not None:
                # Fetch model name
                model_type = match_whitelist_section.group(1)
                model_name = match_whitelist_section.group(2)

                self.whitelist_literals_per_model[(model_type, model_name)] = \
                    self._extract_whitelist_literals_from_settings_section(config_section_name)

    def _extract_whitelist_regex_from_settings_section(self, settings_section):
        whitelist_regexps_config_items = list(dict(self.config.items(settings_section)).values())
        list_whitelist_regexps = list()
        failing_regular_expressions = set()

        # Verify that all regular expressions in the whitelist are valid.
        # If this is not the case, log an error to the user, as these will be ignored.
        for each_whitelist_configuration_file_value in whitelist_regexps_config_items:
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            list_compile_regex_whitelist_value = list()
            for whitelist_val_to_check in whitelist_values_to_check:
                try:
                    list_compile_regex_whitelist_value.append(re.compile(whitelist_val_to_check.strip(), re.IGNORECASE))
                except Exception:
                    failing_regular_expressions.add(whitelist_val_to_check)

            list_whitelist_regexps.append(list_compile_regex_whitelist_value)
        return list_whitelist_regexps, failing_regular_expressions

    def _load_model_whitelist_regexps(self):
        regex_match = "whitelist_regexps_([^\\_]+)_(.*)"

        for config_section_name in self.config.sections():
            match_whitelist_section = re.search(regex_match, config_section_name)
            if match_whitelist_section is not None:
                # Fetch model name
                model_type = match_whitelist_section.group(1)
                model_name = match_whitelist_section.group(2)

                list_whitelist_regexps, failing_regular_expressions = \
                    self._extract_whitelist_regex_from_settings_section(config_section_name)
                self.whitelist_regexps_per_model[(model_type, model_name)] = list_whitelist_regexps

    def get_whitelist_literals(self, extra_whitelist_section):
        if self.whitelist_literals_per_model is not None and \
                extra_whitelist_section in self.whitelist_literals_per_model:
            return self.whitelist_literals_config + self.whitelist_literals_per_model[extra_whitelist_section]

        return self.whitelist_literals_config

    def get_whitelist_regexps(self, extra_whitelist_section):
        if self.whitelist_regexps_per_model is not None and \
                extra_whitelist_section in self.whitelist_regexps_per_model:
            return self.whitelist_regexps_config + self.whitelist_regexps_per_model[extra_whitelist_section]

        return self.whitelist_regexps_config
