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

        # Literal whitelist
        self.whitelist_literals_config = list()
        self._append_whitelist_literals_from_settings_section("whitelist_literals")
        self._load_model_whitelist_literals()
        # Regex whitelist
        self.whitelist_regexps_config = list()
        self._append_whitelist_regex_from_settings_section("whitelist_regexps")
        self._load_model_whitelist_regexps()

    def _append_whitelist_literals_from_settings_section(self, settings_section, extra_whitelist_fields_list=None):
        fetch_whitelist_literals_elements = list(dict(self.config.items(settings_section)).values())

        for each_whitelist_configuration_file_value in fetch_whitelist_literals_elements:
            list_whitelist_element = list()
            for one_whitelist_config_file_value in str(each_whitelist_configuration_file_value).split(','):
                list_whitelist_element.append(one_whitelist_config_file_value.strip())

            if extra_whitelist_fields_list is not None:
                list_whitelist_element += extra_whitelist_fields_list

            # Append to global whitelist (and remove duplicate (transform in set))
            self.whitelist_literals_config.append(set(list_whitelist_element))

    def _load_model_whitelist_literals(self):
        regex_match = "whitelist_literals_([^\\_]+)_(.*)"
        for config_section_name in self.config.sections():
            match_whitelist_section = re.search(regex_match, config_section_name)
            if match_whitelist_section is not None:
                # Fetch model name
                model_type = match_whitelist_section.group(1)
                model_name = match_whitelist_section.group(2)

                self._append_whitelist_literals_from_settings_section(config_section_name,
                                                                      extra_whitelist_fields_list=[model_type,
                                                                                                   model_name])

    def _append_whitelist_regex_from_settings_section(self, settings_section, extra_whitelist_fields_list=None):
        whitelist_regexps_config_items = list(dict(self.config.items(settings_section)).values())

        # Verify that all regular expressions in the whitelist are valid.
        # If this is not the case, log an error to the user, as these will be ignored.
        for each_whitelist_configuration_file_value in whitelist_regexps_config_items:
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            list_compile_regex_whitelist_value = list()
            for whitelist_val_to_check in whitelist_values_to_check:
                try:
                    list_compile_regex_whitelist_value.append(re.compile(whitelist_val_to_check.strip(), re.IGNORECASE))
                except Exception:
                    self.failing_regular_expressions.add(whitelist_val_to_check)

            if extra_whitelist_fields_list is not None:
                list_compiled_extra_whitelist_field = list()
                for extra_whitelist_field in extra_whitelist_fields_list:
                    try:
                        list_compiled_extra_whitelist_field.append(re.compile(extra_whitelist_field.strip(),
                                                                             re.IGNORECASE))
                    except Exception:
                        self.failing_regular_expressions.add(extra_whitelist_field)
                list_compile_regex_whitelist_value += list_compiled_extra_whitelist_field

            self.whitelist_regexps_config.append(list_compile_regex_whitelist_value)

    def _load_model_whitelist_regexps(self):
        regex_match = "whitelist_regexps_([^\\_]+)_(.*)"

        for config_section_name in self.config.sections():
            match_whitelist_section = re.search(regex_match, config_section_name)
            if match_whitelist_section is not None:
                # Fetch model name
                model_type = match_whitelist_section.group(1)
                model_name = match_whitelist_section.group(2)

                self._append_whitelist_regex_from_settings_section(config_section_name,
                                                                   extra_whitelist_fields_list=[model_type, model_name])
