import configparser
import argparse

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

        self.args = parser.parse_args()
        self.backup_args_config = self.args.config[:]
        self.process_configuration_files()

    def process_configuration_files(self):
        config_paths = self.args.config

        # Read configuration files
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str  # preserve case sensitivity in config keys, important for derived field names

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config

    def _change_configuration_path(self, new_path):
        """
        Only use by tests

        :param new_path: the new path of the configuration
        """
        self.args.config = [new_path]
        self.process_configuration_files()

    def _restore_default_configuration_path(self):
        """
        Only use by tests to restore the default configuration
        """
        self.args.config = self.backup_args_config
        self.process_configuration_files()
