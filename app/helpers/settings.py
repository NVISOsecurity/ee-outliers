import configparser
import argparse

from helpers.singleton import singleton  # type: ignore

from typing import List, Set, Optional, Tuple


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

        self.args: argparse.Namespace = parser.parse_args()
        self.process_configuration_files()

    def process_configuration_files(self) -> None:
        config_paths = self.args.config

        # Read configuration files
        config: configparser.ConfigParser = configparser.ConfigParser(interpolation=None)
        # preserve case sensitivity in config keys, important for derived field names
        config.optionxform = str  # type: ignore

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config

        self.whitelist_literals_config = self.config.items("whitelist_literals")
        self.whitelist_regexps_config = self.config.items("whitelist_regexps")
