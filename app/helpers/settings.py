import configparser
import argparse

from helpers.singleton import singleton

from typing import Dict, List, Set


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
        self.args: argparse.Namespace
        self.config: configparser.ConfigParser

        self.loaded_config_paths: List[str]
        self.failed_config_paths: Set[str]

        self.search_range_start: str
        self.search_range_end: str
        self.search_range: Dict[str, Dict]

        self.process_arguments()

    def process_arguments(self) -> None:
        args: argparse.Namespace = parser.parse_args()
        self.args = args

        self.process_configuration_files(args.config)

    def reload_configuration_files(self) -> None:
        self.process_configuration_files(self.args.config)

    def process_configuration_files(self, config_paths: str) -> None:
        # Read configuration files
        config: configparser.ConfigParser = configparser.ConfigParser(interpolation=None)
        # preserve case sensitivity in config keys, important for derived field names
        config.optionxform = str # type: ignore

        self.loaded_config_paths = config.read(config_paths)
        self.failed_config_paths = set(config_paths) - set(self.loaded_config_paths)

        self.config = config
