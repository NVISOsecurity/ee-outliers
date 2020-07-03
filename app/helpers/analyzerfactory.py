from helpers.singletons import logging
import configparser
import os
import re

from analyzers.metrics import MetricsAnalyzer
from analyzers.simplequery import SimplequeryAnalyzer
from analyzers.terms import TermsAnalyzer
from analyzers.word2vec import Word2VecAnalyzer

CLASS_MAPPING = {
    "simplequery": SimplequeryAnalyzer,
    "metrics": MetricsAnalyzer,
    "word2vec": Word2VecAnalyzer,
    "terms": TermsAnalyzer
}


class AnalyzerFactory:

    @staticmethod
    def section_to_analyzer(section_name, section):
        """
        Converts the name of a section in the configuration file into
        :param section_name: name of section in the parsed configuration file, i.e. metrics_numerical_value_dummy_test
        :param section: the section object representing the section called section_name
        :return:
        """
        # Check if the config section matches any of the defined "analyzer" prefixes
        for prefix, _class in CLASS_MAPPING.items():
            if section_name.startswith(prefix):
                return _class(model_name=section_name[(len(prefix) + 1):], config_section=section)
        return None

    @staticmethod
    def create(config_file):
        """
        Creates an analyzer based on a configuration file
        Deprecated in favor of `create_multi`
        :param config_file: configuration file containing a single analyzer
        :return: returns the analyzer object
        """
        analyzers = AnalyzerFactory.create_multi(config_file)

        # File should only contain 1 analyzer
        if len(analyzers) != 1:
            raise ValueError("Config file must contain exactly one use case (found %d)" % len(analyzers))

        analyzer = analyzers[0]

        return analyzer

    @staticmethod
    def create_multi(config_file, configparser_options={}):
        """
        Creates a list of analyzers based on a configuration file
        :param config_file: configuration file containing one or multiple analyzers
        :param configparser_options: Optional parameters to configparser.RawConfigParser(...)
        :return: returns the analyzer objects in a list
        """
        if not os.path.isfile(config_file):
            raise ValueError("Use case file %s does not exist" % config_file)

        # Read the ini file from disk
        config = configparser.RawConfigParser(**configparser_options)
        config.read(config_file)

        # Create a list of all analyzers found in the config file
        analyzers = [AnalyzerFactory.section_to_analyzer(section_name, section)
                     for section_name, section in config.items()]
        analyzers = list(filter(None, analyzers))

        for analyzer in analyzers:
            if "whitelist_literals" in config.sections():
                for _, value in config["whitelist_literals"].items():
                    analyzer.model_whitelist_literals.append(
                        set([x.strip() for x in value.split(",")]))

            if "whitelist_regexps" in config.sections():
                for _, value in config["whitelist_regexps"].items():
                    analyzer.model_whitelist_regexps.append(
                        (set([re.compile(x.strip(), re.IGNORECASE) for x in value.split(",")])))

        return analyzers