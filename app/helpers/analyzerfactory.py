import configparser
import os
import re

from analyzers.metrics import MetricsAnalyzer
from analyzers.simplequery import SimplequeryAnalyzer
from analyzers.terms import TermsAnalyzer
from analyzers.word2vec import Word2VecAnalyzer

class_mapping = {
    "simplequery": SimplequeryAnalyzer,
    "metrics": MetricsAnalyzer,
    "word2vec": Word2VecAnalyzer,
    "terms": TermsAnalyzer
}

class AnalyzerFactory:
    
    @staticmethod
    def section_to_analyzer(section_name, section):
        # Check if the config section matches any of the defined "analyzer" prefixes
        for prefix, _class in class_mapping.items():
            if section_name.startswith(prefix):
                return _class(model_name=section_name[(len(prefix) + 1):], config_section=section)
        return None

    @staticmethod
    def create(config_file):
        if not os.path.isfile(config_file):
            raise ValueError("Use case file %s does not exist" % config_file)

        # Read the ini file from disk
        config = configparser.ConfigParser()
        config.read(config_file)

        # Create a list of all analyzers found in the config file
        analyzers = [AnalyzerFactory.section_to_analyzer(section_name, section) for section_name, section in config.items()]
        analyzers = list(filter(None, analyzers))

        # File should only contain 1 analyzer
        if len(analyzers) != 1:
            raise ValueError("Config file must contain exactly one use case (found %d)" % len(analyzers))

        analyzer = analyzers[0]

        if "whitelist_literals" in config.sections():
            for _, value in config["whitelist_literals"].items():
                analyzer.add_whitelist_literal(set([x.strip() for x in value.split(",")]))
        
        if "whitelist_regexps" in config.sections():
            for _, value in config["whitelist_regexps"].items():
                analyzer.add_whitelist_regexp(set([re.compile(x.strip(), re.IGNORECASE) for x in value.split(",")]))

        return analyzer