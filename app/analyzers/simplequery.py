from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
import re
from configparser import NoSectionError, NoOptionError


class SimplequeryAnalyzer(Analyzer):

    def __init__(self, model_name, config_section):
        super(SimplequeryAnalyzer, self).__init__("simplequery", model_name, config_section)

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        self.model_settings["process_documents_chronologically"] = self.config_section.getboolean("process_documents_chronologically")
        self.model_settings["highlight_match"] = self.config_section.getboolean("highlight_match")
        if self.model_settings["highlight_match"] is None:
            try:
                self.model_settings["highlight_match"] = settings.config.getboolean("simplequery", "highlight_match")
            except (NoSectionError, NoOptionError):
                self.model_settings["highlight_match"] = False

    def evaluate_model(self):

        model_filter = {
            "bool": {
                "filter": [{
                    "term": {
                        "outliers.model_name.keyword": {
                            "value": self.model_name
                        }
                    }
                },
                    {
                    "term": {
                        "outliers.model_type.keyword": {
                            "value": "simplequery"
                        }
                    }
                }]
            }
        }

        exclude_hits_filter = {
            "bool": {
                "must_not": model_filter
            }
        }

        query = self.search_query

        if "filter" in query:
            query["filter"].append(exclude_hits_filter)
        else:
            query["filter"] = [exclude_hits_filter]

        self.total_events, documents = es.count_and_scan_documents(index=self.model_settings["es_index"],
                                                                   search_query=query,
                                                                   model_settings=self.model_settings)

        self.print_analysis_intro(event_type="evaluating " + self.model_type + "_" + self.model_name,
                                  total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events,
                            desc=self.model_name + " - evaluating " + self.model_type + " model")
        if self.total_events > 0:
            for doc in documents:
                logging.tick()
                outlier = self._create_outlier(doc)
                self.process_outlier(outlier)

        self.print_analysis_summary()

    def _create_outlier(self, raw_doc):
        """
        Create outlier from raw_doc

        :param raw_doc: raw document representing one hit event from an Elasticsearch request
        :return: the created outlier
        """
        extra_outlier_information = dict()
        if self.model_settings["highlight_match"]:
            extra_outlier_information["matched_fields"] = raw_doc["highlight"]

            matched_values = dict()
            for key, fields in raw_doc["highlight"].items():
                matched_values[key] = list()
                for field in fields:
                    # Find values between tags <value> and </value>
                    values = re.findall("<value>((.|\n)*?)</value>", field)
                    matched_values[key] = [value for value, _ in values]
            extra_outlier_information["matched_values"] = str(matched_values)
        fields = es.extract_fields_from_document(raw_doc,
                                                 extract_derived_fields=self.model_settings["use_derived_fields"])
        return self.create_outlier(fields, raw_doc, extra_outlier_information=extra_outlier_information)
