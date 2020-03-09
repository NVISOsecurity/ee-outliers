from helpers.singletons import es, logging
from helpers.analyzer import Analyzer

from typing import Dict, List, Any

from helpers.outlier import Outlier


class SimplequeryAnalyzer(Analyzer):

    def __init__(self, model_name, config_section):
        super(SimplequeryAnalyzer, self).__init__("simplequery", model_name, config_section)

    def evaluate_model(self) -> None:

        model_filter: Dict[str, Any] = {
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

        exclude_hits_filter: Dict[str, Any] = {
            "bool": {
                "must_not": model_filter
            }
        }

        query: Dict[str, List] = self.search_query

        if "filter" in query:
            query["filter"].append(exclude_hits_filter)
        else:
            query["filter"] = [exclude_hits_filter]

        documents: List[Dict[str, Any]]
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
                fields: Dict = es.extract_fields_from_document(
                                                doc, extract_derived_fields=self.model_settings["use_derived_fields"])
                outlier: Outlier = self.create_outlier(fields, doc)
                self.process_outlier(outlier)

        self.print_analysis_summary()
