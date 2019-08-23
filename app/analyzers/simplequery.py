from helpers.singletons import es, logging
from helpers.analyzer import Analyzer


class SimplequeryAnalyzer(Analyzer):

    def evaluate_model(self):

        model_filter = {
            "bool": {
                "filter": [{
                    "term": {
                        "outliers.model_name.raw": {
                            "value": self.model_name
                        }
                    }
                },
                    {
                    "term": {
                        "outliers.model_type.raw": {
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
        self.print_analysis_intro(event_type="evaluating " + self.config_section_name,
                                  total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events,
                            desc=self.model_name + " - evaluating " + self.model_type + " model")
        if self.total_events > 0:
            for doc in documents:
                logging.tick()
                fields = es.extract_fields_from_document(
                                                doc, extract_derived_fields=self.model_settings["use_derived_fields"])
                outlier = self.create_outlier(fields, doc)
                self.process_outlier(outlier)

        self.print_analysis_summary()
