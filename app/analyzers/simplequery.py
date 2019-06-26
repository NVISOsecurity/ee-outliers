from helpers.singletons import es, logging
from helpers.analyzer import Analyzer


class SimplequeryAnalyzer(Analyzer):

    def evaluate_model(self):

        model_filter = {
            "bool": {
                "filter": [
                    {
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

        self.total_events = es.count_documents(search_query=query)
        logging.print_analysis_intro(event_type="evaluating " + self.config_section_name,
                                     total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating " + self.model_type + \
                                                                " model")
        for doc in es.scan(search_query=query):
            logging.tick()
            fields = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
            self.process_outlier(fields, doc)

        self.print_analysis_summary()
