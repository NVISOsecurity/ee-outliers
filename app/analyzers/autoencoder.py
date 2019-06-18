from helpers.singletons import es, logging
from helpers.analyzer import Analyzer


class AutoencoderAnalyzer(Analyzer):

    def evaluate_model(self):
        self.total_events = es.count_documents(index=self.es_index, search_query=self.search_query)
        logging.print_analysis_intro(event_type="evaluating " + self.config_section_name, total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating " + self.model_type + " model")
        for doc in es.scan(index=self.es_index, search_query=self.search_query):
            logging.tick()
            fields = es.extract_fields_from_document(doc, extract_derived_fields=self.model_settings["use_derived_fields"])
            self.process_outlier(fields, doc)

        self.print_analysis_summary()
