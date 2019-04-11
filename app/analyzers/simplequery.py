from helpers.singletons import es, logging
from helpers.analyzer import Analyzer


class SimplequeryAnalyzer(Analyzer):

    def evaluate_model(self):
        self.total_events = es.count_documents(lucene_query=self.lucene_query)
        logging.print_analysis_intro(event_type="evaluating " + self.config_section_name, total_events=self.total_events)

        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating " + self.model_type + " model")
        for doc in es.scan(lucene_query=self.lucene_query):
            logging.tick()
            fields = es.extract_fields_from_document(doc)
            self.process_outlier(fields, doc)

        self.print_analysis_summary()
