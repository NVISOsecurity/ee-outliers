from helpers.singletons import es, logging
from helpers.analyzer import Analyzer


class SimplequeryAnalyzer(Analyzer):

    def evaluate_model(self):
        outliers = list()

        logging.init_ticker(total_steps=self.total_events, desc=self.model_name + " - evaluating simplequery model")
        for doc in es.scan(lucene_query=self.lucene_query):
            logging.tick()
            fields = es.extract_fields_from_document(doc)
            outlier = self.create_outlier(fields)
            outliers.append(outlier)

            es.process_outliers(doc=doc, outliers=[outlier], should_notify=self.model_settings["should_notify"])

        self.print_batch_summary()