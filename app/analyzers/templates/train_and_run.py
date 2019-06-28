from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer

from typing import List, Dict


class TemplateAnalyzer(Analyzer):

    def evaluate_model(self) -> None:
        self.extract_extra_model_settings()

        if self.model_settings["train_model"]:
            self.train_model()

        if self.model_settings["run_model"] or self.model_settings["test_model"]:
            self.run_model()

    def extract_extra_model_settings(self) -> None:
        self.model_settings["train_model"] = settings.config.getboolean(self.config_section_name, "train_model")

    def train_model(self) -> None:
        search_query: Dict[str, List] = es.filter_by_query_string(self.model_settings["es_query_filter"])

        train_data: List[Dict] = list()

        self.total_events = es.count_documents(index=self.es_index, search_query=search_query)
        training_data_size_pct: int = settings.config.getint("machine_learning", "training_data_size_pct")
        training_data_size: float = self.total_events / 100 * training_data_size_pct

        logging.print_analysis_intro(event_type="training " + self.model_name, total_events=self.total_events)
        total_training_events: int = int(min(training_data_size, self.total_events))

        logging.init_ticker(total_steps=total_training_events, desc=self.model_name + " - preparing SVM training set")
        for doc in es.scan(index=self.es_index, search_query=search_query):
            if len(train_data) < total_training_events:
                logging.tick()
                fields: Dict = es.extract_fields_from_document(doc,
                                                     extract_derived_fields=self.model_settings["use_derived_fields"])
                train_data.append(fields)
            else:
                # We have collected sufficient training data
                break

        # Now, train the model
        if len(train_data) > 0:
            pass  # Train!!
        else:
            logging.logger.warning("no sentences to train model on. Are you sure the sentence configuration is " + \
                                   "correctly defined?")
