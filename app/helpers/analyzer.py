import abc
from configparser import NoOptionError
from helpers.singletons import settings, es, logging
import helpers.utils
from helpers.outlier import Outlier


class Analyzer(abc.ABC):

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_settings = self._extract_model_settings()

        self.lucene_query = es.filter_by_query_string(self.model_settings["es_query_filter"])
        self.total_events = es.count_documents(lucene_query=self.lucene_query)

        logging.print_analysis_intro(event_type="evaluating " + self.model_name, total_events=self.total_events)

    def _extract_model_settings(self):
        model_settings = dict()
        model_settings["es_query_filter"] = settings.config.get(self.model_name, "es_query_filter")
        model_settings["outlier_reason"] = settings.config.get(self.model_name, "outlier_reason")
        model_settings["outlier_type"] = settings.config.get(self.model_name, "outlier_type")
        model_settings["outlier_summary"] = settings.config.get(self.model_name, "outlier_summary")

        try:
            model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(self.model_name, "should_notify")
        except NoOptionError:
            model_settings["should_notify"] = False

        return model_settings

    def print_batch_summary(self, outliers):
        if len(outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in outliers))
            logging.logger.info("total outliers processed in batch: " + str(len(outliers)) + " [" + str(unique_summaries) + " unique]")

    def create_outlier(self, fields):
        outlier_summary = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_summary"], fields)
        outlier_type = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_type"], fields)
        outlier_reason = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_reason"], fields)

        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        outlier = Outlier(type=outlier_type, reason=outlier_reason, summary=outlier_summary)

        if len(outlier_assets) > 0:
            outlier.outlier_dict["assets"] = outlier_assets

        return outlier

    @abc.abstractmethod
    def evaluate_model(self):
        raise NotImplementedError()
