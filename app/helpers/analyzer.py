import abc
from configparser import NoOptionError
from helpers.singletons import settings, es, logging
import helpers.utils
from helpers.outlier import Outlier


class Analyzer(abc.ABC):

    def __init__(self, config_section_name):
        # the configuration file section for the use case, for example [simplequery_test_model]
        self.config_section_name = config_section_name

        # split the configuration section into the model type ("simplequery") and the model nalem ("test_model")
        self.model_type = self.config_section_name.split("_")[0]
        self.model_name = "_".join((self.config_section_name.split("_")[1:]))

        # extract all settings for this use case
        self.model_settings = self._extract_model_settings()

        self.lucene_query = es.filter_by_query_string(self.model_settings["es_query_filter"])

        self.total_events = 0
        self.outliers = list()

    def _extract_model_settings(self):
        model_settings = dict()
        model_settings["es_query_filter"] = settings.config.get(self.config_section_name, "es_query_filter")
        model_settings["outlier_reason"] = settings.config.get(self.config_section_name, "outlier_reason")
        model_settings["outlier_type"] = settings.config.get(self.config_section_name, "outlier_type")
        model_settings["outlier_summary"] = settings.config.get(self.config_section_name, "outlier_summary")

        try:
            model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(self.config_section_name, "should_notify")
        except NoOptionError:
            model_settings["should_notify"] = False

        self.should_test_model = settings.config.getboolean("general", "run_models") and settings.config.getboolean(self.config_section_name, "run_model")
        self.should_run_model = settings.config.getboolean("general", "test_models") and settings.config.getboolean(self.config_section_name, "test_model")

        try:
            model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and settings.config.getboolean(self.config_section_name, "should_notify")
        except NoOptionError:
            model_settings["should_notify"] = False

        return model_settings

    def print_analysis_summary(self):
        if len(self.outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in self.outliers))
            logging.logger.info("total outliers processed for use case: " + str(len(self.outliers)) + " [" + str(unique_summaries) + " unique]")
        else:
            logging.logger.info("no outliers detected")

    def print_batch_summary(self):
        if len(self.outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in self.outliers))
            logging.logger.info("total outliers processed in batch: " + str(len(self.outliers)) + " [" + str(unique_summaries) + " unique]")
        else:
            logging.logger.info("no outliers detected in batch")

    def process_outlier(self, fields, doc, extra_outlier_information=dict()):
        outlier_summary = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_summary"], fields)
        outlier_type = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_type"], fields)
        outlier_reason = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_reason"], fields)

        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        outlier = Outlier(type=outlier_type, reason=outlier_reason, summary=outlier_summary)

        if len(outlier_assets) > 0:
            outlier.outlier_dict["assets"] = outlier_assets

        for k, v in extra_outlier_information.items():
            outlier.outlier_dict[k] = v

        self.outliers.append(outlier)
        es.process_outliers(doc=doc, outliers=[outlier], should_notify=self.model_settings["should_notify"])

        return outlier

    @abc.abstractmethod
    def evaluate_model(self):
        raise NotImplementedError()
