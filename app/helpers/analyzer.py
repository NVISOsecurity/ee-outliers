import abc
from configparser import NoOptionError
from helpers.singletons import settings, es, logging
import helpers.utils
from helpers.outlier import Outlier

from typing import Dict, Any, List


class Analyzer(abc.ABC):

    def __init__(self, config_section_name: str) -> None:
        # the configuration file section for the use case, for example [simplequery_test_model]
        self.config_section_name: str = config_section_name

        # split the configuration section into the model type ("simplequery") and the model nalem ("test_model")
        self.model_type: str = self.config_section_name.split("_")[0]
        self.model_name: str = "_".join((self.config_section_name.split("_")[1:]))

        # extract all settings for this use case
        self.model_settings: Dict[str, Any] = self._extract_model_settings()

        self.search_query: Dict[str, List]
        if self.model_settings["es_query_filter"]:
            self.search_query = es.filter_by_query_string(self.model_settings["es_query_filter"])

        if self.model_settings["es_dsl_filter"]:
            self.search_query = es.filter_by_dsl_query(self.model_settings["es_dsl_filter"])

        self.total_events: int = 0
        self.outliers: List[Outlier] = list()

    def _extract_model_settings(self) -> Dict[str, Any]:
        model_settings: Dict[str, Any] = dict()

        try:
            model_settings["es_query_filter"] = settings.config.get(self.config_section_name, "es_query_filter")
        except NoOptionError:
            model_settings["es_query_filter"] = None

        try:
            model_settings["es_dsl_filter"] = settings.config.get(self.config_section_name, "es_dsl_filter")
        except NoOptionError:
            model_settings["es_dsl_filter"] = None

        try:
            model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and \
                                              settings.config.getboolean(self.config_section_name, "should_notify")
        except NoOptionError:
            model_settings["should_notify"] = False

        try:
            model_settings["use_derived_fields"] = settings.config.getboolean(self.config_section_name,
                                                                              "use_derived_fields")
        except NoOptionError:
            model_settings["use_derived_fields"] = False

        try:
            model_settings["should_notify"] = settings.config.getboolean("notifier", "email_notifier") and \
                                              settings.config.getboolean(self.config_section_name, "should_notify")
        except NoOptionError:
            model_settings["should_notify"] = False

        model_settings["outlier_reason"] = settings.config.get(self.config_section_name, "outlier_reason")
        model_settings["outlier_type"] = settings.config.get(self.config_section_name, "outlier_type")
        model_settings["outlier_summary"] = settings.config.get(self.config_section_name, "outlier_summary")

        self.should_test_model = settings.config.getboolean("general", "run_models") and \
                                 settings.config.getboolean(self.config_section_name, "run_model")
        self.should_run_model = settings.config.getboolean("general", "test_models") and \
                                settings.config.getboolean(self.config_section_name, "test_model")

        return model_settings

    def print_analysis_summary(self) -> None:
        if len(self.outliers) > 0:
            unique_summaries = len(set(o.outlier_dict["summary"] for o in self.outliers))
            logging.logger.info("total outliers processed for use case: " + str(len(self.outliers)) + \
                                " [" + str(unique_summaries) + " unique summaries]")
        else:
            logging.logger.info("no outliers detected for use case")

    def process_outlier(self, fields: Dict, doc: Dict[str, Any], extra_outlier_information: Dict=dict()) -> Outlier:
        extra_outlier_information["model_name"] = self.model_name
        extra_outlier_information["model_type"] = self.model_type

        fields_and_extra_outlier_information = fields.copy()
        fields_and_extra_outlier_information.update(extra_outlier_information)

        outlier_summary = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_summary"],
                                                                               fields_and_extra_outlier_information)
        outlier_type = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_type"],
                                                                            fields_and_extra_outlier_information)
        outlier_reason = helpers.utils.replace_placeholder_fields_with_values(self.model_settings["outlier_reason"],
                                                                              fields_and_extra_outlier_information)

        outlier_assets = helpers.utils.extract_outlier_asset_information(fields, settings)
        outlier: Outlier = Outlier(type=outlier_type, reason=outlier_reason, summary=outlier_summary)

        if len(outlier_assets) > 0:
            outlier.outlier_dict["assets"] = outlier_assets

        for k, v in extra_outlier_information.items():
            outlier.outlier_dict[k] = v

        self.outliers.append(outlier)
        es.process_outliers(doc=doc, outliers=[outlier], should_notify=self.model_settings["should_notify"])

        return outlier

    @abc.abstractmethod
    def evaluate_model(self) -> None:
        raise NotImplementedError()
