import abc
from configparser import NoOptionError

import dateutil

from helpers.singletons import settings, es, logging
import helpers.utils
from helpers.outlier import Outlier

from typing import Dict, Set, Tuple, Any, List, Union, Optional, cast


# pylint: disable=too-many-instance-attributes
class Analyzer(abc.ABC):
    """
    The base class for all analyzers.
    This class keeps state around the model type, processed events, identified outliers, etc.
    It is also resposible for extracting the model settings from the configuration files.
    """
    DEFAULT_CONFIG_KEY = {"run_model", "test_model", "es_index"}

    def __init__(self, config_section_name: str) -> None:
        # the configuration file section for the use case, for example [simplequery_test_model]
        self.config_section_name: str = config_section_name

        # split the configuration section into the model type ("simplequery") and the model name ("test_model")
        self.model_type: str = self.config_section_name.split("_")[0]
        self.model_name: str = "_".join((self.config_section_name.split("_")[1:]))

        self.total_events: int = 0
        self.total_outliers: int = 0
        self.outlier_summaries: Set[str] = set()

        self.analysis_start_time: Optional[float] = None
        self.analysis_end_time: Optional[float] = None

        self.completed_analysis: bool = False
        self.index_not_found_analysis: bool = False
        self.unknown_error_analysis: bool = False

        self.nr_whitelisted_elements: int = 0

        # extract all settings for this use case
        self.configuration_parsing_error: bool = False

        try:
            self.model_settings: Dict[str, Any] = self._extract_model_settings()
            self._extract_additional_model_settings()
            self.extra_model_settings = self._extract_arbitrary_config()
        except Exception:  # pylint: disable=broad-except
            logging.logger.error("error while parsing use case configuration for %s", self.config_section_name,
                                 exc_info=True)
            self.configuration_parsing_error = True

    @property
    def analysis_time_seconds(self) -> Optional[float]:
        """
        Return the time that was needed to complete the analysis, in seconds.

        :return: float value that represent the time in seconds. In case the analysis has not completed,
        this will return None.
        """
        if self.completed_analysis:
            return float(self.analysis_end_time - self.analysis_start_time)

        return None

    def _extract_model_settings(self) -> Dict[str, Any]:
        model_settings: Dict[str, Optional[Union[int, str, bool]]] = dict()

        # by default, we don't process documents chronologically when analyzing the model, as it
        # has a high impact on performance when scanning in Elasticsearch
        model_settings["process_documents_chronologically"] = True

        try:
            model_settings["es_query_filter"] = settings.config.get(self.config_section_name, "es_query_filter")
            self.search_query = es.filter_by_query_string(cast(str, model_settings["es_query_filter"]))

        except NoOptionError:
            model_settings["es_query_filter"] = None

        try:
            model_settings["es_dsl_filter"] = settings.config.get(self.config_section_name, "es_dsl_filter")
            self.search_query = es.filter_by_dsl_query(cast(str, model_settings["es_dsl_filter"]))

        except NoOptionError:
            model_settings["es_dsl_filter"] = None

        try:
            model_settings["timestamp_field"] = settings.config.get(self.config_section_name, "timestamp_field")
        except NoOptionError:
            model_settings["timestamp_field"] = settings.config.get("general", "timestamp_field", fallback="timestamp")

        try:
            model_settings["history_window_days"] = settings.config.getint(self.config_section_name,
                                                                           "history_window_days")
        except NoOptionError:
            model_settings["history_window_days"] = settings.config.getint("general", "history_window_days")

        try:
            model_settings["history_window_hours"] = settings.config.getint(self.config_section_name,
                                                                            "history_window_hours")
        except NoOptionError:
            model_settings["history_window_hours"] = settings.config.getint("general", "history_window_hours")

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
            self.es_index = settings.config.get(self.config_section_name, "es_index")
        except NoOptionError:
            self.es_index = settings.config.get("general", "es_index_pattern")

        model_settings["outlier_reason"] = settings.config.get(self.config_section_name, "outlier_reason")
        model_settings["outlier_type"] = settings.config.get(self.config_section_name, "outlier_type")
        model_settings["outlier_summary"] = settings.config.get(self.config_section_name, "outlier_summary")

        self.should_run_model: bool = settings.config.getboolean("general",
                                                                 "run_models") and settings.config.getboolean(
            self.config_section_name, "run_model")
        self.should_test_model: bool = settings.config.getboolean("general",
                                                                  "test_models") and settings.config.getboolean(
            self.config_section_name, "test_model")
        return model_settings

    def _extract_additional_model_settings(self) -> None:
        """
        Method call in the construction to load all parameters of this analyzer
        This method can be overridden by children to load content linked to a specific analyzer
        """

    def _extract_arbitrary_config(self) -> Dict[str, Any]:
        """
        Extract all other key in the model section to copied verbatim to the outlier

        :return: dictionary with arbitrary key config
        """
        extra_model_settings: Dict[str, Any] = dict()

        all_items = settings.config.items(self.config_section_name)
        for key, value in all_items:
            if key not in self.model_settings and key not in Analyzer.DEFAULT_CONFIG_KEY:
                extra_model_settings[key] = value

        return extra_model_settings

    def print_analysis_summary(self) -> None:
        """
        Print information about the analyzer. Must be call at the end of processing
        """
        if self.total_outliers > 0:
            unique_summaries: int = len(self.outlier_summaries)
            message: str = "total outliers processed for use case: " + "{:,}".format(self.total_outliers) + " [" + \
                           "{:,}".format(unique_summaries) + " unique summaries]"
            if self.nr_whitelisted_elements > 0:
                message += " - ignored " + "{:,}".format(self.nr_whitelisted_elements) + " whitelisted outliers"
            logging.logger.info(message)
        else:
            logging.logger.info("no outliers detected for use case")

    def _prepare_outlier_parameters(self, extra_outlier_information: Dict[str, Any],
                                    fields: Dict) -> Tuple[List[str], List[str], str, List[str]]:
        """
        Compute different parameters to create outlier

        :param extra_outlier_information: information about outlier
        :param fields: fields information
        :return: outlier type, outlier reason, outlier summary, outlier assets
        """
        extra_outlier_information["model_name"] = self.model_name
        extra_outlier_information["model_type"] = self.model_type

        fields_and_extra_outlier_information: Dict = fields.copy()
        fields_and_extra_outlier_information.update(extra_outlier_information)

        outlier_summary: str = helpers.utils.replace_placeholder_fields_with_values(
            self.model_settings["outlier_summary"], fields_and_extra_outlier_information)

        # for both outlier types and reasons, we also allow the case where multiples values are provided at once.
        # example: type = malware, IDS
        outlier_type: List[str] = helpers.utils.replace_placeholder_fields_with_values(
            self.model_settings["outlier_type"], fields_and_extra_outlier_information).split(",")
        outlier_reason: List[str] = helpers.utils.replace_placeholder_fields_with_values(
            self.model_settings["outlier_reason"], fields_and_extra_outlier_information).split(",")

        # remove any leading or trailing whitespace from either. For example: "type = malware,  IDS" should just
        # return ["malware","IDS"] instead of ["malware", "  IDS"]
        outlier_type = [item.strip() for item in outlier_type]
        outlier_reason = [item.strip() for item in outlier_reason]

        outlier_assets: List[str] = helpers.utils.extract_outlier_asset_information(fields, settings)
        return outlier_type, outlier_reason, outlier_summary, outlier_assets

    def create_outlier(self, fields: Dict, doc: Dict[str, Any],
                       extra_outlier_information: Optional[Dict[str, Any]]=None) -> Outlier:
        """
        Create an outlier object with all the correct fields & parameters

        :param fields: extracted fields
        :param doc: document linked to this outlier
        :param extra_outlier_information: extra information that should be added to the outlier
        :return: the created outlier object
        """
        if extra_outlier_information is None:
            extra_outlier_information = dict()

        outlier_type, outlier_reason, outlier_summary, outlier_assets = \
            self._prepare_outlier_parameters(extra_outlier_information, fields)
        outlier: Outlier = Outlier(outlier_type=outlier_type, outlier_reason=outlier_reason,
                                   outlier_summary=outlier_summary, doc=doc)

        if outlier_assets:
            outlier.outlier_dict["assets"] = outlier_assets

        for key, value in self.extra_model_settings.items():
            outlier.outlier_dict[key] = value

        if extra_outlier_information:
            for outlier_key, outlier_value in extra_outlier_information.items():
                outlier.outlier_dict[outlier_key] = outlier_value

        return outlier

    def process_outlier(self, outlier: Outlier) -> None:
        """
        Save outlier (in statistic and) in ES database if not whitelisted (and if settings is configured to save in ES)

        :param outlier: outlier to save
        """
        self.total_outliers += 1
        self.outlier_summaries.add(outlier.outlier_dict["summary"])

        es.process_outlier(outlier=outlier, should_notify=self.model_settings["should_notify"])

    def print_analysis_intro(self, event_type: str, total_events: int) -> None:
        """
        Print an introduction before starting the analysis
        :param event_type: can be any string, typically evaluating or training
        :param total_events: the total number of events being processed
        """
        logging.logger.info("")
        logging.logger.info("===== %s [%s model] ===", event_type, self.model_type)
        logging.logger.info("analyzing %s events", "{:,}".format(total_events))
        logging.logger.info(self.get_time_window_info(history_days=self.model_settings["history_window_days"],
                                                      history_hours=self.model_settings["history_window_hours"]))

        if total_events == 0:
            logging.logger.warning("no events to analyze!")

    @staticmethod
    def get_time_window_info(history_days: Optional[float] = None, history_hours: Optional[float] = None) -> str:
        """
        Convert an absolute number of days and hours into a string that represents the start and end dates covering
        the time window between [now- (history_days + history_hours)] and now.
        :param history_days: total number of days to look back
        :param history_hours: total number of hours to look back
        :return:
        """
        search_range: Dict[str, Any] = es.get_time_filter(days=history_days, hours=history_hours,
                                          timestamp_field=settings.config.get("general", "timestamp_field",
                                                                              fallback="timestamp"))

        search_range_start = search_range["range"][str(settings.config.get("general", "timestamp_field",
                                                                           fallback="timestamp"))]["gte"]
        search_range_end = search_range["range"][str(settings.config.get("general", "timestamp_field",
                                                                         fallback="timestamp"))]["lte"]

        search_start_range_printable: str = dateutil.parser.parse(search_range_start).strftime('%Y-%m-%d %H:%M:%S')
        search_end_range_printable: str = dateutil.parser.parse(search_range_end).strftime('%Y-%m-%d %H:%M:%S')
        return "processing events between " + search_start_range_printable + " and " + search_end_range_printable

    @abc.abstractmethod
    def evaluate_model(self) -> None:
        """
        This is the method that will be called whenever all the default analyzer parameters have been initialized.
        This is the only method that should be implemented in the subclasses and will be responsible for actual
        detection of outliers.
        """
        raise NotImplementedError()
