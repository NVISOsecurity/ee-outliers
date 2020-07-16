from helpers.singletons import settings, es, logging
from helpers.analyzer import Analyzer
import datetime as dt
import dateutil.parser
import math


class SuddenAppearanceAnalyzer(Analyzer):

    def __init__(self, model_name, config_section):
        super(SuddenAppearanceAnalyzer, self).__init__("sudden_appearance", model_name, config_section)
        self.end_time = dt.datetime.now()
        self.num_event_proc = 0  # Current number of events processed

    def _extract_additional_model_settings(self):
        """
        Override method from Analyzer
        """
        self.model_settings["timestamp_field"] = settings.config.get("general",
                                                                     "timestamp_field",
                                                                     fallback="@timestamp")

        self.model_settings["target"] = self.config_section["target"].replace(' ', '').split(",")
        self.model_settings["aggregator"] = self.config_section["aggregator"].replace(' ', '').split(",")

        self.model_settings["trigger_slide_window_proportion"] = self.config_section.getfloat(
            "trigger_slide_window_proportion")

        self.delta_history_win = self._extract_delta_window(day_parameter_name="history_window_days",
                                                            hour_parameter_name="history_window_hours")

        self.delta_slide_win = self._extract_delta_window(day_parameter_name="slide_window_days",
                                                          hour_parameter_name="slide_window_hours",
                                                          min_parameter_name="slide_window_mins")

        self.jump_win = self._extract_delta_window(day_parameter_name="slide_jump_days",
                                                   hour_parameter_name="slide_jump_hours",
                                                   min_parameter_name="slide_jump_mins")

        error_if_win1_smaller_than_win2(win1=self.delta_history_win,
                                        win2=self.delta_slide_win,
                                        win1_name="history window",
                                        win2_name="slide window")
        error_if_win1_smaller_than_win2(win1=self.delta_slide_win,
                                        win2=self.jump_win,
                                        win1_name="slide window",
                                        win2_name="jump window")

    def _extract_delta_window(self, day_parameter_name, hour_parameter_name, min_parameter_name=None):
        """
        Return the time window in datetime.timedelta format from the day, hour and minute parameter names.
        Raise a error if time window is equal to 0

        :param day_parameter_name: day parameter name
        :param hour_parameter_name: hour parameter name
        :param min_parameter_name: minute parameter name
        :return: datetime.timedelta format of the time window
        """
        self.model_settings[day_parameter_name] = self.extract_parameter(day_parameter_name,
                                                                         param_type="int",
                                                                         section_name="general")
        self.model_settings[hour_parameter_name] = self.extract_parameter(hour_parameter_name,
                                                                          param_type="int",
                                                                          section_name="general")
        if min_parameter_name is not None:
            self.model_settings[min_parameter_name] = self.extract_parameter(min_parameter_name,
                                                                             param_type="int",
                                                                             section_name="general",
                                                                             default=0)
            delta_win = dt.timedelta(days=self.model_settings[day_parameter_name],
                                     hours=self.model_settings[hour_parameter_name],
                                     minutes=self.model_settings[min_parameter_name])
            if delta_win == dt.timedelta(days=0, hours=0, minutes=0):
                raise ValueError("%s, %s and %s parameters shouldn't not be all set to 0." %
                                 (day_parameter_name, hour_parameter_name, min_parameter_name))
        else:
            delta_win = dt.timedelta(days=self.model_settings[day_parameter_name],
                                     hours=self.model_settings[hour_parameter_name])
            if delta_win == dt.timedelta(days=0, hours=0, minutes=0):
                raise ValueError("%s and %s parameters shouldn't not be all set to 0." %
                                 (day_parameter_name, hour_parameter_name))
        return delta_win

    def evaluate_model(self):
        self.total_events = es.count_documents(index=self.model_settings["es_index"],
                                               search_query=self.search_query,
                                               model_settings=self.model_settings)

        self.print_analysis_intro(event_type="evaluating " + self.model_type + "_" + self.model_name,
                                  total_events=self.total_events)

        # Compute the number of times we will scan the slide window
        num_jump = math.ceil((self.delta_history_win - self.delta_slide_win) / self.jump_win)
        num_scan = num_jump + 1

        logging.init_ticker(total_steps=num_scan,
                            desc=self.model_name + " - evaluating " + self.model_type + " model")

        start_slide_win = self.end_time - self.delta_history_win
        end_slide_win = start_slide_win + self.delta_slide_win

        if end_slide_win == self.end_time:
            self.find_sudden_appearance(start_slide_win, end_slide_win)

        while end_slide_win < self.end_time:
            self.find_sudden_appearance(start_slide_win, end_slide_win)
            start_slide_win += self.jump_win
            end_slide_win += self.jump_win
            if end_slide_win >= self.end_time:
                end_slide_win = self.end_time
                start_slide_win = self.end_time - self.jump_win
                self.find_sudden_appearance(start_slide_win, end_slide_win)

        self.print_analysis_summary()

    def find_sudden_appearance(self, start_slide_win, end_slide_win):
        """
        Find sudden apparition in aggregation defined by self.model_settings["aggregator"] of a term field defined by
        self.model_settings["target"] in events within the time window defined by start_slide_win and en_slide_win
        and create outliers. An event is considered as outlier when a term field appear for the first time after
        a certain proportion of the time window. This proportion is defined by
        self.model_settings["trigger_slide_window_proportion"].

        :param start_slide_win: start time of the time window
        :param end_slide_win: end time of the time window
        """
        aggregator_buckets = es.scan_first_occur_documents(search_query=self.search_query,
                                                           start_time=start_slide_win,
                                                           end_time=end_slide_win,
                                                           model_settings=self.model_settings)
        # Loop over the aggregations
        for aggregator_bucket in aggregator_buckets:
            target_buckets = aggregator_bucket["target"]["buckets"]
            logging.logger.debug(aggregator_bucket["key"])
            # Loop over the documents in aggregation
            for doc in target_buckets:
                self.num_event_proc += doc["doc_count"]
                raw_doc = doc["top_doc"]["hits"]["hits"][0]
                fields = es.extract_fields_from_document(raw_doc,
                                                         extract_derived_fields=self.model_settings[
                                                             "use_derived_fields"])
                # convert the event timestamp in the right format
                event_timestamp = dateutil.parser.parse(fields[self.model_settings["timestamp_field"]],
                                                        ignoretz=True)
                # Compute the proportion, compared to the time window, the event appear
                prop_time_win_event_appears = (event_timestamp - start_slide_win) / (end_slide_win - start_slide_win)
                if prop_time_win_event_appears > self.model_settings["trigger_slide_window_proportion"]:
                    # retrieve extra information
                    extra_outlier_information = dict()
                    extra_outlier_information["prop_first_appear_in_time_window"] = prop_time_win_event_appears
                    trigger_slide_window_proportion_str = str(self.model_settings["trigger_slide_window_proportion"])
                    extra_outlier_information["trigger_slide_window_proportion"] = trigger_slide_window_proportion_str
                    extra_outlier_information["size_time_window"] = str(self.delta_slide_win)
                    extra_outlier_information["start_time_window"] = str(start_slide_win)
                    extra_outlier_information["end_time_window"] = str(end_slide_win)
                    extra_outlier_information["aggregator"] = self.model_settings["aggregator"]
                    extra_outlier_information["aggregator_value"] = aggregator_bucket["key"]
                    extra_outlier_information["target"] = self.model_settings["target"]
                    extra_outlier_information["target_value"] = doc["key"]
                    extra_outlier_information["num_target_value_in_window"] = doc["doc_count"]
                    extra_outlier_information["resume"] = "In aggregator '%s: %s', the field(s) '%s: %s' appear(s) " \
                                                          "suddenly at %s%%(%s) of the time window of size %s days, " \
                                                          "%s hours, %s minutes" % \
                                                          (", ".join(self.model_settings["aggregator"]),
                                                           aggregator_bucket["key"],
                                                           " ,".join(self.model_settings["target"]),
                                                           doc["key"],
                                                           "{:.2}".format(prop_time_win_event_appears),
                                                           str(event_timestamp),
                                                           self.config_section["slide_window_days"],
                                                           self.config_section["slide_window_hours"],
                                                           self.config_section["slide_window_mins"])

                    outlier = self.create_outlier(fields,
                                                  raw_doc,
                                                  extra_outlier_information=extra_outlier_information)
                    logging.logger.debug(outlier)
                    self.process_outlier(outlier)

        logging.tick(self.num_event_proc)


def error_if_win1_smaller_than_win2(win1, win2, win1_name, win2_name):
    """
    Print an error message if the size of the time window 1 is smaller than the size of the time window 2.

    :param win1: Time window 1
    :param win2: Time window 2
    :param win1_name: String name of the time window 1
    :param win2_name: String name of the time window 2
    """
    if win1 < win2:
        raise ValueError("%s of size %s is bigger than %s of size %s" %
                         (win2_name, win2, win1_name, win1))
