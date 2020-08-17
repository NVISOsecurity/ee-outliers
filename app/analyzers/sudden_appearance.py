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

        self.model_settings["max_num_aggs"] = settings.config.getint("sudden_appearance",
                                                                     "max_num_aggregators",
                                                                     fallback=100000)
        self.model_settings["max_num_targets"] = settings.config.getint("sudden_appearance",
                                                                        "max_num_targets",
                                                                        fallback=100000)

        self.model_settings["target"] = self.config_section["target"].replace(' ', '').split(",")
        self.model_settings["aggregator"] = self.config_section["aggregator"].replace(' ', '').split(",")

        self.delta_history_win = dt.timedelta(days=self.model_settings["history_window_days"],
                                              hours=self.model_settings["history_window_hours"])

        self.model_settings["sliding_window_size"] = self.config_section.get("sliding_window_size")
        days, hours, minutes = map(int, self.model_settings["sliding_window_size"].split(':'))
        if days + hours + minutes == 0:
            raise ValueError("The sliding_window_size should be bigger than 0")
        self.delta_slide_win = dt.timedelta(days=days,
                                            hours=hours,
                                            minutes=minutes)

        self.model_settings["sliding_window_step_size"] = self.config_section.get("sliding_window_step_size")
        days, hours, minutes = map(int, self.model_settings["sliding_window_step_size"].split(':'))
        if days + hours + minutes == 0:
            raise ValueError("The sliding_window_step_size should be bigger than 0")
        self.jump_win = dt.timedelta(days=days,
                                     hours=hours,
                                     minutes=minutes)

        if self.delta_history_win < self.delta_slide_win:
            raise ValueError("sliding_window_size of %s should not be bigger than history_window of %s"
                             % (str(self.delta_slide_win), str(self.delta_history_win)))

        if self.delta_slide_win < self.jump_win:
            raise ValueError("sliding_window_step_size of %s should not be bigger than sliding_window_size of %s"
                             % (str(self.jump_win), str(self.delta_slide_win)))

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
        the (end_slide_win - self.jump_win)

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

                if event_timestamp > (end_slide_win - self.jump_win):
                    # retrieve extra information
                    extra_outlier_information = dict()
                    extra_outlier_information["size_time_window"] = str(self.delta_slide_win)
                    extra_outlier_information["start_time_window"] = str(start_slide_win)
                    extra_outlier_information["end_time_window"] = str(end_slide_win)
                    extra_outlier_information["aggregator"] = self.model_settings["aggregator"]
                    extra_outlier_information["aggregator_value"] = aggregator_bucket["key"]
                    extra_outlier_information["target"] = self.model_settings["target"]
                    extra_outlier_information["target_value"] = doc["key"]
                    extra_outlier_information["num_target_value_in_window"] = doc["doc_count"]

                    outlier = self.create_outlier(fields,
                                                  raw_doc,
                                                  extra_outlier_information=extra_outlier_information)
                    self.process_outlier(outlier)

                    summary = "\nIn aggregator '%s: %s',\n the field(s) '%s: %s',\n appear(s) " \
                              "suddenly the %s,\n in a time window of size %s." % \
                              (", ".join(self.model_settings["aggregator"]),
                               aggregator_bucket["key"],
                               " ,".join(self.model_settings["target"]),
                               doc["key"],
                               str(event_timestamp),
                               self.delta_slide_win)
                    logging.logger.debug(summary)
                    # logging.logger.debug(outlier)

        logging.tick(self.num_event_proc)
