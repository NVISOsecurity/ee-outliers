import json
import datetime as dt
import math
import copy
import os

from collections import defaultdict
from itertools import chain

from pygrok import Grok
from elasticsearch import helpers as eshelpers, Elasticsearch
from elasticsearch.exceptions import AuthenticationException, ConnectionError, RequestError

from configparser import NoOptionError

import helpers.utils
import helpers.logging

from helpers.singleton import singleton
from helpers.notifier import Notifier
from helpers.outlier import Outlier

DEFAULT_TIMESTAMP_FIELD = "@timestamp"


@singleton
class ES:
    """
    This is the singleton class that holds the connection object with Elasticsearch.
    It also serves as the queue for all the queued bulk actions.

    This class is responsible for performing all operations that directly interact with Elasticsearch,
    or that create the data structures needed to work with Elasticsearch.

    This includes removing all the outliers from Elasticsearch, scanning, counting documents, etc.
    """
    index = None
    conn = None
    settings = None

    grok_filters = dict()

    notifier = None

    bulk_actions = []

    BULK_FLUSH_SIZE = 1000

    def __init__(self, settings=None, logging=None):
        self.settings = settings
        self.logging = logging

        if self.settings.config.getboolean("notifier", "email_notifier"):
            self.notifier = Notifier(settings, logging)

    def init_connection(self):
        """
        Initialize the connection with Elasticsearch

        :return: Connection object if connection with Elasticsearch succeeded, False otherwise
        """

        http_auth = (os.getenv("es_username",""),os.getenv("es_password",""))
        verify_certs = os.getenv("verify_certs", "true").lower() in ["yes", "true", "1"]
        ca_certs = os.getenv("ca_certs", None)
        url = self.settings.config.get("general", "es_url")

        self.logging.logger.debug(f"url: {url}, username: {http_auth[0]}, verify_certs: {verify_certs}, ca_certs: {ca_certs}")

        self.conn = Elasticsearch([url],
                                  http_auth=http_auth,
                                  timeout=self.settings.config.getint("general", "es_timeout"),
                                  verify_certs=verify_certs,
                                  ca_certs=ca_certs,
                                  retry_on_timeout=True)

        try:
            self.conn.info()
            self.logging.logger.info("connected to Elasticsearch on host %s" %
                                     (self.settings.config.get("general", "es_url")))
            return self.conn
        except AuthenticationException as e:
            self.logging.logger.error("could not connect to Elasticsearch on host %s due to authentication error." %
                                      (self.settings.config.get("general", "es_url")))
            self.logging.logger.debug(str(e))
            return False
        except ConnectionError as e:
            self.logging.logger.error("could not connect to Elasticsearch on host %s" %
                                      (self.settings.config.get("general", "es_url")))
            self.logging.logger.debug(str(e))
            return False

    def _get_history_window(self, model_settings=None):
        """
        Get different parameters about windows settings

        :param model_settings: model configuration part
        :return: timtestamp field name, number of recover day , number of recover hours (in addition to the days)
        """
        if model_settings is None:
            timestamp_field = self.settings.config.get("general", "timestamp_field", fallback=DEFAULT_TIMESTAMP_FIELD)
            history_window_days = self.settings.config.getint("general", "history_window_days")
            history_window_hours = self.settings.config.getint("general", "history_window_hours")
        else:
            timestamp_field = model_settings["timestamp_field"]
            history_window_days = model_settings["history_window_days"]
            history_window_hours = model_settings["history_window_hours"]
        return timestamp_field, history_window_days, history_window_hours

    def _scan(self, index, search_range, bool_clause=None, sort_clause=None, query_fields=None, search_query=None,
              model_settings=None):
        """
        Scan and get documents in Elasticsearch

        :param index: on which index the request must be done
        :param search_range: the range of the search
        :param bool_clause: boolean condition
        :param sort_clause: request to sort results
        :param query_fields: the query field
        :param search_query: the search query
        :param model_settings: part of the configuration linked to the model
        :return: generator to fetch documents
        """
        preserve_order = False

        highlight_settings = self._get_highlight_settings(model_settings)

        if model_settings is not None and model_settings["process_documents_chronologically"]:
            sort_clause = {"sort": [{model_settings["timestamp_field"]: "asc"}]}
            preserve_order = True

        return eshelpers.scan(self.conn, request_timeout=self.settings.config.getint("general", "es_timeout"),
                              index=index, query=build_search_query(bool_clause=bool_clause,
                                                                    sort_clause=sort_clause,
                                                                    search_range=search_range,
                                                                    query_fields=query_fields,
                                                                    search_query=search_query,
                                                                    highlight_settings=highlight_settings),
                              size=self.settings.config.getint("general", "es_scan_size"),
                              scroll=self.settings.config.get("general", "es_scroll_time"),
                              preserve_order=preserve_order, raise_on_error=False)

    def _count_documents(self, index, search_range, bool_clause=None, query_fields=None, search_query=None):
        """
        Count number of document in Elasticsearch that match the query

        :param index: on which index the request must be done
        :param search_range: the range of research
        :param bool_clause: boolean condition
        :param query_fields: the query field
        :param search_query: the search query
        :return: number of document
        """
        res = self.conn.count(index=index, body=build_search_query(bool_clause=bool_clause, search_range=search_range,
                                                                   query_fields=query_fields,
                                                                   search_query=search_query))

        return res["count"]

    def count_and_scan_documents(self, index, bool_clause=None, sort_clause=None, query_fields=None, search_query=None,
                                 model_settings=None):
        """
        Count the number of document and fetch them from Elasticsearch

        :param index: on which index the request must be done
        :param bool_clause: boolean condition
        :param sort_clause: request to sort result
        :param query_fields: the query field
        :param search_query: the search query
        :param model_settings: part of the configuration linked to the model
        :return: the number of document and a generator/list of all documents
        """
        timestamp_field, history_window_days, history_window_hours = self._get_history_window(model_settings)
        search_range = self.get_time_filter(days=history_window_days, hours=history_window_hours,
                                            timestamp_field=timestamp_field)
        total_events = self._count_documents(index, search_range, bool_clause, query_fields, search_query)
        if total_events > 0:
            return total_events, self._scan(index, search_range, bool_clause, sort_clause, query_fields, search_query,
                                            model_settings)
        return total_events, []

    def count_documents(self, index, search_query, model_settings, bool_clause=None, query_fields=None):
        """
        Count the number of document

        :param index: on which index the request must be done
        :param search_query: the search query
        :param model_settings: part of the configuration linked to the model
        :param bool_clause: boolean condition
        :param query_fields: the query field
        :return: the number of document
        """
        timestamp_field, history_window_days, history_window_hours = self._get_history_window(model_settings)
        search_range = self.get_time_filter(days=history_window_days, hours=history_window_hours,
                                            timestamp_field=timestamp_field)
        return self._count_documents(index, search_range, bool_clause, query_fields, search_query)

    def scan_first_occur_documents(self, search_query, start_time, end_time, model_settings):
        """
        Retrieve from Elasticsearch in aggregation defined by model_settings["aggregator"], the first occurrence of a
        term field defined by model_settings["target"], of events within the time window defined by start_time and
        end_time.

        :param search_query: the search query
        :param start_time: start time of the time window
        :param end_time: end time of the time window
        :param model_settings: part of the configuration linked to the model
        :return aggregator_buckets: list of aggregation buckets
        """
        search_range = self.get_time_filter(start_time=start_time,
                                            end_time=end_time,
                                            timestamp_field=model_settings["timestamp_field"])
        first_occur_search_query = build_first_occur_search_query(search_query=search_query,
                                                                  search_range=search_range,
                                                                  target_list=model_settings["target"],
                                                                  aggregator_list=model_settings["aggregator"],
                                                                  timestamp=model_settings["timestamp_field"],
                                                                  max_num_aggs=model_settings["max_num_aggs"],
                                                                  max_num_targets=model_settings["max_num_targets"])
        try:
            results = self.conn.search(index=model_settings["es_index"], body=first_occur_search_query)
        except RequestError as e:
            self.logging.logger.error(e.error)
            self.logging.logger.error(json.dumps(e.info, indent=4))
            return []

        if results["_shards"]["failed"] > 0:
            self.logging.logger.warning("Search query failure(s): ")
            for failure in results["_shards"]["failures"]:
                self.logging.logger.warning("Failed script: %s caused by %s: %s" %
                                            (failure["reason"]["script"],
                                             failure["reason"]["caused_by"]["type"],
                                             failure["reason"]["caused_by"]["reason"]))

        aggregator_buckets = results["aggregations"]["aggregator"]["buckets"]
        return aggregator_buckets

    @staticmethod
    def filter_by_query_string(query_string=None):
        """
        Format a query request

        :param query_string: the query request
        :return: query request formatted for Elasticsearch
        """
        filter_clause = {"filter": [
            {"query_string": {"query": query_string}}
        ]}

        return filter_clause

    @staticmethod
    def filter_by_dsl_query(dsl_query=None):
        """
        Format a DSL query

        :param dsl_query: the DSL query
        :return: the formatted request
        """
        dsl_query = json.loads(dsl_query)

        if isinstance(dsl_query, list):
            filter_clause = {"filter": []}
            for query in dsl_query:
                filter_clause["filter"].append(query["query"])
        else:
            filter_clause = {"filter": [
                dsl_query["query"]
            ]}
        return filter_clause

    # this is part of housekeeping, so we should not access non-threat-save objects, such as logging progress to
    # the console using ticks!
    def remove_all_whitelisted_outliers(self, dict_with_analyzer):
        """
        Remove all whitelisted outliers present in Elasticsearch.
        This method is normally only call by housekeeping

        :return: the number of outliers removed
        """
        outliers_filter_query = {"filter": [{"term": {"tags": "outlier"}}]}

        total_outliers_whitelisted = 0
        total_outliers_processed = 0

        idx = self.settings.config.get("general", "es_index_pattern")
        total_nr_outliers, documents = self.count_and_scan_documents(index=idx, bool_clause=outliers_filter_query)

        if total_nr_outliers > 0:
            self.logging.logger.info("going to analyze %s outliers and remove all whitelisted items", "{:,}"
                                     .format(total_nr_outliers))
            start_time = dt.datetime.today().timestamp()

            for doc in documents:
                total_outliers_processed = total_outliers_processed + 1
                total_outliers_in_doc = int(doc["_source"]["outliers"]["total_outliers"])
                # generate all outlier objects for this document
                total_whitelisted = 0

                for i in range(total_outliers_in_doc):
                    outlier_type = doc["_source"]["outliers"]["type"][i]
                    outlier_reason = doc["_source"]["outliers"]["reason"][i]
                    outlier_summary = doc["_source"]["outliers"]["summary"][i]

                    # Extract information and get analyzer linked to this outlier
                    model_name = doc["_source"]["outliers"]["model_name"][i]
                    model_type = doc["_source"]["outliers"]["model_type"][i]
                    config_section_name = model_type + "_" + model_name
                    if config_section_name not in dict_with_analyzer:
                        self.logging.logger.debug("Outlier '" + config_section_name + "' " +
                                                  " was not found in configuration, could not check whitelist")
                        break  # If one outlier is not whitelisted, we keep all other outliers
                    analyzer = dict_with_analyzer[config_section_name]

                    outlier = Outlier(outlier_type=outlier_type, outlier_reason=outlier_reason,
                                      outlier_summary=outlier_summary, doc=doc)
                    if outlier.is_whitelisted(extra_literals_whitelist_value=analyzer.model_whitelist_literals,
                                              extra_regexps_whitelist_value=analyzer.model_whitelist_regexps):
                        total_whitelisted += 1

                # if all outliers for this document are whitelisted, removed them all. If not, don't touch the document.
                # this is a limitation in the way our outliers are stored: if not ALL of them are whitelisted, we
                # can't remove just the whitelisted ones
                # from the Elasticsearch event, as they are stored as array elements and potentially contain
                # observations that should be removed, too.
                # In this case, just don't touch the document.
                if total_whitelisted == total_outliers_in_doc:
                    total_outliers_whitelisted += 1
                    doc = remove_outliers_from_document(doc)
                    self.add_remove_outlier_bulk_action(doc)

                # we don't use the ticker from the logger singleton, as this will be called from the housekeeping thread
                # if we share a same ticker between multiple threads, strange results would start to appear in
                # progress logging
                # so, we duplicate part of the functionality from the logger singleton
                if self.logging.verbosity >= 5:
                    should_log = True
                else:
                    should_log = total_outliers_processed % max(1,
                                                                int(math.pow(10, (6 - self.logging.verbosity)))) == 0 \
                                 or total_outliers_processed == total_nr_outliers

                if should_log:
                    # avoid a division by zero
                    time_diff = max(float(1), float(dt.datetime.today().timestamp() - start_time))
                    ticks_per_second = "{:,}".format(round(float(total_outliers_processed) / time_diff))

                    self.logging.logger.info("whitelisting historical outliers " + " [" + ticks_per_second + " eps." +
                                             " - " + '{:.2f}'.format(round(float(total_outliers_processed) /
                                                                           float(total_nr_outliers) * 100, 2)) +
                                             "% done" + " - " + "{:,}".format(total_outliers_whitelisted) +
                                             " outliers whitelisted]")

            self.flush_bulk_actions()

        return total_outliers_whitelisted

    def remove_all_outliers(self):
        """
        Remove all outliers present in Elasticsearch
        """
        idx = self.settings.config.get("general", "es_index_pattern")

        must_clause = {"filter": [{"term": {"tags": "outlier"}}]}
        timestamp_field, history_window_days, history_window_hours = self._get_history_window()
        search_range = self.get_time_filter(days=history_window_days,
                                            hours=history_window_hours,
                                            timestamp_field=timestamp_field)
        total_outliers = self._count_documents(index=idx, search_range=search_range, bool_clause=must_clause)

        if total_outliers > 0:
            query = build_search_query(bool_clause=must_clause, search_range=search_range)

            script = {
                "source": "ctx._source.remove(\"outliers\"); " +
                          "ctx._source.tags.remove(ctx._source.tags.indexOf(\"outlier\"))",
                "lang": "painless"
            }

            query["script"] = script

            self.logging.logger.info("wiping %s existing outliers", "{:,}".format(total_outliers))
            self.conn.update_by_query(index=idx, body=query, refresh=True, wait_for_completion=True)
            self.logging.logger.info("wiped outlier information of " + "{:,}".format(total_outliers) + " documents")
        else:
            self.logging.logger.info("no existing outliers were found, so nothing was wiped")

    def process_outlier(self, outlier=None, should_notify=False, extract_derived_fields=False):
        """
        Save outlier (if configuration is setup for that), notify (also depending of configuration) and print.

        :param outlier: the detected outlier
        :param should_notify: True if notification need to be send
        :param extract_derived_fields: True to save derived fields
        """

        if self.settings.es_save_results:
            self.save_outlier(outlier=outlier, extract_derived_fields=extract_derived_fields)

        if should_notify:
            self.notifier.notify_on_outlier(outlier=outlier)

        if self.settings.print_outliers_to_console:
            self.logging.logger.info("outlier - " + outlier.outlier_dict["summary"])

    def add_update_bulk_action(self, document):
        """
        Add a bulk action of "update" type

        :param document: document that need to be update
        """
        action = {
            '_op_type': 'update',
            '_index': document["_index"],
            '_type': document["_type"],
            '_id': document["_id"],
            'retry_on_conflict': 10,
            'doc': document["_source"]
        }
        self.add_bulk_action(action)

    def add_remove_outlier_bulk_action(self, document):
        """
        Creates the bulk action to remove all the outlier traces from all events.
        Removing an outlier means that the "outlier" tag is removed, as well as the "outlier" dictionary in the event.
        :param document: the document from which the outlier information should be removed
        """
        action = {
            '_op_type': 'update',
            '_index': document["_index"],
            '_type': document["_type"],
            '_id': document["_id"],
            'retry_on_conflict': 10,
            '_source': {
                "script": {
                    "source": "ctx._source.remove(\"outliers\"); " +
                              "if (ctx._source.tags != null && ctx._source.tags.indexOf(\"outlier\") > -1) { " +
                              "ctx._source.tags.remove(ctx._source.tags.indexOf(\"outlier\")); " +
                              "}",
                    "lang": "painless"
                }
            }
        }
        self.add_bulk_action(action)

    def add_bulk_action(self, action):
        """
        Add a bluk action

        :param action: action that need to be added
        """
        self.bulk_actions.append(action)
        if len(self.bulk_actions) > self.BULK_FLUSH_SIZE:
            self.flush_bulk_actions()

    def flush_bulk_actions(self, refresh=False):
        """
        Force bulk action to be process

        :param refresh: refresh or not in Elasticsearch
        """
        if not self.bulk_actions:
            return
        try:
            eshelpers.bulk(self.conn, self.bulk_actions, stats_only=True, refresh=refresh)
        except eshelpers.BulkIndexError:
            self.logging.logger.error("BulkIndexError: Unable to write on index %s" % self.bulk_actions[0]["_index"])

        self.bulk_actions = []

    def save_outlier(self, outlier=None, extract_derived_fields=False):
        """
        Complete (with derived fields) and save outlier to Elasticsearch (via bulk action)

        :param outlier: the outlier that need to be save
        :param extract_derived_fields: True to save derived fields
        """
        if extract_derived_fields:
            # add the derived fields as outlier observations
            derived_fields = self.extract_derived_fields(outlier.doc["_source"])
            for derived_field, derived_value in derived_fields.items():
                outlier.outlier_dict["derived_" + derived_field] = derived_value
                # delete temporary derived fields
                del outlier.doc["_source"][derived_field]

        doc = add_outlier_to_document(outlier)
        self.add_update_bulk_action(doc)

    def extract_derived_fields(self, doc_fields):
        """
        Extract derived field based on a document

        :param doc_fields: document information used to extract derived fields
        :return: all derived fields
        """
        derived_fields = dict()
        for field_name, grok_pattern in self.settings.list_derived_fields:
            try:
                # If key doesn't exist, an exception is raise
                doc_value = helpers.utils.get_dotkey_value(doc_fields, field_name, case_sensitive=False)

                if grok_pattern in self.grok_filters.keys():
                    grok = self.grok_filters[grok_pattern]
                else:
                    grok = Grok(grok_pattern)
                    self.grok_filters[grok_pattern] = grok

                match_dict = grok.match(doc_value)

                if match_dict:
                    for match_dict_k, match_dict_v in match_dict.items():
                        derived_fields[match_dict_k] = match_dict_v

            except KeyError:
                pass  # Ignore, value not found...

        return derived_fields

    def extract_fields_from_document(self, doc, extract_derived_fields=False):
        """
        Extract fields information of a document (and also extract derived field if specified)

        :param doc: document where information are fetch
        :param extract_derived_fields: True to extract derived fields
        :return: all documents fields
        """
        doc_fields = doc["_source"]

        if extract_derived_fields:
            derived_fields = self.extract_derived_fields(doc_fields)

            for derived_field_key, derived_field_value in derived_fields.items():
                doc_fields[derived_field_key] = derived_field_value

        return doc_fields

    @staticmethod
    def get_time_filter(start_time=None, end_time=None, days=None, hours=None, timestamp_field=DEFAULT_TIMESTAMP_FIELD):
        """
        Create a filter to limit the time

        :param start_time: start time
        :param end_time: end time
        :param days: number of days of the filter
        :param hours: number of hours of the filter
        :param timestamp_field: the name of the timestamp field
        :return: the query
        """
        if start_time is None or end_time is None:
            start_time = (dt.datetime.now() - dt.timedelta(days=days, hours=hours))
            end_time = dt.datetime.now()

        start_time_iso = start_time.isoformat()
        end_time_iso = end_time.isoformat()

        # Construct absolute time range filter, increases cacheability
        time_filter = {
            "range": {
                str(timestamp_field): {
                    "gte": start_time_iso,
                    "lte": end_time_iso
                }
            }
        }
        return time_filter

    @staticmethod
    def _get_highlight_settings(model_settings):
        """
        Build the highlight settings to include into the Elasticsearch query body.
        This settings will search in all the fields and return fields that contain query matches.
        Each value inside the field that match the search query will be tag as follow:
         <value>value_that_match_search_query</values>

        :param model_settings: part of the configuration linked to the model
        :return: highlight_settings: Highlight settings
        """
        highlight_settings = None
        if model_settings is not None and "highlight_match" in model_settings and model_settings["highlight_match"]:
            highlight_settings = dict()
            # Pre and post tag definition
            highlight_settings["pre_tags"] = ["<value>"]
            highlight_settings["post_tags"] = ["</value>"]
            highlight_settings["fields"] = dict()
            # Search in all the fields
            highlight_settings["fields"]["*"] = dict()
        return highlight_settings


def add_outlier_to_document(outlier):
    """
    Add outliers information to a document (this method also add tag to the document)

    :param outlier: the outlier that need to be added (note that document is contain in the outlier)
    :return: the modified document
    """
    doc = add_tag_to_document(outlier.doc, "outlier")

    if "outliers" in doc["_source"]:
        if outlier.outlier_dict["summary"] not in doc["_source"]["outliers"]["summary"]:
            merged_outliers = defaultdict(list)
            for outlier_key, outlier_value in chain(doc["_source"]["outliers"].items(),
                                                    outlier.get_outlier_dict_of_arrays().items()):

                # merge ["reason 1"] and ["reason 2"]] into ["reason 1", "reason 2"]
                if isinstance(outlier_value, list):
                    merged_outliers[outlier_key].extend(outlier_value)
                else:
                    merged_outliers[outlier_key].append(outlier_value)

            merged_outliers["total_outliers"] = doc["_source"]["outliers"]["total_outliers"] + 1
            doc["_source"]["outliers"] = merged_outliers
    else:
        doc["_source"]["outliers"] = outlier.get_outlier_dict_of_arrays()
        doc["_source"]["outliers"]["total_outliers"] = 1

    return doc


def remove_outliers_from_document(doc):
    """
    Remove all outliers information from a document (reverse of "add_outlier_to_document")

    :param doc: document that need to be modified
    :return: the modified document
    """
    doc = remove_tag_from_document(doc, "outlier")

    if "outliers" in doc["_source"]:
        doc["_source"].pop("outliers")

    return doc


def add_tag_to_document(doc, tag):
    """
    Add a tag to a document

    :param doc: document that need to be modified
    :param tag: the tag that need to be added
    :return: modified document
    """
    if "tags" not in doc["_source"]:
        doc["_source"]["tags"] = [tag]
    else:
        if tag not in doc["_source"]["tags"]:
            doc["_source"]["tags"].append(tag)
    return doc


def remove_tag_from_document(doc, tag):
    """
    Remove a tag from a document (reverse of "add_tag_to_document")

    :param doc: document that need to be modified
    :param tag: tag that need to be added
    :return: modified document
    """
    if "tags" in doc["_source"] and tag in doc["_source"]["tags"]:
        doc["_source"]["tags"].remove(tag)
    return doc


def build_search_query(bool_clause=None,
                       sort_clause=None,
                       search_range=None,
                       query_fields=None,
                       search_query=None,
                       highlight_settings=None):
    """
    Create a query for Elasticsearch

    :param bool_clause: boolean condition
    :param sort_clause: sort query
    :param search_range: search range
    :param query_fields: query fields
    :param search_query: search query
    :param highlight_settings: highlight settings
    :return: the building query
    """
    query = dict()
    query["query"] = dict()
    query["query"]["bool"] = dict()
    query["query"]["bool"]["filter"] = list()

    if query_fields:
        query["_source"] = query_fields

    if bool_clause:
        # To avoid side effects (multiple search_range) when calling multiple times the function on the same bool_clause
        query["query"]["bool"]["filter"] = copy.deepcopy(bool_clause["filter"])

    if sort_clause:
        query.update(sort_clause)

    if search_range:
        if "bool" not in query["query"]:
            query["bool"] = dict()

        if "filter" not in query["query"]["bool"]:
            query["query"]["bool"]["filter"] = list()
        query["query"]["bool"]["filter"].append(search_range)

    if search_query:
        query["query"]["bool"]["filter"].extend(copy.deepcopy(search_query["filter"]))

    if highlight_settings:
        query["highlight"] = highlight_settings

    return query


def build_first_occur_search_query(search_query,
                                   search_range,
                                   target_list,
                                   aggregator_list,
                                   timestamp,
                                   max_num_aggs,
                                   max_num_targets):
    """
    Build specific Elasticsearch query for retrieving aggregation of first occurrence of a term field in events matching
    with the search_query.

    :param search_query: search query
    :param search_range: search range
    :param target_list: list of target fields
    :param aggregator_list: list of fields that we want to aggregate
    :param timestamp: timestamp field name
    :param max_num_aggs: number maximum of estimate aggregation
    :param max_num_targets: number maximum of estimate targets
    :return: the build Elasticsearch query
    """
    order_dict = dict()
    order_dict[timestamp] = dict()
    order_dict[timestamp]["order"] = "asc"

    filter_list = list()
    filter_list.append(search_range)

    aggr_and_target_exists_query = build_aggr_and_target_exists_query(target_list, aggregator_list)
    filter_list.append(aggr_and_target_exists_query)

    filter_list.extend(copy.deepcopy(search_query["filter"]))

    query = {"size": 0,
             "query": {
                 "bool": {
                     "filter": filter_list
                 }
             },
             "aggs": {
                 "aggregator": {
                     "terms": {
                         "script": build_script(aggregator_list),
                         "size": max_num_aggs,
                         "collect_mode": "breadth_first"
                     },
                     "aggs": {
                         "target": {
                             "terms": {
                                 "script": build_script(target_list),
                                 "size": max_num_targets,
                                 "collect_mode": "breadth_first"
                             },
                             "aggs": {
                                 "top_doc": {
                                     "top_hits": {
                                         "sort": [order_dict],
                                         "size": 1
                                     }
                                 }
                             }
                         }
                     }
                 }
             }}

    return query


def build_aggr_and_target_exists_query(target_list, aggregator_list):
    """
    Build query to select events where fields defined in target_list and aggregator_list exist.

    :param target_list: list of target fields
    :param aggregator_list: list of fields that we want to aggregate
    :return aggr_and_target_exists_query: Elasticsearch query that select event where fields defined in target_list and
    aggregator_list exist
    """
    field_list = target_list + aggregator_list
    query = " AND ".join(["_exists_: " + field for field in field_list])
    aggr_and_target_exists_query = {"query_string": {"query": query}}
    return aggr_and_target_exists_query


def build_script(field_list):
    """
    Build painless script to create aggregation from multiple fields

    :param field_list: list of event field
    :return:  painless script format
    """
    return " + ' - ' + ".join(["doc['" + field + "'].value" for field in field_list])
