from elasticsearch import helpers as eshelpers, Elasticsearch
import datetime
import helpers.utils
import helpers.logging
import json
from pygrok import Grok

from helpers.singleton import singleton
from helpers.notifier import Notifier
from collections import defaultdict
from itertools import chain

from typing import Dict, List, AnyStr, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from helpers.settings import Settings
    from helpers.logging import Logging
    from helpers.outlier import Outlier

BULK_FLUSH_SIZE = 1000


@singleton
class ES:
    index = None # Always None ?
    # conn: Elasticsearch = None
    # settings: Settings = None

    grok_filters: Dict[str, Grok] = dict()

    notifier: Notifier

    bulk_actions: List[Dict] = []

    def __init__(self, settings: 'Settings', logging: 'Logging') -> None:
        self.settings= settings
        self.logging= logging

        if self.settings.config.getboolean("notifier", "email_notifier"):
            self.notifier = Notifier(settings, logging)

    def init_connection(self) -> Elasticsearch:
        self.conn: Elasticsearch = Elasticsearch([self.settings.config.get("general", "es_url")], use_ssl=False,
                                  timeout=self.settings.config.getint("general", "es_timeout"),
                                  verify_certs=False, retry_on_timeout=True)

        if self.conn.ping():
            self.logging.logger.info("connected to Elasticsearch on host %s" %
                                     (self.settings.config.get("general", "es_url")))
        else:
            self.logging.logger.error("could not connect to to host %s. Exiting!" %
                                      (self.settings.config.get("general", "es_url")))

        return self.conn

    def scan(self, index: str, bool_clause: Dict[str, List]=None, sort_clause: Dict=None, query_fields: Dict=None,
             search_query: Dict[str, List]=None) -> Dict:
        preserve_order: bool = True if sort_clause is not None else False
        return eshelpers.scan(self.conn, request_timeout=self.settings.config.getint("general", "es_timeout"), 
                              index=index, query=build_search_query(bool_clause=bool_clause, 
                                                                    sort_clause=sort_clause, 
                                                                    search_range=self.settings.search_range, 
                                                                    query_fields=query_fields, 
                                                                    search_query=search_query), 
                              size=self.settings.config.getint("general", "es_scan_size"), 
                              scroll=self.settings.config.get("general", "es_scroll_time"), 
                              preserve_order=preserve_order, raise_on_error=False)

    def count_documents(self, index: str, bool_clause: Dict[str, List]=None, query_fields: Dict=None,
                        search_query: Dict[str, List]=None) -> int:
        res = self.conn.search(index=index, body=build_search_query(bool_clause=bool_clause, 
                                                       search_range=self.settings.search_range, 
                                                       query_fields=query_fields, search_query=search_query), 
                               size=self.settings.config.getint("general", "es_scan_size"), 
                               scroll=self.settings.config.get("general", "es_scroll_time"))
        return res["hits"]["total"]

    def filter_by_query_string(self, query_string: str=None) -> Dict[str, List]:
        bool_clause: Dict[str, List] = {"filter": [
            {"query_string": {"query": query_string}}
        ]}
        return bool_clause

    def filter_by_dsl_query(self, dsl_query_path: AnyStr) -> Dict[str, List]:
        dsl_query = json.loads(dsl_query_path)

        bool_clause: Dict[str, List]
        if isinstance(dsl_query, list):
            bool_clause = {"filter": []}
            for query in dsl_query:
                bool_clause["filter"].append(query["query"])
        else:
            bool_clause = {"filter": [
                dsl_query["query"]
            ]}
        return bool_clause

    # this is part of housekeeping, so we should not access non-threat-save objects, such as logging progress to
    # the console using ticks!
    def remove_all_whitelisted_outliers(self) -> int:
        from helpers.outlier import Outlier  # import goes here to avoid issues with singletons & circular
        # requirements ... //TODO: fix this

        outliers_filter_query: Dict[str, List] = {"filter": [{"term": {"tags": "outlier"}}]}
        total_docs_whitelisted: int = 0

        idx: str = self.settings.config.get("general", "es_index_pattern")
        total_nr_outliers: int = self.count_documents(index=idx, bool_clause=outliers_filter_query)
        self.logging.logger.info("going to analyze %s outliers and remove all whitelisted items", "{:,}"\
                                 .format(total_nr_outliers))

        for doc in self.scan(index=idx, bool_clause=outliers_filter_query):
            total_outliers: int = int(doc["_source"]["outliers"]["total_outliers"])
            # Generate all outlier objects for this document
            total_whitelisted: int = 0

            for i in range(total_outliers):
                outlier_type: str = doc["_source"]["outliers"]["type"][i]
                outlier_reason: str = doc["_source"]["outliers"]["reason"][i]
                outlier_summary: str = doc["_source"]["outliers"]["summary"][i]

                outlier: Outlier = Outlier(outlier_type=outlier_type, outlier_reason=outlier_reason, 
                                           outlier_summary=outlier_summary)
                if outlier.is_whitelisted(additional_dict_values_to_check=doc):
                    total_whitelisted += 1

            # if all outliers for this document are whitelisted, removed them all. If not, don't touch the document.
            # this is a limitation in the way our outliers are stored: if not ALL of them are whitelisted, we can't
            # remove just the whitelisted ones from the Elasticsearch event, as they are stored as array elements and
            # potentially contain observations that should be removed, too.
            # In this case, just don't touch the document.
            if total_whitelisted == total_outliers:
                total_docs_whitelisted += 1
                doc = remove_outliers_from_document(doc)

                self.conn.delete(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], refresh=True)
                self.conn.create(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], body=doc["_source"],
                                 refresh=True)

        return total_docs_whitelisted

    def remove_all_outliers(self) -> None:
        idx = self.settings.config.get("general", "es_index_pattern")

        must_clause: Dict[str, Any] = {"filter": [{"term": {"tags": "outlier"}}]}
        total_outliers: int = self.count_documents(index=idx, bool_clause=must_clause)

        query: Dict[str, Dict] = build_search_query(bool_clause=must_clause, search_range=self.settings.search_range)

        script: Dict[str, str] = {
            "source": "ctx._source.remove(\"outliers\"); ctx._source.tags.remove(ctx._source.tags.indexOf(\"outlier\"))",
            "lang": "painless"
        }

        query["script"] = script

        if total_outliers > 0:
            self.logging.logger.info("wiping %s existing outliers", "{:,}".format(total_outliers))
            self.conn.update_by_query(index=idx, body=query, refresh=True, wait_for_completion=True)
            self.logging.logger.info("wiped outlier information of " + "{:,}".format(total_outliers) + " documents")
        else:
            self.logging.logger.info("no existing outliers were found, so nothing was wiped")

    def process_outliers(self, doc: Dict[str, Any], outliers: List['Outlier'], should_notify: bool=False) -> None:
        for outlier in outliers:
            if outlier.is_whitelisted(additional_dict_values_to_check=doc):
                if self.settings.config.getboolean("general", "print_outliers_to_console"):
                    self.logging.logger.info(outlier.outlier_dict["summary"] + " [whitelisted outlier]")
            else:
                if self.settings.config.getboolean("general", "es_save_results"):
                    self.save_outlier(doc=doc, outlier=outlier)

                if should_notify:
                    self.notifier.notify_on_outlier(doc=doc, outlier=outlier)

                if self.settings.config.getboolean("general", "print_outliers_to_console"):
                    self.logging.logger.info("outlier - " + outlier.outlier_dict["summary"])

    def add_bulk_action(self, action: Dict[str, Any]) -> None:
        self.bulk_actions.append(action)
        if len(self.bulk_actions) > BULK_FLUSH_SIZE:
            self.flush_bulk_actions()

    def flush_bulk_actions(self, refresh: bool=False) -> None:
        if len(self.bulk_actions) == 0:
            return
        eshelpers.bulk(self.conn, self.bulk_actions, stats_only=True, refresh=refresh)
        self.bulk_actions = []

    def save_outlier(self, doc: Dict[str, Any], outlier: 'Outlier') -> None:
        # add the derived fields as outlier observations
        derived_fields = self.extract_derived_fields(doc["_source"])
        for derived_field, derived_value in derived_fields.items():
            outlier.outlier_dict["derived_" + derived_field] = derived_value

        doc = add_outlier_to_document(doc, outlier)

        action: Dict[str, Any] = {
            '_op_type': 'update',
            '_index': doc["_index"],
            '_type': doc["_type"],
            '_id': doc["_id"],
            'retry_on_conflict': 10,
            'doc': doc["_source"]
        }
        self.add_bulk_action(action)

    def extract_derived_fields(self, doc_fields: Dict) -> Dict:
        derived_fields: Dict = dict()
        for field_name, grok_pattern in self.settings.config.items("derivedfields"):
            if helpers.utils.dict_contains_dotkey(doc_fields, field_name, case_sensitive=False):
                grok: Grok
                if grok_pattern in self.grok_filters.keys():
                    grok = self.grok_filters[grok_pattern]
                else:
                    grok = Grok(grok_pattern)
                    self.grok_filters[grok_pattern] = grok

                match_dict = grok.match(helpers.utils.get_dotkey_value(doc_fields, field_name, case_sensitive=False))

                if match_dict:
                    for match_dict_k, match_dict_v in match_dict.items():
                        derived_fields[match_dict_k] = match_dict_v

        return derived_fields

    def extract_fields_from_document(self, doc: Dict[str, Dict], extract_derived_fields: bool= False) -> Dict:
        doc_fields: Dict = doc["_source"]

        if extract_derived_fields:
            derived_fields = self.extract_derived_fields(doc_fields)

            for k, v in derived_fields.items():
                doc_fields[k] = v

        return doc_fields


def add_outlier_to_document(doc: Dict[str, Any], outlier: 'Outlier') -> Dict[str, Any]:
    doc = add_tag_to_document(doc, "outlier")

    if "outliers" in doc["_source"]:
        if outlier.outlier_dict["summary"] not in doc["_source"]["outliers"]["summary"]:
            merged_outliers: defaultdict = defaultdict(list)
            for k, v in chain(doc["_source"]["outliers"].items(), outlier.get_outlier_dict_of_arrays().items()):

                # merge ["reason 1"] and ["reason 2"]] into ["reason 1", "reason 2"]
                if isinstance(v, list):
                    merged_outliers[k].extend(v)
                else:
                    merged_outliers[k].append(v)

            merged_outliers["total_outliers"] = doc["_source"]["outliers"]["total_outliers"] + 1
            doc["_source"]["outliers"] = merged_outliers
    else:
        doc["_source"]["outliers"] = outlier.get_outlier_dict_of_arrays()
        doc["_source"]["outliers"]["total_outliers"] = 1

    return doc


def remove_outliers_from_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = remove_tag_from_document(doc, "outlier")

    if "outliers" in doc["_source"]:
        doc["_source"].pop("outliers")

    return doc


def add_tag_to_document(doc: Dict[str, Any], tag: str) -> Dict[str, Any]:
    if "tags" not in doc["_source"]:
        doc["_source"]["tags"] = [tag]
    else:
        if tag not in doc["_source"]["tags"]:
            doc["_source"]["tags"].append(tag)
    return doc


def remove_tag_from_document(doc: Dict[str, Any], tag: str) -> Dict[str, Any]:
    if "tags" not in doc["_source"]:
        pass
    else:
        tags = doc["_source"]["tags"]
        if tag in tags:
            tags.remove(tag)
            doc["_source"]["tags"] = tags
    return doc


def build_search_query(bool_clause: Dict[str, Any]=None, sort_clause: Dict=None, search_range: Dict[str, Dict]=None,
                       query_fields: Dict=None, search_query: Dict[str, Any]=None) -> Dict[str, Dict]:
    query: Dict[str, Dict] = dict()
    query["query"] = dict()
    query["query"]["bool"] = dict()
    query["query"]["bool"]["filter"] = list()

    if query_fields:
        query["_source"] = query_fields

    if bool_clause:
        query["query"]["bool"]["filter"] = bool_clause["filter"].copy()  # To avoid side effects (multiple search_
        # range) when calling multiple times the function on the same bool_clause

    if sort_clause:
        query.update(sort_clause)

    if search_range:
        if "bool" not in query["query"]:
            query["bool"] = dict()

        if "filter" not in query["query"]["bool"]:
            query["query"]["bool"]["filter"] = list()
        query["query"]["bool"]["filter"].append(search_range)

    if search_query:
        query["query"]["bool"]["filter"].append(search_query["filter"].copy())

    return query


def get_time_filter(days: int, hours: int, timestamp_field: str="timestamp") -> Dict[str, Dict]:
    time_start: str = (datetime.datetime.now() - datetime.timedelta(days=days, hours=hours)).isoformat()
    time_stop: str = datetime.datetime.now().isoformat()

    # Construct absolute time range filter, increases cacheability
    time_filter: Dict[str, Dict] = {
        "range": {
            str(timestamp_field): {
                "gte": time_start,
                "lte": time_stop
            }
        }
    }
    return time_filter
