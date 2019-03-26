from elasticsearch import helpers as eshelpers, Elasticsearch
import datetime
import numpy
import helpers.utils
import helpers.logging
from pygrok import Grok

from functools import reduce
from helpers.singleton import singleton
from helpers.notifier import Notifier
from collections import defaultdict
from itertools import chain


@singleton
class ES:
    index = None
    conn = None
    settings = None

    outliers_dict = []
    grok_filters = dict()

    notifier = None

    def __init__(self, settings=None, logging=None):
        self.settings = settings
        self.logging = logging

        if self.settings.config.getboolean("notifier", "email_notifier"):
            self.notifier = Notifier(settings, logging)

    def init_connection(self):
        self.conn = Elasticsearch([self.settings.config.get("general", "es_url")], use_ssl=False, timeout=self.settings.config.getint("general", "es_timeout"), verify_certs=False, retry_on_timeout=True)

        if self.conn.ping():
            self.logging.logger.info("connected to Elasticsearch on host %s" % (self.settings.config.get("general", "es_url")))
        else:
            self.logging.logger.error("could not connect to to host %s. Exiting!" % (self.settings.config.get("general", "es_url")))

        return self.conn

    def get_type(self, field):
        '''
        Params
        - field (str) : Field we want to get the type
                        (OsqueryFilter.name for example)
        '''
        mappings = self.conn.indices.get_mapping(
            index=self.settings.config.get("general", "es_index_pattern"))

        mappings = mappings[self.settings.config.get(
            "general", "es_index_pattern")]["mappings"]
        mappings = mappings[next(iter(mappings))]  # type
        mappings = mappings['properties']

        for v in field.split('.'):
            mappings = mappings[v]
            if 'properties' in mappings:
                mappings = mappings['properties']

        return mappings['type']

    def get_field_by_path(self, doc, path):
        '''
        Return the field using the path
        expl: OsqueryFilter.name
        '''
        fields = self.extract_fields_from_document(doc)
        field = reduce(
            (lambda x, y: x[y][-1] if isinstance(x[y], list) else x[y]),
            [fields, *path.split('.')]
        )
        return field

    def scan(
        self, bool_clause=None, query_fields=None,
        lucene_query=None, sort=None
    ):
        if self.settings.config.getint("general", "ignore_history_window"):
            self.settings.search_range = None

        query = build_search_query(
            bool_clause=bool_clause,
            search_range=self.settings.search_range,
            query_fields=query_fields,
            lucene_query=lucene_query
        )

        if sort:
            preserve_order = True
            query['sort'] = [
                {v + '.keyword': {"order": "asc", "unmapped_type": "keyword"}}
                # we don't need keyword to sort these types
                if self.get_type(v) not in ['long', 'date']
                else {v: {"order": "asc"}}
                for v in sort
            ]

        return eshelpers.scan(
            self.conn,
            request_timeout=self.settings.config.getint(
                "general", "es_timeout"),
            index=self.settings.config.get("general", "es_index_pattern"),
            query=query,
            size=self.settings.config.getint("general", "es_scan_size"),
            scroll=self.settings.config.get("general", "es_scroll_time"),
            preserve_order=preserve_order, raise_on_error=True,
            terminate_after=self.settings.config.get(
                "general", "es_terminate_after")
        )

    def scan_aggregations(self, aggregator, es_query_filter):
        '''
        Usage
        =====
        for aggregation, documents in es.scan_aggregations(...):
            for doc in documents:
                pass
        '''
        lucene_query = self.filter_by_query_string(es_query_filter)

        def aggregations_iterator():
            yield aggregations_iterator.doc

            for doc in aggregations_iterator.es_scan:
                agg = self.get_field_by_path(doc, aggregator)

                if agg != aggregations_iterator.agg:
                    aggregations_iterator.agg = agg
                    aggregations_iterator.doc = doc
                    break

                yield doc

            else:
                aggregations_iterator.stop = True

        aggregations_iterator.es_scan = self.scan(
            lucene_query=lucene_query, sort=[aggregator])

        aggregations_iterator.doc = next(aggregations_iterator.es_scan)

        aggregations_iterator.agg = self.get_field_by_path(
            aggregations_iterator.doc,
            aggregator
        )

        aggregations_iterator.stop = False

        while not aggregations_iterator.stop:
            # yield aggregation, documents
            yield aggregations_iterator.agg, aggregations_iterator()

    def count_documents(self, bool_clause=None, query_fields=None, lucene_query=None):
        res = self.conn.search(index=self.index, body=build_search_query(bool_clause=bool_clause, search_range=self.settings.search_range, query_fields=query_fields, lucene_query=lucene_query), size=self.settings.config.getint("general", "es_scan_size"), scroll=self.settings.config.get("general", "es_scroll_time"))
        return res["hits"]["total"]

    def sort_by_field_name(self, field_name, ascending=True):
        order = 'asc' if ascending else 'desc'
        sort_clause = {"sort": [
            {field_name + ".raw": order}
        ]}
        return sort_clause

    def filter_by_query_string(self, query=None):
        bool_clause = {"must": [
            {"query_string": {"query": query}}
        ]}
        return bool_clause

    def filter_by_field_value(self, field_name=None, field_value=None):
        bool_clause = {"must": [
            {"match": {field_name + ".raw": field_value}},
        ]}
        return bool_clause

    def filter_by_field_existance(self, field_name=None):
        bool_clause = {"must": [
            {"exists": {"field": field_name + ".raw"}}
        ]}
        return bool_clause

    def filter_by_field_value_list(self, field_name=None, field_value_list=None):
        bool_clause = {"should": [], "minimum_should_match": 1}

        for field_value in field_value_list:
            should_clause = dict({"match": {field_name + ".raw": field_value}})
            bool_clause.get("should").append(should_clause)

        return bool_clause

    # this is part of housekeeping, so we should not access non-threat-save objects, such as logging progress to the console using ticks!
    def remove_all_whitelisted_outliers(self):
        from helpers.outlier import Outlier  # import goes here to avoid issues with singletons & circular requirements ... //TODO: fix this

        must_clause = {"must": [{"match": {"tags": "outlier"}}]}
        total_docs_whitelisted = 0

        for doc in self.scan(bool_clause=must_clause):
            total_outliers = int(doc["_source"]["outliers"]["total_outliers"])

            # Generate all outlier objects for this document
            total_whitelisted = 0

            for i in range(total_outliers):
                outlier_type = doc["_source"]["outliers"]["type"][i]
                outlier_reason = doc["_source"]["outliers"]["reason"][i]
                outlier_summary = doc["_source"]["outliers"]["summary"][i]

                outlier = Outlier(type=outlier_type, reason=outlier_reason, summary=outlier_summary)
                if outlier.is_whitelisted(additional_dict_values_to_check=doc):
                    total_whitelisted += 1

            # if all outliers for this document are whitelisted, removed them all. If not, don't touch the document.
            # this is a limitation in the way our outliers are stored: if not ALL of them are whitelisted, we can't remove just the whitelisted ones
            # from the Elasticsearch event, as they are stored as array elements and potentially contain observations that should be removed, too.
            # In this case, just don't touch the document.
            if total_whitelisted == total_outliers:
                total_docs_whitelisted += 1
                doc = remove_outliers_from_document(doc)

                self.conn.delete(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], refresh=True)
                self.conn.create(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], body=doc["_source"], refresh=True)

        return total_docs_whitelisted

    def remove_all_outliers(self):
        must_clause = {"must": [{"match": {"tags": "outlier"}}]}
        total_outliers = self.count_documents(bool_clause=must_clause)
        self.logging.init_ticker(total_steps=total_outliers, desc="wiping existing outliers")

        for doc in self.scan(bool_clause=must_clause):
            self.logging.tick()
            doc = remove_outliers_from_document(doc)

            self.conn.delete(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], refresh=True)
            self.conn.create(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], body=doc["_source"], refresh=True)

        if total_outliers > 0:
            self.logging.logger.info("wiped outlier information of " + str(total_outliers) + " documents")
        else:
            self.logging.logger.info("no existing outliers were found, so nothing was wiped")

    def process_outliers(self, doc=None, outliers=None, should_notify=False):
        for outlier in outliers:
            if outlier.is_whitelisted(additional_dict_values_to_check=doc):
                if self.settings.config.getboolean("general", "print_outliers_to_console"):
                    self.logging.logger.info(outlier.get_summary() + " [whitelisted outlier]")
            else:
                if self.settings.config.getboolean("general", "es_save_results"):
                    self.save_outlier(doc=doc, outlier=outlier)

                if should_notify:
                    self.notifier.notify_on_outlier(doc=doc, outlier=outlier)

                if self.settings.config.getboolean("general", "print_outliers_to_console"):
                    self.logging.logger.info("outlier - " + outlier.get_summary())

    def save_outlier(self, doc=None, outlier=None):
        # add the derived fields as outlier observations
        derived_fields = self.extract_derived_fields(doc["_source"])
        for derived_field, derived_value in derived_fields.items():
            outlier.add_observation("derived_" + derived_field, derived_value)

        doc = add_outlier_to_document(doc, outlier)

        if "tags" in doc["_source"]:
            doc_body = dict(doc={"tags": doc["_source"]["tags"], "outliers": doc["_source"]["outliers"]})
        else:
            doc_body = dict(doc={"outliers": doc["_source"]["outliers"]})

        self.conn.update(index=doc["_index"], doc_type=doc["_type"], id=doc["_id"], body=doc_body, refresh=True)

    def extract_derived_fields(self, doc_fields):
        derived_fields = dict()
        for field_name, grok_pattern in self.settings.config.items("derivedfields"):
            if helpers.utils.dict_contains_dotkey(doc_fields, field_name, case_sensitive=False):
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

    def extract_fields_from_document(self, doc):
        doc_fields = doc["_source"]
        derived_fields = self.extract_derived_fields(doc_fields)

        for k, v in derived_fields.items():
            doc_fields[k] = v

        return doc_fields

    # Filters
    def filter_doc_id(self, event_id):
        if isinstance(event_id, numpy.ndarray):
            id_list = event_id.tolist()
        elif isinstance(event_id, list):
            id_list = event_id
        else:
            id_list = [event_id]

        bool_clause = {"must": [
            {"terms": {"_id": id_list}}
        ]}
        return bool_clause


def add_outlier_to_document(doc, outlier):
    doc = add_tag_to_document(doc, "outlier")

    if "outliers" in doc["_source"]:
        if outlier.get_summary() not in doc["_source"]["outliers"]["summary"]:
            merged_outliers = defaultdict(list)
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


def remove_outliers_from_document(doc):
    doc = remove_tag_from_document(doc, "outlier")

    if "outliers" in doc["_source"]:
        doc["_source"].pop("outliers")

    return doc


def add_tag_to_document(doc, tag):
    if "tags" not in doc["_source"]:
        doc["_source"]["tags"] = [tag]
    else:
        if tag not in doc["_source"]["tags"]:
            doc["_source"]["tags"].append(tag)
    return doc


def remove_tag_from_document(doc, tag):
    if "tags" not in doc["_source"]:
        pass
    else:
        tags = doc["_source"]["tags"]
        if tag in tags:
            tags.remove(tag)
            doc["_source"]["tags"] = tags
    return doc


def build_search_query(bool_clause=None, sort_clause=None, search_range=None, query_fields=None, lucene_query=None):
    query = dict()
    query["query"] = dict()
    query["query"]["bool"] = dict()
    query["query"]["bool"]["must"] = list()

    if query_fields:
        query["_source"] = query_fields

    if bool_clause:
        query["query"]["bool"]["must"] = bool_clause["must"].copy()  # To avoid side effects (multiple search_range) when calling multiple times the function on the same bool_clause

    if sort_clause:
        query.update(sort_clause)

    if search_range:
        if "bool" not in query["query"]:
            query["bool"] = dict()

        if "filter" not in query["query"]["bool"]:
            query["query"]["bool"]["filter"] = list()
        query["query"]["bool"]["filter"].append(search_range)

    if lucene_query:
        query["query"]["bool"]["must"].append(lucene_query["must"].copy())

    return query


def get_time_filter(days=None, hours=None, timestamp_field="timestamp"):
    time_start = (datetime.datetime.now() - datetime.timedelta(days=days, hours=hours)).isoformat()
    time_stop = datetime.datetime.now().isoformat()

    # Construct absolute time range filter, increases cacheability
    time_filter = {
        "range": {
            str(timestamp_field): {
                "gte": time_start,
                "lte": time_stop
            }
        }
    }
    return time_filter
