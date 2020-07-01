from helpers.singletons import es
import helpers.es

import dateutil.parser


def _convert_to_elastic_search_format(aggr_dict):
    for aggr_key in aggr_dict:
        tmp_list_target_buckets = list()
        for target_key in aggr_dict[aggr_key]["target"]["buckets"]:
            tmp_list_target_buckets.append(aggr_dict[aggr_key]["target"]["buckets"][target_key])
        aggr_dict[aggr_key]["target"]["buckets"] = tmp_list_target_buckets
    return list(aggr_dict.values())


def _get_values_from_list_field(doc, list_field):
    values = ""
    for field_name in list_field:
        doc_value = doc
        for key_name in field_name.split("."):
            doc_value = doc_value[key_name]
        if values == "":
            values = doc_value
        else:
            values += " - " + doc_value
    return values


class TestStubEs:

    def __init__(self):
        # Init test stub es
        self.list_data = dict()
        self.id = 0
        self.default_es_methods = self._get_default_es_methods()
        self.apply_new_es()

    @staticmethod
    def _get_default_es_methods():
        # could not do deepcopy due to "TypeError: cannot serialize '_io.TextIOWrapper' object"
        return {
                "default_bulk_flush_size": es.BULK_FLUSH_SIZE,
                "default_init": es.__init__,
                "default_init_connection": es.init_connection,
                "default_scan": es._scan,
                "default_count_documents": es._count_documents,
                "default_scan_first_occur_documents": es.scan_first_occur_documents,
                "default_remove_all_outliers": es.remove_all_outliers,
                "default_flush_bulk_actions": es.flush_bulk_actions
            }

    def apply_new_es(self):
        es.BULK_FLUSH_SIZE = 0
        es.__init__ = self.new_init
        es.init_connection = self.init_connection
        es._scan = self._scan
        es._count_documents = self._count_documents
        es.scan_first_occur_documents = self.scan_first_occur_documents
        es.remove_all_outliers = self.remove_all_outliers
        es.flush_bulk_actions = self.flush_bulk_actions

    def restore_es(self):
        es.BULK_FLUSH_SIZE = self.default_es_methods["default_bulk_flush_size"]
        es.__init__ = self.default_es_methods["default_init"]
        es.init_connection = self.default_es_methods["default_init_connection"]
        es._scan = self.default_es_methods["default_scan"]
        es._count_documents = self.default_es_methods["default_count_documents"]
        es.scan_first_occur_documents = self.default_es_methods["default_scan_first_occur_documents"]
        es.remove_all_outliers = self.default_es_methods["default_remove_all_outliers"]
        es.flush_bulk_actions = self.default_es_methods["default_flush_bulk_actions"]
        es.bulk_actions = list()

    def new_init(self, settings=None, logging=None):
        pass

    def init_connection(self):
        return None

    def _scan(self, index="", search_range=None, bool_clause=None, sort_clause=None, query_fields=None,
              search_query=None, model_settings=None):
        for element in self.list_data.values():
            yield element

    def _count_documents(self, index="", bool_clause=None, query_fields=None, search_query=None, model_settings=None):
        return len(self.list_data)

    def scan_first_occur_documents(self, search_query, start_time, end_time, model_settings):
        """
        Function that imitate the helpers.es.scan_first_occur_documents() function behavior.
        """
        aggr_dict = dict()
        aggregators = model_settings["aggregator"]
        targets = model_settings["target"]
        for raw_doc in self.list_data.values():
            doc = raw_doc["_source"]
            doc_timestamp = doc["@timestamp"]
            doc_timestamp = dateutil.parser.parse(doc_timestamp, ignoretz=True)
            if end_time >= doc_timestamp >= start_time:
                aggr_value = _get_values_from_list_field(doc, aggregators)
                target_value = _get_values_from_list_field(doc, targets)

                if aggr_value not in aggr_dict:
                    aggr_dict[aggr_value] = dict()
                    aggr_dict[aggr_value]["key"] = aggr_value
                    aggr_dict[aggr_value]["target"] = dict()
                    aggr_dict[aggr_value]["target"]["buckets"] = dict()
                if target_value not in aggr_dict[aggr_value]["target"]["buckets"]:
                    aggr_dict[aggr_value]["target"]["buckets"][target_value] = dict()
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["key"] = target_value
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["doc_count"] = 1
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["top_doc"] = dict()
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["top_doc"]["hits"] = dict()
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["top_doc"]["hits"]["hits"] = [raw_doc]
                else:
                    aggr_dict[aggr_value]["target"]["buckets"][target_value]["doc_count"] += 1

                    previous_doc = aggr_dict[aggr_value]["target"]["buckets"][target_value]["top_doc"]["hits"]["hits"][0]
                    previous_doc_timestamp = previous_doc["_source"]["@timestamp"]
                    previous_doc_timestamp = dateutil.parser.parse(previous_doc_timestamp, ignoretz=True)

                    if doc_timestamp < previous_doc_timestamp:
                        aggr_dict[aggr_value]["target"]["buckets"][target_value]["top_doc"]["hits"]["hits"] = [raw_doc]

        return _convert_to_elastic_search_format(aggr_dict)

    def remove_all_outliers(self):
        self.list_data = dict()

    def flush_bulk_actions(self, refresh=False):
        if len(es.bulk_actions) == 0:
            return

        for bulk in es.bulk_actions:
            if bulk['_op_type'] == 'update':
                # If it is a script bulk request
                if "_source" in bulk and "script" in bulk["_source"]:
                    data = self.list_data[bulk['_id']]

                    # We supposed here that only the remove of outlier is possible
                    data = helpers.es.remove_outliers_from_document(data)
                    self.list_data[bulk['_id']] = data

                else:  # Else it is only a update request
                    data = {
                        "_source": bulk['doc'],
                        "_id": bulk['_id']
                    }

                    if '_type' in bulk:
                        data['_type'] = bulk['_type']
                    if '_index' in bulk:
                        data['_index'] = bulk['_index']

                    self.list_data[bulk['_id']].update(data)

            else:
                raise KeyError('Unknown bulk action: "' + bulk['_op_type'] + '"')
        es.bulk_actions = []

    def add_data(self, dictionary_data):
        """
        Add "fake" data, that can be return by custom elastic search

        :param dictionary_data: dictionary with key and value (key with the format: key1.key2.key3)
        """
        source = {}
        for key, val in dictionary_data.items():
            self._create_dict_based_on_key(source, key, val)
        data = {
            "_source": source,
            "_id": self.id
        }
        self.id += 1
        self.add_doc(data)

    def add_doc(self, doc):
        if doc['_id'] in self.list_data:
            raise KeyError("Key " + str(doc['_id']) + " already exist in testStubEs")
        self.list_data[doc['_id']] = doc

    def add_multiple_docs(self, list_doc):
        for doc in list_doc:
            self.add_doc(doc)

    def _create_dict_based_on_key(self, doc, key, data):
        self.list_key = key.split(".")

        if len(self.list_key) == 1:
            if key in doc:
                if isinstance(doc[key], list):
                    doc[key].append(data)
                else:
                    old_data = doc[key]
                    doc[key] = [old_data, data]
            else:
                doc[key] = data
        else:
            first_key = self.list_key[0]
            if first_key in doc:
                if isinstance(doc[first_key], dict):
                    self._create_dict_based_on_key(doc[first_key], ".".join(self.list_key[1:]), data)
                else:
                    raise ValueError("Key: " + str(first_key) + " have already a value that isn't a dictionary")
            else:
                doc[first_key] = dict()
                self._create_dict_based_on_key(doc[first_key], ".".join(self.list_key[1:]), data)
