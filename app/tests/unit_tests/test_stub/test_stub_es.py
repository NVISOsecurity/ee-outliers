import time
import random
from datetime import datetime

from helpers.singletons import es


POSSIBLE_SLAVE_NAME = ["ee-slave-lab", "random-name"]
POSSIBLE_META_CMD_NAME = ["get_all_processes_with_listening_conns", "get_all_scheduled_tasks"]
POSSIBLE_TAGS = [["unknown_hashes", "endpoint"], ["endpoint"], ["test", "unknown_hashes"]]

list_data = dict()
es.bulk_actions = list()
id = 0


def new_init(settings=None, logging=None):
    super().__init__(settings, logging)
    global list_data, id
    list_data = dict()
    id = 0


def init_connection():
    return None


def scan(index="", bool_clause=None, sort_clause=None, query_fields=None, search_query=None, model_settings=None):
    global list_data
    for element in list_data.values():
        yield element


def count_documents(index="", bool_clause=None, query_fields=None, search_query=None, model_settings=None):
    global list_data
    return len(list_data)


def _update_es(doc):
    global list_data
    id = doc['_id']
    if id in list_data:
        list_data[id] = doc


def remove_all_outliers():
    global list_data
    list_data = dict()


def flush_bulk_actions(refresh=False):
    global list_data
    if len(es.bulk_actions) == 0:
        return

    for bulk in es.bulk_actions:
        if bulk['_op_type'] == 'update':
            data = {
                "_source": bulk['doc'],
                "_id": bulk['_id']
            }
            if '_type' in bulk:
                data['_type'] = bulk['_type']
            if '_index' in bulk:
                data['_index'] = bulk['_index']
            list_data[bulk['_id']] = data
        else:
            raise KeyError('Unknown bulk action: "' + bulk['_op_type'] + '"')
    es.bulk_actions = []

# Apply new ES functions


default_bulk_flush_size = es.BULK_FLUSH_SIZE
default_init = es.__init__
default_init_connection = es.init_connection
default_scan = es.scan
default_count_documents = es.count_documents
default_update_es = es._update_es
default_remove_all_outliers = es.remove_all_outliers
default_flush_bulk_actions = es.flush_bulk_actions


def apply_new_es():
    es.BULK_FLUSH_SIZE = 0
    es.__init__ = new_init
    es.init_connection = init_connection
    es.scan = scan
    es.count_documents = count_documents
    es._update_es = _update_es
    es.remove_all_outliers = remove_all_outliers
    es.flush_bulk_actions = flush_bulk_actions


def restore_es():
    es.BULK_FLUSH_SIZE = default_bulk_flush_size
    es.__init__ = default_init
    es.init_connection = default_init_connection
    es.scan = default_scan
    es.count_documents = default_count_documents
    es._update_es = default_update_es
    es.remove_all_outliers = default_remove_all_outliers
    es.flush_bulk_actions = default_flush_bulk_actions

    global id, list_data, bulk_actions
    id = 0
    list_data = dict()
    es.bulk_actions = list()

# "Static" functions


def generate_data( nbr_data=1, fixed_infos=dict()):
    for _ in range(nbr_data):
        timestamp = get_random_timestamp()
        slave_name = random.choice(POSSIBLE_SLAVE_NAME)
        meta_cmd_name = random.choice(POSSIBLE_META_CMD_NAME)
        tags = random.choice(POSSIBLE_TAGS)

        dictionary_data = {}
        dictionary_data["@timestamp"] = timestamp
        dictionary_data["timestamp"] = timestamp
        dictionary_data["tags"] = tags
        dictionary_data["slave_name"] = slave_name
        dictionary_data["meta.command.name"] = meta_cmd_name
        dictionary_data.update(fixed_infos)
        add_data(dictionary_data)


def add_data(dictionary_data):
    """
    Add "fake" data, that can be return by custom elastic search

    :param dictionary_data dictionary with key and value (key with the format: key1.key2.key3)
    """
    global id

    source = {}
    for key, val in dictionary_data.items():
        _create_dict_based_on_key(source, key, val)
    data = {
        "_source": source,
        "_id": id
    }
    id += 1
    add_doc(data)


def add_doc(doc):
    global list_data
    if doc['_id'] in list_data:
        raise KeyError("Key " + str(doc['_id']) + " already exist in testStubEs")
    list_data[doc['_id']] = doc


def _create_dict_based_on_key(doc, key, data):
    list_key = key.split(".")

    if(len(list_key) == 1):
        if key in doc:
            if isinstance(doc[key], list):
                doc[key].append(data)
            else:
                old_data = doc[key]
                doc[key] = [old_data, data]
        else:
            doc[key] = data
    else:
        first_key = list_key[0]
        if first_key in doc:
            if isinstance(doc[first_key], dict):
                _create_dict_based_on_key(doc[first_key], ".".join(list_key[1:]), data)
            else:
                raise ValueError("Key: " + str(first_key) + " have already a value that isn't a dictionary")
        else:
            doc[first_key] = dict()
            _create_dict_based_on_key(doc[first_key], ".".join(list_key[1:]), data)


def get_random_timestamp(max_delay=604800):  # == one week
    random_timestamp = random.randint(0, max_delay)
    return datetime.fromtimestamp(int(time.time()) - random_timestamp).isoformat()
