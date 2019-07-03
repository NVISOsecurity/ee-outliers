import time
import random
from datetime import datetime

from helpers.es import ES

POSSIBLE_SLAVE_NAME = ["ee-slave-lab", "random-name"]
POSSIBLE_META_CMD_NAME = ["get_all_processes_with_listening_conns", "get_all_scheduled_tasks"]
POSSIBLE_TAGS = [["unknown_hashes", "endpoint"], ["endpoint"], ["test", "unknown_hashes"]]


class mokup_es(ES):

    BULK_FLUSH_SIZE = 0

    def __init__(self, settings=None, logging=None):
        super().__init__(settings, logging)
        self.list_data = dict()
        self.id = 0

    def init_connection(self):
        return None

    def scan(self, index="", bool_clause=None, sort_clause=None, query_fields=None, search_query=None):
        for element in self.list_data.values():
            yield element

    def count_documents(self, index="", bool_clause=None, query_fields=None, search_query=None):
        return len(self.list_data)

    def _update_es(self, doc):
        id = doc['_id']
        if id in self.list_data:
            self.list_data[id] = doc

    def remove_all_outliers(self):
        self.list_data = dict()

    def flush_bulk_actions(self, refresh=False):
        if len(self.bulk_actions) == 0:
            return

        for bulk in self.bulk_actions:
            if bulk['_op_type'] == 'update':
                data = {
                    "_source": bulk['doc'],
                    "_id": bulk['_id']
                }
                if '_type' in bulk:
                    data['_type'] = bulk['_type']
                if '_index' in bulk:
                    data['_index'] = bulk['_index']
                self.list_data[bulk['_id']] = data
            else:
                raise KeyError('Unknown bulk action: "' + bulk['_op_type'] + '"')
        self.bulk_actions = []

    def generate_data(self, nbr_data=1, fixed_infos=dict()):
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
            self.add_data(dictionary_data)

    """
    Add "fake" data, that can be return by custom elastic search
    
    :param dictionary_data dictionary with key and value (key with the format: key1.key2.key3)
    """
    def add_data(self, dictionary_data):
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
            raise KeyError("Key " + str(doc['_id']) + " already exist in mokup_es")
        self.list_data[doc['_id']] = doc

    def _create_dict_based_on_key(self, doc, key, data):
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
                    self._create_dict_based_on_key(doc[first_key], ".".join(list_key[1:]), data)
                else:
                    raise ValueError("Key: " + str(first_key) + " have already a value that isn't a dictionary")
            else:
                doc[first_key] = dict()
                self._create_dict_based_on_key(doc[first_key], ".".join(list_key[1:]), data)


def get_random_timestamp(max_delay = 604800): # == one week
    random_timestamp = random.randint(0, max_delay)
    return datetime.fromtimestamp(int(time.time()) - random_timestamp).isoformat()
