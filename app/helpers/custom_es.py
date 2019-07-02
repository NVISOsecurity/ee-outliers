import time
import random
from datetime import datetime

from helpers.es import ES

POSSIBLE_SLAVE_NAME = ["ee-slave-lab", "random-name"]
POSSIBLE_META_CMD_NAME = ["get_all_processes_with_listening_conns", "get_all_scheduled_tasks"]
POSSIBLE_TAGS = [["unknown_hashes", "endpoint"], ["endpoint"], ["test", "unknown_hashes"]]

class custom_es(ES):

    def __init__(self, settings=None, logging=None):
        super().__init__(settings, logging)
        self.list_data = list()
        self.id = 0

    def init_connection(self):
        return None

    def scan(self, index = "", bool_clause=None, sort_clause=None, query_fields=None, search_query=None):
        for element in self.list_data:
            yield element

    def count_documents(self, index = "", bool_clause=None, query_fields=None, search_query=None):
        return len(self.list_data)

    def _update_es(self, doc):
        for i in range(len(self.list_data)):
            elem = self.list_data[i]
            if elem['_id'] == doc['_id']:
                self.list_data[i] = doc
                break

    def remove_all_outliers(self):
        self.list_data = []

    def flush_bulk_actions(self, refresh=False):
        self.bulk_actions = []

    def generate_data(self, nbrData = 1, fixedInfos = dict()):
        for _ in range(nbrData):
            slave_name = random.choice(POSSIBLE_SLAVE_NAME)
            meta_cmd_name = random.choice(POSSIBLE_META_CMD_NAME)
            tags = random.choice(POSSIBLE_TAGS)

            self.add_minimum_data(get_random_timestamp(), slave_name, meta_cmd_name, tags, extra_infos=fixedInfos)

    def add_minimum_data(self, timestamp, slave_name, meta_cmd_name, tags = list(), osquery_cmd = None,
                         osquery_name = None, osquery_address = None, osquery_port = None, bro_event_type = None,
                         bro_server_name = None, bro_id_orig_h = None, extra_infos = dict()):
        dictionary_data = {}
        dictionary_data["@timestamp"] = timestamp
        dictionary_data["timestamp"] = timestamp
        dictionary_data["tags"] = tags
        dictionary_data["slave_name"] = slave_name
        dictionary_data["meta.command.name"] = meta_cmd_name

        dictionary_data["OsqueryFilter.cmdline"] = osquery_cmd
        dictionary_data["OsqueryFilter.name"] = osquery_name
        dictionary_data["OsqueryFilter.remote_address.raw"] = osquery_address
        dictionary_data["OsqueryFilter.remote_port.raw"] = osquery_port

        dictionary_data["BroFilter.event_type"] = bro_event_type
        dictionary_data["BroFilter.server_name"] = bro_server_name
        dictionary_data["BroFilter.id_orig_h"] = bro_id_orig_h
        dictionary_data.update(extra_infos)

        self.add_data(dictionary_data)


    """
    Add "fake" data, that can be return by custom elastic search
    
    :param dictionary_data dictionary with key and value (key with the format: key1.key2.key3)
    """
    def add_data(self, dictionary_data):
        source = {}
        for key, val in dictionary_data.items():
            self._add_key_data(source, key, val)
        data = {
            "_source": source,
            "_id": self.id
        }
        self.id += 1

        self.list_data.append(data)

    def _add_key_data(self, doc, key, data):
        listKey = key.split(".")

        if(len(listKey) == 1):
            if key in doc:
                if isinstance(doc[key], list):
                    doc[key].append(data)
                else:
                    oldData = doc[key]
                    doc[key] = [oldData, data]
            else:
                doc[key] = data
        else:
            firstKey = listKey[0]
            if firstKey in doc:
                if isinstance(doc[firstKey], dict):
                    self._add_key_data(doc[firstKey], ".".join(listKey[1:]), data)
                else:
                    raise ValueError("Key: " + str(firstKey) + " have already a value that isn't a dictionary")
            else:
                doc[firstKey] = dict()
                self._add_key_data(doc[firstKey], ".".join(listKey[1:]), data)

def get_random_timestamp(max_delay = 604800): # == one week
    randomTimestamp = random.randint(0, max_delay)
    return datetime.fromtimestamp(int(time.time()) - randomTimestamp).isoformat()
