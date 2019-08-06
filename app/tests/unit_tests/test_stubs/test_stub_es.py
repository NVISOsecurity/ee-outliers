from helpers.singletons import es


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
                "default_remove_all_outliers": es.remove_all_outliers,
                "default_flush_bulk_actions": es.flush_bulk_actions
            }

    def apply_new_es(self):
        es.BULK_FLUSH_SIZE = 0
        es.__init__ = self.new_init
        es.init_connection = self.init_connection
        es._scan = self._scan
        es._count_documents = self._count_documents
        es.remove_all_outliers = self.remove_all_outliers
        es.flush_bulk_actions = self.flush_bulk_actions

    def restore_es(self):
        es.BULK_FLUSH_SIZE = self.default_es_methods["default_bulk_flush_size"]
        es.__init__ = self.default_es_methods["default_init"]
        es.init_connection = self.default_es_methods["default_init_connection"]
        es._scan = self.default_es_methods["default_scan"]
        es._count_documents = self.default_es_methods["default_count_documents"]
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

    def remove_all_outliers(self):
        self.list_data = dict()

    def flush_bulk_actions(self, refresh=False):
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
                self.list_data[bulk['_id']] = data
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
