import datetime
import random
import numpy as np

from typing import List

all_possible_index_prefix = ["", "logstash-eagleeye-osqueryfilter-"]
all_possible_tags = ["endpoint", "tag"]
all_possible_slave_name = ["eaglesnest.cloud", "eaglesnest.local"]
all_possible_doc_source_type = ["eagleeye", "dummy"]
all_possible_filename = ["osquery_get_all_scheduled_tasks.log", "osquery_get_all_scheduled_actions.log"]
all_possible_deployment_name = ["NVISO Workstations", "NVISO Localhost"]
all_possible_toolname = ['osquery']
all_possible_hostname = ['NVISO-WIN10-JVZ', 'NVISO-LINUX-JVZ', 'NVISO-WIN10-RDE', 'NVISO-WIN10-DRA']


class GenerateDummyDocuments:

    def __init__(self):
        self.id = 0
        now = datetime.datetime.today()
        now -= datetime.timedelta(weeks=3)
        now = now.replace(hour=0, minute=0, second=0)
        self.start_timestamp = now

    def _date_time_to_timestamp(self, date_time):
        return str(date_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))

    def _generate_date(self):
        self.start_timestamp += datetime.timedelta(seconds=1)
        return self.start_timestamp

    def _generate_document(self, create_outlier=False, nbr_tags=1, index=None, slave_name=None, hostname=None):
        doc_date_time = self._generate_date()
        str_date = doc_date_time.strftime("%Y.%m.%d")

        if index is None:
            index = random.choice(all_possible_index_prefix) + str(str_date)

        doc = {
            '_index': index,
            '_type': "doc",
            '_id': self.id,
            '_version': 2,
            '_source': self._generate_source(doc_date_time, create_outlier, nbr_tags, slave_name, hostname)
        }
        self.id += 1
        return doc

    def _generate_source(self, doc_date_time, create_outlier, nbr_tags, slave_name, hostname):
        # Example: 2018-08-23T10:48:16.200315+00:00
        str_timestamp = self._date_time_to_timestamp(doc_date_time)
        filename = random.choice(all_possible_filename)

        if slave_name is None:
            slave_name = random.choice(all_possible_slave_name)

        source = {
            'tags': self._generate_tag(nbr_tags, create_outlier),
            'timestamp': str_timestamp,
            '@timestamp': str_timestamp,
            '@version': '1',
            'slave_name': slave_name,
            'type': random.choice(all_possible_doc_source_type),
            'filename': filename,
            'meta': self._generate_meta(doc_date_time, filename, hostname)
        }
        if create_outlier:
            source['outliers'] = dict()  # TODO
        return source

    def _generate_tag(self, nbr_tags, create_outlier=False):
        list_tags = []
        if nbr_tags > len(all_possible_tags):
            raise ValueError("Not enough possible tags, number of expected tags too high (max: " +
                             str(len(all_possible_tags)) + ")")
        for _ in range(nbr_tags):
            list_tags.append(random.choice(all_possible_tags))

        if create_outlier:
            list_tags.append('outlier')

        return list_tags

    def _generate_meta(self, doc_date_time, filename, hostname):
        if hostname is None:
            hostname = random.choice(all_possible_hostname)

        return {
            'timestamp': self._date_time_to_timestamp(doc_date_time),
            'command': self._generate_query_command(),
            'deployment_name': random.choice(all_possible_deployment_name),
            'toolname': random.choice(all_possible_toolname),
            'filename': filename,
            'hostname': hostname,
            'output_file_path': filename
        }

    def _generate_query_command(self):
        return {
            'name': "get_all_scheduled_tasks",
            'query': "SELECT * FROM scheduled_tasks;",
            'mode': "base_scan"
        }

    def create_documents(self, nbr_document):
        all_doc = []
        for _ in range(nbr_document):
            all_doc.append(self._generate_document())
        return all_doc

    def _compute_number_document_respect_std(self, std_max: float, number_element: int, min_value: int = 0,
                                             max_value: int = 10) -> np.ndarray:
        """
        :param std_max: maximum value accepted for std
        :param number_element: number of samples to generate:
        :param min_value: minimum document
        :param max_value: maximum document
        :return: a list of number of document that must be generated
        """
        list_nbr_documents = np.random.randint(min_value, max_value + 1, size=number_element)
        while list_nbr_documents.std() > std_max:
            index = np.argmax(np.abs(list_nbr_documents - list_nbr_documents.mean()))
            if list_nbr_documents[index] < list_nbr_documents.mean():
                list_nbr_documents[index] += 1
            else:
                list_nbr_documents[index] -= 1
        return list_nbr_documents

    def create_doc_time_variable_sensitivity(self, nbr_val, max_trigger_sensitivity, max_difference, default_value):
        nbr_doc_generated_per_hours = self._compute_number_document_respect_std(max_trigger_sensitivity, nbr_val,
                                                                                default_value-max_difference,
                                                                                default_value+max_difference)
        all_doc = []
        hostname = random.choice(all_possible_hostname)

        for nbr_doc in nbr_doc_generated_per_hours:
            for _ in range(nbr_doc):
                all_doc.append(self._generate_document(hostname=hostname))
            self.start_timestamp += datetime.timedelta(hours=1)
            self.start_timestamp = self.start_timestamp.replace(minute=0, second=0)

        return all_doc
