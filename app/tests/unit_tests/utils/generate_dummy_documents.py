import datetime
import random

all_possible_tags = ["endpoint", "tag"]
all_possible_slave_name = ["eaglesnest.cloud", "eaglesnest.local"]
all_possible_doc_source_type = ["eagleeye", "dummy"]
all_possible_filename = ["osquery_get_all_scheduled_tasks.log", "osquery_get_all_scheduled_actions.log"]
all_possible_deployment_name = ["NVISO Workstations", "NVISO Localhost"]
all_possible_toolname = ['osquery']
all_possible_hostname = ['NVISO-WIN10-JVZ', 'NVISO-LINUX-JVZ', 'NVISO-WIN10-RDE', 'NVISO-WIN10-DRA']


class GenerateDummyDocuments:

    def __init__(self):
        self.index = 0
        now = datetime.datetime.today()
        now -= datetime.timedelta(weeks=3)
        self.start_timestamp = now

    def _date_time_to_timestamp(self, date_time):
        return str(date_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))

    def _generate_date(self):
        self.start_timestamp += datetime.timedelta(seconds=1)
        return self.start_timestamp

    def _generate_document(self, create_outlier=False, nbr_tags=1):
        doc_date_time = self._generate_date()
        str_date = doc_date_time.strftime("%Y.%m.%d")
        doc = {
            '_index': "logstash-eagleeye-osqueryfilter-" + str(str_date),
            '_type': "doc",
            '_id': self.index,
            '_version': 2,
            '_source': self._generate_source(doc_date_time, create_outlier, nbr_tags)
        }
        self.index += 1
        return doc

    def _generate_source(self, doc_date_time, create_outlier, nbr_tags):
        # Example: 2018-08-23T10:48:16.200315+00:00
        str_timestamp = self._date_time_to_timestamp(doc_date_time)
        filename = random.choice(all_possible_filename)

        source = {
            'tags': self._generate_tag(nbr_tags, create_outlier),
            'timestamp': str_timestamp,
            '@timestamp': str_timestamp,
            '@version': '1',
            'slave_name': random.choice(all_possible_slave_name),
            'type': random.choice(all_possible_doc_source_type),
            'filename': filename,
            'meta': self._generate_meta(doc_date_time, filename)
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

    def _generate_meta(self, doc_date_time, filename):
        return {
            'timestamp': self._date_time_to_timestamp(doc_date_time),
            'command': self._generate_query_command(),
            'deployment_name': random.choice(all_possible_deployment_name),
            'toolname': random.choice(all_possible_toolname),
            'filename': filename,
            'hostname': random.choice(all_possible_hostname),
            'output_file_path': filename
        }

    def _generate_query_command(self):
        return {
            'name': "get_all_scheduled_tasks",
            'query': "SELECT * FROM scheduled_tasks;",
            'mode': "base_scan"
        }

    def create_document(self, nbr_document):
        all_doc = []
        for _ in range(nbr_document):
            all_doc.append(self._generate_document())
        return all_doc
