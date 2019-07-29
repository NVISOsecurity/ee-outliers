import datetime
import random
import numpy as np

all_possible_index_prefix = ["", "logstash-eagleeye-osqueryfilter-"]
all_possible_tags = ["endpoint", "tag"]
all_possible_slave_name = ["eaglesnest.cloud", "eaglesnest.local"]
all_possible_doc_source_type = ["eagleeye", "dummy"]
all_possible_filename = ["osquery_get_all_scheduled_tasks.log", "osquery_get_all_scheduled_actions.log"]
all_possible_deployment_name = ["Company Workstations", "Company Localhost", "Localhost", "Company Test", "Testhost",
                                "Google", "Dummy Environment", "Deployment system", "New Company", "Super Company",
                                "New deployment", "Test deployment"]
all_possible_toolname = ['osquery']
all_possible_hostname = ['DUMMY-WIN10-JVZ', 'DUMMY-LINUX-JVZ', 'DUMMY-WIN10-RDE', 'DUMMY-WIN10-DRA', 'LOCAL-WIN-RDE',
                         'TEST-LINUX-XYZ', 'TEST-WIN-XYZ', 'localhost', 'abcdefghijklmno', 'aaaaaaa', "here",
                         "DUMMY-VERY-LONG-HOSTNAME-TO-ENHANCE-TEST"]
all_test_hex_values = ["not hex value", "12177014F73", "5468697320697320612074657374",
                       "The same text 5468652073616d652074657874"]
all_test_base64_values = ["QVlCQUJUVQ==", "VGhpcyBpcyBhIHRleHQ=", "not base"]
all_test_url_values = ["http://google.be", "This is a test without URL", "Example: http://www.dance.com/",
                       "http://nviso.be", "http://long-url-example-to-test.brussels"]


class DummyDocumentsGenerate:

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

    def generate_document(self, create_outlier=False, nbr_tags=1, index=None, slave_name=None, hostname=None,
                          deployment_name=None, user_id=None, test_hex_value=None, test_base64_value=None,
                          test_url_value=None):
        doc_date_time = self._generate_date()
        str_date = doc_date_time.strftime("%Y.%m.%d")

        if index is None:
            index = random.choice(all_possible_index_prefix) + str(str_date)

        doc = {
            '_index': index,
            '_type': "doc",
            '_id': self.id,
            '_version': 2,
            '_source': self._generate_source(doc_date_time, create_outlier, nbr_tags, slave_name, hostname,
                                             deployment_name, user_id, test_hex_value, test_base64_value,
                                             test_url_value)
        }
        self.id += 1
        return doc

    def _generate_source(self, doc_date_time, create_outlier, nbr_tags, slave_name, hostname, deployment_name,
                         user_id, test_hex_value, test_base64_value, test_url_value):
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
            'meta': self._generate_meta(doc_date_time, filename, hostname, deployment_name, user_id),
            'test': self._generate_test_data(test_hex_value, test_base64_value, test_url_value)
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

    def _generate_meta(self, doc_date_time, filename, hostname, deployment_name, user_id):
        if hostname is None:
            hostname = random.choice(all_possible_hostname)

        if deployment_name is None:
            deployment_name = random.choice(all_possible_deployment_name),

        if user_id is None:
            user_id = random.randint(0, 5)

        return {
            'timestamp': self._date_time_to_timestamp(doc_date_time),
            'command': self._generate_query_command(),
            'deployment_name': deployment_name,
            'toolname': random.choice(all_possible_toolname),
            'filename': filename,
            'hostname': hostname,
            'output_file_path': filename,
            'user_id': user_id
        }

    def _generate_query_command(self):
        return {
            'name': "get_all_scheduled_tasks",
            'query': "SELECT * FROM scheduled_tasks;",
            'mode': "base_scan"
        }

    def _generate_test_data(self, test_hex_value, test_base64_value, test_url_value):
        if test_hex_value is None:
            test_hex_value = random.choice(all_test_hex_values)

        if test_base64_value is None:
            test_base64_value = random.choice(all_test_base64_values)

        if test_url_value is None:
            test_url_value = random.choice(all_test_url_values)

        return {
            'hex_value': test_hex_value,
            'base64_value': test_base64_value,
            'url_value': test_url_value
        }

    def create_documents(self, nbr_document):
        all_doc = []
        for _ in range(nbr_document):
            all_doc.append(self.generate_document())
        return all_doc

    def _compute_number_document_have_at_least_specific_coef_variation(self, coef_var_min: float,
                                                                       min_number_element: int, min_value: int = 0,
                                                                       max_value: int = 10) -> np.ndarray:
        list_nbr_documents = np.random.randint(min_value, max_value + 1, size=min_number_element)

        # While current list doesn't have the minimum coef
        while list_nbr_documents.std()/list_nbr_documents.mean() < coef_var_min:
            index = np.argmax(list_nbr_documents)
            if list_nbr_documents[index] < list_nbr_documents.mean():
                list_nbr_documents[index] -= 1
            else:
                list_nbr_documents[index] += 1

        return list_nbr_documents

    def _compute_number_document_have_at_most_specific_coef_variation(self, coef_var_max: float,
                                                                      min_number_element: int, min_value: int = 0,
                                                                      max_value: int = 10) -> np.ndarray:
        list_nbr_documents = np.random.randint(min_value, max_value + 1, size=min_number_element)

        # While current list doesn't have the minimum coef
        while list_nbr_documents.std()/list_nbr_documents.mean() > coef_var_max:
            index = np.argmax(list_nbr_documents)
            if list_nbr_documents[index] < list_nbr_documents.mean():
                list_nbr_documents[index] += 1
            else:
                list_nbr_documents[index] -= 1

        return list_nbr_documents

    def _generate_doc_time_variable_sensitivity(self, nbr_doc_generated_per_hours):
        all_doc = []
        hostname = random.choice(all_possible_hostname)

        for nbr_doc in nbr_doc_generated_per_hours:
            for _ in range(nbr_doc):
                all_doc.append(self.generate_document(hostname=hostname))
            self.start_timestamp += datetime.timedelta(hours=1)
            self.start_timestamp = self.start_timestamp.replace(minute=0, second=0)

        return all_doc

    def create_doc_uniq_target_variable_at_least_specific_coef_variation(self, nbr_val, coef_var_min, max_difference,
                                                                         default_value):
        nbr_doc_to_generate = self._compute_number_document_have_at_least_specific_coef_variation(
            coef_var_min, nbr_val, default_value - max_difference, default_value + max_difference)

        return self._generate_doc_time_variable_sensitivity(nbr_doc_to_generate)

    def create_doc_uniq_target_variable_at_most_specific_coef_variation(self, nbr_val, coef_var_max, max_difference,
                                                                        default_value):
        nbr_doc_to_generate = self._compute_number_document_have_at_most_specific_coef_variation(
            coef_var_max, nbr_val, default_value - max_difference, default_value + max_difference)

        return self._generate_doc_time_variable_sensitivity(nbr_doc_to_generate)
