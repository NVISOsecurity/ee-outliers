import json
import unittest

test_doc = json.load(open("/app/tests/unit_tests/files/doc_without_outlier.json"))


class TestDicts(unittest.TestCase):
    def test_dotdict(self):
        pass
