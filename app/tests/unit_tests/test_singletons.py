import unittest
from helpers.singletons import settings, logging, es


class TestSingletons(unittest.TestCase):
    def setUp(self):
        pass

    def test_settings_singleton_args_not_none(self):
        self.assertIsNotNone(settings.args)

    def test_logging_singleton_logger_not_none(self):
        self.assertIsNotNone(logging.logger)

    # TODO: add unit tests for Elasticsearch singleton

