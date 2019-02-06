import unittest


class TestSingletons(unittest.TestCase):
    def setUp(self):
        pass

    def test_singleton_creations(self):
        from helpers.singletons import settings, logging, es
        type(settings)
        type(logging)
        type(es)
