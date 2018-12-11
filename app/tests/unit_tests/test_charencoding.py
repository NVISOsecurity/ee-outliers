import unittest


class TestUnicode(unittest.TestCase):
    def setUp(self):
        pass

    def test_print_string(self):
        self.assertEqual("ボールト","ボールト")

