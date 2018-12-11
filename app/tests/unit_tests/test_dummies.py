import unittest


class TestDummies(unittest.TestCase):
    # Mock unit tests, which will always succeed.
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a  String - testing Daan
        with self.assertRaises(TypeError):
            s.split(2)
