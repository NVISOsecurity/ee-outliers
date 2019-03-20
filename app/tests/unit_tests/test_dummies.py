import unittest


class TestDummies(unittest.TestCase):
    # Mock unit tests, which will always succeed.
    def test_upper_1(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper_1(self):
        self.assertTrue('FOO'.isupper())

    def test_isupper_2(self):
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])

    def test_split_fails(self):
        s = 'hello world'
        # check that s.split fails when the separator is not a  String - testing Daan
        with self.assertRaises(TypeError):
            s.split(2)
