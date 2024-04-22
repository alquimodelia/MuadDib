import unittest

from util4tests import log, run_single_test

from muaddib import MyModel


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        log.info("testing str.upper()")
        self.assertEqual("foo".upper(), "FOO")

    def test_isupper(self):
        self.assertTrue("FOO".isupper())
        self.assertFalse("Foo".isupper())

    def test_split(self):
        s = "hello world"
        self.assertEqual(s.split(), ["hello", "world"])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_modelError(self):
        m = MyModel()
        self.assertAlmostEqual(
            m.my_method(10), 0.01
        )  # when comparing floats one should allow for some margin
        with self.assertRaises(
            AssertionError
        ):  # to trigger and test for intended failure
            m.my_method(200)


if __name__ == "__main__":
    run_single_test(__file__)
