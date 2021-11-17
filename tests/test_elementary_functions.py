import unittest
from src.variable import Variable
from src.elementary_functions import *


class TestElementaryFunctions(unittest.TestCase):

    def setUp(self):
        """Prepare the test fixture. Executed before each test method."""
        pass

    def test_log(self):
        """Test the logarithm function with the following arguments:
            a Variable object,
            a Variable object and a base > 0, /= e, /= 1,
            a Variable object and base 1,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_exp(self):
        """Test the exponential function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_root(self):
        """Test the root function with the following arguments:
            a Variable object,
            a Variable object and n /= 2,
            a floating point number,
            a floating point number and n /= 2,
            an integer,
            an integer and n /= 2,
            an object of an invalid type."""
        pass

    def test_sin(self):
        """Test the sin function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_sinh(self):
        """Test the sinh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_arcsin(self):
        """Test the arcsin function with the following arguments:
            a Variable object with -1 <= value < 1,
            a Variable object with value < -1,
            a Variable object with value > 1,
            a floating point number between -1 and 1 (inclusive),
            a floating point number less than -1,
            a floating point number greater than 1,
            an integer between -1 and 1 (inclusive),
            an integer less than -1,
            an integer greater than 1,
            an object of an invalid type."""
        pass

    def test_cos(self):
        """Test the cos function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_cosh(self):
        """Test the cosh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_arccos(self):
        """Test the arccos function with the following arguments:
            a Variable object with -1 <= value < 1,
            a Variable object with value < -1,
            a Variable object with value > 1,
            a floating point number between -1 and 1 (inclusive),
            a floating point number less than -1,
            a floating point number greater than 1,
            an integer between -1 and 1 (inclusive),
            an integer less than -1,
            an integer greater than 1,
            an object of an invalid type."""
        pass

    def test_tan(self):
        """Test the tan function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_tanh(self):
        """Test the tanh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass

    def test_arctan(self):
        """Test the arctan function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type."""
        pass


if __name__ == "__main__":
    unittest.main()