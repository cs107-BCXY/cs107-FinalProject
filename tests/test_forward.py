import unittest
import copy
from src.variable import Variable
from src.elementary_functions import *
from src.forward import Forward


class TestForward(unittest.TestCase):

    def setUp(self):
        """Prepare the test fixture. Executed before each test method."""
        self.x = Variable(3)
        self.y = Variable(4., 5.)
        self.simple_scalar_func = lambda t: t**2
        self.fmode1 = Forward(self.simple_scalar_func, self.x)
    
    def test_constructor(self):
        """Check the constructor with simple function and single variable.
        Will also test Variables and function getter methods as they are
        necessary to test constructor."""
        func = self.fmode1.func
        vars = self.fmode1.vars
        self.assertEqual(self.simple_scalar_func(3), func(3))
        self.assertEqual([self.x], vars)

    def test_setter(self):
        """Test the variable and function setters."""
        fmode = copy.deepcopy(self.fmode1)
        func = fmode.func
        vars = fmode.vars
        self.assertEqual(self.simple_scalar_func(3), func(3))
        self.assertEqual([self.x], vars)
        newf = lambda t: ((3*t)**2)/4
        fmode.func = newf
        fmode.vars = self.y
        self.assertEqual(newf(4), fmode.func(4))
        self.assertEqual([self.y], fmode.vars)

    def test_attribute_error(self):
        """Test that Attribute error is raised if 'calculate' method has
        not been called yet."""
        with self.assertRaises(AttributeError):
            self.fmode1.value
        with self.assertRaises(AttributeError):
            self.fmode1.derivative

    def test_simple_single_variable(self):
        """Test forward mode on a simple scalar function with a single variable
        input. Will also test value and derivative getter methods."""
        self.fmode1.calculate()
        self.assertEqual(self.fmode1.value, 9)
        self.assertEqual(self.fmode1.derivative, 6)

        fmode = Forward(self.simple_scalar_func, self.y)
        fmode.calculate()
        self.assertEqual(fmode.value, 16.0)
        self.assertEqual(fmode.derivative, 40.0)

    def test_complicated_single_variable(self):
        """Test forward mode on a more complicated function with a single variable input."""
        f = lambda x: (exp(cos(x)))/(sin(x)**2)
        fmode1 = Forward(f, self.x)
        fmode1.calculate()
        self.assertEqual(fmode1.value, 18.658405892057388)
        self.assertEqual(fmode1.derivative, 259.153784690042)

        fmode2 = Forward(f, self.y)
        fmode2.calculate()
        self.assertEqual(fmode2.value, 0.9081572861687339)
        self.assertEqual(fmode2.derivative, -4.407195647615257)


if __name__ == "__main__":
    unittest.main()