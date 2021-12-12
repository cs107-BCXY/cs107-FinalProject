import unittest
import copy
from src.pyadbcxy.elementary_functions import *
from src.pyadbcxy.variable import Variable
from src.pyadbcxy.forward import Forward


class TestForward(unittest.TestCase):

    def setUp(self):
        """
        Prepare the test fixture. Executed before each test method.
    `   Inside, we set our x, y, z variables for the convenience of later functions.
        """
        self.x = Variable(3)
        self.y = Variable(4., 5.)
        self.z = Variable(1, 2)
        self.simple_single_var = lambda x: x**2
        self.simple_two_vars = lambda x, y: x + y
        self.fmode1 = Forward(self.simple_single_var, self.x)
        self.fmode2 = Forward(self.simple_two_vars, (self.x, self.y))
    
    def test_constructor(self):
        """
        Check the constructor with simple function and single variable.
        Will also test Variables and function getter methods as they are
        necessary to test constructor.
        """
        func1 = self.fmode1.func
        vars1 = self.fmode1.vars
        self.assertEqual(self.simple_single_var(3), func1(3))
        self.assertEqual([self.x], vars1)

        func2 = self.fmode2.func
        vars2 = self.fmode2.vars
        self.assertEqual(self.simple_two_vars(3, 4.), func2(3, 4.))
        self.assertEqual((self.x, self.y), vars2)

    def test_setter(self):
        """
        Test the variable and function setters.
        """
        fmode = copy.deepcopy(self.fmode1)
        func = fmode.func
        vars = fmode.vars
        self.assertEqual(self.simple_single_var(3), func(3))
        self.assertEqual([self.x], vars)

        newf = lambda t: ((3*t)**2)/4
        fmode.func = newf
        fmode.vars = self.y
        self.assertEqual(newf(4), fmode.func(4))
        self.assertEqual([self.y], fmode.vars)

        fmode.vars = (self.x, self.y)
        self.assertEqual(fmode.vars, (self.x, self.y))
        

    def test_attribute_error(self):
        """
        Test that Attribute error is raised if 'calculate' method has
        not been called yet.
        Although getting the derivative for multiple variable entries will not rely on calling the calculate,
        we still decided to ranse attribute error when calculate is not called. 
        """
        with self.assertRaises(AttributeError):
            self.fmode1.value
        with self.assertRaises(AttributeError):
            self.fmode1.derivative

    def test_simple_single_variable(self):
        """
        Test forward mode on a simple scalar function with a single variable
        input. Will also test value and derivative getter methods.
        """
        self.fmode1.calculate()
        self.assertEqual(self.fmode1.value, 9)
        self.assertEqual(self.fmode1.derivative, 6)

        fmode = Forward(self.simple_single_var, self.y)
        fmode.calculate()
        self.assertEqual(fmode.value, 16.0)
        self.assertEqual(fmode.derivative, 40.0)

    def test_complicated_single_variable(self):
        """
        Test forward mode on a more complicated function with a single variable input.
        """
        f = lambda x: (exp(cos(x)))/(sin(x)**2)
        fmode1 = Forward(f, self.x)
        fmode1.calculate()
        self.assertEqual(fmode1.value, 18.658405892057388)
        self.assertEqual(fmode1.derivative, 259.153784690042)

        fmode2 = Forward(f, self.y)
        fmode2.calculate()
        self.assertEqual(fmode2.value, 0.9081572861687339)
        self.assertEqual(fmode2.derivative, -4.407195647615257)

    def test_simple_two_variables(self):
        """
        Test forward mode on a simple scalar function with a two variable inputs.
        Essentially, the target function that we are trying to calculate towards
        can have any length of input arrays of variables.
        Here, we specifically test two variables.
        """
        self.fmode2.calculate()
        self.assertEqual(self.fmode2.value, 7.0)
        self.assertEqual(self.fmode2.derivative, [1, 5.0])

        f = lambda x, y: x*y
        fmode = Forward(f, (self.x, self.y))
        fmode.calculate()
        self.assertEqual(fmode.value, 12.0)
        self.assertEqual(fmode.derivative, [4.0, 15.0])

    def test_three_variables(self):
        """
        Test forward mode on a simple scalar function with a two variable inputs.
        Essentially, the target function that we are trying to calculate towards
        can have any length of input arrays of variables.
        Here, we specifically test three variables.
        """
        f = lambda x, y, z: cos(x)*y-sin(z)
        fmode = Forward(f, (self.x, self.y, self.z))
        fmode.calculate()
        self.assertAlmostEqual(fmode.value, -4.801440971209678, 7)
        self.assertEqual(fmode.derivative, [-0.5644800322394689, -4.949962483002227, -1.0806046117362795])




if __name__ == "__main__":
    unittest.main()