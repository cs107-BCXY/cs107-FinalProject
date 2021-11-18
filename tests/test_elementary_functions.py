import unittest
import sys
sys.path.append(sys.path[0][:-5])
from src.variable import Variable
from src.elementary_functions import *
import math


class TestElementaryFunctions(unittest.TestCase):

    def setUp(self):
        """
        Prepare the test fixture. 
        Executed before each test method.
        """
        self.var1 = Variable(3)
        self.var2 = Variable(4., 5.)
        self.fp = 3.5
        self.i = 2


    def test_log(self):
        """
        Test the logarithm function with the following arguments:
            a Variable object,
            a Variable object and a base > 0, /= e, /= 1,
            a Variable object and base 1,
            a floating point number,
            an integer,
            an object of an invalid type.
        """

        # a variable object with base e
        log_result = log(self.var2)
        self.assertEqual(log_result.val, math.log(self.var2.val))
        self.assertEqual(log_result.der, self.var2.der / (self.var2.val * math.log(math.e)))

        # a variable object with a non-default base=8
        base = 8
        log_result = log(self.var2, base)
        self.assertEqual(log_result.val, math.log(self.var2.val, base))
        self.assertEqual(log_result.der, self.var2.der / (self.var2.val * math.log(base)))

        var = 10
        self.assertEqual(log(var), math.log(var))


    def test_exp(self):
        """
        Test the exponential function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        exp_result = exp(self.var2)
        self.assertEqual(exp_result.val, math.exp(self.var2.val))
        self.assertEqual(exp_result.der, math.exp(self.var2.val) * self.var2.der)

        var = 20
        exp_result = exp(var)
        self.assertEqual(exp_result, math.exp(20))

    def test_root(self):
        """
        Test the root function with the following arguments:
            a Variable object,
            a Variable object and n /= 2,
            a floating point number,
            a floating point number and n /= 2,
            an integer,
            an integer and n /= 2,
            an object of an invalid type.
        """
        # normal sqaure root
        root_result = root(self.var2)
        self.assertEqual(root_result.val, self.var2.val ** 0.5)
        self.assertEqual(root_result.der, 0.5 * self.var2.val ** (-0.5))

        var = 20
        self.assertEqual(root(20, 3), 20 ** (1.0/3))

        # nth root

    def test_sin(self):
        """
        Test the sin function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        sin_result = sin(self.var2)
        self.assertEqual(sin_result.val, math.sin(self.var2.val))
        self.assertEqual(sin_result.der, math.cos(self.var2.val) * self.var2.der)

        var = math.pi
        self.assertAlmostEqual(sin(var), 0, places=7)

        var = 0
        self.assertAlmostEqual(sin(var), 0, places=7)
        

    def test_sinh(self):
        """
        Test the sinh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        sinh_result = sinh(self.var2)
        self.assertEqual(sinh_result.val, math.sinh(self.var2.val))
        self.assertEqual(sinh_result.der, cosh(self.var2).val * self.var2.der)

        var = math.pi
        self.assertEqual(sinh(var), math.sinh(var))


    def test_arcsin(self):
        """
        Test the arcsin function with the following arguments:
            a Variable object with -1 <= value < 1,
            a Variable object with value < -1,
            a Variable object with value > 1,
            a floating point number between -1 and 1 (inclusive),
            a floating point number less than -1,
            a floating point number greater than 1,
            an integer between -1 and 1 (inclusive),
            an integer less than -1,
            an integer greater than 1,
            an object of an invalid type.
        """
        var2 = Variable(0.9, 0.5)
        arcsin_result = arcsin(var2)
        self.assertEqual(arcsin_result.val, math.asin(var2.val))
        self.assertEqual(arcsin_result.der, var2.der/math.sqrt(1-var2.val**2))


        var2 = Variable(-0.7, 0.5)
        arcsin_result = arcsin(var2)
        self.assertEqual(arcsin_result.val, math.asin(var2.val))
        self.assertEqual(arcsin_result.der, var2.der/math.sqrt(1-var2.val**2))

        var = 0.8
        self.assertEqual(arcsin(var), math.asin(var))


    def test_cos(self):
        """
        Test the cos function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        cos_result = cos(self.var2)
        self.assertEqual(cos_result.val, math.cos(self.var2.val))
        self.assertEqual(cos_result.der, -math.sin(self.var2.val) * self.var2.der)

        var = math.pi
        self.assertAlmostEqual(cos(var), -1, places=7)

        var = 0
        self.assertAlmostEqual(cos(var), 1, places=7)


    def test_cosh(self):
        """
        Test the cosh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        cosh_result = cosh(self.var2)
        self.assertEqual(cosh_result.val, math.cosh(self.var2.val))
        self.assertEqual(cosh_result.der, math.sinh(self.var2.val) * self.var2.der)

        var = math.pi
        self.assertEqual(cosh(var), math.cosh(var))


    def test_arccos(self):
        """
        Test the arccos function with the following arguments:
            a Variable object with -1 <= value < 1,
            a Variable object with value < -1,
            a Variable object with value > 1,
            a floating point number between -1 and 1 (inclusive),
            a floating point number less than -1,
            a floating point number greater than 1,
            an integer between -1 and 1 (inclusive),
            an integer less than -1,
            an integer greater than 1,
            an object of an invalid type.
        """
        var2 = Variable(0.9, 0.5)
        arccos_result = arccos(var2)
        self.assertEqual(arccos_result.val, math.acos(var2.val))
        self.assertEqual(arccos_result.der, -var2.der/math.sqrt(1-var2.val**2))


        var2 = Variable(-0.7, 0.5)
        arccos_result = arccos(var2)
        self.assertEqual(arccos_result.val, math.acos(var2.val))
        self.assertEqual(arccos_result.der, -var2.der/math.sqrt(1-var2.val**2))

        var = 0.8
        self.assertEqual(arccos(var), math.acos(var))


    def test_tan(self):
        """
        Test the tan function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        var2 = Variable(0.9, 0.5)
        tan_result = tan(var2)
        self.assertEqual(tan_result.val, math.tan(var2.val))
        self.assertEqual(tan_result.der, (1 + (math.sin(0.9)/math.cos(0.9)) ** 2) * 0.5)


        var2 = Variable(-0.7, 0.5)
        tan_result = tan(var2)
        self.assertEqual(tan_result.val, math.tan(var2.val))
        self.assertEqual(tan_result.der, (1 + (math.sin(-0.7)/math.cos(-0.7)) ** 2) * 0.5)

        var = 0.8
        self.assertEqual(tan(var), math.tan(var))


    def test_tanh(self):
        """
        Test the tanh function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        var2 = Variable(0.9, 0.5)
        tanh_result = tanh(var2)
        self.assertEqual(tanh_result.val, math.tanh(var2.val))
        self.assertEqual(tanh_result.der, (1 - math.tanh(var2.val)**2) * var2.der)


        var2 = Variable(-0.7, 0.5)
        tanh_result = tanh(var2)
        self.assertEqual(tanh_result.val, math.tanh(var2.val))
        self.assertEqual(tanh_result.der, (1 - math.tanh(var2.val)**2) * var2.der)

        var = 0.8
        self.assertEqual(tanh(var), math.tanh(var))


    def test_arctan(self):
        """
        Test the arctan function with the following arguments:
            a Variable object,
            a floating point number,
            an integer,
            an object of an invalid type.
        """
        pass



if __name__ == "__main__":
    unittest.main()