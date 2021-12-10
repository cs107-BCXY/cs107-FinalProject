## Test cases for reverse mode of Autodiff

import unittest
from src.pyadbcxy.reverse import Reverse


## first Testing addition case

class TestReverseMode(unittest.TestCase):
    def setup(self):
        '''fixing values'''
        self.x = Reverse(3)
        self.y = Reverse(4)

    def test_repr_str(self):
        self.x = Reverse(3)
        print(repr(self.x))
        print(str(self.x))

    def test_self_val(self):
        self.x = Reverse(3)
        self.assertEqual(self.x.get_val(), 3)

    def set_grad(self):
        """For the case that the gradient needs to be set"""
        x = Reverse(5)
        x.grad = 100
        self.assertEqual(x.grad, 100)

    def test_grad(self):
        x = Reverse(3)
        self.assertEqual(x.val, 3)
        self.assertEqual(x.gradient(), 1)

    def test_mul_reverse(self):
        """ Testing when multiplication on class RevMod """
        x = Reverse(3)
        y = Reverse(5)
        z = x * y
        self.assertEqual(z.val, 15)
        self.assertEqual(x.gradient(), 5)
        self.assertEqual(y.gradient(), 3)

    def test_mul_value(self):
        x = Reverse(3)
        y = 5
        z = x * y
        self.assertEqual(z.val, 15)
        self.assertEqual(x.gradient(), 5)

    def test_mul_errorcatching(self):
        x = Reverse(3)
        y = []
        with self.assertRaises(AttributeError):
            z = x * y

    def test_add(self):
        """Testing When addition on Class RevMod objects"""
        x = Reverse(3, 1)
        y = Reverse(5, 7)
        summed = x + y
        self.assertEqual(float(summed.get_val()), 8)
        self.assertEqual(summed.gradient(), 15)

    def test_add_value(self):
        """Testing When addition on Class RevMod objects"""
        x = Reverse(3, 1)
        y = 5
        summed = x + y
        self.assertEqual(float(summed.get_val()), 8)
        self.assertEqual(summed.gradient(), 1)

    def test_add_errorcatching(self):
        """Testing When addition on Class RevMod objects"""
        x = Reverse(3, 1)
        y = []
        with self.assertRaises(AttributeError):
            z = x + y

    def test_sub(self):
        """Testing when subtraction on Class RevMod objects"""
        x = Reverse(3)
        y = Reverse(5)
        z = y - x
        self.assertTrue(float(z.get_val()) == 2) and (float(z.gradient()) == 1.0)

    def test_sub_value(self):
        """Testing When addition on Class RevMod objects"""
        x = Reverse(3, 1)
        y = 5
        summed = x - y
        self.assertEqual(float(summed.get_val()), -2)

    def test_sub_errorcatching(self):
        """Testing When addition on Class RevMod objects"""
        x = Reverse(3, 1)
        y = []
        with self.assertRaises(AttributeError):
            z = x - y

    def test_div(self):
        """ Testing when division is on Revmod"""
        x = Reverse(3)
        y = Reverse(5)
        z = x / y
        self.assertEqual(z.val, (0.6))
        self.assertEqual(float(x.gradient()), 0.2)
        self.assertEqual(y.gradient(), -0.12)

    def test_div_zero_div(self):
        """ Testing when division is on Revmod"""
        x = Reverse(3)
        y = Reverse(0)
        with self.assertRaises(ZeroDivisionError):
            z = x / y

    def test_div_value(self):
        """ Testing when division is on Revmod"""
        x = Reverse(3)
        y = 5
        z = x / y
        self.assertEqual(z.val, (0.6))

    def test_div_errorcatching(self):
        """ Testing when division is on Revmod"""
        x = Reverse(3)
        y = []
        with self.assertRaises(AttributeError):
            z = x / y

    def test_radd(self):
        """Testing when addition on Class RevMod and number"""
        x = Reverse(3)
        summed = 3 + x
        self.assertTrue(float(summed.get_val()) == 6 and float(summed.gradient()) == 1)

    def test_rmul(self):
        """ Testing when multiplication is on Revmod and int number"""
        x = Reverse(3)
        z = -5 * x
        self.assertEqual(z.val, -15)
        self.assertEqual(z.gradient(), 1)


    def test_rdiv(self):
        """ Testing when division is on Revmod and int number"""
        x = Reverse(5)
        z = 3 / x
        self.assertEqual(z.val, (0.6))
        self.assertEqual(float(x.gradient()), 0.2)

    def test_rsub(self):
        """Testing When subtraction on Class RevMod and number"""
        y = Reverse(2)
        z = 5 - y
        self.assertTrue(float(z.get_val()) == 3) and (float(z.gradient()) == 1.0)








if __name__ == "__main__":
    unittest.main()
