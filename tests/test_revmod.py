## Test cases for reverse mode of Autodiff

import unittest
from src.ReverseMode import RevMod


## first Testing addition case

class TestReverseMode(unittest.TestCase):
    def setup(self):
        '''fixing values'''
        self.x = RevMod(3)
        self.y = RevMod(4)

    def test_self_val(self):
        self.x = RevMod(3)
        self.assertEqual(self.x.get_val(), 3)

    def set_grad(self):
        """For the case that the gradient needs to be set"""
        x = RevMod(5)
        x.grad = 100
        self.assertEqual(x.grad, 100)

    def test_grad(self):
        x = RevMod(3)
        self.assertEqual(x.val, 3)
        self.assertEqual(x.gradient(), 1)

    def test_add(self):
        """Testing When addition on Class RevMod objects"""
        x = RevMod(3)
        y = RevMod(5)
        summed = x + y
        self.assertEqual(float(summed.get_val()), 8)

    def test_radd(self):
        """Testing when addition on Class RevMod and number"""
        x = RevMod(3)
        summed = x + 3
        self.assertTrue(float(summed.get_val()) == 6 and float(summed.gradient()) == 1)

    def test_sub(self):
        """Testing when subtraction on Class RevMod objects"""
        x = RevMod(3)
        y = RevMod(5)
        z = y - x
        self.assertTrue(float(z.get_val()) == 2) and (float(z.gradient()) == 1.0)

    def test_rsub(self):
        """Testing When subtraction on Class RevMod and number"""
        y = RevMod(5)
        z = y - 2
        self.assertTrue(float(z.get_val()) == 3) and (float(z.gradient()) == 1.0)

    def test_mul(self):
        """ Testing when multiplication on class RevMod """
        x = RevMod(3)
        y = RevMod(5)
        z = x * y
        self.assertEqual(z.val, 15)
        self.assertEqual(x.gradient(), 5)
        self.assertEqual(y.gradient(), 3)

    def test_rmul(self):
        """ Testing when multiplication is on Revmod and int number"""
        x = RevMod(3)
        z = x * -5
        self.assertEqual(z.val, -15)
        self.assertEqual(x.gradient(), -5)

    def test_div(self):
        x = RevMod(3)
        y = RevMod(5)
        z = x / y
        self.assertEqual(z.val, (0.6))
        self.assertEqual(float(x.gradient()), 0.2)
        self.assertEqual(y.gradient(), -0.12)

    def test_rdiv(self):
        x = RevMod(3)
        z = x / 5
        self.assertEqual(z.val, (0.6))
        self.assertEqual(float(x.gradient()), 0.2)














if __name__ == "__main__":
    unittest.main()

