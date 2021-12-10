import math
import unittest
from src.pyadbcxy.reverse import Reverse


class TestReverseMode(unittest.TestCase):
    def setUp(self):
        """Prepare the test fixture. Executed before each test method."""
        self.x = Reverse(3)
        self.y = Reverse(4)

    def test_repr_str(self):
        self.assertEqual(repr(self.x), "Reverse(val = 3)")
        self.assertEqual(str(self.x), "Reverse(val = 3, grad = 1)")
        self.assertEqual(repr(self.y), "Reverse(val = 4)")
        self.assertEqual(str(self.y), "Reverse(val = 4, grad = 1)")

    def test_getters(self):
        self.assertEqual(self.x.val, 3)
        self.assertEqual(self.x.grad, 1)
        self.assertEqual(self.y.val, 4)
        self.assertEqual(self.y.grad, 1)

    def test_setters(self):
        self.y.val = 5
        self.y.grad = 2
        self.assertEqual(self.y.val, 5)
        self.assertEqual(self.y.grad, 2)

    def test_mul(self):
        z = self.x * self.y
        self.assertEqual(z.val, 12)
        # self.assertEqual(z.grad, 7) # TODO: are we sure this is correct?

        z = self.x * 5
        self.assertEqual(z.val, 15)
        # self.assertEqual(z.grad, 5)

        with self.assertRaises(TypeError):
            self.x * []

    def test_add(self):
        z = self.x + self.y
        self.assertEqual(z.val, 7)
        # self.assertEqual(z.grad, 2) # TODO: are we sure this is correct?
        
        z = self.x + 5
        self.assertEqual(z.val, 8)
        self.assertEqual(z.grad, 1)

        z = self.y + 2.0
        self.assertEqual(z.val, 6.0)
        self.assertEqual(z.grad, 1)

        with self.assertRaises(TypeError):
            self.x + []

    def test_sub(self):
        z = self.x - self.y
        self.assertEqual(z.val, -1)
        # self.assertEqual(z.grad, 0) # TODO: are we sure this is correct?

        z = self.x - 5
        self.assertEqual(z.val, -2)
        self.assertEqual(z.grad, 1)

        z = self.y - 2.0
        self.assertEqual(z.val, 2.0)
        self.assertEqual(z.grad, 1)

        with self.assertRaises(TypeError):
            self.x - []

    # def test_div(self):
    #     """ Testing when division is on Revmod"""
    #     x = Reverse(3)
    #     y = Reverse(5)
    #     z = x / y
    #     self.assertEqual(z.val, (0.6))
    #     self.assertEqual(float(x.gradient()), 0.2)
    #     self.assertEqual(y.gradient(), -0.12)

    # def test_div_zero_div(self):
    #     """ Testing when division is on Revmod"""
    #     x = Reverse(3)
    #     y = Reverse(0)
    #     with self.assertRaises(ZeroDivisionError):
    #         z = x / y

    # def test_div_value(self):
    #     """ Testing when division is on Revmod"""
    #     x = Reverse(3)
    #     y = 5
    #     z = x / y
    #     self.assertEqual(z.val, (0.6))

    # def test_div_errorcatching(self):
    #     """ Testing when division is on Revmod"""
    #     x = Reverse(3)
    #     y = []
    #     with self.assertRaises(AttributeError):
    #         z = x / y

    # def test_radd(self):
    #     """Testing when addition on Class RevMod and number"""
    #     x = Reverse(3)
    #     summed = 3 + x
    #     self.assertTrue(float(summed.get_val()) == 6 and float(summed.gradient()) == 1)

    # def test_rmul(self):
    #     """ Testing when multiplication is on Revmod and int number"""
    #     x = Reverse(3)
    #     z = -5 * x
    #     self.assertEqual(z.val, -15)
    #     self.assertEqual(z.gradient(), 1)


    # def test_rdiv(self):
    #     """ Testing when division is on Revmod and int number"""
    #     x = Reverse(5)
    #     z = 3 / x
    #     self.assertEqual(z.val, (0.6))


    # def test_rsub(self):
    #     """Testing When subtraction on Class RevMod and number"""
    #     y = Reverse(2)
    #     z = 5 - y
    #     self.assertTrue(float(z.get_val()) == 3) and (float(z.gradient()) == 1.0)

    # def test_cos(self):
    #     """
    #     Test the cos function.
    #     """
    #     v = Reverse(math.pi, 1)
    #     cos_result = v.cos()
    #     self.assertEqual(cos_result.val, math.cos(v.val))
    #     self.assertEqual(cos_result.grad, 1)

    # def test_tan(self):
    #     """
    #     Test the tan function.
    #     """
    #     v = Reverse(0.9, 0.5)
    #     result = v.tan()
    #     self.assertEqual(result.val, math.tan(v.val))
    #     self.assertAlmostEqual(result.grad, 1.2939993666298242, places=6)

    # def test_sin(self):
    #     """
    #     Test the sin function.
    #     """
    #     v = Reverse(4, 5)
    #     result = v.sin()
    #     self.assertAlmostEqual(result.val, math.sin(v.val), places=6)
    #     self.assertEqual(result.grad, -3.2682181043180596)

    # def test_cosh(self):
    #     """
    #     Test the cosh.
    #     """
    #     v = Reverse(4, 5)
    #     cosh_result = v.cosh()
    #     self.assertEqual(cosh_result.val, math.cosh(v.val))
    #     self.assertEqual(cosh_result.grad, math.sinh(v.val) * v.grad)

    # def test_sinh(self):
    #     """
    #     Test the sinh.
    #     """
    #     v = Reverse(4, 5)
    #     sinh_result = v.sinh()
    #     self.assertEqual(sinh_result.val, math.sinh(v.val))
    #     self.assertEqual(sinh_result.grad, v.cosh().val * v.grad)

    # def test_tanh(self):
    #     """
    #     Test the cosh.
    #     """
    #     v = Reverse(0.9, 0.5)
    #     tanh_result = v.tanh()
    #     self.assertEqual(tanh_result.val, math.tanh(v.val))
    #     self.assertEqual(tanh_result.grad, (1 - math.tanh(v.val)**2) * v.grad)


if __name__ == "__main__":
    unittest.main()
