import math
import unittest
from src.pyadbcxy.reverse import Reverse


class TestReverseMode(unittest.TestCase):
    def setUp(self):
        """Prepare the test fixture. Executed before each test method."""
        self.x = Reverse(3)
        self.y = Reverse(4)

    def test_repr_str(self):
        self.assertEqual(repr(self.x), "Reverse(val = 3, grad = 1)")
        self.assertEqual(str(self.x), "Reverse(val = 3, grad = 1)")
        self.assertEqual(repr(self.y), "Reverse(val = 4, grad = 1)")
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
        self.assertEqual(self.x.grad, 4)
        self.assertEqual(self.y.grad, 3) # TODO: are we sure this is correct?

        x = Reverse(3)
        z = x * 5
        self.assertEqual(z.val, 15)
        self.assertEqual(x.grad, 5)

        with self.assertRaises(TypeError):
            self.x * []

    def test_add(self):
        z = self.x + self.y
        self.assertEqual(z.val, 7)
        self.assertEqual(self.x.grad, 1) # TODO: are we sure this is correct?
        self.assertEqual(self.y.grad, 1)

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
        self.assertEqual(self.x.grad, 1)
        self.assertEqual(self.y.grad, -1) # TODO: are we sure this is correct?

        z = self.x - 5
        self.assertEqual(z.val, -2)
        self.assertEqual(z.grad, 1)

        z = self.y - 2.0
        self.assertEqual(z.val, 2.0)
        self.assertEqual(z.grad, 1)

        with self.assertRaises(TypeError):
            self.x - []

    def test_div(self):
        x = Reverse(8)
        z = x / self.y
        self.assertEqual(z.val, 2.0)
        self.assertEqual(x.grad, 0.25) # TODO: are we sure this is correct?

        x = Reverse(8)
        z = x / 2
        self.assertEqual(z.val, 4.0)
        self.assertEqual(x.grad, 0.5)

        y = Reverse(4)
        z = y / 2.0
        self.assertEqual(z.val, 2.0)
        self.assertEqual(y.grad, 0.5)

        with self.assertRaises(ZeroDivisionError):
            x / Reverse(0)
        with self.assertRaises(ZeroDivisionError):
            x / 0
        
        with self.assertRaises(TypeError):
            x / []

    def test_radd(self):
        z = 3 + self.x
        self.assertEqual(z.val, 6)
        self.assertEqual(z.grad, 1)

    def test_rmul(self):
        z = -5 * self.x
        self.assertEqual(z.val, -15)
        self.assertEqual(self.x.grad, -5)

    def test_rsub(self):
        z = 5 - self.x
        self.assertEqual(z.val, 2)
        self.assertEqual(z.grad, 1)

        with self.assertRaises(TypeError):
            [] - self.x

    def test_rdiv(self):
        z = 9 / self.x
        self.assertEqual(z.val, 3.0)
        self.assertEqual(self.x.grad, -1.0)

        with self.assertRaises(ZeroDivisionError):
            9 / Reverse(0)

        with self.assertRaises(TypeError):
            [] / self.x

    def test_cos(self):
        """
        Test the cos function.
        """
        v = Reverse(math.pi)
        z = v.cos()
        self.assertEqual(z.val, math.cos(v.val))
        self.assertEqual(v.grad, -math.sin(v.val))

    def test_tan(self):
        """
        Test the tan function.
        """
        v = Reverse(math.pi/4)
        z = v.tan()
        self.assertEqual(z.val, math.tan(v.val))
        self.assertEqual(v.grad, 1/math.cos(v.val)**2)

    def test_sin(self):
        """
        Test the sin function.
        """
        v = Reverse(math.pi/2)
        z = v.sin()
        self.assertEqual(z.val, math.sin(v.val))
        self.assertEqual(v.grad, math.cos(v.val))

    def test_cosh(self):
        """
        Test the cosh.
        """
        x = Reverse(1)
        z = x.cosh()
        self.assertEqual(z.val, math.cosh(x.val))
        self.assertEqual(x.grad, math.sinh(x.val))

    def test_tanh(self):
        """
        Test the tanh.
        """
        x = Reverse(1)
        z = x.tanh()
        self.assertEqual(z.val, math.tanh(x.val))
        self.assertAlmostEqual(x.grad, (1 - math.tanh(x.val)**2))

    def test_sinh(self):
        """
        Test the sinh.
        """
        x = Reverse(2)
        z = x.sinh()
        self.assertEqual(z.val, math.sinh(x.val))
        self.assertEqual(x.grad, math.cosh(x.val))

    def test_arccos(self):
        """
        Test the arccos
        """
        x = Reverse(0.5)
        z = x.arccos()
        self.assertEqual(z.val, math.acos(x.val))
        self.assertAlmostEqual(x.grad, (-1 / math.sqrt( 1 - x.val ** 2)))

    def test_arctan(self):
        """
        Test the arctan
        """
        x = Reverse(0.5)
        z = x.arctan()
        self.assertEqual(z.val, math.atan(x.val))
        self.assertAlmostEqual(x.grad, (1 / (1 + x.val ** 2)))        

    def test_arcsin(self):
        """
        Test the arcsin
        """
        x = Reverse(0.5)
        z = x.arcsin()
        self.assertEqual(z.val, math.asin(x.val))
        self.assertAlmostEqual(x.grad, (1 / math.sqrt(1 - x.val ** 2 )))

    def test_exp(self):
        """
        Test the exponential
        """
        x = Reverse(3)
        z = x.exp()
        self.assertEqual(z.val, math.exp(x.val))
        self.assertAlmostEqual(x.grad, math.exp(x.val))

    def test_log(self):
        """
        Test the logarithm
        """
        x = Reverse(3)
        z = x.log()
        self.assertEqual(z.val, math.log(x.val))
        self.assertAlmostEqual(x.grad, (1 / (x.val * math.log(math.e))))

    def test_pow(self):
        """
        Test power, revpower
        """
        x = Reverse(3)
        y = Reverse(4)
        z = x ** 3
        self.assertEqual(z.val, 27)
        self.assertEqual(x.grad, 27)

        x = Reverse(3)
        y = Reverse(4)
        z = x ** y
        self.assertEqual(z.val, 81)
        self.assertEqual(x.grad, 108)

        x = Reverse(3)
        y = Reverse(4)
        z = 3 ** y
        self.assertEqual(z.val, 81)
        self.assertEqual(y.grad, math.log(3) * (3 ** 4))

    # TODO: test __eq__
    # TODO: test __ne__

if __name__ == "__main__":
    unittest.main()
