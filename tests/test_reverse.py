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
        self.assertEqual(z.grad, 7) # TODO: are we sure this is correct?

        z = self.x * 5
        self.assertEqual(z.val, 15)
        self.assertEqual(z.grad, 5)

        with self.assertRaises(TypeError):
            self.x * []

    def test_add(self):
        z = self.x + self.y
        self.assertEqual(z.val, 7)
        self.assertEqual(z.grad, 2) # TODO: are we sure this is correct?
        
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
        self.assertEqual(z.grad, 0) # TODO: are we sure this is correct?

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
        self.assertEqual(z.grad, -0.25) # TODO: are we sure this is correct?

        z = x / 2
        self.assertEqual(z.val, 4.0)
        self.assertEqual(z.grad, 4.0)

        z = self.y / 2.0
        self.assertEqual(z.val, 2.0)
        self.assertEqual(z.grad, 0.5)

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

    # TODO: test arccos
    # TODO: test arctan
    # TODO: test arcsin
    # TODO: test exp
    # TODO: test log
    # TODO: test __pow__
    # TODO: test __rpow__
    # TODO: test __eq__
    # TODO: test __ne__


if __name__ == "__main__":
    unittest.main()
