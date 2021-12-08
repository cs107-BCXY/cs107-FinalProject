import math
import unittest
from src.pyadbcxy.variable import Variable


class TestVariable(unittest.TestCase):
    def setUp(self):
        """Prepare the test fixture. Executed before each test method."""
        self.x = Variable(3)
        self.y = Variable(4., 5.)

    def test_constructor(self):
        """Check constructor with single argument."""
        self.assertEqual(self.x.val, 3)
        self.assertEqual(self.x.der, 1)

    def test_property(self):
        """
        Check that Variable is initialized correctly.
        """
        x = Variable(3)
        self.assertEqual(x.val, 3)
        self.assertEqual(x.der, 1)

    def test_set_val(self):
        """
        Testing setting val function.
        """
        x = Variable(3)
        x.val = 5
        self.assertEqual(x.val, 5)

    def test_set_der(self):
        """
        testing setting der function.
        """
        x = Variable(3)
        x.der = 7
        self.assertEqual(x.der, 7)

    def test_add_radd(self):
        """
        testing add and radd function.
        """
        x = Variable(3)
        y = Variable(4)
        result = x + y
        self.assertEqual(result.val, 7)
        self.assertEqual(result.der, 2)

        x = Variable(3)
        y = 4
        result = x + y
        self.assertEqual(result.val, 7)
        self.assertEqual(result.der, 1)

        x = 3
        y = Variable(4)
        result = x + y
        self.assertEqual(result.val, 7)
        self.assertEqual(result.der, 1)

    def test_constructor_getter(self):
        """Test both input argument and getter."""
        self.assertEqual(self.y.val, 4.)
        self.assertEqual(self.y.der, 5.)

    def test_setter(self):
        """Test setter."""
        self.x.val = 6
        self.x.der = 7
        self.assertEqual(self.x.val, 6)
        self.assertEqual(self.x.der, 7)

    def test_add(self):
        """Test addition of the following combinations
            2 Variable objects,
            a Variable object and a constant number,
            a Variable and an object of an invalid type."""
        z = self.x + self.y
        self.assertEqual(z.val, 7)
        self.assertEqual(z.der, 6)

        y = 6
        z1 = self.x + y
        self.assertEqual(z1.val, 9)
        self.assertEqual(z1.der, 1)

        z2 = y + self.x
        self.assertEqual(z2.val, 9)
        self.assertEqual(z2.der, 1)

        with self.assertRaises(TypeError):
            self.x + None
        with self.assertRaises(TypeError):
            None + self.x
        with self.assertRaises(TypeError):
            self.x + list()
        with self.assertRaises(TypeError):
            tuple() + self.x

    def test_mul(self):
        """Test multiplication of the following combinations
            2 Variable objects,
            a Variable object and a constant number,
            a Variable and an object of an invalid type."""
        z = self.x * self.y
        self.assertEqual(z.val, 12)
        self.assertEqual(z.der, 19)

        y = 6
        z1 = self.x * y
        self.assertEqual(z1.val, 18)
        self.assertEqual(z1.der, 6)

        z2 = y * self.x
        self.assertEqual(z2.val, 18)
        self.assertEqual(z2.der, 6)

        with self.assertRaises(TypeError):
            self.x * None
        with self.assertRaises(TypeError):
            None * self.x
        with self.assertRaises(TypeError):
            self.x * list()
        with self.assertRaises(TypeError):
            tuple() * self.x

    def test_neg(self):
        """Test negation."""
        z = -self.x
        self.assertEqual(z.val, -3)
        self.assertEqual(z.der, -1)

    def test_sub(self):
        """Test subtraction of the following combinations
            2 Variable objects,
            a Variable object and a constant number,
            a Variable and an object of an invalid type."""
        z = self.x - self.y
        self.assertEqual(z.val, -1)
        self.assertEqual(z.der, -4)

        y = 6
        z1 = self.x - y
        self.assertEqual(z1.val, -3)
        self.assertEqual(z1.der, 1)

        z2 = y - self.x
        self.assertEqual(z2.val, 3)
        self.assertEqual(z2.der, -1)

        with self.assertRaises(TypeError):
            self.x - None
        with self.assertRaises(TypeError):
            None - self.x
        with self.assertRaises(TypeError):
            self.x - list()
        with self.assertRaises(TypeError):
            tuple() - self.x

    def test_div_nonzero(self):
        """Test division of the following nonzero combinations
            2 Variable objects,
            a Variable object and a constant number,
            a Variable and an object of an invalid type."""
        z = self.x / self.y
        self.assertEqual(z.val, 3 / 4)
        self.assertEqual(z.der, -11 / 16)

        y = 6
        z1 = self.x / y
        self.assertEqual(z1.val, 1 / 2)
        self.assertEqual(z1.der, 1 / 6)

        z2 = y / self.x
        self.assertEqual(z2.val, 2)
        self.assertEqual(z2.der, -2 / 3)

        with self.assertRaises(TypeError):
            self.x / None
        with self.assertRaises(TypeError):
            None / self.x
        with self.assertRaises(TypeError):
            self.x / list()
        with self.assertRaises(TypeError):
            tuple() / self.x

    def test_zero_division_error(self):
        """Test division when the denominator is 0."""
        y = Variable(0)
        with self.assertRaises(ZeroDivisionError):
            self.x / y

        y = 0
        with self.assertRaises(ZeroDivisionError):
            self.x / y

    def test_pow(self):
        """Test power of the following combinations
            2 Variable objects,
            a Variable object and a constant number,
            a Variable and an object of an invalid type."""
        z = self.x ** self.y
        self.assertEqual(z.val, 3 ** 4)
        self.assertEqual(z.der, 3 ** 4 * (5 * math.log(3) + 4 * 1 / 3))

        y = 6
        z1 = self.x ** y
        self.assertEqual(z1.val, 3 ** 6)
        self.assertEqual(z1.der, 6 * 3 ** 5)

        z2 = y ** self.x
        self.assertEqual(z2.val, 6 ** 3)
        self.assertEqual(z2.der, 6 ** 3 * math.log(6))

        with self.assertRaises(TypeError):
            self.x ** None
        with self.assertRaises(TypeError):
            None ** self.x
        with self.assertRaises(TypeError):
            self.x ** list()
        with self.assertRaises(TypeError):
            tuple() ** self.x

    def test_pow_negative_base_error(self):
        """Test exponentiation with non-positive base and variable exponent."""
        y = Variable(0)
        with self.assertRaises(ValueError):
            y ** self.x
        y = Variable(-10)
        with self.assertRaises(ValueError):
            y ** self.x
        with self.assertRaises(ValueError):
            0 ** self.x
        with self.assertRaises(ValueError):
            (-3) ** self.x

    def test_eq(self):
        """Test equality operator."""
        self.assertNotEqual(self.x, self.y)
        x = Variable(3, 1)
        self.assertEqual(self.x, x)
        self.assertNotEqual(self.x, 3)
        self.assertNotEqual(self.x, list())

    def test_str_repr(self):
        """Test str and repr."""
        self.assertEqual(str(self.x), "Variable(val = 3, der = 1)")
        self.assertEqual(repr(self.x), "Variable(val = 3, der = 1)")
        self.assertEqual(str(self.y), "Variable(val = 4.0, der = 5.0)")
        self.assertEqual(repr(self.y), "Variable(val = 4.0, der = 5.0)")


if __name__ == "__main__":
    unittest.main()
