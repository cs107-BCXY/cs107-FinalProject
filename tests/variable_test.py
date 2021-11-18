import unittest
import sys
sys.path.append(sys.path[0][:-5] + 'src')
from variable import Variable


class TestVariable(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
