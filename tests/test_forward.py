import unittest
from src.variable import Variable
from src.elementary_functions import *
from src.forward import Forward


class TestForward(unittest.TestCase):
    
    def test_constructor(self):
        """Check the constructor with simple function and single variable.
        Will also check getter methods as they are necessary to test constructor."""
        x = Variable(3)
        f = lambda t: t**2
        fmode = Forward(f, x)
        func = fmode.func
        vars = fmode.vars
        self.assertEqual(f(3), func(3))
        self.assertEqual([x], vars)


if __name__ == "__main__":
    unittest.main()