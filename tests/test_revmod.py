## Test cases for reverse mode of Autodiff

import numpy as np
import unittest
from src.ReverseMode import RevMod




## first Testing addition case

class TestReverseMode(unittest.TestCase):
    def setup(self):
        '''fixing values'''
        self.x = RevMod(3)

    def test_self_val(self):
        self.x = RevMod(3)
        self.assertEqual(self.x.get_val(), 3)

    def test_grad(self):
        x = RevMod(3)
        self.assertEqual(x.val, 3)
        self.assertEqual(x.gradient(), 1)

    def test_add(self):
        x = RevMod(3)
        y = RevMod(5)
        summed = x + y
        self.assertEqual(float(summed.get_val()), 8)

    def test_radd(self):
        x = RevMod(3)
        summed = x + 3
        self.assertTrue(float(summed.get_val()) == 6 and float(summed.gradient()) == 1)























if __name__ == "__main__":
    unittest.main()

