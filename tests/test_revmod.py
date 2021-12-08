## Test cases for reverse mode of Autodiff

import numpy as np
import unittest
from src.ReverseMode import RevMod




## first Testing addition case

def test_add_func():
    x = RevMod(20)
    y = RevMod(5)
    summed = x + y
    assertEqual(float(summed.get_value()), 25)


