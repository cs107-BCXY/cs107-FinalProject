"""
This file contains all of the elementary functions for the cs107-BCXY package. It implements the behavior
of basic functions on the Variable objects that are not dunder methods. Such functions include trigonometric
functions, logarithms, etcetera.
"""
import math
# from src.variable import Variable
import sys
sys.path.append(sys.path[0][:-5] + 'src')
from variable import Variable
# TODO: remove path editing, using for development purposes


def log(input, base=math.e):
    """Executes logarithm operation (log()) on Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply logarithm to
        base (int or float, optional): logarithm base. Defaults to math.e which uses natural logarithm.

    Returns:
        Variable, int, or float: resulting logarithm value

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.log(input, base)
    elif isinstance(input, Variable):
        if base == 1:
            # as per math.log standard
            raise ZeroDivisionError("float division by zero")
        elif base > 0:
            # this will still apply when base = math.e because math.log(math.e) == 1
            # if Variable.val < 0, this will be caught by math.log()
            return Variable(val = math.log(input.val, base), der = input.der/(input.val * math.log(base)))
        else:
            raise ValueError("math domain error")
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def exp(input):
    """Executes exponential operation (exp()) on Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply exponential to

    Returns:
        Variable, int, or float: resulting exponential value

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.exp(input)
    elif isinstance(input, Variable):
        return Variable(val = math.exp(input.val), der = math.exp(input.val)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")


if __name__ == "__main__":
    print(log('a'))