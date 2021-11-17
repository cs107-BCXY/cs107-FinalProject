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
    """Calculates logarithm (log()) of Variable, int, or float and returns the result.

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
    """Calculates exponential (exp()) of Variable, int, or float and returns the result.

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

def root(input, n=2):
    """Calculates nth root (square root, cube root, etc.) of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply square root to
        n (int or float, optional): root base. Defaults to 2 which is the square root.

    Returns:
        Variable, int, or float: resulting root value

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float) or isinstance(input, Variable):
        return input**(1.0/n)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def sin(input):
    """Calculates trigonometric sine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply sine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.sin(input)
    elif isinstance(input, Variable):
        return Variable(val = math.sin(input.val), der = math.cos(input.val)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def sinh(input):
    """Calculates hyperbolic sine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply hyperbolic sine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.sinh(input)
    elif isinstance(input, Variable):
        return Variable(val = math.sinh(input.val), der = math.cosh(input.val)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def arcsin(input):
    """Calculates arc sine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply arc sine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    # don't need to check for -1 <= input <= 1, math.asin will handle it
    if isinstance(input, int) or isinstance(input, float):
        return math.asin(input)
    elif isinstance(input, Variable):
        return Variable(val = math.asin(input.val), der = input.der/math.sqrt(1 - input.val**2))
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def cos(input):
    """Calculates trigonometric cosine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply cosine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.cos(input)
    elif isinstance(input, Variable):
        return Variable(val = math.cos(input.val), der = -1*math.sin(input.val)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def cosh(input):
    """Calculates hyperbolic cosine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply hyperbolic cosine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.cosh(input)
    elif isinstance(input, Variable):
        return Variable(val = math.cosh(input.val), der = math.sinh(input.val)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def arccos(input):
    """Calculates arc cosine of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply arc cosine function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    # don't need to check for -1 <= input <= 1, math.acos will handle it
    if isinstance(input, int) or isinstance(input, float):
        return math.acos(input)
    elif isinstance(input, Variable):
        return Variable(val = math.acos(input.val), der = -1*input.der/math.sqrt(1 - input.val**2))
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def tan(input):
    """Calculates trigonometric tangent of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply tangent function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.tan(input)
    elif isinstance(input, Variable):
        return Variable(val = math.tan(input.val), der = input.der*(1/math.cos(input.val)**2))
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def tanh(input):
    """Calculates hyperbolic tangent of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply hyperbolic tangent function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.tanh(input)
    elif isinstance(input, Variable):
        return Variable(val = math.tanh(input.val), der = (1 - math.tanh(input.val)**2)*input.der)
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def arctan(input):
    """Calculates arc tangent of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply arc tangent function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    """
    # TODO: write examples for docstring
    if isinstance(input, int) or isinstance(input, float):
        return math.atan(input)
    elif isinstance(input, Variable):
        return Variable(val = math.atan(input.val), der = input.der/(1 + input.val**2))
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")