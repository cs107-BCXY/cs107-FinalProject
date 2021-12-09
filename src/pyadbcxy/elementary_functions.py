"""
This file contains all of the elementary functions for the cs107-BCXY package.
It implements the behavior of basic functions on the Variable objects that are
not dunder methods. Such functions include trigonometric functions, logarithms,
etcetera.
"""
import numpy as np
from .variable import Variable


__all__ = ["log", "exp", "root", "sin", "sinh", "arcsin", "cos", "cosh",
           "arccos", "tan", "tanh", "arctan", "logistic"]


def log(input, base=np.e):
    """Calculates logarithm (log()) of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply logarithm to
        base (int or float, optional): logarithm base. Defaults to np.e which uses natural logarithm.

    Returns:
        Variable, int, or float: resulting logarithm value

    Examples
    --------
    >>> log(Variable(4., 5.))
    Variable(val = 1.3862943611198906, der = 1.25)
    """
    if isinstance(input, int) or isinstance(input, float):
        if base == 1:
            # as per math.log standard
            raise ZeroDivisionError("float division by zero")
        elif base > 0:
            # this will still apply when base = np.e because np.log(np.e) == 1
            if input <= 0:
                # as per math.log standard
                raise ValueError("math domain error")
            return np.log(input)/np.log(base)
        else:
            raise ValueError("math domain error")
    elif isinstance(input, Variable):
        if base == 1:
            # as per math.log standard
            raise ZeroDivisionError("float division by zero")
        elif base > 0:
            # this will still apply when base = np.e because np.log(np.e) == 1
            if input.val <= 0:
                # as per math.log standard
                raise ValueError("math domain error")
            return Variable(val = np.log(input.val)/np.log(base), der = input.der/(input.val * np.log(base)))
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
    >>> exp(Variable(4., 5.))
    Variable(val = 54.598150033144236, der = 272.9907501657212)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.exp(input)
    elif isinstance(input, Variable):
        return Variable(val = np.exp(input.val), der = np.exp(input.val)*input.der)
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
    >>> root(Variable(4., 5.))
    Variable(val = 2.0, der = 0.25)
    """
    if isinstance(input, int) or isinstance(input, float):
        return input**(1.0/n)
    elif isinstance(input, Variable):
        return Variable(val = input.val**(1.0/n), der = (1.0/n) * input.val ** (1.0/n - 1))
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
    >>> sin(Variable(4., 5.))
    Variable(val = -0.7568024953079282, der = -3.2682181043180596)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.sin(input)
    elif isinstance(input, Variable):
        return Variable(val = np.sin(input.val), der = np.cos(input.val)*input.der)
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
    >>> sinh(Variable(4., 5.))
    Variable(val = 27.28991719712775, der = 136.54116418008243)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.sinh(input)
    elif isinstance(input, Variable):
        return Variable(val = np.sinh(input.val), der = np.cosh(input.val)*input.der)
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
    >>> arcsin(Variable(0.9, 0.5))
    Variable(val = 1.1197695149986342, der = 1.147078669352809)
    """
    if isinstance(input, int) or isinstance(input, float):
        if input < -1 or input > 1:
            raise ValueError("math domain error")
        return np.arcsin(input)
    elif isinstance(input, Variable):
        if input.val < -1 or input.val > 1:
            raise ValueError("math domain error")
        return Variable(val = np.arcsin(input.val), der = input.der/np.sqrt(1 - input.val**2))
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
    >>> cos(Variable(4., 5.))
    Variable(val = -0.6536436208636119, der = 3.7840124765396412)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.cos(input)
    elif isinstance(input, Variable):
        return Variable(val = np.cos(input.val), der = -1*np.sin(input.val)*input.der)
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
    >>> cosh(Variable(4., 5.))
    Variable(val = 27.308232836016487, der = 136.44958598563875)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.cosh(input)
    elif isinstance(input, Variable):
        return Variable(val = np.cosh(input.val), der = np.sinh(input.val)*input.der)
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
    >>> arccos(Variable(0.9, 0.5))
    Variable(val = 0.45102681179626236, der = -1.147078669352809)
    """
    if isinstance(input, int) or isinstance(input, float):
        if input < -1 or input > 1:
            raise ValueError("math domain error")
        return np.arccos(input)
    elif isinstance(input, Variable):
        if input.val < -1 or input.val > 1:
            raise ValueError("math domain error")
        return Variable(val = np.arccos(input.val), der = -1*input.der/np.sqrt(1 - input.val**2))
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
    >>> tan(Variable(0.9, 0.5))
    Variable(val = 1.2601582175503392, der = 1.2939993666298242)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.tan(input)
    elif isinstance(input, Variable):
        return Variable(val = np.tan(input.val), der = input.der*(1/np.cos(input.val)**2))
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
    >>> tanh(Variable(0.9, 0.5))
    Variable(val = 0.7162978701990245, der = 0.24345868057417075)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.tanh(input)
    elif isinstance(input, Variable):
        return Variable(val = np.tanh(input.val), der = (1 - np.tanh(input.val)**2)*input.der)
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
    >>> arctan(Variable(0.9, 0.5))
    Variable(val = 0.7328151017865066, der = 0.27624309392265195)
    """
    if isinstance(input, int) or isinstance(input, float):
        return np.arctan(input)
    elif isinstance(input, Variable):
        return Variable(val = np.arctan(input.val), der = input.der/(1 + input.val**2))
    else:
        raise TypeError(f"must be a real number or Variable object, not {type(input)}")

def logistic(input):
    """Calculates logistic [1/(1 + e^-x)] of Variable, int, or float and returns the result.

    Args:
        input (Variable, int, or float): item to apply logistic function to

    Returns:
        Variable, int, or float: resulting value object

    Examples
    --------
    >>> logistic(Variable(3))
    Variable(val = 0.9525741268224334, der = 0.045176659730912144)
    """
    return 1/(1 + exp(-1*input))
