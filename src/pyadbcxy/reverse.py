"""
This file contains the Reverse module for the PyADBCXY package. It includes the Reverse class,
which implements the reverse mode of automatic differentiation.
"""
import numpy as np


__all__ = ["Reverse"]


class Reverse(object):
    """
    Reverse is the class for implementing the reverse mode auto differentiation including 
    instantiating the class, storing children, evaluating gradient and all required basic
    operations. 

    >>> x = Reverse(3)
    >>> y = Reverse(3)
    >>> z = x + y
    >>> z.val
    6
    >>> z.grad
    1
    """

    def __init__(self, val, grad=1):
        """Constructor for Reverse class

        Args:
            val (int or float): value of the Reverse
            grad (int or float, optional): derivative of the Reverse. Defaults to 1.
        """
        self._val = val
        self._grad = grad
        self._children = []

    def __repr__(self):
        return f"Reverse(val = {self.val})"

    def __str__(self):
        return f"Reverse(val = {self.val}, grad = {self.grad})"

    @property
    def val(self):
        """Get the value of the Reverse

        Examples
        --------
        >>> x = Reverse(3)
        >>> x.val
        3
        """
        return self._val

    @property
    def grad(self):
        """Get the gradient of the Reverse object

        Examples
        --------
        >>> x = Reverse(3)
        >>> x.grad
        1
        """
        if self._grad == None:
            grad = 0
            for der, child in self._children:
                grad += der * child.grad
            self._grad = grad
        return self._grad

    @val.setter
    def val(self, val):
        """Set the value of the Reverse object
        
        Args:
            val (int or float): new value of the Reverse object

        Examples
        --------
        >>> x = Reverse(3)
        >>> x.val
        3
        >>> x.val = 4
        >>> x.val
        4
        """
        self._val = val

    @grad.setter
    def grad(self, grad):
        """Set the gradient of the Reverse object

        Args:
            grad (int, float, or array): new gradient of the Reverse object

        Examples
        --------
        >>> x.Variable(3)
        >>> x.grad
        1
        >>> x.grad = 2
        >>> x.grad
        2
        """
        self._grad = grad

    def __mul__(self, other):
        """Overload of the '*' operator (Reverse * other). Calculates the value and gradient resulting
        from the multiplication of two Reverse objects or a Reverse object and other object.

        Args:
            other (Reverse object, int, or float): item to be added to the Reverse

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x1 = Reverse(3)
        >>> x2 = Reverse(4)
        >>> x3 = x1 * x2
        >>> print(x3)
        Reverse(val = 12, grad = 7)
        >>> x4 = x1 * 3
        >>> print(x4)
        "Reverse(val = 9, der = 3)"
        >>> x5 = x2 * 2.0
        >>> print(x5)
        "Reverse(val = 8.0, der = 2.0)"
        """
        if isinstance(other, Reverse):
            val_mul = self.val * other.val
            new_RevMod = Reverse(val_mul)
            self._children.append((other.val, new_RevMod))
            self.grad = None
            other._children.append((self.val, new_RevMod))
            other.grad = None
            return new_RevMod
        elif isinstance(other, float) or isinstance(other, int):
            val_mul = self.val * other
            new_RevMod = Reverse(val_mul) # instantiate class
            self._children.append((other, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def __add__(self, other):
        """Overload of the '+' operator (Reverse + other). Calculates the value and gradient resulting
        from the addition of two Reverse objects or a Reverse object and other object.

        Args:
            other (Reverse object, int, or float): item to be added to the Reverse

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x1 = Reverse(3)
        >>> x2 = Reverse(4)
        >>> x3 = x1 + x2
        >>> print(x3)
        Reverse(val = 7, grad = 2)
        >>> x4 = x1 + 5
        >>> print(x4)
        Reverse(val = 8, grad = 1)
        >>> x5 = x2 + 2.0
        >>> print(x5)
        Reverse(val = 6.0, grad = 1)
        """
        if isinstance(other, Reverse):
            val_add = self.val + other.val
            new_RevMod = Reverse(val_add)
            self._children.append((1, new_RevMod))
            self.grad = None
            other._children.append((1, new_RevMod))
            other.grad = None
            return new_RevMod
        elif isinstance(other, float) or isinstance(other, int):
            val_add = self.val + other
            new_RevMod = Reverse(val_add)
            self._children.append((1, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def __sub__(self, other):
        """Overload of the '-' operator (Reverse - other). Calculates the value and derivative resulting
        from the addition of two Reverse objects or a Reverse object and other object.

        Args:
            other (Reverse object, int, or float): item to be added to the Reverse

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x1 = Reverse(3)
        >>> x2 = Reverse(4)
        >>> x3 = x1 - x2
        >>> print(x3)
        Reverse(val = -1, grad = 0)
        >>> x4 = x1 - 5
        >>> print(x4)
        Reverse(val = -2, grad = 1)
        >>> x5 = x2 - 2.0
        >>> print(x5)
        Reverse(val = 2.0, grad = 1)
        """
        if isinstance(other, Reverse):
            val_sub = self.val - other.val
            new_RevMod = Reverse(val_sub)
            self._children.append((1, new_RevMod))
            self.grad = None
            other._children.append((-1, new_RevMod))
            other.grad = None
            return new_RevMod
        elif isinstance(other, float) or isinstance(other, int):
            val_sub = self.val - other
            new_RevMod = Reverse(val_sub)
            self._children.append((1, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def __truediv__(self, other):
        """Overload of the '/' operator (Reverse / other). Calculates the value and derivative resulting
        from the division of one Reverse (or other object) from a Reverse.

        Args:
            other (Reverse, int, or float): item the Reverse is to be divided by

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x1 = Reverse(8)
        >>> x2 = Reverse(4)
        >>> x3 = x1 / x2
        >>> print(x3)
        Reverse(val = 2.0, grad = 1)
        >>> x4 = x1 / 2
        >>> print(x4)
        Reverse(val = 4.0, grad = 4.0)
        >>> x5 = x2 / 2.0
        >>> print(x5)
        Reverse(val = 2.0, grad = 0.5)
        """
        if isinstance(other, Reverse):
            if other.val == 0:
                raise ZeroDivisionError("Cannot divide the variable with 0.")
            val_div = self.val / other.val
            new_RevMod = Reverse(val_div)
            self._children.append(( 1 / other.val, new_RevMod))
            self.grad = None
            other._children.append(( - self.val / (other.val ** 2) , new_RevMod))  # need confirmation
            other.grad = None
            return new_RevMod
        elif isinstance(other, float) or isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot divide the variable with 0.")
            val_div = self.val / other
            new_RevMod = Reverse(val_div)
            self._children.append(( 1 / other, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def __radd__(self, other):
        """Overload of the '+' operator (other + Reverse). Calculates the value and derivative resulting
        from the addition of two Reverse objects or a other object and Reverse object.

        Args:
            other (Reverse object, int, or float): item to add Reverse to

        Returns:
            Reverse: resulting Reverse object
        
        Examples
        --------
        >>> x = Reverse(3)
        >>> z = 3 + x
        >>> print(z)
        Reverse(val = 6, grad = 1)
        """
        return self.__add__(other)

    def __rmul__(self, other):
        """Overload of the '*' operator (other * Reverse). Calculates the value and derivative resulting
        from the multiplication of two Reverse objects or other object and Reverse object.

        Args:
            other (Reverse object, int, or float): item to be multiplied with the Reverse

        Returns:
            Reverse: resulting Reverse object
        
        Examples
        --------
        >>> x = Reverse(3)
        >>> z = -5 * x
        >>> print(z)
        Reverse(val = -15, grad = -5)
        """
        return self.__mul__(other)

    def __rsub__(self, other):
        """Overload of the '-' operator (other - Reverse). Calculates the value and derivative resulting
        from the subtraction of two Reverse objects or a Reverse object from another object.

        Args:
            other (Reverse object, int, or float): item to subtract Reverse from

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x = Reverse(3)
        >>> z = 5 - x
        >>> print(z)
        Reverse(val = 2, grad = 1)
        """
        if isinstance(other, Reverse) or isinstance(other, int) or isinstance(other, float):
            new_val = other - self.val
            new_RevMod= Reverse(new_val)
            self._children.append((1, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def __rtruediv__(self, other):
        """Overload of the '/' operator (other / Reverse). Calculates the value and derivative resulting
        from the division of one object by a Reverse object.

        Args:
            other (Reverse, int, or float): item to divide by Reverse object

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> x = Reverse(3)
        >>> z = 9 / x
        >>> print(z)
        Reverse(val = 3.0, grad = -1.0)
        """
        if isinstance(other, Reverse) or isinstance(other, int) or isinstance(other, float):
            if self.val == 0:
                raise ZeroDivisionError("Cannot divide the variable with 0.")
            new_val = other / self.val
            new_RevMod = Reverse(new_val)
            self._children.append(( - other / self.val ** 2, new_RevMod))
            self.grad = None
            return new_RevMod
        else:
            raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")

    def cos(self):
        """Calculates trigonometric cosine of the current Reverse object.

        Args:
            none

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(np.pi, 1)
        >>> z = x.cos()
        >>> print(z)
        Reverse(val = -1, grad = -1.2246467991473532e-16)
        """
        new_val = np.cos(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((-np.sin(self.val), new_RevMod)) # -sinx
        self.grad = None
        return new_RevMod

    def tan(self):
        """Calculates trigonometric tangent of the current Reverse object.

        Args:
            none

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(np.pi / 4)
        >>> z = x.tan()
        >>> print(z)
        Reverse(val = 0.9999999999999999, grad = 1.9999999999999996)
        """
        new_val = np.tan(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append(( 1/(np.cos(self.val) ** 2), new_RevMod)) # 1/ sec **2 x
        self.grad = None
        return new_RevMod

    def sin(self):
        """Calculates trigonometric sine of Reverse and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(np.pi / 2)
        >>> z = x.sin()
        >>> print(z)
        Reverse(val = 1.0, grad = 6.123233995736766e-17)
        """
        new_val = np.sin(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((np.cos(self.val), new_RevMod)) #cosx
        self.grad = None
        return new_RevMod

    def cosh(self):
        """Calculates hyperbolic cosine of Reverse and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(1)
        >>> z = x.cosh()
        >>> print(z)
        Reverse(val = 1.5430806348152437, grad = 1.1752011936438014)
        """
        new_val = np.cosh(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((np.sinh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    def tanh(self):
        """Calculates hyperbolic tanh of Reverse and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(1)
        >>> z = x.tanh()
        >>> print(z)
        Reverse(val = 0.7615941559557649, grad = 0.41997434161402614)
        """
        new_val = np.tanh(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((1 / np.cosh(self.val) ** 2, new_RevMod))  
        self.grad = None
        return new_RevMod

    def sinh(self):
        """Calculates hyperbolic sinh of Reverse and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> x = Reverse(2)
        >>> z = x.sinh()
        >>> print(z)
        Reverse(val = 3.626860407847019, grad = 3.7621956910836314)
        """
        new_val = np.sinh(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((np.cosh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    def arccos(self):
        """Calculates arc arccos of Reverse object and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> r = Reverse(0.9, 0.5)
        >>> r.arccos()
        Reverse(val = 0.45102681179626236, grad = -1.147078669352809)
        """
        # TODO: raise ValueError if self.val outside of domain (NumPy won't, it just returns nan)
        new_val = np.arccos(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((-1 / np.sqrt( 1- self.val ** 2), new_RevMod ))
        self.grad = None
        return new_RevMod
        
    def arctan(self):
        """Calculates arc tangent of Reverse object and returns the result.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> r = Reverse(0.9, 0.5)
        >>> r.arctan()
        Reverse(val = 0.7328151017865066, grad = 0.27624309392265195)
        """
        new_val = np.arctan(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((1 / (1 + self.val ** 2), new_RevMod ))
        self.grad = None
        return new_RevMod

    def arcsin(self):
        """Calculates arc sine of Reverse and returns the result.

        Args:
            None

        Returns:
            Reverse object.

        Examples
        --------
        >>> r = Reverse(0.9, 0.5)
        Reverse(val = 1.1197695149986342, grad = 1.147078669352809)
        """
        # TODO: raise ValueError if self.val outside of domain (NumPy won't, it just returns nan)
        new_val = np.arcsin(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((1 / np.sqrt(1 - self.val**2 ), new_RevMod ))
        self.grad = None
        return new_RevMod

    def exp(self):
        """Calculates exponential (exp()) of Reverse object and returns a Reverse object back.

        Args:
            None

        Returns:
            Reverse object

        Examples
        --------
        >>> r = Reverse(4., 5.)
        >>> r.exp()
        Reverse(val = 54.598150033144236, grad = 272.9907501657212)
        """
        new_val = np.exp(self.val)
        new_RevMod = Reverse(new_val)
        self._children.append((np.exp(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    def log(self, base=np.e):
        """Calculates logarithm (log()) of Reverse, int, or float and returns the result.

        Args:
            base (int or float, optional): logarithm base. Defaults to np.e which uses natural logarithm.

        Returns:
            Reverse: resulting logarithm value

        Examples
        --------
        >>> r = Reverse(4., 5.)
        >>> r.log()
        Reverse(val = 1.3862943611198906, grad = 1.25)
        """
        # TODO: raise ZeroDivisionError if base == 1
        if self.val <= 0:
            raise ValueError(f"Log cannot be negative for this implementation")
        else:
            new_val = np.log(self.val)/ np.log(base)
            new_RevMod = Reverse(new_val)
            self._children.append((1 / (self.val * np.log(base)), new_RevMod))
            self.grad = None
            return new_RevMod

    def __pow__(self, other):
        """Overload of the '**' or 'pow()' operator (Reverse**other). Calculates the value and derivative resulting
        from raising Reverse to the power of other.

        Args:
            other (Reverse, int, or float): item the Reverse is to be raised to

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> Reverse(3) ** Reverse(4., 5.)
        Reverse(val = 81.0, der = 552.9379769105844)
        """
        if self.val > 0:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val ** other.val
                # der_div = other.val * self.val**(other.val-1) * self.der + np.log(self.val) * self.val**other.val * other.der
                new_RevMod = Reverse(new_val)
                self._children.append((other.val * self.val ** (other.val -1 ), new_RevMod))
                self.grad = None
                other._children.append((np.log(self.val) * self.val ** other.val , new_RevMod))
                other.grad = None
                return new_RevMod
            elif isinstance(other, Reverse):
                new_val = self.val ** other
                new_RevMod = Reverse(new_val)
                self._children.append((other*self.val**(other-1), new_RevMod))
                self.grad = None
                return new_RevMod
            else:
                raise TypeError("Reverse mode calculation only accepts Reverse object, int, float types.")
        else:
            raise ValueError('math domain error: the base of exponentiation cannot be non-positive')

    def __rpow__(self, other):
        """Overload of the '**' or 'pow()' operator (other**Reverse). Calculates the value and derivative resulting
        from raising other to the power of the Reverse.

        Args:
            other (Reverse, int, or float): item to raise to the power of the Reverse

        Returns:
            Reverse: resulting Reverse object

        Examples
        --------
        >>> 6 ** Reverse(3)
        Reverse(val = 216, grad = 387.0200453532599)
        """
        if isinstance(other, int) or isinstance(other, float):
            if other > 0:
                new_val = other ** self.val
                new_RevMod = Reverse(new_val)
                self._children.append((np.log(other) * (other ** self.val), new_RevMod))
                self.grad = None
                return new_RevMod
            else:
                raise ValueError('math domain error: the base of exponentiation cannot be non-positive')
        else:
            raise TypeError(f"unsupported operand type(s) for ** or power: '{type(other)}' and '{type(self)}'")

    def __eq__(self, other):
        """Overload of the '==' operator. Determines whether Reverse is equal to
        another object.

        Args:
            other (Reverse, int, or float): item to check equality with Reverse

        Returns:
            tup(bool): tuple whether the Reverse and other object are equal, first
                       index corresponds to the value, second to the derivative

        Examples
        --------
        >>> Reverse(3, 4) == Reverse(3, 4)
        True
        >>> Reverse(3, 4) == Reverse(7, 4)
        False
        >>> Reverse(3, 4) == Reverse(7, 8)
        False
        >>> 7 == Reverse(7, 4)
        False
        """
        if not isinstance(other, Reverse):
            return False
        if self.val != other.val:
            return False
        if not self.grad or not other.grad:
            return False
        else:
            return self.grad == other.grad
        
    def __ne__(self, other):
        return not self.__eq__(other)
