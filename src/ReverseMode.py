import numpy as np

"""
For reverse mode autodiff implementation

"""


class RevMod:
    """
    RevMod is the class for implementing the reverse mode auto differentiation including 
    instantiating the class, storing children, evaluating gradient and all required basic
    operations. 

    >>> x = RevMod(3)
    >>> y = RevMod(3)
    >>> z = x + y
    >>> z.val
    6
    >>> z.gradient()
    1
    """

    def __init__(self, val):
        self.val = val
        self.grad = 1
        self.children = []

## Adding a repr rev class
    def __repr__(self):
        return (f'RevMod({self.val})')

    def __str__(self):
        return (f'RevMod({self.val}), Its Gradient: {self.grad}')


## will just return the value
    def get_val(self):
        return self.val

### function creating gradient ***

    def gradient(self):
        if self.grad == None:
            grad = 0
            for der, child in self.children:
                grad += der * child.gradient()
            self.grad = grad
        return self.grad # returns or updates gradient


## basic operations in rev mode

    def __mul__(self, other):
        try:
            val_mul = self.val * other.val
            new_RevMod = RevMod(val_mul)
            self.children.append((other.val, new_RevMod))
            self.grad = None
            other.children.append((self.val, new_RevMod))
            other.grad = None
            # now include when attribute error
        except AttributeError:
            val_mul = self.val * other
            new_RevMod = RevMod(val_mul) # instantiate class
            self.children.append((other, new_RevMod))
            self.grad = None
        return new_RevMod

    def __add__(self, other):
        try:
            val_add = self.val + other.val
            new_RevMod = RevMod(val_add)
            self.children.append((1, new_RevMod))
            self.grad = None
            other.children.append((1, new_RevMod))
            other.grad = None
        except AttributeError:
            val_add = self.val + other
            new_RevMod = RevMod(val_add)
            self.children.append((1, new_RevMod))
            self.grad = None
        return new_RevMod

    # subtraction: adding "-"
    def __sub__(self, other):
        try:
            val_sub = self.val - other.val
            new_RevMod = RevMod(val_sub)
            self.children.append((1, new_RevMod))
            self.grad = None
            other.children.append((-1, new_RevMod))
            other.grad = None
        except AttributeError:
            val_sub = self.val - other
            new_RevMod = RevMod(val_sub)
            self.children.append((1, new_RevMod))
            self.grad = None

        return new_RevMod

    # division
    def __truediv__(self, other):
        try:
            val_div = self.val / other.val
            new_RevMod = RevMod(val_div)
            self.children.append(( 1 / other.val, new_RevMod))
            self.grad = None
            other.children.append(( - self.val / (other.val ** 2) , new_RevMod))  # need confirmation
            other.grad = None
        except AttributeError:
            val_div = self.val / other
            new_RevMod = RevMod(val_div)
            self.children.append(( 1 / other, new_RevMod))
            self.grad = None

        return new_RevMod

    ## adding section to address reversed operands:

    # raddition
    def __radd__(self, other):
        return self.__add__(other)

    # rmultiplication
    def __rmul__(self, other):
        return self.__mul__(other)

    # rsubtraction
    def __rsub__(self,other):
        new_val = other - self.val
        new_RevMod= RevMod(new_val)
        self.children.append((1, new_RevMod))
        self.grad = None
        return new_RevMod

    # rtruedivision
    def __rtruediv__(self, other):
        new_val = other / self.val
        new_RevMod = RevMod(new_val)
        self.children.append(( - other / self.val ** 2, new_RevMod))
        self.grad = None
        return new_RevMod


    # cosine
    def cos(self):
        new_RevMod = RevMod(np.cos(val))
        self.grad = None
        self.children.append((-np.sin(self.val), new_RevMod)) # -sinx
        return new_RevMod

    # tangent
    def tan(self):
        new_RevMod = RevMod(np.tan(val))
        self.children.append(( 1/(np.cos(self.val) ** 2), new_RevMod)) # 1/ sec **2 x
        self.grad = None
        return new_RevMod

    # sin function
    def sin(self):
        new_RevMod = RevMod(np.sin(val))
        self.grad = None
        self.children.append((np.cos(self.val), new_RevMod)) #cosx
        return new_RevMod

    # adding hyperbolic functions:

    def cosh(self):
        """hyperbolic cosine function"""
        new_RevMod = RevMod(np.cosh(self.val))
        self.children.append((np.sinh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod


    def tanh(self):
        """hyperbolic tangent function"""
        new_RevMod = RevMod(np.tanh(self.val))
        self.children.append((1 / np.cosh(self.val) ** 2, new_RevMod))  
        self.grad = None
        return new_RevMod

    def sinh(self):
        """hyperbolic sine function"""
        new_RevMod = RevMod(np.sinh(self.val))
        self.children.append((np.cosh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod


    # handling exponential function
    def __pow__(self, other):
        if not isinstance(other, RevMod):
            pass














# if __name__ == '__main__':
#     x = RevMod(4)
#     y = RevMod(4)
#     z = x + y
#     print(z.val)
#Still needed to add
# negation 
# exp
# eq. ne etc.. comparison dunder methods?

