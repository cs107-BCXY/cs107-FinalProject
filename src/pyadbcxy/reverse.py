import numpy as np

"""
For reverse mode autodiff implementation

"""


__all__ = ["Reverse"]


class Reverse:
    """
    RevMod is the class for implementing the reverse mode auto differentiation including 
    instantiating the class, storing children, evaluating gradient and all required basic
    operations. 

    >>> x = Reverse(3)
    >>> y = Reverse(3)
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
            new_RevMod = Reverse(val_mul)
            self.children.append((other.val, new_RevMod))
            self.grad = None
            other.children.append((self.val, new_RevMod))
            other.grad = None
            # now include when attribute error
        except AttributeError:
            val_mul = self.val * other
            new_RevMod = Reverse(val_mul) # instantiate class
            self.children.append((other, new_RevMod))
            self.grad = None
        return new_RevMod

    def __add__(self, other):
        try:
            val_add = self.val + other.val
            new_RevMod = Reverse(val_add)
            self.children.append((1, new_RevMod))
            self.grad = None
            other.children.append((1, new_RevMod))
            other.grad = None
        except AttributeError:
            val_add = self.val + other
            new_RevMod = Reverse(val_add)
            self.children.append((1, new_RevMod))
            self.grad = None
        return new_RevMod

    # subtraction: adding "-"
    def __sub__(self, other):
        try:
            val_sub = self.val - other.val
            new_RevMod = Reverse(val_sub)
            self.children.append((1, new_RevMod))
            self.grad = None
            other.children.append((-1, new_RevMod))
            other.grad = None
        except AttributeError:
            val_sub = self.val - other
            new_RevMod = Reverse(val_sub)
            self.children.append((1, new_RevMod))
            self.grad = None

        return new_RevMod

    # division
    def __truediv__(self, other):
        try:
            val_div = self.val / other.val
            new_RevMod = Reverse(val_div)
            self.children.append(( 1 / other.val, new_RevMod))
            self.grad = None
            other.children.append(( - self.val / (other.val ** 2) , new_RevMod))  # need confirmation
            other.grad = None
        except AttributeError:
            val_div = self.val / other
            new_RevMod = Reverse(val_div)
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
        new_RevMod= Reverse(new_val)
        self.children.append((1, new_RevMod))
        self.grad = None
        return new_RevMod

    # rtruedivision
    def __rtruediv__(self, other):
        new_val = other / self.val
        new_RevMod = Reverse(new_val)
        self.children.append(( - other / self.val ** 2, new_RevMod))
        self.grad = None
        return new_RevMod

    ## Trigonomic Functions:

    # cosine
    def cos(self):
        new_val = np.cos(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((-np.sin(self.val), new_RevMod)) # -sinx
        self.grad = None
        return new_RevMod

    # tangent
    def tan(self):
        new_val = np.tan(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append(( 1/(np.cos(self.val) ** 2), new_RevMod)) # 1/ sec **2 x
        self.grad = None
        return new_RevMod

    # sin function
    def sin(self):
        new_val = np.csin(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((np.cos(self.val), new_RevMod)) #cosx
        self.grad = None
        return new_RevMod

    # adding hyperbolic functions:

    def cosh(self):
        """hyperbolic cosine function"""
        new_val = np.cosh(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((np.sinh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod


    def tanh(self):
        """hyperbolic tangent function"""
        new_val = np.tanh(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((1 / np.cosh(self.val) ** 2, new_RevMod))  
        self.grad = None
        return new_RevMod

    def sinh(self):
        """hyperbolic sine function"""
        new_val = np.sinh(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((np.cosh(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    def arccos(x):
        """arc cosine func"""
        new_val = np.arccos(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((-1 / np.sqrt( 1- self.val ** 2), new_RevMod ))
        self.grad = None
        return new_RevMod
        
    def arctan(x):
        """arc tangent func"""
        new_val = np.arctan(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((1 / (1 + self.val ** 2), new_RevMod ))
        self.grad = None
        return new_RevMod

    def arcsin(x):
        """arc sine func"""
        new_val = np.arcsin(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((1 / np.sqrt(1 - self.val **2 ), new_RevMod ))
        self.grad = None
        return new_RevMod



    # Other Functions as needed or required:

    def sqrt(self): #square root 
        new_val = np.sqrt(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append(( 0.5 * self.val ** (-0.5), new_RevMod))  
        self.grad = None
        return new_RevMod

    def exp(self): #exponential func
        new_val = np.exp(self.val)
        new_RevMod = Reverse(new_val)
        self.children.append((np.exp(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    def log(self, base): # log function
        if self.val < 0:
            raise ValueError(f"Log cannot be negative for this implementation")
        else:
            new_val = np.log(self.val)/ np.log(base)
            new_RevMod = Reverse(new_val)
            self.children.append((1 / (self.val * np.log(base)), new_RevMod))
            self.grad = None
            return new_RevMod
 
    def __pow__(self, other):
        try:
            new_val = self.val ** other.val
            # der_div = other.val * self.val**(other.val-1) * self.der + np.log(self.val) * self.val**other.val * other.der
            new_RevMod = Reverse(new_val)
            self.children.append((other.val * self.val ** (other.val -1 ), new_RevMod))
            self.grad = None
            other.children.append((np.log(self.val) * self.val ** other.val , new_RevMod))
            other.grad = None
        # when not RevMod class 
        except AttributeError: 
            new_val = self.val ** other
            new_RevMod = Reverse(new_val)
            self.children.append((other*self.val**(other-1), new_RevMod))
            self.grad = None
        return new_RevMod


    def __rpow__(self, other):
        new_val = other ** self.val
        new_RevMod = Reverse(new_val)
        self.children.append((np.log(other) * (other ** self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    # _equal and not equal
    def __eq__(self, other):
        if not isinstance(other, Reverse):
            return False
        if self.val != other.val:
            return False
        if not self.grad or not other.grad:
            return False
        else:
            return self.gradient() == other.gradient()
        
    def __ne__(self, other):
        return not self.__eq__(other)




if __name__ == '__main__':
    x = Reverse(2)
    y = Reverse(5)
    z = x.log(2)
    print(z.val, y.children, x.gradient(), y.grad)


