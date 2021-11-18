import numpy as np

"""
For reverse mode autodiff implementation

"""


class RevMod:
    def __init__(self, val):
        self.val = val
        self.grad = 1
        self.children = []

    def __mul__(self, other):
        try:
            val_mul = self.val * other.val
            new_RevMod = RevMod(val_mul)
            self.children.append((other.val, new_RevMod))
            self.grad = None
            other.children.append((self.val, new_RevMod))
            other.grad = None
        except AttributeError:
            val_mul = self.val * other
            new_RevMod = RevMod(val_mul)
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

    def get_value(self):
        return self.val

    def __repr__(self):
        return 'RevMod({})'.format(self.val)

    def __str__(self):
        return 'RevMod({})'.format(self.val)

    # cosine
    def cos(self):
        val = np.cos(self.val)
        new_RevMod = RevMod(val)
        self.children.append((-np.sin(self.val), new_RevMod))
        self.grad = None
        return new_RevMod

    # tangent
    def tan(self):
        val = np.tan(self.val)
        new_RevMod = RevMod(val)
        self.children.append(( 1/(np.cos(self.val)**2), new_RevMod)) 
        self.grad = None
        return new_RevMod

    # sin function
    def sin(self):
        new_RevMod = RevMod(np.sin(self.val))
        self.children.append((np.cos(self.val), new_RevMod))
        self.grad = None
        return new_RevMod




        