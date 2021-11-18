import numpy as np

"""
For reverse mode autodiff implementation

"""


class RevMod:
    def __init__(self, val):
        self.val = val
        self.grad = 1
        self.derivs = []

    def __mul__(self, other):
        try:
            val_mul = self.val * other.val
            new_RevMod = RevMod(val_mul)
            self.derivs.append((other.val, new_RevMod))
            self.grad = None
            other.derivs.append((self.val, new_RevMod))
            other.grad = None
        except AttributeError:
            val_mul = self.val * other
            new_RevMod = RevMod(val_mul)
            self.derivs.append((other, new_RevMod))
            self.grad = None
        return new_RevMod

    def __add__(self, other):
        try:
            val_add = self.val + other.val
            new_RevMod = RevMod(val_add)
            self.derivs.append((1, new_RevMod))
            self.grad = None
            other.derivs.append((1, new_RevMod))
            other.grad = None
        except AttributeError:
            val_add = self.val + other
            new_RevMod = RevMod(val_add)
            self.derivs.append((1, new_RevMod))
            self.grad = None
        return new_RevMod