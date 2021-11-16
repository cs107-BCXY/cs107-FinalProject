"""
This file contains the Variable module for the cs107-BCXY package. It includes the Variable class,
which implements the creation of the Variable object, as well as numerous basic operations on the
Variable.
"""
import math


class Variable(object):
	"""
	This class implements all variables, to include the basic operations necessary
	to perform automatic differentiation.

	Examples
	--------

	# Compute the derivative of a function consisting of basic operations
	>>> x = Variable(3)
	>>> f = x**2
	>>> f.val
	9
	>>> f.der
	6
	"""

	def __init__(self, val, der=1):
		"""Constructor for Variable class

		Args:
			val (int or float): value of the variable
			der (int or float, optional): derivative of the variable. Defaults to 1.
		"""
		self._val = val
		self._der = der

	@property
	def val(self):
		"""Get the value of the Variable

		Examples
		--------
		>>> x = Variable(3)
		>>> x.val
		3
		"""
		return self._val

	@property
	def der(self):
		"""Get the derivative of the Variable

		Examples
		--------
		>>> x = Variable(3)
		>>> x.der
		1
		"""
		return self._der

	@val.setter
	def val(self, val):
		"""Set the value of the Variable

		Args:
			val (int or float): new value of the variable

		Examples
		--------
		>>> x = Variable(3)
		>>> x.val
		3
		>>> x.val = 4
		>>> x.val
		4
		"""
		self._val = val

	@der.setter
	def der(self, der):
		"""Set the derivative of the Variable.
		[WARNING] Should not be used unless there is legitimate reason.

		Args:
			val (int or float): new derivative of the variable

		>>> x = Variable(3)
		>>> x.der
		1
		>>> x.der = 2
		>>> x.der
		2
		"""
		self._der = der

	def __add__(self, other):
		"""Overload of the '+' operator (Variable + other). Calculates the value and derivative resulting
		from the addition of two variables or a variable and other object.

		Args:
			other (Variable, int, or float): item to be added to the Variable

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		>>> x1 = Variable(3)
		>>> x2 = Variable(4)
		>>> x3 = x1 + x2
		>>> print(x3)
		"Variable(val = 7, der = 2)"
		>>> x4 = x1 + 5
		>>> print(x4)
		"Variable(val = 8, der = 1)
		>>> x5 = x2 + 2.0
		>>> print(x5)
		"Variable(val = 6.0, der = 1)
		"""
		if isinstance(other, int) or isinstance(other, float):
			return Variable(val = self.val + other, der = self.der)
		elif isinstance(other, Variable):
			return Variable(val = self.val + other.val, der = self.der + other.der)
		else:
			# other is not Variable, int, or float
			raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

	def __radd__(self, other):
		"""Overload of the '+' operator (other + Variable). Calculates the value and derivative resulting
		from the addition of two variables or a variable and other object.

		Args:
			other (Variable, int, or float): item to be added to the Variable

		Returns:
			Variable: resulting Variable object
		"""
		return self.__add__(other)

	def __mul__(self, other):
		"""Overload of the '*' operator (Variable * other). Calculates the value and derivative resulting
		from the multiplication of two variables or a variable and other object.

		Args:
			other (Variable, int, or float): item to be multiplied with the Variable

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		>>> x1 = Variable(3)
		>>> x2 = Variable(4)
		>>> x3 = x1 * x2
		>>> print(x3)
		"Variable(val = 12, der = 1)"
		>>> x4 = x1 * 3
		>>> print(x4)
		"Variable(val = 9, der = 3)"
		>>> x5 = x2 * 2.0
		>>> print(x5)
		"Variable(val = 8.0, der = 2.0)"
		"""
		if isinstance(other, int) or isinstance(other, float):
				return Variable(val = self.val*other, der = self.der*other)
		elif isinstance(other, Variable):
			return Variable(val = self.val*other.val, der = self.der*other.val + self.val*other.der)
		else:
			# other is not Variable, int, or float
			raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

	def __rmul__(self, other):
		"""Overload of the '*' operator (other * Variable). Calculates the value and derivative resulting
		from the multiplication of two variables or a variable and other object.

		Args:
			other (Variable, int, or float): item to be multiplied with the Variable

		Returns:
			Variable: resulting Variable object
		"""
		return self.__mul__(other)

	def __neg__(self):
		"""Overload of the negation '-' operator. Calculates the value and derivative resulting from the
		negation operator.

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		>>> x = Variable(3)
		>>> print(-x)
		"Variable(val = -3, der = -1)"
		"""
		return -1*self

	def __sub__(self, other):
		"""Overload of the subtraction '-' operator (Variable - other). Calculates the value and derivative resulting
		from the subtraction of one Variable (or other object) from a Variable.

		Args:
			other (Variable, int, or float): item to be subtracted from the Variable

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		>>> x1 = Variable(4)
		>>> x2 = Variable(3)
		>>> x3 = x1 - x2
		>>> print(x3)
		"Variable(val = 1, der = 2)"
		>>> x4 = x1 - 3
		>>> print(x4)
		"Variable(val = 1, der = 1)
		>>> x5 = x2 - 1.0
		>>> print(x5)
		"Variable(val = 2.0, der = 1)
		"""
		if isinstance(other, int) or isinstance(other, float) or isinstance(other, Variable):
			return self + (-1)*other
		else:
			raise TypeError(f"unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

	def __rsub__(self, other):
		"""Overload of the subtraction '-' operator (other - Variable). Calculates the value and derivative resulting
		from the subtraction of one Variable (or other object) from a Variable.

		Args:
			other (Variable, int, or float): item to be subtracted from the Variable

		Returns:
			Variable: resulting Variable object
		"""
		if isinstance(other, int) or isinstance(other, float):
			return (-1)*self + other
		else:
			raise TypeError(f"unsupported operand type(s) for -: '{type(other)}' and '{type(self)}'")

	def __div__(self, other):
		"""Overload of the '/' operator (Variable / other). Calculates the value and derivative resulting
		from the division of one Variable (or other object) from a Variable.

		Args:
			other (Variable, int, or float): item the Variable is to be divided by

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		>>> x1 = Variable(8)
		>>> x2 = Variable(4)
		>>> x3 = x1 - x2
		>>> print(x3)
		"Variable(val = 1, der = 2)"
		>>> x4 = x1 - 3
		>>> print(x4)
		"Variable(val = 1, der = 1)
		>>> x5 = x2 - 1.0
		>>> print(x5)
		"Variable(val = 2.0, der = 1)
		"""
		if isinstance(other, int) or isinstance(other, float):
			if other == 0:
				raise ZeroDivisionError("division by zero")
			else:
				return self*(other**-1)
		elif isinstance(other, Variable):
			if other.val == 0:
				raise ZeroDivisionError("division by zero")
			else:
				return self*(other**-1)
		else:
			raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

	def __rdiv__(self, other):
		"""Overload of the '/' operator (other / Variable). Calculates the value and derivative resulting
		from the division of one Variable (or other object) from a Variable.

		Args:
			other (Variable, int, or float): item to be divided by the Variable

		Returns:
			Variable: resulting Variable object
		"""
		if isinstance(other, int) or isinstance(other, float):
			return other*(self**-1)
		else:
			raise TypeError(f"unsupported operand type(s) for /: '{type(other)}' and '{type(self)}'")

	def __truediv__(self, other):
		return self.__div__(other)

	def __rtruediv__(self, other):
		return self.__rdiv__(other)

	def __pow__(self, other):
		"""Overload of the '**' or 'pow()' operator (Variable**other). Calculates the value and derivative resulting
		from raising Variable to the power of other.

		Args:
			other (Variable, int, or float): item the Variable is to be raised to

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		"""
		# TODO: write examples for docstring
		if isinstance(other, int) or isinstance(other, float):
			return Variable(self.val**other, other*self.val**(other - 1)*self.der)
		elif isinstance(other, Variable):
			if self.val > 0:
				return Variable(self.val**other.val,
								self.val**other.val * (
										math.log(self.val) * other.der
										+ self.der / self.val * other.val))
			else:
				raise ValueError('math domain error: the base of exponentiation cannot be non-positive')
		else:
			raise TypeError(f"unsupported operand type(s) for ** or pow(): '{type(self)}' and '{type(other)}'")

	def __rpow__(self, other):
		"""Overload of the '**' or 'pow()' operator (other**Variable). Calculates the value and derivative resulting
		from raising other to the power of the Variable.

		Args:
			other (Variable, int, or float): item to raise to the power of the Variable

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		"""
		# TODO: write examples for docstring
		if isinstance(other, int) or isinstance(other, float):
			if other > 0:
				return Variable(other**self.val, other**self.val*math.log(other)*self.der)
			else:
				raise ValueError('math domain error: the base of exponentiation cannot be non-positive')
		else:
			raise TypeError(f"unsupported operand type(s) for ** or pow(): '{type(other)}' and '{type(self)}'")

	def __abs__(self):
		"""Overload of the 'abs()' operator. Calculates the value and derivative resulting
		from taking the absolute value of the Variable.

		Returns:
			Variable: resulting Variable object

		Examples
		--------
		"""
		# TODO: write examples for docstring
		if self.val < 0:
			val = -1*self.val
		else:
			val = self.val
		if self.der < 0:
			der = -1*self.der
		else:
			der = self.der
		return Variable(val, der)

	def __eq__(self, other):
		"""Overload of the '==' operator. Determines whether Variable is equal to
		another object.

		Args:
			other (Variable, int, or float): item to check equality with Variable

		Returns:
			tup(bool): tuple whether the Variable and other object are equal, first
					   index corresponds to the value, second to the derivative

		Examples
		--------
		"""
		# TODO: write examples for docstring
		if isinstance(other, Variable):
			return self.val == other.val and self.der == other.der
		return False

	def __str__(self) -> str:
		return f"Variable(val = {self.val}, der = {self.der})"

	def __repr__(self) -> str:
		return str(self)
