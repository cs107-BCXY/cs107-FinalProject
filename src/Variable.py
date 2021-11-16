class Variable:

	def __init__(self, val, der=1):
		self.val = val
		self.der = der

	def __add__(self, other):
		try:
			return Variable(value=self.val+other.val, der=self.der+other.der)

		except AttributeError:
			# it is not an autodiff object
			if isinstance(other, int) or isinstance(other, float):
				return Variable(val=self.val+other, der=self.der)
			raise Exception("Input invalid: The input is not one of the below: {AutoDiffToy object, int, float.")

	def __radd__(self, other):
		return self.__add__(other)


	def __mul__(self, other):
		try:
			return Variable(val=self.val*other.val, der=self.der*other.val + self.val*other.der)
		except AttributeError:
			# it is not an autodiff object
			if isinstance(other, int) or isinstance(other, float):
				return Variable(val=self.val*other, der=self.der*other)
			raise Exception("Input invalid: The input is not one of the below: {AutoDiffToy object, int, float.")


	def __rmul__(self, other):
		return self.__mul__(other)

	def __neg__(self):
		return -1*self

	def __sub__(self, other):
		return self + (-1)*other

	def __rsub__(self, other):
		return (-1)*self + other

	def __div__(self, other):
		return self*(other**-1)

	def __rdiv__(self, other):
		return other*(self**-1)

	def __truediv__(self, other):
		return self.__div__(other)

	def __rtruediv__(self, other):
		return self.__rdiv__(other)

	def __pow__(self, other):
		if isinstance(other, ad):
			return ad(self.val**other.val, 
				other.val*self.val**(other.val - 1)*self.der + \
				self.val**other.val*log(self.val)*other.der)
		else:
			return ad(self.val**other, other*self.val**(other - 1)*self.der)

	def __rpow__(self, other):
		return ad(other**self.val, other**self.val*log(other)*self.der)

	def __abs__(self):
		if self==0:
			return ad(0*self.val, 0*self.der)
		else:
			return (self**2)**0.5
    
	def __eq__(self, other):
		if isinstance(other, ad):
			return self.nom==other.nom
		else:
			return self.nom==other

	def d(self, n):
		"""
		Get the nth derivative

		Example
		-------
		A 3rd-order differentiable object at 1.5::

		>>> x = adn(1.5, 3)
		>>> y = x**2
		>>> y.d(1)
		3.0
		>>> y.d(2)
		2.0
		>>> y.d(3)
		0.0

		"""
		assert n>=0, 'Derivative order must not be negative.'
		if n==0:
			return self.nom
		else:
			derivs = taylorderivatives(self)
			assert len(derivs)>=n, \
				"You didn't track derivatives of order = {}".format(n)
			return derivs[n - 1]



if __name__ == '__main__':
	v1 = Variable(30, 3)
	v2 = Variable(20, 1)
	print("Multiply ", (v1*v2).val)
	