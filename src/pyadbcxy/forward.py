"""
This file contains the Forward module for the cs107-BCXY package. It includes the Forward class,
which implements the forward mode of automatic differentiation.
"""


__all__ = ["Forward"]


class Forward(object):
    """
    This class implements the forward mode of automatic differentiation. The user inputs the
    function and the variables for that function and then can access the value and derivative
    of the function.

    Examples
    --------

    # Compute the value and derivative of a function consisting of basic operations
    >>> x = Variable(3)
    >>> f = lambda t: t**2
    >>> fmode = Forward(f, x)
    >>> fmode.calculate()
    >>> fmode.value
    9
    >>> fmode.derivative
    6
    # A rather complicated function with a single variable
    >>> fmode = Forward(lambda x: (exp(cos(x)))/(sin(x)**2), Variable(3))
    >>> fmode.calculate()
    >>> fmode.value
    18.658405892057388
    >>> fmode.derivative
    259.153784690042
    """

    def __init__(self, func, vars):
        """Constructor for the Forward class.

        Args:
            func (function): function of interest
            vars (Variable, list, or tuple): Variable object or list/tuple of Variables
                                             to evaluate the function
        """
        self._func = func
        if not isinstance(vars, list) and not isinstance(vars, tuple):
            self._vars = [vars]
        else:
            self._vars = vars
        self._res = None

    @property
    def func(self):
        """Get the function of the Forward instance."""
        return self._func

    @property
    def vars(self):
        """Get the Variable(s) of the Forward instance."""
        return self._vars

    @func.setter
    def func(self, func):
        """Set the function for the forward mode. Used to quickly make changes to
        function if Variables are the same.

        Args:
            func (function): new function to implement forward mode on

        Example
        -------
        >>> x = Variable(3)
        >>> f = lambda t: t**2
        >>> fmode = Forward(f, x)
        >>> g = lambda t: exp(-t ** 2)
        >>> fmode.func = g
        """
        self._func = func

    @vars.setter
    def vars(self, vars):
        """Set the Variables for the forward mode. Used to quickly make changes to
        Variables if function is the same.

        Args:
            vars (Variable or tuple/list of Variables): new Variable(s) to implement forward mode on

        Example
        -------
        >>> x = Variable(3)
        >>> f = lambda t: t**2
        >>> fmode = Forward(f, x)
        >>> y = Variable(4, 5)
        >>> fmode.vars = y
        >>> z = [Variable(6), Variable(7, 8)]
        >>> fmode.vars = z
        """
        if not isinstance(vars, list) and not isinstance(vars, tuple):
            self._vars = [vars]
        else:
            self._vars = vars

    def calculate(self):
        """Evaluate the given function with the Variables"""
        self._res = self._func(*self._vars) # will be a Variable object


    @property
    def value(self):
        """Get the value of the function evaluated at the Variables.

        Raises:
            ValueError: if 'calculate' method has not been called

        Returns:
            float or int: value of the function evaluated at the Variable
        """
        if not self._res:
            raise AttributeError("value and derivative have not been calculated yet, call 'calculate' method")
        return self._res.val

    @property
    def derivative(self, non_differential=False):
        """Get the derivative of the function evaluated at the Variables.

        Raises:
            ValueError: if 'calculate' method has not been called

        Returns:
            float or int: derivative of the function evaluated at the Variable
        """
        if not self._res:
            raise AttributeError("value and derivative have not been calculated yet, call 'calculate' method")
        if non_differential:
            return self._res.der
        if isinstance(self.vars, list) or isinstance(self.vars, tuple):
            var_count = len(self.vars)
            der_vector = []

            # goes into a loop where only the ith partial derivatives is treated as a variable
            # all other variables are treated as only numerical values
            for i in range(var_count):
                input_arr = []
                for j in range(var_count):
                    if j != i:
                        input_arr.append(self.vars[j].val)
                    else:
                        input_arr.append(self.vars[j])
                result = self._func(*input_arr)
                der_vector.append(result.der)
            if len(der_vector) == 1:
                return der_vector[0]
            else:
                return der_vector
        else:
            return self._res.der
