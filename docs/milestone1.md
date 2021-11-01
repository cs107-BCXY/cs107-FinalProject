## Introduction

A [`python`](https://www.python.org/) [**automatic differentiation**](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) library.

Many methods in science for example machine learning require the evaluation of derivatives and most of the traditional learning algorithms have relied on the computation of gradients of an objective function. Manual differentiation is cumbersome and impractical in computational models. Automatic differentiation addresses the need for a better way than manual derivations and doing so in a more accurate manner than numerical differentiation. Hence we are hoping to create a useful and easy to implement AD package.


## Background
Derivatives are required at the core of many numerical algorithms.  

However, they are usually computed inefficiently and approximately by some variant of the finite difference approach
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?f'(x)\approx\frac{f(x+h)-f(x)}{h},"> 
</p>
<p>for <img src="https://latex.codecogs.com/svg.latex?h"> small.</p>
The method is ineffecient, since it requires 
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\Omega(n)"> 
</p>  
evaluations of 
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?f:\mathbb{R}^n\to\mathbb{R}">
</p>
to compute the graident
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla%20f(x)=\left(\frac{\partial%20f}{\partial%20x_1}(x),\cdots,\frac{\partial%20f}{\partial%20x_n}(x)\right),">
</p>
for example.

### What can we do instead?
One option is to explicitly write down a function which computes the exact derivatives by using the rules that we know from calculus. However, this quickly becomes an error-prone and tedious exercise. **There is another way!** The field of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) provides methods for automatically computing *exact* derivatives (up to floating-point error) given only the function <img src="https://latex.codecogs.com/svg.latex?f"> itself. Some methods use many fewer evaluations of <img src="https://latex.codecogs.com/svg.latex?f"> than would be required when using finite differences. In the best case, the exact gradient of <img src="https://latex.codecogs.com/svg.latex?f"> can be evaluated for the cost of <img src="https://latex.codecogs.com/svg.latex?\mathcal{O}(1)"> evaluations of <img src="https://latex.codecogs.com/svg.latex?f"> itself. The caveat is that <img src="https://latex.codecogs.com/svg.latex?f"> cannot be considered a black box; instead, we require either access to the source code of <img src="https://latex.codecogs.com/svg.latex?f"> or a way to plug in a special type of number using operator overloading.

## How to use the package [BCXY](https://github.com/cs107-BCXY/cs107-FinalProject)

For this Python package we will aim to be easily implementable where calling for example `python -m pip install cs107-BCXY` will download the package and address dependencies such as [`NumPy`](https://numpy.org/).  

To use the package, users first need to instantiate variables, whether real or dual. This generates the specified number of variables with their symbolic representations, their current values, and derivatives.
```{python}
import cs107-BCXY as ad

vars = ad.variables(n = 1, vals = (1))
x = vars[0]
```
Once they have done this, they can define their function of interest. Having completed this, they can then execute automatic differentiation with their chosen method. The user can then look at the value and derivative of the function using the instance attributes.
```{python}
func = ad.exp(x)
fmode = ad.forward(func)
rmode = ad.reverse(func)
print(fmode.get_value())
print(fmode.get_derivative())
```

## Software Organization

### Directory Structure 
Our directory structure will be the following.
```
├── docs
│   └── milestone1
├── src
│   ├── __init__.py
│   ├── __main__.py
│   ├── autodiff.py
│   ├── elementary_functions.py
│   ├── variables.py
│   ├── dual_variables.py
│   └── ...
├── tests
├── .gitignore
├── .travis.yml
├── LICENSE
├── README.md
├── requirements.txt
└── ...
```
Our main source code are placed in the directory [`src`](/src) and our tests are put in the directory [`tests`](/tests). Our package documentation will located in the [`docs`](/docs) directory. Top-level package information and documents (e.g. our licensing) will be in the root directory.

### Modules
We will have three modules within our package:  
1. `autodiff.py` - this module contains our implementation of automatic differntiation, including both forward and reverse modes.  
2. `elementary_functions.py` - this module contains our definitions of all elementary functions.  
3. `variables.py` - this module handles our implementation of real variables within automatic differentiation.  
4. `dual_variables.py` - this module handles our implementation of dual variables within automatic differentiation.

### Testing
Our testing suite is located within our [`tests`](/tests) directory. We will use both [`TravisCI`](https://travis-ci.org/) and [`Codecov`](https://about.codecov.io/) for automated testing and coverage reporting respectively.

### Distribution

The package is currently under development and will be distributed to [`PyPI`](https://pypi.org/).

### Packaging

For packaging the software, we can look into utilizing Wheels as shown in lecture material since we do not expect our package will not be extremely complex and will not need many dependencies. In this case, the installation can be done simply with `pip`. Alternatively, we may also be able to use Conda-Forge as the [conda package system](https://docs.conda.io/en/latest/) is known to be quite good at supporting multiple applications with different dependencies.

## Implementation

### Data Structures

Our primary data structure for automatic differentiation will be the standard Python dictionary. This will allow us to easily track and access each element within the trace. A single node in the trace will have the following format (items in brackets represent placeholders):
```{python}
{[trace]: "elem_op":[elementary operation], "value":[value], "elem_der":[elementary derivative], "dd_x1":[directional derivative]}
```
The entire trace will be a dictionary of dictionaries - one for each node in the trace. Of note, the size of the node dictionary shown above can alter depending on the number of variables.

### Classes

- Variables - this class will instantiate the variales for a given function, if said variables are real-valued.  
- DualVariables - this class will instantiate the variables for a given function, if said variables are dual.  
- AutoDiff - this class will be the interface through which the user executes automatic differentiation. They will specify whether to use forward or reverse mode.  

We will also create classes for each elementary function (e.g. +, sin, etc.) to for use in function definition. These classes will contain both the function evaluation and its derivative evaluation.

### Method and Attributes

Every class will contain ``__init__``, ``__repr__``, and ``__str__`` methods. Our ``DualVariables`` class will also contain the classes ``__add__``, ``__radd__``, ``__mul__``, ``__rmul__``, ``__truediv__``, and ``__pow__`` to handle all of these operations for dual numbers. Our ``Variables`` and ``DualVariables`` classes will have ``value`` and ``derivate`` attributes. Finally, our ``AutoDiff`` class will have a ``forward`` method for forward automatic differentiation, and ``reverse`` for reverse mode.

### External Dependencies

At this point, we have identified [`NumPy`](https://numpy.org/) and [`SymPy`](https://www.sympy.org/en/index.html) as two dependencies that our package will have. It is possible that more dependencies will occur as we develop, but we expect the total amount to be relatively small.

## Licensing
[MIT License](/LICENSE) is chosen since it puts only very limited restriction and we would like to follow the spirit of open source.

## Feedback

### Milestone 1

The only feedback that we recieved on Milestone 1 was "Why depend on [`SymPy`](https://www.sympy.org/en/index.html)?", though no points were deducted. We intend to use [`SymPy`](https://www.sympy.org/en/index.html) primarily for its function printing - it displays mathematical equations in a symbolically aesthetic manner. We also will potentially use [`SymPy`](https://www.sympy.org/en/index.html) in our testing, utilizing its [calculus](https://docs.sympy.org/latest/modules/calculus/index.html) module to confirm that we computing the correct derivatives.