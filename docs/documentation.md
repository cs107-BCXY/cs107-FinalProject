# PyAD-BCXY

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
to compute the gradient
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla%20f(x)=\left(\frac{\partial%20f}{\partial%20x_1}(x),\cdots,\frac{\partial%20f}{\partial%20x_n}(x)\right),">
</p>
for example.

### What can we do instead?
One option is to explicitly write down a function which computes the exact derivatives by using the rules that we know from calculus. However, this quickly becomes an error-prone and tedious exercise. **There is another way!** The field of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) provides methods for automatically computing *exact* derivatives (up to floating-point error) given only the function <img src="https://latex.codecogs.com/svg.latex?f"> itself. Some methods use many fewer evaluations of <img src="https://latex.codecogs.com/svg.latex?f"> than would be required when using finite differences. In the best case, the exact gradient of <img src="https://latex.codecogs.com/svg.latex?f"> can be evaluated for the cost of <img src="https://latex.codecogs.com/svg.latex?\mathcal{O}(1)"> evaluations of <img src="https://latex.codecogs.com/svg.latex?f"> itself. The caveat is that <img src="https://latex.codecogs.com/svg.latex?f"> cannot be considered a black box; instead, we require either access to the source code of <img src="https://latex.codecogs.com/svg.latex?f"> or a way to plug in a special type of number using operator overloading.

## How to Use the Package [BCXY](https://github.com/cs107-BCXY/cs107-FinalProject)

Eventually the package will be uploaded via PyPi for download. It is ready to be used by simply cloning our project git __(see Installation Instructions below for details)__. 

### Demo

```python
from src.variable import Variable
from src.elementary_functions import *
from src.forward import Forward

x = Variable(3)       # instantiate Variable object
x.val                 # gives the value of x
x.der                 # gives the derivative of x

z = x ** 2            # create new Variable object equal to x^2
z.val                 # gives value of z
z.der                 # gives the derivative of x

f = lambda x: sin(x)  # define a function
fmode = Forward(f, x) # instantiate Forward object with f and x
fmode.calculate()     # evaluate f at x
fmode.value           # gives the value of f at x 
fmode.derivative      # gives the derivative of f at x

```

## Software Organization 

### Directory Structure 
Our directory structure will be the following.
<div class="highlight"><pre><span></span><code>cs107-FinalProject/
├── .github
│   └── workflows
│       └── workflow.yml
├── docs
│   └── milestone1.md
│   └── documentation.md
├── src
│   └── __init__.py
│   └── elementary_functions.py
│   └── forward.py
│   └── variable.py
├── tests
│   └── __init__.py
│   └── test_elementary_functions.py 
│   └── test_forward.py
│   └── test_variable.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
</code></pre></div>

Our main source code are placed in the directory [`src`](/src) and our tests are put in the directory [`tests`](/tests). Our package documentation will located in the [`docs`](/docs) directory. Top-level package information and documents (e.g. our licensing) will be in the root directory.

### Modules
- `elementary_functions.py` - this module contains our definitions of all elementary functions. 
- `forward.py` - this module facilitates the forward mode of automatic differentiation. 
- `variables.py` - this module handles our implementation of real variables within automatic differentiation.  

### Installation Instructions
The package will be released on [`PyPI`](https://pypi.org/) and can be easily installed using the command
```
pip install cs107-BCXY
```
For now the package isn't on [`PyPI`](https://pypi.org/) yet, so one can download the repository with the command
```
git clone https://github.com/cs107-BCXY/cs107-FinalProject.git
```
and install the dependencies via
```
pip install -r requirements.txt
```

## Implementation Details

### Data Structures

Our primary data structore for our implementation of automatic differentiation is our ``Variable`` class. With our current implementation, every sub-function simply updates the variable's value and derivative. Other than this, the only other data structures used are the standard Python list and tuple. These are used to pass in multiple Variables to the ``Forward`` class and are then unpacked and evaluated in the given function when the ``calculate()`` method is called.

### Classes

There are two classes within our package: ``Variable`` and ``Forward``. ``Variable`` is used to define a variable for input into a function. It is initialized with a specified value and derivative, or a derivative of one if none is given. The ``Forward`` class is used as an interface for the user to execute forward mode to compute the function's value and derivative when evaluated with the Variable.

### Important Attributes

Both the ``Variable`` and ``Forward`` classes have getter and setter methods that allow the user to easily access and edit attributes of these classes if necessary. For example, if a user creates an instance of the ``Forward`` class, but then realizes that they have a typo in their function definition, they can easily set the instance's function to the correct one without having to create a new instance of the object.  

``Variable`` has a number of important dunder methods that overload standard operations such as "+", "-", "*", "/", "**", and "==". Each of these methods calculates the new value and derivative of the variable, using appropriate derivative rules (e.g. the product rule) whenever applicable.  

Aside from its getter and setter methods, the ``Forward`` class has only one other method: ``calculate()``. The reason why the function is not evaluated at the variable immediately when the ``Forward`` instance is created is to allow the user to make quick changes/edits if they need to, as previously described. The ``calculate()`` method evaluates the function, storing the resulting variable and derivative in the object.

### External Dependencies

At this point in development, there are no external dependencies to our package. We are relying heavily on the built-in [`math`](https://docs.python.org/3/library/math.html) library.

### Elementary Functions

All of our elementary functions that are not dunder methods are contained within our [`elementary_functions`](/src/elementary_functions.py) module. Each defined function can take a Variable, floating point number, or integer as an input. If the input is a Variable, a new Variable object will be returned with the updated value and derivative.

## Future Features

At this point, we are not tracking the computational graph of the automatic differentiation process. That is, we are need keeping a log of the sub-values and sub-derivatives for every sub-function within the function of interest. Instead, the variable's value and derivative are merely updated, as explained in [Data Structures](#data-structures). This is a potential area of future development, and either the standard Python list or dictionary could be used for this. Instead of simply updating the variable's value and derivative, a running record could be appended/added to.  

As [External Dependencies](#external-dependencies) states, we are currently not relying on any external libraries and instead are using the built-in [`math`](https://docs.python.org/3/library/math.html) library. In order to extend our package to handle multi-function and multi-variable inputs, we recognize that we will need to replace this with [`NumPy`](https://numpy.org/doc/stable/index.html).  

Finally, we plan on adding another module to handle automatic differentiation using reverse mode. The reverse mode module will work by instantiating a `RevMod` object with some value. It will eventually be able to support running the reverse mode autodifferentiation for all arithmetic operations, trigonometric functions, exponential, and comparison methods, for which we will look into all `__ne__`, `__eq__`, `__lt__` etc.. as was suggested for the forward mode in the final deliverables. We may be also considering briefly reorganizing our test and src directories to separate the forward and reverse mode subsections



