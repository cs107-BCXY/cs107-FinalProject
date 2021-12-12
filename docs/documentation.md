# PyADBCXY

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

## How to Use the Package [PyADBCXY](https://github.com/cs107-BCXY/cs107-FinalProject)

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
│   ├── documentation.md
│   ├── milestone1.md
│   └── milestone2.md
├── src
│   └── pyadbcxy
│       ├── __init__.py
│       ├── elementary_functions.py
│       ├── forward.py
│       ├── reverse.py
│       └── variable.py
├── tests
│   ├── __init__.py
│   ├── run_tests.sh
│   ├── test_elementary_functions.py
│   ├── test_forward.py
│   ├── test_reverse.py
│   └── test_variable.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.cfg
</code></pre></div>

Our main source code are placed in the directory [`src`](/src) and our tests are put in the directory [`tests`](/tests). Our package documentation will located in the [`docs`](/docs) directory. Top-level package information and documents (e.g. our licensing) will be in the root directory.

### Modules
- `elementary_functions.py` - this module contains our definitions of all elementary functions. 
- `forward.py` - this module facilitates the forward mode of automatic differentiation. 
- `reverse.py` - extension of the project, the reverse mode.
- `variables.py` - this module handles our implementation of real variables within automatic differentiation.  

### Installation Instructions
The package is now released on [`PyPI`](https://pypi.org/) and can be easily installed using the command
```
pip install pyadbcxy
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

Our package lists `numpy` as our only external dependency.

### Elementary Functions

All of our elementary functions that are not dunder methods are contained within our [`elementary_functions`](/src/elementary_functions.py) module. Each defined function can take a Variable, floating point number, or integer as an input. If the input is a Variable, a new Variable object will be returned with the updated value and derivative.

## Extension - `Reverse Mode` 

Finally, we have created the extension feature to handle automatic differentiation using reverse mode. The reverse mode module works by instantiating a `Reverse` object for each variable with some assigned value. It is capable of supporting reverse mode autodifferentiation for all basic arithmetic operations, trigonometric-- including hyperbolic and inverse trig functions, exponential, and comparison methods `__ne__` and `__eq__`, and logarithmic operations. As needed, the reverse mode feature appends the values of adjoints and child within each relative variable for computing the derivative. Additionally, the derivatives are saved in original variables by calling the `grad` decorator. 

## Broader Impact and Inclusivity Statement

We strove to create a convenient way to automatically differentiate smoothly and accurately. The automatic differentiation package is able to efficiently compute the derivatives of functions of any numerical inputs granted they are mathematically valid in the constraints of functions, including integers, floats, single and multiple variables. Traditionally in finite differentiation, users need to select an epsilon value for the algorithm that calculates the difference of slope. The choice of epsilon will impact the accuracy of the derivative especially since computationally the rounding error may be a specific problem. Our package eliminates this process for users by adopting autodifferentation method.

While automatic differentiation is proven to be powerful in calculating accurate derivatives, such function does not prevail in common machine learning packages. In neural networks and regression based models, gradient descent is widely used to find the optimal parameters. Automatic differentiation assists this process so that any differentiation, even when the algebraic form is hard to compute, can be done easily. This broadens the range of models one can choose from without concerning the complexity of their derivatives. If used responsibly, the benefit of a wider range of models and increasing accuracy can be broadcast to many fields including public health and medicine, where models are rather complicated.

We strongly believe in the importance of inclusivity of our package. We worked to ensure that our package is accessible to all and is licensed as such. We ensured proper documentation and simple and straightforward usage instruction that is easy to follow. The creation of this package was conducted through teamwork where every member was respected and represented, and contributed to the outcome. The coding process was discussed among members of the team as well as researched online through open source.

Our purpose in this development was to at once enrichment or own understanding of the mathematical and programming grounds of a commonly used and powerful tool as well as provide a basis for others to refer to and build upon. We discourage any illegal and unethical use of our package in projects that harm a particular group based on attributes including (but not limited to) age, culture, ethnicity, gender identity or expression, national origin, physical or mental difference, politics, race, religion, sex, sexual orientation, socio-economic status, and subculture.


## Future Features

One area of development is the higher order derivatives in which we can compute the Hessian matrix, or an arbitrary order of derivatives. Additionally, batch differentiation may be a further area of implementation so that instead of differentiating hte function at one set of variable values [x,y...] at a time, we can provide the option to differentiate a matrix of values where each row is one set of variable values. At this point, we are not tracking the computational graph of the automatic differentiation process. That is, we are need keeping a log of the sub-values and sub-derivatives for every sub-function within the function of interest. Instead, the variable's value and derivative are merely updated, as explained in [Data Structures](#data-structures). This is a potential area of future development, and either the standard Python list or dictionary could be used for this. Instead of simply updating the variable's value and derivative, a running record could be appended/added to. Additionally, as a further case of reverse mode, back propagation may be a consideration while leveraging and/or adjusting our current design for its application.
