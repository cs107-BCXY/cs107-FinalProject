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

Eventually the package will be uploaded via PyPi for download. It is ready to be used by simply cloning our project git. 

#### Demo

```python
from src.variable import Variable
from src.forward import Forward
import math

x = Variable(3)

x.val
x.der

z = x ** 2

z.val # gives value
z.der # gives the derivative of Variable
f = lambda x: math.sin(x)

fmode = Forward(f, x)
fmode.calculate() #run calculate method to 

fmode.value # yields the 
fmode.derivative

```

## Software organization 

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

### Installation instructions
The package will be released on [`PyPI`](https://pypi.org/) and can be easily installed using the command
```
pip install cs107-BCXY
```
For now the package isn't on [`PyPI`](https://pypi.org/) yet, so one can download the reposity with the command
```
git clone https://github.com/cs107-BCXY/cs107-FinalProject.git
```
and install the dependencies via
```
pip install -r requirements.txt
```

## Implementation details
