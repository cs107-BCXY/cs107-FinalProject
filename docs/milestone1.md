
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

For this Python package we will aim to be easily implementable where calling for example `python -m pip install [package-name]` will download the package and address dependencies such as [`NumPy`](https://numpy.org/).

To instantiate the AD objects, simply running autodiff class and inputting parameters options `method`, `func`, and `values` will run.
D
-----
     import autodiff as ad

     ad1 = ad(values = [], func = f_examples, method = "reverse")
     ad1.calculate()
-----

## Software Organization
[`TravisCI`](https://travis-ci.org/) and [`Codecov`](https://about.codecov.io/) are used for automated testing and coverage report.

The package is currently under development and will be distributed to [`PyPI`](https://pypi.org/).

For packaging the software, we can look into utilizing Wheels as shown in lecture material since we do not expect our package will not be extremely complex and will not need many dependencies. In this case, the installation can be done simply with `pip`. Alternatively, we may also be able to use Conda-Forge as the [conda package system](https://docs.conda.io/en/latest/) is known to be quite good at supporting multiple applications with different dependencies.


## Directory Structure 
<div class="highlight"><pre><span></span><code>cs107-FinalProject/
├── docs
│   └── milestone1
├── src
├── tests
├── .gitignore
├── .travis.yml
├── LICENSE
├── README.md
├── requirements.txt
└── ...
</code></pre></div>

Main source code are placed in the directory [`src`](/src). Tests are put in the directory [`tests`](/tests).

## Implementation
We will be able to take in lists and/or tuples of inputs in our implementation.

If f is a composite function of the following forms:

* f = g + h
* f = g - h
* f = g * h
* f = g / h
* f = g(h)
where g and f are also functions, then these will be addressed in our package.

For elementary functions `sin`, `sqrt`, `log`, `exp`, etc.. mentioned in the prompt, the package can rely on the `numpy` and  [`SymPy`](https://www.sympy.org/en/index.html) module as it holds a well curated list of basic functions for differentiation.

### Licensing
[MIT License](/LICENSE) is chosen since it puts only very limited restriction and we would like to follow the spirit of open source.

