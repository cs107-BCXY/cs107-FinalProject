# Introduction
A [`python`](https://www.python.org/) [**automatic differentiation**](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) library.

# Background
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

# How to use [BCXY](https://github.com/cs107-BCXY/cs107-FinalProject)

# Software Organization
[`TravisCI`](https://travis-ci.org/) and [`Codecov`](https://about.codecov.io/) are used for automated testing and coverage report.

The package is currently under development and will be distributed to [`PyPI`](https://pypi.org/).

For packaging the software, we can look into utilizing Wheels as shown in lecture material, in which case, the installation can be done simply with 'pip'.


### Directory Structure 
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

Main source code are placed in the directory [`src`](https://github.com/cs107-BCXY/cs107-FinalProject/tree/main/src). Tests are put in the directory [`tests`](https://github.com/cs107-BCXY/cs107-FinalProject/tree/main/tests).

# Implementation

# Licensing
MIT License is chose since it puts only very limited restriction and we would like to follow the spirit of open source.

