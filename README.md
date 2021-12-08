# Welcome to PyADBCXY  

[![build](https://github.com/cs107-BCXY/cs107-FinalProject/actions/workflows/workflow.yml/badge.svg?branch=main)](https://github.com/cs107-BCXY/cs107-FinalProject/actions/workflows/workflow.yml)
[![codecov](https://codecov.io/gh/cs107-BCXY/cs107-FinalProject/branch/main/graph/badge.svg?token=LJX9AH62PE)](https://codecov.io/gh/cs107-BCXY/cs107-FinalProject)  

PyADBCXY is an automatic differentiation software package that can calculate the values and derivatives of complex functions while maintaining high levels of accuracy.

## Contributors  

Group Name: cs107-BCXY  
Group Number: 12  
Group Members:  
* Charlie Harrington, Harvard University, <charlesharrington@g.harvard.edu>  
* Bowen Zhu, Harvard University, <bszhu@fas.hardvard.edu>  
* Yaxin Lei, Harvard University, <yaxin_lei@g.harvard.edu>  
* Xiang Bai, Harvard University, <xbai@hsph.harvard.edu>

## Installation

Install the package with pip:

    pip install pyadbcxy

Then you can run a simple example such as

```python
import pyadbcxy as ad
x = ad.Variable(3)            # instantiate x variable
y = ad.Variable(4., 5.)       # instantiate y variable
f = lambda x, y: x + y              # define function of interest
fmode = ad.Forward(f, (x, y)) # instantiate forward mode
fmode.calculate()                   # evaluate function at x and y
print(fmode.value)                  # print value of f at x and y
print(fmode.derivative)             # print derivative of f and x and y
```

For further details on package usage as well as the math behind it, please see the [documentation](/docs/documentation.md).

## For developers

To install the package for further development, clone the repository to your machine:

    git clone https://github.com/cs107-BCXY/cs107-FinalProject.git

Then, move into the repository and install the package dependencies

    cd cs107-FinalProject
    pip install -r requirements.txt

#### Testing

To run the tests, you can execute

    python -m unittest discover -s tests -p 'test_*.py'

Alternatively, you can run the test driver script. The default testing framework is [`unittest`](https://docs.python.org/3/library/unittest.html) and can be run with:

    bash tests/run_tests.sh

If you'd like to execute the tests using [`pytest`](https://docs.pytest.org/en/6.2.x/), simply specify in the command:

    bash tests/run_tests.sh pytest

Finally, if you'd like to see the code coverage report, you can do so with the [`coverage`](https://coverage.readthedocs.io/en/6.2/) keyword:

    bash tests/run_tests.sh coverage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) for more details.