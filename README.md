# pyQuEST README

PyQuEST-cffi is a python interface to [QuEST](https://github.com/QuEST-Kit/QuEST) based on [cffi](https://cffi.readthedocs.io/en/latest/index.html) developed by HQS Quantum Simulations. QuEST is an open source toolkit for the simulation of quantum circuits (quantum computers). 

PyQuEST-cffi provides two main functionalities:

1. An interactive python to QuEST interface based on cffi, mapping QuEST functions to python and executing them during runtime.

2. A compile function generating a complete QuEST c-source-file from python calls, building it and importing it into python via cffi ).

For more information see the detailed code [documentation](https://pyquest_cffi.readthedocs.io/en/latest/)

## Note

Please note, pyQuEST is currently in the alpha stage and not an official QuEST project. 

PyQuEST-cffi currently depends on a forked version of the development branch of QuEST. We plan to move dependency to the official QuEST master and bring pyQuEST to beta stage after the next official QuEST release.

In the developing branches of QuEST the QuEST project has implemented a [ctypes](https://docs.python.org/3.6/library/ctypes.html)-based python interface [QuestPy](https://github.com/QuEST-Kit/QuEST/tree/PythonTesting/tests/QuESTPy) for unit testing.

Do not assume that any bugs occuring using pyQuEST are QuEST bugs unless the same bug occurs when compiling/using a QuEST c-programm with the official release version of [QuEST](https://github.com/QuEST-Kit/QuEST).

## Installation

At the moment we do not provide PyPi packages. The recommended way to install pyQuEST is
```shell
# pyQuEST add QuEST as a git submodule
git clone --recurse-submodules https://github.com/HQSquantumsimulations/pyquest_cffi.git
pip install -e pyquest_cffi/
```
