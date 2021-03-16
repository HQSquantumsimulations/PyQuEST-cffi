# PyQuEST-cffi README

[![Documentation Status](https://readthedocs.org/projects/pyquest/badge/?version=latest)](https://pyquest.readthedocs.io/en/latest/?badge=latest)
[![GitHub Workflow Status](https://github.com/HQSquantumsimulations/PyQuEST-cffi/workflows/Python%20package/badge.svg)](https://github.com/HQSquantumsimulations/PyQuEST-cffi/actions)
[![PyPI](https://img.shields.io/pypi/v/pyquest_cffi)](https://pypi.org/project/pyquest_cffi/)
![PyPI - License](https://img.shields.io/pypi/l/pyquest_cffi)
[![PyPI - Format](https://img.shields.io/pypi/format/pyquest_cffi)](https://pypi.org/project/pyquest_cffi/)

PyQuEST-cffi is a python interface to [QuEST](https://github.com/QuEST-Kit/QuEST) based on [cffi](https://cffi.readthedocs.io/en/latest/index.html) developed by HQS Quantum Simulations. QuEST is an open source toolkit for the simulation of quantum circuits (quantum computers).

PyQuEST-cffi provides an interactive python to QuEST interface based on cffi, mapping QuEST functions to python and executing them during runtime.

For more information see the detailed code [documentation](https://pyquest.readthedocs.io/en/latest/)

## Note

Please note, PyQuEST-cffi is not an official QuEST project.

In the developing branches of QuEST the QuEST project has implemented a [ctypes](https://docs.python.org/3.6/library/ctypes.html)-based python interface [QuestPy](https://github.com/QuEST-Kit/QuEST/tree/master/utilities/QuESTPy) for unit testing.

Do not assume that any bugs occuring using PyQuEST-cffi are QuEST bugs unless the same bug occurs when compiling/using a QuEST c-program with the official release version of [QuEST](https://github.com/QuEST-Kit/QuEST).

## Installation

We do provide a PyPi source packages. The recommended way to install PyQuEST-cffi is

```shell
pip install pyquest_cffi
```

If you want to install PyQuEST-cffi in development mode we recommend

```shell
# PyQuEST-cffi add QuEST as a git submodule
git clone --recurse-submodules https://github.com/HQSquantumsimulations/pyquest_cffi.git
pip install -e pyquest_cffi/
```
