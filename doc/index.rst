.. PyQuEST-cffi documentation master file

Welcome to PyQuEST-cffi's documentation!
======================================

PyQuEST-cffi is a python interface to QuEST(https://github.com/QuEST-Kit/QuEST) based on cffi(https://cffi.readthedocs.io/en/latest/index.html) developed by HQS Quantum Simulations. QuEST is an open source toolkit for the simulation of quantum circuits (quantum computers). 

PyQuEST-cffi provides two main functionalities:

1. An interactive python to QuEST interface based on cffi, mapping QuEST functions to python and executing them during runtime.

2. A compile function generating a complete QuEST c-source-file from python calls, building it and importing it into python via cffi ).

For more information see the detailed code documentation below

Note
=====

Please note, PyQuEST-cffi is currently in the alpha stage and not an official QuEST project.

PyQuEST-cffi currently depends on a forked version of the development branch of QuEST. We plan to move dependency to the official QuEST master and bring PyQuEST-cffi to beta stage after the next official QuEST release.

In the developing branches of QuEST the QuEST project has implemented a ctypes-based python interface QuestPy(https://github.com/QuEST-Kit/QuEST/tree/PythonTesting/tests/QuESTPy) for unit testing.

Do not assume that any bugs occuring using PyQuEST-cffi are QuEST bugs unless the same bug occurs when compiling/using a QuEST c-programm with the official release version of QuEST(https://github.com/QuEST-Kit/QuEST).

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules
   README.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
