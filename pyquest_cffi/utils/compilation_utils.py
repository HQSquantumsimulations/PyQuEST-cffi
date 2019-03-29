# Copyright 2019 HQS Quantum Simulations GmbH
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from pyquest_cffi.questlib import _PYQUEST
from typing import Union, Optional, List, Tuple
import ctypes
from cffi import FFI
import os
import importlib
import numpy as np
import shutil
import glob
import sys


class defineVariable(_PYQUEST):
    """
    Define variables for a compiled QuEST programm

    Args:
        vartype: the type of variable that is defined
        length: optional length of the variable
        name: the name of the variable that is defined
        start_value: optional start value
        local: is the variable only declared locally or not
    """

    def call_interactive(self) -> None:
        """
        Not implemented: No need to define variables in interactive mode
        """
        raise NotImplementedError

    def call_static(self, vartype: str, name: str,
                    length: Optional[int] = None,
                    start_value: Optional[Union[float, str]] = None,
                    local: bool = True) -> List[str]:
        """
        Defined variables can be used in the code after this has been called
        """
        if local:
            call = "{vartype}".format(vartype=vartype)
            call += " {name}".format(name=name)
            if length is not None:
                call += "[{}]".format(length)
            if start_value is None:
                call += ";"
            else:
                call += " = {};".format(start_value)
        else:
            if (start_value is not None) or (length is None):
                raise RuntimeError(
                    "for non local variables length must be set and no start_value is allowed")
            call = "{vartype}".format(vartype=vartype)
            call += " *{name}".format(name=name)
            call += "=malloc(sizeof({vartype})*{length});".format(vartype=vartype, length=length)
        return [call]


class createProgrammPreamble(_PYQUEST):
    """
    Create preamble (include statements, function definition and input arguments) for a compiled QuEST programm

    Args:
        return_type: the return type of the function
        function_name: the name of the function
        arguments: A list of tuples definting the arguments of the function, 
                    with the type as the first element in each tuple
                     and the name as the second element (assumes pointers)
    """

    def call_interactive(self) -> None:
        """
        Not implemented: No need to create preamble in interactive mode
        """
        raise NotImplementedError

    def call_static(self,
                    return_type: str = 'Complex',
                    function_name: str = 'tmp_QuEST_function',
                    arguments: Optional[List[Tuple[str, str]]] = None) -> List[str]:
        call_list = list()
        call_list.append('#include <stdlib.h>')
        call_list.append("#include <stdio.h>")
        call_list.append('#include "QuEST.h"')
        call = "{return_type} * ".format(return_type=return_type)
        call += " {}(".format(function_name)
        if arguments is not None:
            call += "{} * {}".format(arguments[0][0], arguments[0][1])
            for vartype, name in arguments[1:]:
                call += ",{} * {}".format(vartype, name)
        call += ")"
        call_list.append(call)
        call_list.append("{")
        return call_list



def write_code_to_disk(lines=List[str], file_name: str = 'tmp_QuEST_code.c'):
    """
    Write the QuEST c-programm to file

    Args:
        lines: Lines of code in the .c file
        file_name: Name of .c file
    """
    with open(file_name, "w") as f:
        for line in lines:
            f.write(line+'\n')


class createProgrammEnd(_PYQUEST):
    """
    Create return statement at end of QuEST Programm

    Args:
        return_name: the name of the Complex return register
    """

    def call_interactive(self) -> None:
        """
        Not implemented: No need to create return statement in interactive mode
        """
        raise NotImplementedError

    def call_static(self,
                    return_name: str = 'readout',
                    ) -> List[str]:
        call_list = list()
        call_list.append("return {return_name};".format(return_name=return_name))
        call_list.append("}")
        return call_list


class QuESTCompiler():
    """
    Class that compiles a QuEST programm with cffi

    Args:
        file_name: file_name of the c-file with the QuEST programm
        destination_directory: where python Quest programm should be compiled
        code_lines: the lines of c-code as a list of strings,
                    optional for skipping writing the code to file
                    if this is set, file_name is ignored
        return_type: the return type of the function
        function_name: the name of the function
        arguments: A list of tuples definting the arguments of the function, 
                    with the type as the first element in each tuple
                     and the name as the second element (assumes pointers)
    """

    def __init__(self,
                 file_name: str = 'tmp_QuEST_code.c',
                 destination_directory: Optional[str] = None,
                 code_lines: Optional[List[str]] = None,
                 return_type: str = 'Complex',
                 function_name: str = 'tmp_QuEST_function',
                 arguments: Optional[List[Tuple[str, str]]] = None,
                 ) -> None:

        self._destination_directory = destination_directory
        self.lib_path = os.path.dirname(os.path.realpath(__file__))
        quest_path = os.path.join(self.lib_path, "../../QuEST/QuEST")
        self.questlib_path = os.path.join(self.lib_path, "../questlib/")
        questlib = os.path.join(self.questlib_path, "libQuEST.so")
        self.questlib = questlib
        self.include = [os.path.join(quest_path, "include")]

        _questlib = ctypes.CDLL(questlib)
        QuESTPrecFunc = _questlib['getQuEST_PREC']
        QuESTPrecFunc.restype = ctypes.c_int
        QuESTPrec = QuESTPrecFunc()

        if QuESTPrec == 1:
            qreal = "float"
        elif QuESTPrec == 2:
            qreal = "double"
        elif QuESTPrec == 4:
            qreal = "longdouble"
        else:
            raise TypeError('Unable to determine precision of qreal')
        del(QuESTPrec)
        del(QuESTPrecFunc)

        with open(os.path.join(self.include[0], "QuEST.h"), "r") as f:
            def_lines = [line for line in f]
        def_lines += [
            "void statevec_setAmps(Qureg qureg, long long int startInd, qreal* reals, qreal* imags, long long int numAmps);"]
        def_lines += [
            "qreal densmatr_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome);"]
        def_lines += [
            "qreal statevec_calcProbOfOutcome(Qureg qureg, const int measureQubit, int outcome);"]
        def_lines += ["int generateMeasurementOutcome(qreal zeroProb, qreal *outcomeProb);"]

        if code_lines is not None:
            lines = code_lines
        else:
            with open(os.path.join(self.include[0], file_name), "r") as f:
                lines = [line for line in f]

        _lines = []
        no_def = True
        for l in lines:
            l += '\n'
            if not l.find("getEnvironmentString") >= 0:
                if no_def and not l.startswith("#"):
                    _lines.append(l)
                elif l.startswith("#ifdef"):
                    no_def = False
                elif l.startswith("#endif"):
                    no_def = True
        _lines = "".join(_lines).replace('qreal', qreal)

        _def_lines = []
        no_def = True
        for l in def_lines:
            if not l.find("getEnvironmentString") >= 0:
                if no_def and not l.startswith("#"):
                    _def_lines.append(l)
                elif l.startswith("#ifdef"):
                    no_def = False
                elif l.startswith("#endif"):
                    no_def = True
        _def_lines = "".join(_def_lines).replace('qreal', qreal)

        call = "{return_type} * ".format(return_type=return_type)
        call += " {}(".format(function_name)
        if arguments is not None:
            call += "{} * {}".format(arguments[0][0], arguments[0][1])
            for vartype, name in arguments[1:]:
                call += ",{} * {}".format(vartype, name)
        call += ");\n"

        _def_lines += (call.replace('qreal', qreal))

        self.lines = _lines
        self.def_lines = _def_lines

    def compile(self, compiled_module_name: str = "_compiled_tmp_quest_programm") -> None:
        """
        Starts the actual cffi compilation

        Args:
            compiled_module_name: Name of the compiled module. The same compiled_module_name can not be used twice
            when being imported (even at different times) in one python module or package
        """
        current_directory = os.getcwd()

        if self._destination_directory is not None:
            os.chdir(self._destination_directory)
        # remove compiled files
        for ending in ['o', 'so', 'c']:
            remove_list = glob.glob('{a}*.{b}'.format(a=compiled_module_name, b=ending))
            for file in remove_list:
                os.remove(file)
        shutil.copy2(self.questlib, os.getcwd())
        ffibuilder = FFI()
        ffibuilder.cdef(self.def_lines)
        source = '# include <QuEST.h>\n'+'# include <stdio.h>\n'+'# include <stdlib.h>\n'
        # for line in self.lines:
        #    source += line+'\n'
        source += self.lines
        ffibuilder.set_source(
            "{}".format(compiled_module_name), source,
            libraries=["QuEST"],
            include_dirs=self.include,
            library_dirs=[self.questlib_path],
            extra_link_args=['-Wl,-rpath=$ORIGIN'],
            # extra_link_args=['-Wl,-rpath={}'.format(self.questlib_path)],
        )
        ffibuilder.compile(verbose=False)
        os.chdir(current_directory)


class callCompiledQuestProgramm(_PYQUEST):
    """
    Class providing the interface to the  compiled cffi QuEST prgramm.
    Provided __call__ magic function like all other pyquest_cffi gate classes

    Args:
        compiled_module_name: The name of the  cffi module compiled fromt the QuEST programm
    """

    def __init__(self,
                 compiled_module_name: str = '_compiled_tmp_quest_programm',
                 **kwargs):
        self._compiled_module_name = compiled_module_name
        super().__init__(**kwargs)
        try:
            sys.path.append(os.getcwd())
            # importing module
            self._compiled_module = importlib.import_module(self._compiled_module_name)
            importlib.reload(self._compiled_module)
        except ModuleNotFoundError:
            print(os.getcwd())
            raise ModuleNotFoundError("Could not find compiled cffi python module {}".format(
                self._compiled_module_name))

    def call_interactive(self, function_name: str = 'tmp_QuEST_function',
                         length_result: int = 2, **kwargs):
        """
        Call the compiled progamm in interactive mode

        Args:
            function_name: The name of the function called in the compiled programm.
            length_result: The length of the Complex array returned by the compiled QuEST programm
            kwargs: The arguments passed to the compiled function
        """
        function = getattr(self._compiled_module.lib, function_name)
        result = function(*[kwargs[key] for key in kwargs.keys()])
        output = np.zeros((length_result,), dtype=np.complex)
        for co in range(length_result):
            output[co] = result[co].real+1j*result[co].imag
        return output

    def call_static(self) -> None:
        raise RuntimeError(
            "This class only provides calls to already compiled function. Can't be compiled again")
