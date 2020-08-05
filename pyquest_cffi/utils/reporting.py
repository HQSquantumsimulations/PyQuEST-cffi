"""Reporting of QuEST states"""
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

from typing import List
from pyquest_cffi.questlib import quest, _PYQUEST, tquestenv, tqureg, paulihamil


class reportQuESTEnv(_PYQUEST):
    r"""Report the properties of the QuEST simulation environment to stdout

    Args:
        env: QuEST environment for which the parameter are reported

    """

    def call_interactive(self, env: tquestenv) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            env: QuEST environment for which the parameter are reported
        """
        quest.reportQuESTEnv(env)

    @property
    def restype(self) -> str:
        r"""Return result type

        Returns:
            str
        """
        return "void"

    @property
    def argtype(self) -> List[str]:
        r"""Return type of argument

        Returns:
            List[str]
        """
        return ["QuESTEnv"]


class reportQuregParams(_PYQUEST):
    r"""Reports the parameters of a quantum register to stdout

    Args:
        qureg: Quantum register for which the parameter are reported

    """

    def call_interactive(self, qureg: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: Quantum register for which the parameter are reported
        """
        quest.reportQuregParams(qureg)

    @property
    def restype(self) -> str:
        r"""Return result type

        Returns:
            str
        """
        return "void"

    @property
    def argtype(self) -> List[str]:
        r"""Return argument type

        Returns:
            List[str]
        """
        return ["Qureg"]


class reportState(_PYQUEST):
    r"""Report QuEST state

    Args:
        qureg: Quantum Register (qureg)

    """

    def call_interactive(self, qureg: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: Quantum Register (qureg)
        """
        quest.reportState(qureg)

    @property
    def restype(self) -> str:
        r"""Return result type

        Returns:
            str
        """
        return "void"

    @property
    def argtype(self) -> List[str]:
        r"""Return argument type

        Returns:
            List[str]
        """
        return ["Qureg"]


class reportStateToScreen(_PYQUEST):
    r"""Report statevector or density matrix in a qureg to stdout

    Args:
        qureg: the quantum register
        env: the environment of the quantum register

    """

    def call_interactive(self,
                         qureg: tqureg,
                         env: tquestenv,
                         a: int = 0) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: the quantum register
            env: the environment of the quantum register
            a: integer
        """
        quest.reportStateToScreen(qureg, env, a)

    @property
    def restype(self) -> str:
        r"""Return result type

        Returns:
            str
        """
        return "void"

    @property
    def argtype(self) -> List[str]:
        r"""Return argument type

        Returns:
            List[str]
        """
        return ["Qureg", "QuESTEnv", "int"]


class reportPauliHamil(_PYQUEST):
    r"""Report PauliHamil to stdout

    The output features a new line for each term, each with format:
    "c p1 p2 p3 ... pN",
    where c is the real coefficient of the term, and p1 ... pN are
    numbers 0, 1, 2, 3 to indicate identity, pauliX, pauliY and pauliZ
    operators respectively, acting on qubits 0 through N-1 (all qubits).
    A tab character separates c and p1, single spaces separate the Pauli operators

    Args:
        pauli_hamil: instatiated PauliHamil

    """

    def call_interactive(self, pauli_hamil: paulihamil) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            pauli_hamil: instatiated PauliHamil
        """
        quest.reportPauliHamil(pauli_hamil)

    @property
    def restype(self) -> str:
        r"""Return result type

        Returns:
            str
        """
        return "floats/integers"

    @property
    def argtype(self) -> List[str]:
        r"""Return argument type

        Returns:
            List[str]
        """
        return ["Coefficient", "PauliMatrix"]
