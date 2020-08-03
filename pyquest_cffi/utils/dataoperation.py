"""Manage QuEST environments and quantum registers"""
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

from pyquest_cffi.questlib import quest, _PYQUEST, tquestenv, tqureg, paulihamil
from typing import List


class createQuestEnv(_PYQUEST):
    """Creates the QuEST simulator environment, needed for all simulations"""

    def call_interactive(self) -> tquestenv:
        """Interactive call of PyQuest-cffi

        Returns:
            tquestenv
        """
        return quest.createQuESTEnv()


class destroyQuestEnv(_PYQUEST):
    """
    Deallocate QuEST simulation environment

    Args:
        env: QuEST environment to be deallocated

    """

    def call_interactive(self, env: tquestenv) -> None:
        """Interactive call of PyQuest-cffi

        Args:
            env: QuEST environment to be deallocated
        """
        quest.destroyQuESTEnv(env)


class createQureg(_PYQUEST):
    """
    Allocate memory for a wavefunction qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        """Interactive call of PyQuest-cffi

        Args:
            num_qubits: number of qubits in the quantum register
            env: QuEST environment in which the qureg exists

        Returns:
            tqureg
        """
        return quest.createQureg(num_qubits, env)


class createDensityQureg(_PYQUEST):
    """
    Allocate memory for a density matrix qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        """Interactive call of PyQuest-cffi

        Args:
            num_qubits: number of qubits in the quantum register
            env: QuEST environment in which the qureg exists

        Returns:
            tqureg
        """
        return quest.createDensityQureg(num_qubits, env)


class destroyQureg(_PYQUEST):
    """Deallocate memory for a qubit register"""

    def call_interactive(self, qubits: List[int], env: tquestenv) -> None:
        """Interactive call of PyQuest-cffi

        Args:
            qubits: Qubits in system
            env: QuEST environment in which the qureg exists
        """
        quest.destroyQureg(qubits, env)


class createCloneQureg(_PYQUEST):
    """Create a clone of the qureg in a certain environment

    Args:
        qureg: Qureg to be cloned
        env: QuEST environment the clone is created in

    Returns:
        cloned qureg

    """

    def call_interactive(self, qureg: tqureg, env: tquestenv) -> tqureg:
        """Interactive call of PyQuest-cffi

        Args:
            qureg: Qureg to be cloned
            env: QuEST environment the clone is created in

        Returns:
            tqureg: cloned qureg
        """
        return quest.createCloneQureg(qureg, env)


class cloneQureg(_PYQUEST):
    """Clone a qureg state into another one

    Args:
        qureg_original: Qureg to be cloned
        qureg_clone: Cloned qureg

    """

    def call_interactive(self, qureg_clone: tqureg, qureg_original: tqureg) -> None:
        """Interactive call of PyQuest-cffi

        Args:
            qureg_original: Qureg to be cloned
            qureg_clone: Cloned qureg
        """
        quest.cloneQureg(qureg_clone, qureg_original)


class createPauliHamil(_PYQUEST):
    """Create a clone of the qureg in a certain environment

    Args:
        number_qubits: the number of qubits on which this Hamiltonian acts 
        number_pauliprods: the number of weighted terms in the sum, or the number of Pauli products

    Returns:
        cloned qureg

    """

    def call_interactive(self, number_qubits: int, number_pauliprods: int) -> paulihamil:
        """Interactive call of PyQuest-cffi

        Args:
            number_qubits: the number of qubits on which this Hamiltonian acts 
            number_pauliprods: the number of weighted terms in the sum, or the number of Pauli products

        Returns:
            PauliHamil: created Pauli Hamiltonian
        """
        return quest.createPauliHamil(number_qubits, number_pauliprods)


class destroyPauliHamil(_PYQUEST):
    """Create a DiagonalOp on the full Hilbert space of a Qureg.

    Args:
        pauli_hamil: PauliHamil to be destroyed
    """

    def call_interactive(self, pauli_hamil: paulihamil) -> None:
        """Interactive call of PyQuest-cffi

        Args:
            pauli_hamil: PauliHamil to be destroyed
        """
        return quest.destroyPauliHamil(pauli_hamil)
