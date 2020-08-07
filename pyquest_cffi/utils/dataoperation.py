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
from pyquest_cffi import cheat


class createQuestEnv(_PYQUEST):
    r"""Creates the QuEST simulator environment, needed for all simulations"""

    def call_interactive(self) -> tquestenv:
        r"""Interactive call of PyQuest-cffi

        Returns:
            tquestenv
        """
        return quest.createQuESTEnv()


class destroyQuestEnv(_PYQUEST):
    r"""Deallocate QuEST simulation environment

    Args:
        env: QuEST environment to be deallocated

    """

    def call_interactive(self, env: tquestenv) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            env: QuEST environment to be deallocated
        """
        quest.destroyQuESTEnv(env)


class createQureg(_PYQUEST):
    r"""Allocate memory for a wavefunction quantum/qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        r"""Interactive call of PyQuest-cffi

        Args:
            num_qubits: number of qubits in the quantum register
            env: QuEST environment in which the qureg exists

        Returns:
            tqureg
        """
        return quest.createQureg(num_qubits, env)


class createDensityQureg(_PYQUEST):
    r"""Allocate memory for a density matrix quantum/qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        r"""Interactive call of PyQuest-cffi

        Args:
            num_qubits: number of qubits in the quantum register
            env: QuEST environment in which the qureg exists

        Returns:
            tqureg
        """
        return quest.createDensityQureg(num_qubits, env)


class destroyQureg(_PYQUEST):
    r"""Deallocate memory for a quantum/qubit register

    Args:
        qubits: Qubits in system
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, qubits: List[int], env: tquestenv) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qubits: Qubits in system
            env: QuEST environment in which the qureg exists
        """
        quest.destroyQureg(qubits, env)


class createCloneQureg(_PYQUEST):
    r"""Create a clone of the qureg in a certain environment

    Args:
        qureg: Qureg to be cloned
        env: QuEST environment the clone is created in

    """

    def call_interactive(self, qureg: tqureg, env: tquestenv) -> tqureg:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: Qureg to be cloned
            env: QuEST environment the clone is created in

        Returns:
            tqureg: cloned qureg
        """
        return quest.createCloneQureg(qureg, env)


class cloneQureg(_PYQUEST):
    r"""Clone a qureg state into another one

    Set qureg_clone to be a clone of qureg_original

    Args:
        qureg_original: Qureg to be cloned
        qureg_clone: Cloned qureg

    """

    def call_interactive(self, qureg_clone: tqureg, qureg_original: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg_original: Qureg to be cloned
            qureg_clone: Cloned qureg

        Raises:
            TypeError: The quregs need to be of the same type, so either both density
                matrices OR both wave functions
            ValueError: The quregs need to contain the same number of qubits
        """
        if qureg_clone.isDensityMatrix and qureg_original.isDensityMatrix:
            if cheat.getNumQubits()(qureg=qureg_clone) == cheat.getNumQubits()(qureg=qureg_clone):
                quest.cloneQureg(qureg_clone, qureg_original)
            else:
                raise ValueError("The quregs need to contain the same number of qubits")
        elif not qureg_clone.isDensityMatrix and not qureg_original.isDensityMatrix:
            if cheat.getNumQubits()(qureg=qureg_clone) == cheat.getNumQubits()(qureg=qureg_clone):
                quest.cloneQureg(qureg_clone, qureg_original)
            else:
                raise ValueError("The quregs need to contain the same number of qubits")
        else:
            raise TypeError("The quregs need to be of the same type, so either both "
                            + "density matrices OR both wave functions")


class createPauliHamil(_PYQUEST):
    r"""Create a Hamiltonian expressed as a real-weighted sum of products of Pauli operators

    Args:
        number_qubits: the number of qubits on which this Hamiltonian acts
        number_pauliprods: the number of weighted terms in the sum, or the number of Pauli products

    """

    def call_interactive(self, number_qubits: int, number_pauliprods: int) -> 'quest.PauliHamil':
        r"""Interactive call of PyQuest-cffi

        Args:
            number_qubits: the number of qubits on which this Hamiltonian acts
            number_pauliprods: the number of weighted terms in the sum/number of Pauli products

        Returns:
            PauliHamil: created Pauli Hamiltonian

        Raises:
            RuntimeError: number_qubits and number_pauliprods need to be positive integers
        """
        if number_pauliprods <= 0 or number_pauliprods <= 0:
            raise RuntimeError("number_qubits and number_pauliprods need to be positive integers")
        return quest.createPauliHamil(number_qubits, number_pauliprods)


class destroyPauliHamil(_PYQUEST):
    r"""Destroy a PauliHamil instance

    Args:
        pauli_hamil: PauliHamil to be destroyed

    """

    def call_interactive(self, pauli_hamil: paulihamil) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            pauli_hamil: PauliHamil to be destroyed
        """
        quest.destroyPauliHamil(pauli_hamil)
