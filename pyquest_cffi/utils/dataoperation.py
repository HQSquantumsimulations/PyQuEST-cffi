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

from pyquest_cffi.questlib import quest, _PYQUEST, tquestenv, tqureg


class createQuestEnv(_PYQUEST):
    """Creates the QuEST simulator environment, needed for all simulations"""

    def call_interactive(self,) -> tquestenv:
        """Call interactive Pyquest-cffi function"""
        return quest.createQuESTEnv()


class destroyQuestEnv(_PYQUEST):
    """
    Deallocate QuEST simulation environment

    Args:
        env: QuEST environment to be deallocated

    """

    def call_interactive(self, env: tquestenv) -> None:
        """Call interactive Pyquest-cffi function"""
        quest.destroyQuESTEnv(env)


class createQureg(_PYQUEST):
    """
    Allocate memory for a wavefunction qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        """Call interactive Pyquest-cffi function"""
        return quest.createQureg(num_qubits, env)


class createDensityQureg(_PYQUEST):
    """
    Allocate memory for a density matrix qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists

    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        """Call interactive Pyquest-cffi function"""
        return quest.createDensityQureg(num_qubits, env)


class destroyQureg(_PYQUEST):
    """Deallocate memory for a qubit register"""

    def call_interactive(self, qubits, env: tquestenv) -> None:
        """Call interactive Pyquest-cffi function"""
        quest.destroyQureg(qubits, env)


class createCloneQureg(_PYQUEST):
    """Create a clone of the qureg in a certain environment

    Args:
        qureg: Qureg to be cloned
        env: QuEST environment the clone is created in

    Returns:
        cloned qureg

    """

    def call_interactive(self, qureg, env: tquestenv) -> tqureg:
        """Call interactive Pyquest-cffi function"""
        return quest.createCloneQureg(qureg, env)


class cloneQureg(_PYQUEST):
    """Clone a qureg state into another one

    Args:
        qureg_original: Qureg to be cloned
        qureg_clone: Cloned qureg

    """

    def call_interactive(self, qureg_clone, qureg_original) -> None:
        """Call interactive Pyquest-cffi function"""
        return quest.cloneQureg(qureg_clone, qureg_original)
