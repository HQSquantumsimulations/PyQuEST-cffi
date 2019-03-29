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
from pyquest_cffi.questlib import quest, _PYQUEST, tquestenv, tqureg
from typing import Union, List


class createQuestEnv(_PYQUEST):
    """
    Creates the QuEST simulator environment, needed for all simulations
    """

    def call_interactive(self) -> tquestenv:
        return quest.createQuESTEnv()

    def call_static(self, env_name: str) -> List[str]:
        """
        Static call of createQuestEnv

        Args:
            env: The name of the created environment as a string, can be now be used at later points in the programm
        """
        call = "QuESTEnv {env} = createQuESTEnv();".format(
            env=env_name)
        return [call]


class destroyQuestEnv(_PYQUEST):
    """
    Deallocate QuEST simulation environment

    Args:
        env: QuEST environment to be deallocated
    """

    def call_interactive(self, env: tquestenv) -> None:
        quest.destroyQuESTEnv(env)

    def call_static(self, env_name: str) -> List[str]:
        """
        Static call of destroyQestEnv

        Args:
            env: The name of the previously created quantum environment as a string
        """
        call = "destroyQuestEnv({env});".format(
            env=env_name)
        return [call]


class createQureg(_PYQUEST):
    """
    Allocate memory for a wavefunction qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists
    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        return quest.createQureg(num_qubits, env)

    def call_static(self, num_qubits: Union[int, str], env: str, qureg_name: str):
        """
        Static call of createQureg

        Args:
            qureg: The name of the created quantum register as a string, can be now be used at later points in the programm
            env: The name of the previously created quantum environment as a string
            num_qubits: number of qubits, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "Qureg {qureg_name} = createQureg({num_qubits},{env});".format(
            qureg_name=qureg_name, num_qubits=num_qubits, env=env)
        return [call]


class createDensityQureg(_PYQUEST):
    """
    Allocate memory for a density matrix qubit register (qureg)

    Args:
        num_qubits: number of qubits in the quantum register
        env: QuEST environment in which the qureg exists
    """

    def call_interactive(self, num_qubits: int, env: tquestenv) -> tqureg:
        return quest.createDensityQureg(num_qubits, env)

    def call_static(self, num_qubits: Union[int, str], env: str, qureg_name: str):
        """
        Static call of createDensityQureg

        Args:
            qureg: The name of the created quantum register as a string, can be now be used at later points in the programm
            env: The name of the previously created quantum environment as a string
            num_qubits: number of qubits, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "Qureg {qureg_name} = createDensityQureg({N},{env});".format(
            qureg_name=qureg_name, N=num_qubits, env=env)
        return [call]


class destroyQureg(_PYQUEST):
    """
    Deallocate memory for a qubit register
    """

    def call_interactive(self, qubits, env: tquestenv) -> None:
        quest.destroyQureg(qubits, env)

    def call_static(self, qureg_name: str):
        """
        Static call of destroyQureg

        Args:
            qureg: The name of the previously created quantum register as a string
        """
        call = "destroyQureg({qureg_name});".format(
            qureg_name=qureg_name)
        return [call]
