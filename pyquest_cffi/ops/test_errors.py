"""Testing error operations of PyQuest-cffi"""
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

import pytest
import sys
import numpy as np
import numpy.testing as npt
from pyquest_cffi import ops
from pyquest_cffi import cheat
from pyquest_cffi import utils


@pytest.mark.parametrize("prob", list(np.arange(0, 1, 0.05)))
@pytest.mark.parametrize("gate_def", [(ops.applyOneQubitDampingError, 1),
                                      (ops.applyOneQubitDephaseError, 1 / 2),
                                      (ops.applyOneQubitDepolariseError, 3 / 4),
                                      (ops.mixDamping, 1),
                                      (ops.mixDephasing, 1 / 2),
                                      (ops.mixDepolarising, 3 / 4),
                                      (ops.mixDensityMatrix, 1 / 4)])
def test_one_qubit_errors(prob, gate_def) -> None:
    """Testing one qubit errors and the mixDensityMatrix error"""
    op = gate_def[0]
    prob = prob * gate_def[1]
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(1, env)
    state = np.random.random((2, 1)) + 1j * np.random.random((2, 1))
    state = state / np.linalg.norm(state)
    state_dm = state @ state.conjugate().T
    cheat.setDensityAmps()(dm,
                           reals=np.real(state_dm), imags=np.imag(state_dm))
    if gate_def[1] == 1 / 4:
        dm_other = utils.createDensityQureg()(1, env)
        op()(qureg=dm, probability=prob, qureg_other=dm_other)
    else:
        op()(qureg=dm, qubit=0, probability=prob)
    try:
        superop = op().superoperator_matrix(probability=prob)
        state_dm = state_dm.reshape((4, 1))
        end_matrix = (superop @ state_dm).reshape((2, 2), order='F')
        matrix = cheat.getDensityMatrix()(dm)
        npt.assert_array_almost_equal(matrix, end_matrix)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("prob", list(np.arange(0, 1, 0.05)))
@pytest.mark.parametrize("gate_def", [(ops.applyTwoQubitDephaseError, 1 / 2),
                                      (ops.applyTwoQubitDepolariseError, 3 / 4),
                                      (ops.mixTwoQubitDephasing, 1 / 2),
                                      (ops.mixTwoQubitDepolarising, 3 / 4)
                                      ])
def test_two_qubit_errors(prob, gate_def) -> None:
    """Testing two qubit errors"""
    op = gate_def[0]
    prob = prob * gate_def[1]
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(2, env)
    state = np.random.random((4, 1)) + 1j * np.random.random((4, 1))
    state = state / np.linalg.norm(state)
    state_dm = state @ state.conjugate().T
    state_dm = state_dm.reshape((16, 1))
    cheat.initStateFromAmps()(dm,
                              reals=np.real(state_dm),
                              imags=np.imag(state_dm))
    op()(qureg=dm, qubit1=0, qubit2=1, probability=prob)


def test_mix_pauli():
    """Test pauli errors"""
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(2, env)
    ops.mixPauli()(dm, qubit=0, probX=0.1, probY=0.1, probZ=0.1)


def test_mix_kraus_map():
    """Test Kraus operator error"""
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(1, env)
    operators = [np.array([[1, 0], [0, 1]]), ]
    ops.mixKrausMap()(dm, qubit=0, operators=operators)


def test_mix_two_qubit_kraus_map():
    """Test Kraus operator error acting on two qubits"""
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(2, env)
    operators = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                 ]
    ops.mixTwoQubitKrausMap()(dm, target_qubit_1=0, target_qubit_2=1, operators=operators)


def test_mix_multi_qubit_kraus_map():
    """Test Kraus operator error acting on multiple qubits"""
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(2, env)
    operators = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                 ]
    ops.mixMultiQubitKrausMap()(dm, qubits=[0, 1], operators=operators)


if __name__ == '__main__':
    pytest.main(sys.argv)
