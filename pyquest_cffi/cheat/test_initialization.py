"""Testing measurement"""
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
import sys
import pytest
import numpy.testing as npt
from pyquest_cffi import cheat
from pyquest_cffi import utils
import numpy as np


@pytest.mark.parametrize("init", [
    (cheat.initZeroState, ['qureg']),
    (cheat.initPlusState, ['qureg']),
    (cheat.initDebugState, ['qureg']),
    (cheat.initBlankState, ['qureg']),
    (cheat.initClassicalState, ['qureg', 1]),
    (cheat.initPureState, ['qureg', utils.createQureg()(3, utils.createQuestEnv()())]),
    (cheat.initStateFromAmps, ['qureg', [1, 4, 2, 3, 0, 7, 6, 3], [9, 2, 3, 3, 0, 2, 0, 5]]),
    (cheat.initPauliHamil, ['pauli',
                            [0.3, 0.2, 0.5],
                            [[2, 1, 0], [1, 2, 0], [2, 0, 3]]])
    ])
def test_init_functions(init) -> None:
    """Test init functions

    initZeroState, initPlusState, initDebugState, initBlankState, initClassicalState,
    initPureState, initStateFromAmps, initPauliHamil
    """
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(3, env)
    initialisation = init[0]()
    args = init[1]

    if args[0] == 'qureg':
        args[0] = qubits
    elif args[0] == 'pauli':
        args[0] = utils.createPauliHamil()(number_qubits=3, number_pauliprods=3)

    initialisation(*args)


def test_set_amps_qureg() -> None:
    """Testing set functions

    setAmps, setDensityAmps, setWeightedQureg
    """
    env = utils.createQuestEnv()()
    qureg_statevec = utils.createQureg()(2, env)
    qureg_dens = utils.createDensityQureg()(2, env)
    cheat.initZeroState()(qureg_statevec)
    cheat.initZeroState()(qureg_dens)

    cheat.setAmps()(qureg=qureg_statevec,
                    startind=0,
                    reals=[1, 2, 0, 1],
                    imags=[0, 3, 4, 3],
                    numamps=4)
    state_vec = cheat.getStateVector()(qureg=qureg_statevec)
    state_array = np.array([1 + 0j, 2 + 3j, 0 + 4j, 1 + 3j])
    assert np.all(state_vec == state_array)

    cheat.setDensityAmps()(qureg=qureg_dens,
                           reals=[[1, 2, 1, 2], [0, 1, 0, 1], [1, 0, 1, 0], [2, 1, 2, 1]],
                           imags=[[4, 3, 4, 3], [3, 2, 3, 2], [2, 3, 2, 3], [3, 4, 3, 4]],)
    dens_mat = cheat.getDensityMatrix()(qureg=qureg_dens)
    dens_array = np.array([[1 + 4j, 2 + 3j, 1 + 4j, 2 + 3j],
                           [3j, 1 + 2j, 3j, 1 + 2j],
                           [1 + 2j, 3j, 1 + 2j, 3j],
                           [2 + 3j, 1 + 4j, 2 + 3j, 1 + 4j]]).T
    assert np.all(dens_mat == dens_array)

    # for a wavefunction qureg:
    quregout = utils.createQureg()(2, env)
    cheat.setWeightedQureg()(fac1=2.5, qureg1=qureg_statevec,
                             fac2=3.5j, qureg2=qureg_statevec,
                             facout=1, quregout=quregout)
    state_vec_combined = cheat.getStateVector()(qureg=quregout)
    comparison_array = 2.5 * state_array + 3.5j * state_array + 1 * np.array([1, 0, 0, 0])
    assert np.all(state_vec_combined == comparison_array)

    # for a density matrix qureg:
    quregout = utils.createDensityQureg()(2, env)
    cheat.setWeightedQureg()(fac1=2.5, qureg1=qureg_dens,
                             fac2=3.5j, qureg2=qureg_dens,
                             facout=0.5, quregout=quregout)
    dens_mat_combined = cheat.getDensityMatrix()(qureg=quregout)
    comparison_array = 2.5 * dens_array + 3.5j * dens_array + 0.5 * np.array([[1, 0, 0, 0],
                                                                              [0, 0, 0, 0],
                                                                              [0, 0, 0, 0],
                                                                              [0, 0, 0, 0]])
    assert np.all(dens_mat_combined == comparison_array)


if __name__ == '__main__':
    pytest.main(sys.argv)
