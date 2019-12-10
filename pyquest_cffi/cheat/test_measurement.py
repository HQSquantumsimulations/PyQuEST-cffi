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

import pytest
import sys
import numpy.testing as npt
from pyquest_cffi import cheat
from pyquest_cffi import utils


def test_calcPurity():
    """Testing calcPurity"""
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    cheat.initZeroState()(qureg)
    with npt.assert_warns(RuntimeWarning):
        purity = cheat.measurement.calcPurity()(qureg)
    assert(purity is None)
    qureg = utils.createDensityQureg()(2, env)
    cheat.initZeroState()(qureg)
    purity = cheat.measurement.calcPurity()(qureg)
    npt.assert_equal(purity, 1)


def test_calc_Expec_Pauli_Sum():
    """Test calculating the expectation value of a pauli sum"""
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(4, env)
    workspace = utils.createQureg()(4, env)
    a = cheat.calcExpecPauliSum()(
        qureg=qubits,
        paulis=[[0, 1, 2, 3], [3, 2, 1, 0]],
        coefficients=[0.4, 0.3],
        workspace=workspace,
    )
    print(a)


def test_calc_Hilbert_Schmidt_distance():
    """Test calculating the Hilbert Schmidt distance"""
    env = utils.createQuestEnv()()
    qureg1 = utils.createDensityQureg()(4, env)
    qureg2 = utils.createDensityQureg()(4, env)
    a = cheat.calcHilberSchmidtDistance()(
        qureg1=qureg1,
        qureg2=qureg2,
    )
    print(a)


def test_calc_density_inner_product():
    """Test calculating the inner product for density matrices"""
    env = utils.createQuestEnv()()
    qureg1 = utils.createDensityQureg()(4, env)
    qureg2 = utils.createDensityQureg()(4, env)
    a = cheat.calcDensityInnerProduct()(
        qureg1=qureg1,
        qureg2=qureg2,
    )
    print(a)


def test_calc_Expec_Pauli_Prod():
    """Test calculating the expectation value of a pauli product"""
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(4, env)
    workspace = utils.createQureg()(4, env)
    a = cheat.calcExpecPauliProd()(
        qureg=qubits,
        qubits=[0, 1, 2, 3],
        paulis=[0, 1, 3, 2],
        workspace=workspace,
    )
    print(a)


def test_basis_state_to_index():
    """Testing conversion basis state to index"""
    basis_state = [0, 0, 1, 0, 1]
    index = cheat.basis_state_to_index(basis_state, endianness='little')
    npt.assert_array_equal(index, 20)
    index = cheat.basis_state_to_index(basis_state, endianness='big')
    npt.assert_array_equal(index, 5)


if __name__ == '__main__':
    pytest.main(sys.argv)
