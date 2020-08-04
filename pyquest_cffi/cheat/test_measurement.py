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
import numpy.testing as npt
from pyquest_cffi import cheat
from pyquest_cffi import utils


def test_calc_simple() -> None:
    """Testing simple calc functions.
    
    calcPurity, calcFidelity, calcInnerProduct, calcProbofOutcome, calcTotalProb
    """
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    qureg_main = utils.createQureg()(2, env)
    cheat.initZeroState()(qureg)

    with npt.assert_warns(RuntimeWarning):
        purity = cheat.measurement.calcPurity()(qureg)
        density_inner_product = cheat.measurement.calcDensityInnerProduct()(qureg1=qureg,
                                                                            qureg2=qureg_main)
    fidelity = cheat.measurement.calcFidelity()(qureg=qureg_main,
                                                qureg_reference=qureg)
    inner_product = cheat.measurement.calcInnerProduct()(qureg1=qureg,
                                                         qureg2=qureg_main)
    prob_of_outcome = cheat.measurement.calcProbOfOutcome()(qureg, 1, 0)
    total_prob = cheat.measurement.calcTotalProb()(qureg)

    assert purity is None
    assert fidelity == 1
    assert (inner_product.real == 1 and inner_product.imag == 0)
    assert prob_of_outcome == 1.0
    assert total_prob == 1
    assert density_inner_product is None

    qureg = utils.createDensityQureg()(2, env)
    cheat.initZeroState()(qureg)

    with npt.assert_warns(RuntimeWarning):
        fidelity = cheat.measurement.calcFidelity()(qureg=qureg_main,
                                                    qureg_reference=qureg)
        inner_product_1 = cheat.measurement.calcInnerProduct()(qureg1=qureg,
                                                               qureg2=qureg_main)
        inner_product_2 = cheat.measurement.calcInnerProduct()(qureg1=qureg_main,
                                                               qureg2=qureg)
        inner_product_3 = cheat.measurement.calcInnerProduct()(qureg1=qureg,
                                                               qureg2=qureg)
        density_inner_product_1 = cheat.measurement.calcDensityInnerProduct()(qureg1=qureg,
                                                                            qureg2=qureg_main)
        density_inner_product_2 = cheat.measurement.calcDensityInnerProduct()(qureg1=qureg_main,
                                                                            qureg2=qureg)
    density_inner_product_3 = cheat.measurement.calcDensityInnerProduct()(qureg1=qureg,
                                                                        qureg2=qureg)
    purity = cheat.measurement.calcPurity()(qureg)
    prob_of_outcome = cheat.measurement.calcProbOfOutcome()(qureg, 1, 0)
    total_prob = cheat.measurement.calcTotalProb()(qureg)

    assert purity == 1
    assert fidelity is None
    for inner_product in [inner_product_1, inner_product_2, inner_product_3]:
        assert inner_product is None
    assert prob_of_outcome == 1.0
    assert total_prob == 1
    assert [density_inner_product_1, density_inner_product_2] == [None, None]
    assert density_inner_product_3 == 1


def test_calc_Expec_Pauli_Sum() -> None:
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
    assert a == 0.0


def test_calc_Hilbert_Schmidt_distance() -> None:
    """Test calculating the Hilbert Schmidt distance"""
    env = utils.createQuestEnv()()
    qureg1 = utils.createDensityQureg()(4, env)
    qureg2 = utils.createDensityQureg()(4, env)
    a = cheat.calcHilbertSchmidtDistance()(
        qureg1=qureg1,
        qureg2=qureg2,
    )
    assert a == 0.0


def test_calc_Expec_Pauli_Prod() -> None:
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
    assert a == 0.0


def test_seed() -> None:
    """Test seeding functions: seedQuEST, seedQuESTDefault"""
    cheat.seedQuEST()(seed_array=[0, 1, 2])
    cheat.seedQuESTDefault()()


def test_sync() -> None:
    """Test syncing functions: syncQuESTEnv, syncQuESTSuccess"""
    env = utils.createQuestEnv()()
    cheat.syncQuESTEnv()(env=env)
    success = cheat.syncQuESTSuccess()(success_code=0)
    assert success == 0


def test_basis_state_index_conversion() -> None:
    """Testing conversion basis state to index and index to basis state"""
    basis_state_1 = [0, 0, 1, 0, 1]
    index_1 = 20
    index_2 = 5

    index = cheat.basis_state_to_index(basis_state_1, endianness='little')
    npt.assert_array_equal(index, 20)
    index = cheat.basis_state_to_index(basis_state_1, endianness='big')
    npt.assert_array_equal(index, 5)

    basis_state = cheat.index_to_basis_state(index_1,
                                             num_qubits_represented=5,
                                             endianness='little')
    npt.assert_array_equal(basis_state, basis_state_1)
    basis_state = cheat.index_to_basis_state(index_2,
                                             num_qubits_represented=5,
                                             endianness='big')
    npt.assert_array_equal(basis_state, basis_state_1)


if __name__ == '__main__':
    pytest.main(sys.argv)
