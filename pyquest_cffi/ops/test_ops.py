"""Testing gate operations in PyQuest-cffi"""
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
from typing import List


@pytest.mark.parametrize("gate", [ops.hadamard,
                                  ops.sGate,
                                  ops.tGate,
                                  ops.pauliX,
                                  ops.pauliY,
                                  ops.pauliZ,
                                  ops.controlledNot,
                                  ops.controlledPhaseFlip,
                                  ops.controlledPauliY,
                                  ops.MolmerSorensenXX,
                                  ops.sqrtISwap,
                                  ops.invSqrtISwap,
                                  ops.swapGate,
                                  ops.sqrtSwapGate
                                  ])
def test_simple_gate_matrices(gate) -> None:
    """Testing gates without parameters"""
    matrix_gate = gate().matrix()
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(gate, {})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(gate, {})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.rotateX,
                                  ops.rotateY,
                                  ops.rotateY,
                                  ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateZ,
                                  ops.phaseShift,
                                  ops.controlledPhaseShift,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_single_parameter_gate_matrices(gate, theta) -> None:
    """Testing gates with one parameter"""
    matrix_gate = gate().matrix(theta=theta)
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(gate, {'theta': theta})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(gate, {'theta': theta})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.compactUnitary,
                                  ops.controlledCompactUnitary
                                  ])
@pytest.mark.parametrize("alpha", list(np.arange(0, 1, 1 / 3)))
@pytest.mark.parametrize("phase", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
def test_compact_unitary_gate_matrices(gate, alpha, phase) -> None:
    """Testing compact unitary gate"""
    matrix_gate = gate().matrix(alpha=alpha, beta=np.sqrt(1 - alpha**2) * np.exp(1j * phase))
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(
            gate, {'alpha': alpha, 'beta': np.sqrt(1 - alpha**2) * np.exp(1j * phase)})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(
            gate, {'alpha': alpha, 'beta': np.sqrt(1 - alpha**2) * np.exp(1j * phase)})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.rotateAroundAxis,
                                  ops.controlledRotateAroundAxis
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("phi", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
@pytest.mark.parametrize("theta_s", list(np.arange(0, np.pi, np.pi / 3)))
def test_axis_rotation_gate_matrices(gate, theta, phi, theta_s) -> None:
    """Testing axis rotation"""
    vector = np.zeros((3,))
    vector[0] = np.sin(theta_s) * np.cos(phi)
    vector[1] = np.sin(theta_s) * np.sin(phi)
    vector[2] = np.cos(theta_s)
    matrix_gate = gate().matrix(vector=vector, theta=theta)
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(
            gate, {'theta': theta, 'vector': vector})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(
            gate, {'theta': theta, 'vector': vector})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.unitary,
                                  ops.controlledUnitary
                                  ])
@pytest.mark.parametrize("alpha", list(np.arange(0, 1, 1 / 3)))
@pytest.mark.parametrize("phase", list(np.arange(0, 2 * np.pi, 2 * np.pi / 3)))
def test_unitary_gate_matrices(gate, alpha, phase) -> None:
    """Testing unitary gate"""
    beta = np.sqrt(1 - alpha**2) * np.exp(1j * phase)
    matrix = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]], dtype=complex)
    matrix_gate = gate().matrix(matrix=matrix)
    if matrix_gate.shape == (2, 2):
        matrix_reconstructed = build_one_qubit_matrix(
            gate, {'matrix': matrix})
    elif matrix_gate.shape == (4, 4):
        matrix_reconstructed = build_two_qubit_matrix(
            gate, {'matrix': matrix})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateZ,
                                  ops.controlledPhaseShift,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_two_qubit_unitary(gate, theta) -> None:
    """Test two qubit unitary gate"""
    matrix_gate = gate().matrix(theta=theta)
    matrix_reconstructed = build_two_qubit_matrix_target(
        ops.twoQubitUnitary, {'matrix': matrix_gate})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateZ,
                                  ops.controlledPhaseShift,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_controlled_two_qubit_unitary(gate, theta) -> None:
    """Test controlled two qubit unitary gate"""
    matrix_gate = gate().matrix(theta=theta)
    matrix_reconstructed = build_two_qubit_matrix_target(
        ops.controlledTwoQubitUnitary, {'matrix': matrix_gate}, control=True)
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_reconstructed = build_two_qubit_matrix_target(
        ops.controlledTwoQubitUnitary, {'matrix': matrix_gate}, control=False)
    npt.assert_array_almost_equal(np.eye(4), matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateZ,
                                  ops.controlledPhaseShift,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_multi_qubit_unitary(gate, theta) -> None:
    """Test multi qubit unitary gate"""
    matrix_gate = gate().matrix(theta=theta)
    matrix_reconstructed = build_two_qubit_matrix_targets(
        ops.multiQubitUnitary, {'matrix': matrix_gate})
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateZ,
                                  ops.controlledPhaseShift,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_controlled_multi_qubit_unitary(gate, theta) -> None:
    """Test controlled multi qubit unitary gate"""
    matrix_gate = gate().matrix(theta=theta)
    matrix_reconstructed = build_two_qubit_matrix_targets(
        ops.controlledMultiQubitUnitary, {'matrix': matrix_gate}, control=True)
    npt.assert_array_almost_equal(matrix_gate, matrix_reconstructed)
    matrix_reconstructed = build_two_qubit_matrix_targets(
        ops.controlledMultiQubitUnitary, {'matrix': matrix_gate}, control=False)
    npt.assert_array_almost_equal(np.eye(4), matrix_reconstructed)


@pytest.mark.parametrize("gate", [ops.rotateX,
                                  ops.rotateY,
                                  ops.rotateY,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_multi_state_controlled_unitary(gate, theta) -> None:
    """Test multiStateControlledUnitary"""
    matrix_gate = gate().matrix(theta=theta)
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(3, env)
    ops.multiStateControlledUnitary()(
        qureg=qubits,
        controls=[0, 1],
        control_states=[0, 1],
        qubit=2,
        matrix=matrix_gate)


def test_multi_controlled_PhaseFlip() -> None:
    """Test multiControlledPhaseFlip"""
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(3, env)
    ops.multiControlledPhaseFlip()(
        qureg=qubits,
        controls=[0, 1],
    )


def test_multi_controlled_PhaseShift() -> None:
    """Test multiControlledPhaseShift"""
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(3, env)
    ops.multiControlledPhaseShift()(
        qureg=qubits,
        controls=[0, 1],
        theta=0.3 * np.pi
    )


@pytest.mark.parametrize("gate", [ops.rotateX,
                                  ops.rotateY,
                                  ops.rotateY,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_multi_controlled_unitary(gate, theta) -> None:
    """Test multiControlledUnitary"""
    matrix_gate = gate().matrix(theta=theta)
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(3, env)
    ops.multiControlledUnitary()(
        qureg=qubits,
        controls=[0, 1],
        qubit=2,
        matrix=matrix_gate)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateY,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_multi_controlled_two_qubit_unitary(gate, theta) -> None:
    """Test multiControlledTwoQubitUnitary"""
    matrix_gate = gate().matrix(theta=theta)
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(4, env)
    ops.multiControlledTwoQubitUnitary()(
        qureg=qubits,
        controls=[0, 1],
        target_qubit_1=2,
        target_qubit_2=3,
        matrix=matrix_gate)


@pytest.mark.parametrize("gate", [ops.controlledRotateX,
                                  ops.controlledRotateY,
                                  ops.controlledRotateY,
                                  ])
@pytest.mark.parametrize("theta", list(np.arange(0, 2 * np.pi, 2 * np.pi / 10)))
def test_multi_controlled_multi_qubit_unitary(gate, theta) -> None:
    """Test multiControlledMultiQubitUnitary"""
    matrix_gate = gate().matrix(theta=theta)
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(4, env)
    ops.multiControlledMultiQubitUnitary()(
        qureg=qubits,
        controls=[0, 1],
        targets=[2, 3],
        matrix=matrix_gate)


def test_multiRotatePauli() -> None:
    """Testing multiRotatePauli"""
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(3, env=env)
    a = ops.multiRotatePauli()
    a(qureg, [0, 1, 2], [1, 2, 3], 0.35)


def test_multiRotateZ() -> None:
    """Testing multiRotateZ"""
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(3, env=env)
    ops.multiRotateZ()(
        qureg=qureg,
        qubits=[0, 1, 2],
        angle=0.35)


def build_one_qubit_matrix(gate, gate_args) -> np.ndarray:
    """Build one qubit matrix for tests"""
    matrix = np.zeros((2, 2), dtype=complex)
    for co, state in enumerate([np.array([1, 0]),
                                np.array([0, 1])]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(1, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        gate()(qureg=qubits, qubit=0, **gate_args)
        for i in range(0, 2):
            cComplex = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = cComplex.real + 1j * cComplex.imag
    return matrix


def build_two_qubit_matrix(gate, gate_args) -> np.ndarray:
    """Build two qubit matrix for tests"""
    matrix = np.zeros((4, 4), dtype=complex)
    for co, state in enumerate([np.array([1, 0, 0, 0]),
                                np.array([0, 1, 0, 0]),
                                np.array([0, 0, 1, 0]),
                                np.array([0, 0, 0, 1]), ]):
        env = utils.createQuestEnv()()
        qubits = utils.createQureg()(2, env)
        cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
        gate()(qureg=qubits, control=1, qubit=0, **gate_args)
        for i in range(0, 4):
            cComplex = cheat.getAmp()(qureg=qubits, index=i)
            matrix[i, co] = cComplex.real + 1j * cComplex.imag
    return matrix


def build_two_qubit_matrix_target(gate, gate_args, control=None) -> np.ndarray:
    """Build two qubit matrix with target for tests"""
    matrix = np.zeros((4, 4), dtype=complex)
    for co, state in enumerate([np.array([1, 0, 0, 0]),
                                np.array([0, 1, 0, 0]),
                                np.array([0, 0, 1, 0]),
                                np.array([0, 0, 0, 1]), ]):
        env = utils.createQuestEnv()()
        if control is None:
            qubits = utils.createQureg()(2, env)
            cheat.initStateFromAmps()(
                qubits,
                np.real(state),
                np.imag(state))
            gate()(qureg=qubits, target_qubit_1=0, target_qubit_2=1, **gate_args)
        elif control is True:
            qubits = utils.createQureg()(3, env)
            control_array = np.array([0, 1])
            cheat.initStateFromAmps()(
                qubits,
                np.real(np.kron(control_array, state)),
                np.imag(np.kron(control_array, state)))
            gate()(qureg=qubits, target_qubit_1=0, target_qubit_2=1, control=2, **gate_args)
        else:
            qubits = utils.createQureg()(3, env)
            control_array = np.array([1, 0])
            cheat.initStateFromAmps()(
                qubits,
                np.real(np.kron(control_array, state)),
                np.imag(np.kron(control_array, state)))
            gate()(qureg=qubits, target_qubit_1=0, target_qubit_2=1, control=2, **gate_args)
        for i in range(0, 4):
            if control is None:
                cComplex = cheat.getAmp()(qureg=qubits, index=i)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
            elif control is True:
                cComplex = cheat.getAmp()(qureg=qubits, index=i + 4)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
            else:
                cComplex = cheat.getAmp()(qureg=qubits, index=i)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
    return matrix


def build_two_qubit_matrix_targets(gate, gate_args, control=None) -> np.ndarray:
    """Build two qubit matrix with multiple targets for tests"""
    matrix = np.zeros((4, 4), dtype=complex)
    for co, state in enumerate([np.array([1, 0, 0, 0]),
                                np.array([0, 1, 0, 0]),
                                np.array([0, 0, 1, 0]),
                                np.array([0, 0, 0, 1]), ]):
        env = utils.createQuestEnv()()
        if control is None:
            qubits = utils.createQureg()(2, env)
            cheat.initStateFromAmps()(qubits, np.real(state), np.imag(state))
            gate()(qureg=qubits, targets=[0, 1], **gate_args)
        elif control is True:
            qubits = utils.createQureg()(3, env)
            control_array = np.array([0, 1])
            cheat.initStateFromAmps()(
                qubits,
                np.real(np.kron(control_array, state)),
                np.imag(np.kron(control_array, state)))
            gate()(qureg=qubits, targets=[0, 1], control=2, **gate_args)
        else:
            qubits = utils.createQureg()(3, env)
            control_array = np.array([1, 0])
            cheat.initStateFromAmps()(
                qubits,
                np.real(np.kron(control_array, state)),
                np.imag(np.kron(control_array, state)))
            gate()(qureg=qubits, targets=[0, 1], control=2, **gate_args)
        for i in range(0, 4):
            if control is None:
                cComplex = cheat.getAmp()(qureg=qubits, index=i)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
            elif control is True:
                cComplex = cheat.getAmp()(qureg=qubits, index=i + 4)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
            else:
                cComplex = cheat.getAmp()(qureg=qubits, index=i)
                matrix[i, co] = cComplex.real + 1j * cComplex.imag
    return matrix


@pytest.mark.parametrize("applied", [
    (ops.applyDiagonalOp, [(5, utils.createQuestEnv()())]),
    (ops.applyMatrix2, [2, np.array([[0, 1], [2, 1]])]),
    (ops.applyMatrix4, [1, 3, 'matrix']),
    (ops.applyMatrixN, [[1, 2], 'matrix']),
    (ops.applyMultiControlledMatrixN, [[1, 2], [3, 4], 'matrix']),
    (ops.applyPauliHamil, [utils.createPauliHamil()(5, 2),
                           utils.createQureg()(5, utils.createQuestEnv()())]),
    (ops.applyPauliSum, [[[0, 1, 2, 3], [3, 2, 1, 0]], [0.4, 0.3],
                         utils.createQureg()(5, utils.createQuestEnv()())]),
    (ops.applyTrotterCircuit, [utils.createPauliHamil()(5, 2), 0.7, 1, 2])
    ])
def test_apply_functions(applied) -> None:
    """Test all non-deprecated apply functions"""
    env = utils.createQuestEnv()()
    qubits = utils.createQureg()(5, env)
    cheat.initZeroState()(qureg=qubits)
    op = applied[0]()
    positional_args = applied[1]
    matrix = np.array([[1, 0, 0, 1], [0, 2, 0, 2], [2, 0, 2, 0], [1, 0, 0, 1]])
    if positional_args[-1] == 'matrix':
        args = [qubits]
        args.extend(positional_args)
        args[-1] = matrix
    else:
        args = [qubits]
        args.extend(positional_args)

    op(*args)
    if positional_args[-1] == 'matrix':
        npt.assert_array_equal(matrix, op.matrix(matrix=matrix))


def test_measure() -> None:
    """Test measuring functions: measure, measureWithStats and collapseToOutcome"""
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(3, env)
    cheat.initZeroState()(qureg=qureg)

    qubit_0 = ops.measure()(qureg=qureg, qubit=0)
    qubit_1 = ops.measureWithStats()(qureg=qureg, qubit=1, outcome_proba=0.5)
    proba_qubit_2 = ops.collapseToOutcome()(qureg=qureg, qubit=2, outcome=0)

    assert qubit_0 == 0
    assert qubit_1 == 0
    assert proba_qubit_2 == 1


if __name__ == '__main__':
    pytest.main(sys.argv)
