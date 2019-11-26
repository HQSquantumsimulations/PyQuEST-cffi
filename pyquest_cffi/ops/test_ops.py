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
                                  ops.invSqrtISwap
                                  ])
def test_simple_gate_matrices(gate):
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


def build_one_qubit_matrix(gate, gate_args):
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


def build_two_qubit_matrix(gate, gate_args):
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


if __name__ == '__main__':
    pytest.main(sys.argv)
