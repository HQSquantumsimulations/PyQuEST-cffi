"""Measurement function in PyQuest-cffi"""
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
from pyquest_cffi.questlib import quest, _PYQUEST, tqureg, ffi_quest, qreal, tquestenv, paulihamil
import numpy as np
from typing import Sequence, Union, List, Tuple
from pyquest_cffi import cheat


class calcFidelity(_PYQUEST):
    r"""Calculate Fidelity of a quantum register

    Determine the fidelity of a qureg (wavefunction :math:`\left| \psi \right\rangle`
        or density matrix :math:`\rho`)
    with respect to a reference_qureg of a wavefunction :math:`\left| \psi_{ref} \right\rangle`
    Fidelity is defined as:

    .. math::
        \mathcal{F} &= \left\langle \psi | \psi_{ref} \right \rangle \\
        \mathcal{F} &= \left\langle \psi_{ref}| \rho | \psi_{ref} \right \rangle

    Args:
        qureg: a qureg containing a wavefunction or a density matrix
        qureg_reference: a qureg containing a wavefunction
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, qureg_reference: tqureg) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or a density matrix
            qureg_reference: a qureg containing a wavefunction

        Returns:
            float

        Raises:
            RuntimeError: Reference qureg has to be a wavefunction qureg
                but density matrix qureg was used
        """
        if not qureg_reference.isDensityMatrix:
            return quest.calcFidelity(qureg, qureg_reference)
        else:
            raise RuntimeError("Reference qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")


class calcInnerProduct(_PYQUEST):
    r"""Calculate the inner-product/overlap of two wavefunction quregs

    Inner product defined as:
    .. math::
         \left\langle \psi_{qureg1} | \psi_{qureg2} \right \rangle

    Args:
        qureg1: a qureg containing a wavefunction
        qureg2: a qureg containing a wavefunction
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg1: tqureg, qureg2: tqureg) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg1: a qureg containing a wavefunction
            qureg2: a qureg containing a wavefunction

        Returns:
            float

        Raises:
            RuntimeError: Qureg1 has to be a wavefunction qureg but density matrix qureg was used
            RuntimeError: Qureg2 has to be a wavefunction qureg but density matrix qureg was used
        """
        if qureg1.isDensityMatrix:
            raise RuntimeError("Qureg1 has to be a wavefunction qureg but "
                               + "density matrix qureg was used")
        elif qureg2.isDensityMatrix:
            raise RuntimeError("Qureg2 has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        return quest.calcInnerProduct(qureg1, qureg2)


class calcProbOfOutcome(_PYQUEST):
    r"""Calculate the probability that qubit #qubit of qureg is measured in state outcome

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        qubit: the index of the qubit for which the probability is determined
        outcome: the outcome of the measurement
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, qubit: int, outcome: int) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or density matrix
            qubit: the index of the qubit for which the probability is determined
            outcome: the outcome of the measurement

        Returns:
            float
        """
        return quest.calcProbOfOutcome(qureg, qubit, outcome)


class calcPurity(_PYQUEST):
    r"""Calculate the purity of a density matrix in qureg

    Purity defined as:
    .. math::
         \mathcal{Tr}\left(\rho^2\right)

    Args:
        qureg: a qureg containing a density matrix
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if qureg.isDensityMatrix:
            return quest.calcPurity(qureg)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")


class calcTotalProb(_PYQUEST):
    r"""Calculate total probability

    Check physicality of system by calculating probability of system to be in any state.
    In other words check that trace of density matrix or norm of state vector is one.

    Args:
        qureg: a qureg containing a density matrix or wavefunction
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix or wavefunction

        Returns:
            float
        """
        return quest.calcTotalProb(qureg)


class getStateVectoratIndex(_PYQUEST):
    r"""Get the value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction
            index: The index either as an int or as a sequence
                of 0 and 1 referencing the corresponding basis state

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        cComplex = quest.getAmp(qureg, index)
        return cComplex.real + 1j * cComplex.imag


getAmp = getStateVectoratIndex


class getDensityMatrixatRowColumn(_PYQUEST):
    r"""Get the value of the density matrix in qureg at row and column

    Args:
        qureg: a qureg containing a density matrix
        row: The row index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        column: The column index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg,
                         row: Union[int, Sequence[int]],
                         column: Union[int, Sequence[int]]) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            row: The row index either as an int of as a sequence
                of 0 and 1 referencing the corresponding basis state
            column: The column index either as an int of as a sequence
                of 0 and 1 referencing the corresponding basis state

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if hasattr(row, '__len__'):
            row = basis_state_to_index(row)
        if hasattr(column, '__len__'):
            column = basis_state_to_index(column)
        if qureg.isDensityMatrix:
            cComplex = quest.getDensityAmp(qureg, row, column)
            return cComplex.real + 1j * cComplex.imag
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")


getDensityAmp = getDensityMatrixatRowColumn


class getAbsoluteValSquaredatIndex(_PYQUEST):
    r"""Get the absulute value squared of a wavefunction/state vector in a quantum register at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state

    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction
            index: The index either as an int or as a sequence
                of 0 and 1 referencing the corresponding basis state

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        return quest.getProbAmp(qureg, index)


getProbAmp = getAbsoluteValSquaredatIndex


class getRealAmp(_PYQUEST):
    r"""Get the real value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction
            index: The index either as an int of as a sequence
                of 0 and 1 referencing the corresponding basis state

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        return quest.getRealAmp(qureg, index)


class getImagAmp(_PYQUEST):
    r"""Get the imaginary value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction
            index: The index either as an int or as a sequence
                of 0 and 1 referencing the corresponding basis state

        Returns:
            float

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        return quest.getImagAmp(qureg, index)


class getNumAmps(_PYQUEST):
    r"""Get the number of probability amplitudes in a qureg object, given by 2**number_qubits

    Args:
        qureg: a qureg containing a wavefunction or a density matrix

    """

    def call_interactive(self, qureg: tqureg) -> int:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or a density matrix

        Returns:
            int
        """
        if qureg.isDensityMatrix:
            return 2 ** cheat.getNumQubits()(qureg=qureg)
        else:
            return quest.getNumAmps(qureg)


class getNumQubits(_PYQUEST):
    r"""Get the number of qubits in a qureg object

    Args:
        qureg: a qureg containing a wavefunction or a density matrix

    """

    def call_interactive(self, qureg: tqureg) -> int:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or a density matrix

        Returns:
            int
        """
        return quest.getNumQubits(qureg)


class getExpectationValue(_PYQUEST):
    r"""Get the expectation value of an operator in matrix form

    Not implemented for static compilation

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        operator_matrix: The operator in matrix form

    """

    def call_interactive(self, qureg: tqureg, operator_matrix: np.ndarray) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or density matrix
            operator_matrix: The operator in matrix form

        Returns:
            float
        """
        density_matrix = getDensityMatrix()(qureg)
        return np.trace(operator_matrix @ density_matrix)


class getDensityMatrix(_PYQUEST):
    r"""Get the full density matrix of a quantum register

    Args:
        qureg: a qureg containing a wavefunction or density matrix

    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or density matrix

        Returns:
            np.ndarray
        """
        N = qureg.numQubitsRepresented
        density_matrix = np.zeros((2**N, 2**N), dtype=np.complex)
        if qureg.isDensityMatrix:
            for row in range(2**N):
                for column in range(2**N):
                    density_matrix[row, column] = getDensityMatrixatRowColumn()(qureg, row, column)
        else:
            state_vec = np.zeros((2**N, 1), dtype=np.complex)
            for index in range(2**N):
                state_vec[index] = getStateVectoratIndex()(qureg, index)
            density_matrix = state_vec @ state_vec.conj().T
        return density_matrix


class getOccupationProbability(_PYQUEST):
    r"""Get the full vector of occupation probabilities for each basis state

    Args:
        qureg: a qureg containing a wavefunction or density matrix

    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or density matrix

        Returns:
            np.ndarray
        """
        N = qureg.numQubitsRepresented
        prob_vec = np.zeros((2**N,), dtype=np.complex)
        if qureg.isDensityMatrix:
            for index in range(2**N):
                prob_vec[index] = (
                    (getDensityMatrixatRowColumn()(qureg, index, index))
                )
        else:
            for index in range(2**N):
                prob_vec[index] = getAbsoluteValSquaredatIndex()(qureg, index)
        return prob_vec


class getRepeatedMeasurement(_PYQUEST):
    r"""Get a measurement record of a repeated measurement

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        number_measurments: The number of measurement repetitions
        qubits_to_readout_index_dict: The mapping of qubit indices to the readout index
            {qubit_index: readout_index}

    Returns:
        A measurement record 2d numpy array with N columns, one or each qubit
        and number_measuremnet rows. Each row contains one complete measurement result
        for each qubit

    """

    def call_interactive(self, qureg: tqureg, number_measurements: int,
                         qubits_to_readout_index_dict: dict) -> np.ndarray:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction or density matrix
            number_measurements: The number of measurement repetitions
            qubits_to_readout_index_dict: The mapping of qubit indices to the readout index
                {qubit_index: readout_index}

        Returns:
            np.ndarray
        """
        N = qureg.numQubitsRepresented
        number_qubits = qureg.numQubitsRepresented
        probabilities = np.zeros((2**N, ))
        measurement_record = np.zeros((number_measurements, N))
        return_record = np.zeros((number_measurements, len(qubits_to_readout_index_dict)))
        if qureg.isDensityMatrix:
            for index in range(2**N):
                probabilities[index] = np.real(getDensityMatrixatRowColumn()(qureg, index, index))
        else:
            for index in range(2**N):
                probabilities[index] = np.real(getAbsoluteValSquaredatIndex()(qureg, index))
        outcomes = np.random.choice(range(2**N), number_measurements, p=probabilities)
        for co, out in enumerate(outcomes):
            measurement_record[co, :] = index_to_basis_state(out, number_qubits)
        for index in qubits_to_readout_index_dict.keys():
            output_index = qubits_to_readout_index_dict[index]
            return_record[:, output_index] = measurement_record[:, index]
        return return_record


class getStateVector(_PYQUEST):
    r"""Get the full statevector of a quantum register

    Args:
        qureg: a qureg containing a wavefunction

    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a wavefunction

        Returns:
            np.ndarray

        Raises:
            RuntimeError: Reference qureg has to be a wavefunction qureg
                but density matrix qureg was used
        """
        N = qureg.numQubitsRepresented
        state_vec = np.zeros((2**N,), dtype=np.complex)
        if qureg.isDensityMatrix:
            raise RuntimeError("Reference qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")
        else:
            for index in range(2**N):
                state_vec[index] = getStateVectoratIndex()(qureg, index)
        return state_vec


class getEnvironmentString(_PYQUEST):
    r"""Set input string to a string

    Set string contains the number of qubits in qureg, and the
    hardware facilities used

    Args:
        env: object representing the execution environment
        qureg: quantum register of which to query the simulating hardware
        string: string to be populated with the output string

    """

    def call_interactive(self, env: tquestenv, qureg: tqureg, string: str) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            env: object representing the execution environment
            qureg: quantum register of which to query the simulating hardware
            string: string to be populated with the output string

        Raises:
            NotImplementedError: Function added to QuEST.h but not QuEST.c
        """
        # quest.getEnvironmentString(env, qureg, string)
        raise NotImplementedError("Function added to QuEST.h but not QuEST.c")


class calcExpecPauliSum(_PYQUEST):
    r"""Get the expectation value of a sum of products of Pauli operators

    A sum of products of Pauli operators (including Identity) is measured.
    For each qubit a Pauli operator must be given in each sum term (can be identity)

    Args:
        qureg: quantum register that is measured
        paulis: List of Lists of Pauli operators in each product
                encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
        coefficients: coefficients of the sum
        workspace: A qureg of same type and size as input qureg, is used as temporary
                   work qureg

    """

    def call_interactive(self,
                         qureg: tqureg,
                         paulis: Sequence[Sequence[int]],
                         coefficients: Sequence[float],
                         workspace: tqureg
                         ) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register that is measured
            paulis: List of Lists of Pauli operators in each product
                    encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
            coefficients: coefficients of the sum
            workspace: A qureg of same type and size as input qureg, is used as temporary
                    work qureg

        Returns:
            float
        """
        flat_list = [p for product in paulis for p in product]
        pointer_paulis = ffi_quest.new("enum pauliOpType[{}]".format(len(flat_list)))
        for co, p in enumerate(flat_list):
            pointer_paulis[co] = p
        pointer = ffi_quest.new("{}[{}]".format(qreal, len(coefficients)))
        for co, c in enumerate(coefficients):
            pointer[co] = c
        return quest.calcExpecPauliSum(qureg,
                                       pointer_paulis,
                                       pointer,
                                       len(coefficients),
                                       workspace
                                       )


class calcExpecPauliProd(_PYQUEST):
    r"""Get the expectation value of a product of Pauli operators

    A product of Pauli operators (including Identity) is measured.
    For each qubit in qubits a Pauli operator must be given in each sum term (can be identity)

    Args:
        qureg: quantum register that is measured
        qubits: target qubits
        paulis: List of Pauli operators in the product
                encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
        workspace: A qureg of same type and size as input qureg, is used as temporary
                   work qureg

    """

    def call_interactive(self,
                         qureg: tqureg,
                         qubits: Sequence[int],
                         paulis: Sequence[Sequence[int]],
                         workspace: tqureg
                         ) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register that is measured
            qubits: target qubits
            paulis: List of Pauli operators in the product
                    encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
            workspace: A qureg of same type and size as input qureg, is used as temporary
                    work qureg

        Returns:
            float

        Raises:
            RuntimeError: Need the number of qubits and pauli products to be equal
        """
        if not len(qubits) == len(paulis):
            raise RuntimeError("Need the number of qubits and pauli products to be equal")
        flat_list = [p for p in paulis]
        pointer_paulis = ffi_quest.new("enum pauliOpType[{}]".format(len(flat_list)))
        for co, p in enumerate(flat_list):
            pointer_paulis[co] = p
        pointer_q = ffi_quest.new("int[{}]".format(len(qubits)))
        for co, q in enumerate(qubits):
            pointer_q[co] = q
        return quest.calcExpecPauliProd(qureg,
                                        pointer_q,
                                        pointer_paulis,
                                        len(qubits),
                                        workspace
                                        )


class calcExpecDiagonalOp(_PYQUEST):
    r"""Computes the expected value of the diagonal operator op for state qureg

    Since op is not necessarily Hermitian, the expected value may be a complex number.

    Args:
        qureg: quantum register that is measured
        operator: operator acting on a certain number of qubits (operator[0]: int)
            and in a certain QuEST environment (operator[1]: tquestenv)

    """

    def call_interactive(self,
                         qureg: tqureg,
                         operator: Tuple[int, tquestenv],
                         ) -> complex:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register that is measured
            operator: operator acting on a certain number of qubits (operator[0]: int)
                and in a certain QuEST environment (operator[1]: tquestenv)

        Returns:
            float
        """
        diagonal_op = quest.createDiagonalOp(operator[0], operator[1])
        complex_return = quest.calcExpecDiagonalOp(qureg,
                                                   diagonal_op)
        return complex(complex_return.real, complex_return.imag)


class calcExpecPauliHamil(_PYQUEST):
    r"""Get the expectation value of a product of Pauli operators

    Args:
        qureg: quantum register that is measured
        pauli_hamil: a PauliHamil created with createPauliHamil()
        workspace: A qureg of same type and size as input qureg, is used as temporary
                   work qureg

    """

    def call_interactive(self,
                         qureg: tqureg,
                         pauli_hamil: paulihamil,
                         workspace: tqureg
                         ) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register that is measured
            pauli_hamil: a PauliHamil created with createPauliHamil()
            workspace: A qureg of same type and size as input qureg, is used as temporary
                    work qureg

        Returns:
            float

        Raises:
            RuntimeError: Qureg and PauliHamil must be defined for the same number of qubits
        """
        if not (cheat.getNumQubits()(qureg=qureg) == pauli_hamil.numQubits):
            raise RuntimeError("Qureg and PauliHamil must be defined for the "
                               + "same number of qubits")
        return quest.calcExpecPauliHamil(qureg,
                                         pauli_hamil,
                                         workspace)


class calcHilbertSchmidtDistance(_PYQUEST):
    r"""Calculate the Hilbert-Schmidt distance between two density matrix quregs

    Args:
        qureg1: first quantum register
        qureg2: first quantum register

    """

    def call_interactive(self,
                         qureg1: tqureg,
                         qureg2: tqureg
                         ) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg1: first quantum register
            qureg2: first quantum register

        Returns:
            float
        """
        return quest.calcHilbertSchmidtDistance(qureg1,
                                                qureg2
                                                )


class calcDensityInnerProduct(_PYQUEST):
    r"""Calculate the Frobenius inner matrix product between two density matrix quregs

    Args:
        qureg1: first quantum register
        qureg2: first quantum register

    """

    def call_interactive(self,
                         qureg1: tqureg,
                         qureg2: tqureg
                         ) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg1: first quantum register
            qureg2: first quantum register

        Returns:
            float

        Raises:
            RuntimeError: Qureg1 has to be a density matrix qureg but wavefunction qureg was used
            RuntimeError: Qureg2 has to be a density matrix qureg but wavefunction qureg was used
        """
        if not qureg1.isDensityMatrix:
            raise RuntimeError("Qureg1 has to be a density matrix qureg but "
                               + "wavefunction qureg was used")
        elif not qureg2.isDensityMatrix:
            raise RuntimeError("Qureg2 has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

        return quest.calcDensityInnerProduct(qureg1,
                                             qureg2)


class seedQuEST(_PYQUEST):
    r"""Seed the Mersenne Twister used for random number generation with a user defined seed

    This function uses the mt19937 init_by_array function with numSeeds keys supplied by the user.
    Subsequent calls to mt19937 genrand functions will use this seeding.
    For a multi process code, the same seed is given to all process, therefore this seeding is only
    appropriate to use for functions e.g measure where all processes require the same random value.

    Args:
        seed_array: Array of integers to use as seed

    """

    def call_interactive(self, seed_array: Sequence[int]) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            seed_array: Array of integers to use as seed
        """
        quest.seedQuEST(seed_array, len(seed_array))


class seedQuESTDefault(_PYQUEST):
    r"""Seed the Mersenne Twister used for random number generation with an example default seed

    This function uses the mt19937 init_by_array function with numSeeds keys supplied by the user.
    Subsequent calls to mt19937 genrand functions will use this seeding.
    For a multi process code, the same seed is given to all process, therefore this seeding is only
    appropriate to use for functions e.g. measure where all processes require the same random value.

    """

    def call_interactive(self) -> None:
        r"""Interactive call of PyQuest-cffi"""
        quest.seedQuESTDefault()


class syncQuESTEnv(_PYQUEST):
    r"""Guarantees that all code up to the given point has been executed on all nodes

    Args:
        env: execution environment

    """

    def call_interactive(self, env: tquestenv) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            env: execution environment
        """
        quest.syncQuESTEnv(env)


class syncQuESTSuccess(_PYQUEST):
    r"""Performs a logical AND on all successCodes held by all processes.

    If any one process has a zero successCode all processes will return a zero success code.

    Args:
        success_code: 1 if process task succeeded, 0 if process task failed

    """

    def call_interactive(self, success_code: int) -> int:
        r"""Interactive call of PyQuest-cffi

        Args:
            success_code: 1 if process task succeeded, 0 if process task failed

        Returns:
            int
        """
        return quest.syncQuESTSuccess(success_code)


def basis_state_to_index(basis_state: Union[int, Sequence[int]], endianness: str = 'little') -> int:
    r"""Convert a basis state to index

    Converts a up/down representation of a basis state to the index of the basis
    depending on the Endian convention of the system

    Args:
        basis_state: a sequence of 0 and one representing the qubit basis stae
        endianness: 'big' or 'little' corresponding to the least significant bit
            being stored in the last or first element of the array respectively
            In little endianness the qubit 0 corresponds to the bit at index 0 of basis_state
            the qubit 1 to the bit at index 1 and so on.
            Note however, that when initialising basis_state = np.array([0, 1, 1, 0,...])
            the sequence of bits in the list needs to be inverted form the binary representation.
            For example 4 which will be 100 in binary
            would correspond to basis_state= np.array([0,0,1])

    Returns:
        int
    """
    b_state = np.array(basis_state)
    if endianness == 'little':
        index = np.sum(np.dot(np.array([2**k for k in range(0, len(b_state))]), b_state))
    elif endianness == 'big':
        index = np.sum(np.dot(np.array([2**k for k in range(len(b_state) - 1, -1, -1)]), b_state))

    return index


def index_to_basis_state(index: int,
                         num_qubits_represented: int,
                         endianness: str = 'little') -> List:
    r"""Converst index to basis state

    Converts an index of the basis to the  up/down representation of a basis state
    depending on the Endian convention of the system

    Args:
        index: the basis index
        num_qubits_represented: Number of qubits represented
        endianness: 'big' or 'little' corresponding to the least significant bit
            being stored in the last or first element of the array respectively
            In little endianness the qubit 0 corresponds to the bit at index 0 of basis_state
            the qubit 1 to the bit at index 1 and so on.
            Note however, that when initialising basis_state = np.array([0, 1, 1, 0,...])
            the sequence of bits in the list needs to be inverted form the binary representation.
            For example 4 which will be 100 in binary
            would correspond to basis_state= np.array([0,0,1])

    Returns:
        List: a sequence of 0 and one representing the qubit basis state

    """
    b_list = list()
    for k in range(0, num_qubits_represented):
        b_list.append((index // 2**k) % 2)
    if endianness == 'little':
        basis_state = b_list
    elif endianness == 'big':
        basis_state = list(reversed(b_list))
    return basis_state
