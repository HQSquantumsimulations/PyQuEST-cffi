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

from pyquest_cffi.questlib import quest, _PYQUEST, ffi_quest, tquestenv, tqureg
import numpy as np
from typing import Sequence, Optional, Union, List
import warnings
import uuid


class calcFidelity(_PYQUEST):
    r"""
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

    def call_interactive(self, qureg: tqureg, qureg_reference: tqureg) -> Optional[float]:
        if not qureg_reference.isDensityMatrix:
            return quest.calcFidelity(qureg, qureg_reference)
        else:
            warnings.warn('reference qureg has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: str, qureg_reference: str, readout: str) -> List[str]:
        """
        Static call of calcFidelity

        Args:
            qureg: The name of the previously created quantum register as a string
            qureg_reference: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
        """
        call = "{readout} = calcFidelity({qureg:s}, {qureg_reference});".format(
            readout=readout, qureg=qureg, qureg_reference=qureg_reference)
        return [call]


class calcInnerProduct(_PYQUEST):
    r"""
   Calculate the inner-product/overlap of two wavefunction quregs:

    .. math::
         \left\langle \psi_{qureg1} | \psi_{qureg2} \right \rangle

    Args:
        qureg1: a qureg containing a wavefunction
        qureg2: a qureg containing a wavefunction
        readout: the readout register for static compilation
    """

    def call_interactive(self, qureg1: tqureg, qureg2: tqureg) -> Optional[float]:
        if qureg1.isDensityMatrix:
            warnings.warn('qureg1 has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        elif qureg2.isDensityMatrix:
            warnings.warn('qureg2 has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        else:
            return quest.calcInnerProduct(qureg1, qureg2)

    def call_static(self, qureg1: str, qureg2: str, readout: str) -> List[str]:
        """
        Static call of calcInnerProduct

        Args:
            qureg1: The name of the previously created quantum register as a string
            qureg2: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
        """
        call = "{readout} = calcInnerProduct({qureg1:s}, {qureg2:s});".format(
            readout=readout, qureg1=qureg1, qureg2=qureg2)
        return [call]


class calcProbOfOutcome(_PYQUEST):
    r"""
   Calculate the probability that qubit #qubit of qureg is measured in state outcome:

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        qubit: the index of the qubit for which the probability is determined
        outcome: the outcome of the measurement
        readout: the readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg, qubit: int, outcome: int) -> float:
        return quest.calcProbOfOutcome(qureg, qubit, outcome)

    def call_static(self, qureg: str, qubit: Union[str, int],
                    outcome: Union[str, int], readout: str) -> List[str]:
        """
        Static call of calcProbOfOutcome

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            outcome: outcome of the measurement, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            readout: The name of the previously created C-variable of type qreal
        """
        if isinstance(outcome, int) and outcome not in [0, 1]:
            outcome = bool(outcome)
            warnings.warn('outcome is not in [0, 1] casting to bool automatically', RuntimeWarning)
        call = "{readout} = calcProbOfOutcome({qureg:s}, {qubit}, {outcome});".format(
            readout=readout, qureg=qureg, qubit=qubit, outcome=outcome)
        return [call]


class calcPurity(_PYQUEST):
    r"""
   Calculate the purity of a density matrix in qureg:

    .. math::
         \mathcal{Tr}\left(\rho^2\right)

    Args:
        qureg: a qureg containing a density matrix
        readout: the readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg) -> Optional[float]:
        if qureg.isDensityMatrix:
            return quest.calcPurity(qureg)
        else:
            warnings.warn('qureg has to be a density matrix qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: str, readout: str) -> List[str]:
        """
        Static call of calcPutiry

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
        """
        call = "{readout} = calcPurity({qureg:s});".format(
            readout=readout, qureg=qureg)
        return [call]


class calcTotalProb(_PYQUEST):
    r"""
    Check physicallity of system by calculating probability of system to be in any state.
    In other words check that trace of density matrix or norm of state vector is one.

    Args:
        qureg: a qureg containing a density matrix or wavefucntion
        readout: the readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg) -> float:
        return quest.calcTotalProb(qureg)

    def call_static(self, qureg: str, readout: str) -> List[str]:
        """
        Static call of calcProbOfOutcome

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
        """
        call = "{readout} =  calcTotalProb({qureg:s});".format(
            readout=readout, qureg=qureg)
        return [call]


class getStateVectoratIndex(_PYQUEST):
    r"""
    Get the value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: the readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> Optional[float]:
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            warnings.warn('qureg has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        else:
            cComplex = quest.getAmp(qureg, index)
            return cComplex.real+1j*cComplex.imag

    def call_static(self, qureg: str, index: Union[int, str, Sequence[int]], readout: str) -> List[str]:
        """
        Static call of getStateVectoratIndex

        Args:
            qureg: The name of the previously created quantum register as a string
            index: The index in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            readout: The name of the previously created C-variable of type qreal
        """
        if not isinstance(index, str) and hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        call = "{readout} =  getAmp({qureg:s}, {index});".format(
            readout=readout, qureg=qureg, index=index)
        return [call]


getAmp = getStateVectoratIndex


class getDensityMatrixatRowColumn(_PYQUEST):
    r"""
    Get the value of the density matrix in qureg at row and column

    Args:
        qureg: a qureg containing a wavefunction
        row: The row index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        column: The column index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg,
                         row: Union[int, Sequence[int]],
                         column: Union[int, Sequence[int]]) -> Optional[float]:
        if hasattr(row, '__len__'):
            row = basis_state_to_index(row)
        if hasattr(column, '__len__'):
            column = basis_state_to_index(column)
        if qureg.isDensityMatrix:
            cComplex = quest.getDensityAmp(qureg, row, column)
            return cComplex.real+1j*cComplex.imag
        else:
            warnings.warn('qureg1 has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: str,
                    row: Union[int, str, Sequence[int]],
                    column: Union[int, str, Sequence[int]],
                    readout: str) -> List[str]:
        """
        Static call of getDensityMatrixatRowColumn

        Args:
            qureg: The name of the previously created quantum register as a string
            row: The row in the density matrix, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            column: The column in the density matrix, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            readout: The name of the previously created C-variable of type qreal
        """
        if not isinstance(row, str) and hasattr(row, '__len__'):
            row = basis_state_to_index(row)
        if not isinstance(column, str) and hasattr(column, '__len__'):
            column = basis_state_to_index(column)
        call = "{readout} =  getDensityAmp({qureg:s}, {row}, {column});".format(
            readout=readout, qureg=qureg, row=row, column=column)
        return [call]


getDensityAmp = getDensityMatrixatRowColumn


class getAbsoluteValSquaredatIndex(_PYQUEST):
    r"""
    Get the absulute value squared of a wavefunction/state vector in a quantum register at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state
    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            raise RuntimeError('getAbsoluteValSquaredatIndex is only defined for statevector qureg')
        return quest.getProbAmp(qureg, index)

    def call_static(self, qureg: str, index: Union[int, str, Sequence[int]], readout: str) -> List[str]:
        """
        Static call of getAbsoluteValSquaredatIndex

        Args:
            qureg: The name of the previously created quantum register as a string
            index: The index in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            readout: The name of the previously created C-variable of type qreal
        """
        if not isinstance(index, str) and hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        call = "{readout} = getProbAmp({qureg:s}, {index});".format(
            readout=readout, qureg=qureg, index=index)
        return [call]


getProbAmp = getAbsoluteValSquaredatIndex


class getRealAmp(_PYQUEST):
    r"""
    Get the real value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int of as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> Optional[float]:
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            return None
        else:
            return quest.getRealAmp(qureg, index)

    def call_static(self, qureg: str, index: Union[int, str, Sequence[int]], readout: str) -> List[str]:
        """
        Static call of getRealAmp

        Args:
            qureg: The name of the previously created quantum register as a string
            index: The index in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            readout: The name of the previously created C-variable of type qreal
        """
        if not isinstance(index, str) and hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        call = "readout = getRealAmp({qureg:s}, {index});".format(
            readout=readout, qureg=qureg, index=index)
        return [call]


class getImagAmp(_PYQUEST):
    r"""
    Get the imaginary value of a wavefunction/state vector in qureg at index

    Args:
        qureg: a qureg containing a wavefunction
        index: The index either as an int or as a sequence
            of 0 and 1 referencing the corresponding basis state
        readout: The readout register for static compilation
    """

    def call_interactive(self, qureg: tqureg, index: Union[int, Sequence[int]]) -> float:
        if hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        if qureg.isDensityMatrix:
            return None
        else:
            return quest.getImagAmp(qureg, index)

    def call_static(self, qureg: str, index: Union[int, str, Sequence[int]], readout: str) -> List[str]:
        """
        Static call of getImagAmp

        Args:
            qureg: The name of the previously created quantum register as a string
            index: The index in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
                    if Sequence of int is assumed to be a basis state representation and converted to index
            readout: The name of the previously created C-variable of type qreal
        """
        if not isinstance(index, str) and hasattr(index, '__len__'):
            index = basis_state_to_index(index)
        call = "{readout} = getImagAmp({qureg:s}, {index});".format(
            readout=readout, qureg=qureg, index=index)
        return [call]


class getExpectationValue(_PYQUEST):
    r"""
    Get the expectation value of an operator in matrix form
    Not implemented for static compilation

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        operator_matrix: The operator in matrix form
    """

    def call_interactive(self, qureg: tqureg, operator_matrix: np.ndarray) -> float:
        density_matrix = getDensityMatrix()(qureg)
        return np.trace(operator_matrix @ density_matrix)

    def call_static(self, **kwargs) -> List[str]:
        """
        Not implemented
        """
        raise NotImplementedError


class getDensityMatrix(_PYQUEST):
    r"""
    Get the full density matrix of a quantum register

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        readout: The readout register for static compilation mode
        number_qubits: The number of qubits in the quantum register, required for compilation mode
        endianness: The endianness of the returned density matrix for compilation mode
    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
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

    def call_static(self, qureg: str, readout: str,
                    number_qubits: int, endianness='little') -> List[str]:
        """
        Static call of getDensityMatrix

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
            number_qubits: The number of qubits int the density matrix
            endianness: The endianness of the returned density matrix
        """
        N = number_qubits
        lines = []
        for little_endian_row in range(2**N):
            for little_endian_column in range(2**N):
                if endianness == 'big':
                    row = 2**N-1-little_endian_row
                    column = 2**N-1-little_endian_column
                else:
                    row = little_endian_row
                    column = little_endian_column
                lines.extend(getDensityMatrixatRowColumn(interactive=False)(
                    qureg,
                    row=row,
                    column=column,
                    readout=readout+'[{}]'.format(row*2**N+column)))
        return lines


class getOccupationProbability(_PYQUEST):
    r"""
    Get the full vector of occupation probabilities for each basis state

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        readout: The readout register for static compilation mode
        number_qubits: The number of qubits in the quantum register, required for compilation mode
        endianness: The endianness of the returned density matrix for compilation mode
    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
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

    def call_static(self, qureg: str, readout: str,
                    number_qubits: int, endianness='little',
                    is_density_matrix=False) -> List[str]:
        """
        Static call of getDensityMatrix

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
            number_qubits: The number of qubits int the density matrix
            endianness: The endianness of the returned density matrix
            is_density_matrix: Does the qureg contain density matrix (True)
                or statvector (False)
        """
        N = number_qubits
        lines = []
        for little_endian_index in range(2**N):
            t = '{}'.format(uuid.uuid4().hex)
            if endianness == 'big':
                index = 2**N-1-little_endian_index
            else:
                index = little_endian_index
            if is_density_matrix:
                temp_readout = 'ro_{}_{}'.format(index, t)
                lines.append('Complex {}'.format(temp_readout))
                lines.extend(getDensityMatrixatRowColumn(interactive=False)(
                    qureg,
                    row=index,
                    column=index,
                    readout=temp_readout))
                lines.append(
                    '{readout}[{index}] = {temp_readout}.real*{temp_readout}.real+'
                    + '{temp_readout}.imag*{temp_readout}.imag'.format(
                        readout=readout, index=index, temp_readout=temp_readout))
            else:
                lines.extend(getAbsoluteValSquaredatIndex(interactive=False)(
                    qureg,
                    row=index,
                    readout=readout+'[{}]'.format(index)))
        return lines


class getRepeatedMeasurement(_PYQUEST):
    r"""
    Get a measurement record of a repeated measurement

    Args:
        qureg: a qureg containing a wavefunction or density matrix
        number_measurments: The number of measurement repetitions
        qubits_to_readout_index_dict: The mapping of qubit indices to the readout index
            {qubit_index: readout_index}
        number_qubits: the number of qubits represendet by qureg, required for compilation mode
        is_density_matrix: is qureg a statevector or density matrix qureg, r
            equired for compilation mode

    Returns:
        A measurement record 2d numpy array with N columns, one or each qubit
        and number_measuremnet rows. Each row contains one complete measurement result
        for each qubit
    """

    def call_interactive(self, qureg: tqureg, number_measurements: int,
                         qubits_to_readout_index_dict: dict) -> np.ndarray:
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

    def call_static(self,
                    qureg: str,
                    readout: str,
                    number_qubits: int,
                    number_measurements: int,
                    is_density_matrix: bool = False) -> List[str]:
        """
        Static call of getRepeatedMeasurement

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
            number_qubits: The number of qubits int the density matrix
            number_qubits: The number of repeated measurements
            is_density_matrix: Does Qureg contain density_matrix (True) or state vector (False)
        """
        call = 'qreal * outcomeprob; qreal = zeroProb;'
        for cqb in range(number_qubits):
            for cm in range(number_measurements):
                if is_density_matrix:
                    call += (
                        'zeroProb = densmatr_calcProbOfOutcome({qureg}, {qubit}, 0);'.format(
                            qureg=qureg, qubit=cqb))
                else:
                    call += (
                        'zeroProb = statevec_calcProbOfOutcome({qureg}, {qubit}, 0);'.format(
                            qureg=qureg, qubit=cqb))
                call += (
                    '{}[{}] = generateMeasurementOutcome(zeroProb, outcomeprob);'.format(
                        readout, cqb+cm*number_qubits))
        return [call]


class getStateVector(_PYQUEST):
    r"""
    Get the full statevector of a quantum register

    Args:
        qureg: a qureg containing a wavefunction
        readout: The readout register for static compilation
        number_qubits: the number of qubits represendet by qureg, required for compilation mode
        is_density_matrix: is qureg a statevector or density matrix qureg, r
            equired for compilation mode
    """

    def call_interactive(self, qureg: tqureg) -> np.ndarray:
        N = qureg.numQubitsRepresented
        state_vec = np.zeros((2**N,), dtype=np.complex)
        if qureg.isDensityMatrix:
            warnings.warn('reference qureg has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        else:
            for index in range(2**N):
                state_vec[index] = getStateVectoratIndex()(qureg, index)
        return state_vec

    def call_static(self, qureg: str, readout: str, number_qubits: int, endianness='little') -> List[str]:
        """
        Static call of getStateVector

        Args:
            qureg: The name of the previously created quantum register as a string
            readout: The name of the previously created C-variable of type qreal
            number_qubits: The number of qubits int the density matrix
            endianness: The endianness of the returned density matrix
        """
        N = number_qubits
        lines = []
        for little_endian_index in range(2**N):
            if endianness == 'big':
                index = 2**N-1-little_endian_index
            else:
                index = little_endian_index
            lines.extend(getStateVectoratIndex()(interactive=False)(
                qureg,
                index=index,
                readout=readout+'[{}]'.format(index)))
        return lines


def basis_state_to_index(basis_state, endianness='little'):
    """
    Converts a up/down representation of a basis state to the index of the basis
    dependin on the Endian convention of the system

    Args:
        basis_state: a sequence of 0 and one representing the qubit basis stae
        enidanness: 'big' or 'little' corresponding to the least significant bit
            being stored in the last or first element of the array respectively
            In little endianness the qubit 0 corresponds to the bit at index 0 of basis_state
            the qubit 1 to the bit at index 1 and so on.
            Note however, that when initialising basis_state = np.array([0, 1, 1, 0,...])
            the sequence of bits in the list needs to be inverted form the binary representation.
            For example 4 which will be 100 in binary would correspond to basis_state= np.array([0,0,1])
    """
    b_state = np.array(basis_state)
    if endianness == 'little':
        index = np.sum(np.dot(np.array([2**k for k in range(0, len(b_state))]), b_state))
    elif endianness == 'big':
        index = np.sum(np.dot(np.array([2**k for k in range(len(b_state)-1, -1, -1)]), b_state))

    return index


def index_to_basis_state(index, num_qubits_represented, endianness='little'):
    """
    Converts an index of the basis to the  up/down representation of a basis state
    depending on the Endian convention of the system

    Args:
        index: the basis index
        enidanness: 'big' or 'little' corresponding to the least significant bit
            being stored in the last or first element of the array respectively
            In little endianness the qubit 0 corresponds to the bit at index 0 of basis_state
            the qubit 1 to the bit at index 1 and so on.
            Note however, that when initialising basis_state = np.array([0, 1, 1, 0,...])
            the sequence of bits in the list needs to be inverted form the binary representation.
            For example 4 which will be 100 in binary would correspond to basis_state= np.array([0,0,1])

    Returns:
        a sequence of 0 and one representing the qubit basis state
    """
    b_list = list()
    for k in range(0, num_qubits_represented):
        b_list.append((index // 2**k) % 2)
    if endianness == 'little':
        basis_state = b_list
    elif endianness == 'big':
        basis_state = list(reversed(b_list))
    return basis_state
