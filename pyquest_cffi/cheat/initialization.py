"""Initialisation objects in PyQuest-cffi"""
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
from typing import Union, List, Sequence
import numpy as np
from pyquest_cffi.questlib import quest, _PYQUEST, tqureg, paulihamil, ffi_quest, qreal
from pyquest_cffi import cheat


class initZeroState(_PYQUEST):
    r"""Initialise zero state in quantum register

    Args:
        qureg: quantum register

    """

    def call_interactive(self, qureg: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
        """
        quest.initZeroState(qureg)


class initPlusState(_PYQUEST):
    r"""Initialise plus state in quantum register

    Args:
        qureg: quantum register

    """

    def call_interactive(self, qureg: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
        """
        quest.initPlusState(qureg)


class initClassicalState(_PYQUEST):
    r"""Initialise classical state in quantum register

    Initialise classical state, a classical integer in binary representation in the quantum register

    Args:
        qureg: The quantum register
        int: The integer that is initialised in binary representation in the quantum register

    """

    def call_interactive(self, qureg: tqureg, state: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            state: The integer that is initialised in binary representation in the quantum register
        """
        quest.initClassicalState(qureg, state)


class initPureState(_PYQUEST):
    r"""Initialise pure state in quantum register

    Initialises a pure state in one wavefunction quantum register
    based on the pure state of another quantum register

    Args:
        qureg_a: Quantum register that is initialised
        qureg_b: Quantum register that contains the reference pure state

    """

    def call_interactive(self, qureg_a: tqureg, qureg_b: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg_a: Quantum register that is initialised
            qureg_b: Quantum register that contains the reference pure state
        """
        quest.initPureState(qureg_a, qureg_b)


class initStateFromAmps(_PYQUEST):
    r"""Initialise a state (from amplitudes) in quantum register

    Initialise a wavefunction in a quantum register based on the real
    and imaginary parts of the statevector

    Args:
        qureg: the quantum register
        reals: The real parts of the statevector
        imags: The imaginary parts of the statevector

    """

    def call_interactive(self,
                         qureg: tqureg,
                         reals: Union[np.ndarray, List[float]],
                         imags: Union[np.ndarray, List[float]]
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: the quantum register
            reals: The real parts of the statevector
            imags: The imaginary parts of the statevector

        Raises:
            RuntimeError: Size of reals and imags needs to match
            RuntimeError: Shape of reals and imags for wavefunction should be: (1, 2**qubits)
            RuntimeError: Shape of reals and imags for density matrix should be:
                (2**qubits, 2**qubits) OR (4**qubits, 1)
            RuntimeError: Shape of reals and imags for density matrix should be:
                (2**qubits, 2**qubits) OR (4**qubits, 1)
            RuntimeError: Need to set real and imaginary amplitudes for each qubit:
                2**qubits for wavefunction qureg, 4**qubits for density matrix qureg
        """
        reals = list(reals)
        imags = list(imags)
        assert len(reals) == np.max(np.shape(reals))
        assert len(imags) == np.max(np.shape(imags))
        size_amps = np.size(np.array(reals))
        if not size_amps == np.size(np.array(imags)):
            raise RuntimeError("Size of reals and imags needs to match")
        num_qubits = cheat.getNumQubits()(qureg=qureg)

        if size_amps == 2**num_qubits:
            if qureg.isDensityMatrix:
                raise RuntimeError("Shape of reals and imags for wavefunction should be: "
                                   + "(1, 2**qubits)")
            pointer_reals = ffi_quest.new("{}[{}]".format(qreal, len(reals)))
            for co, c in enumerate(reals):
                pointer_reals[co] = c
            pointer_imags = ffi_quest.new("{}[{}]".format(qreal, len(imags)))
            for co, c in enumerate(imags):
                pointer_imags[co] = c
            quest.initStateFromAmps(qureg, pointer_reals, pointer_imags)
        elif size_amps == 4**num_qubits:
            if not qureg.isDensityMatrix:
                raise RuntimeError("Shape of reals and imags for density matrix should be:"
                                   + "(2**qubits, 2**qubits) OR (4**qubits, 1)")
            size_amps_rows = np.size(np.array(reals), 0)
            size_amps_columns = np.size(np.array(reals), 1)
            if (not (size_amps_rows == np.size(np.array(imags), 0))
                    or not (size_amps_columns == np.size(np.array(imags), 1))):
                raise RuntimeError("Size of reals and imags needs to match")
            cheat.initZeroState()(qureg=qureg)
            if (size_amps_rows == size_amps_columns == 2**num_qubits):
                cheat.setDensityAmps()(qureg=qureg, reals=reals, imags=imags)
            elif size_amps_rows == 4**num_qubits:
                reals = np.array(reals).reshape((2**num_qubits, 2**num_qubits))
                imags = np.array(imags).reshape((2**num_qubits, 2**num_qubits))
                cheat.setDensityAmps()(qureg=qureg, reals=reals, imags=imags)
            else:
                raise RuntimeError("Shape of reals and imags should be (2**qubits, 2**qubits) OR "
                                   + "(4**qubits, 1)")
        else:
            raise RuntimeError("Need to set real and imaginary amplitudes for each qubit: "
                               + "2**qubits for wavefunction, 4**qubits for density matrix")


class initDebugState(_PYQUEST):
    r"""Initialise debug state in quantum register

    Args:
        qureg: quantum register

    """

    def call_interactive(self, qureg: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
        """
        quest.initDebugState(qureg)


class initBlankState(_PYQUEST):
    r"""Initialise classical zero state in quantum register

    Args:
        qureg: qureg that is set to zero

    """

    def call_interactive(self,
                         qureg: tqureg,
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: qureg that is set to zero
        """
        quest.initBlankState(qureg)


class initPauliHamil(_PYQUEST):
    r"""Initialise a PauliHamil instance

    Args:
        pauli_hamil: PauliHamil instance to initialise
        coeffs: array of coefficients
        codes: array of Pauli codes

    """

    def call_interactive(self,
                         pauli_hamil: paulihamil,
                         coeffs: Sequence[float],
                         codes: Sequence[Sequence[int]]) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            pauli_hamil: PauliHamil instance to initialise
            coeffs: array of coefficients
            codes: array of Pauli codes

        Raises:
            RuntimeError: Need one coeff and one set of codes per qubit of PauliHamil and
                need one term in each set of codes per PauliProduct
        """
        num_qubits = pauli_hamil.numQubits
        num_pauliprods = pauli_hamil.numSumTerms

        if (num_qubits == len(coeffs) == np.size(np.array(codes), 0)
                and num_pauliprods == np.size(np.array(codes), 1)):
            pointer_coeffs = ffi_quest.new("{}[{}]".format(qreal, len(coeffs)))
            for co, c in enumerate(coeffs):
                pointer_coeffs[co] = c
            flat_list = [p for product in codes for p in product]
            pointer_codes = ffi_quest.new("enum pauliOpType[{}]".format(len(flat_list)))
            for co, p in enumerate(flat_list):
                pointer_codes[co] = p

            quest.initPauliHamil(pauli_hamil, pointer_coeffs, pointer_codes)
        else:
            raise RuntimeError("Need one coeff and one set of codes per qubit of PauliHamil and "
                               + "need one term in each set of codes per PauliProduct")


class setAmps(_PYQUEST):
    r"""Class setting the Amplitudes in statevector

    Set the values of elements of the statevector in a quantum register

    Args:
        qureg: The quantum register
        startind: The index of the first element of the statevector that is set
        reals: the new real values of the elements of the statevector
               between startind and startind+numamps
        imgas: the new imaginary values of the elements of the statevector
               between startind and startind+numamps
        numaps: the number of new values that are set in the statevector

    """

    def call_interactive(self,
                         qureg: tqureg,
                         startind: int,
                         reals: Union[np.ndarray, List[float]],
                         imags: Union[np.ndarray, List[float]],
                         numamps: int
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: The quantum register
            startind: The index of the first element of the statevector that is set
            reals: the new real values of the elements of the statevector
                between startind and startind+numamps
            imags: the new imaginary values of the elements of the statevector
                between startind and startind+numamps
            numamps: the number of new values that are set in the statevector

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        reals = list(reals)
        imags = list(imags)
        assert len(reals) == np.max(np.shape(reals))
        assert len(imags) == np.max(np.shape(imags))
        assert len(reals) == numamps
        assert len(reals) == numamps
        if qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a wavefunction qureg but "
                               + "density matrix qureg was used")

        quest.setAmps(qureg, startind, reals, imags, numamps)


# can't find it in the API


class setDensityAmps(_PYQUEST):
    r"""Class setting the Amplitudes in density matrix

    Set the values of elements of the vector representation
    of the density matrix in a quantum register

    Args:
        qureg: The quantum register of a density matrix
        reals: the new real values of the elements of the density matrix
               between startind and startind+numamps
        imags: the new imaginary values of the elements of the density matrix
               between startind and startind+numamps

    """

    def call_interactive(self,
                         qureg: tqureg,
                         reals: Union[np.ndarray, List[List[float]]],
                         imags: Union[np.ndarray, List[List[float]]],
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: The quantum register of a density matrix
            reals: the new real values of the elements of the density matrix
                between startind and startind+numamps
            imags: the new imaginary values of the elements of the density matrix
                between startind and startind+numamps

        Raises:
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        reals = list(reals)
        imags = list(imags)
        num_amps = cheat.getNumAmps()(qureg=qureg)

        if not qureg.isDensityMatrix:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")
        for i in range(num_amps):
            j = num_amps * i
            reals_flat = reals[i]
            imags_flat = imags[i]
            pointer_reals = ffi_quest.new("{}[{}]".format(qreal, len(reals_flat)))
            for co, c in enumerate(reals_flat):
                pointer_reals[co] = c
            pointer_imags = ffi_quest.new("{}[{}]".format(qreal, len(imags_flat)))
            for co, c in enumerate(imags_flat):
                pointer_imags[co] = c
            quest.statevec_setAmps(qureg, j, pointer_reals, pointer_imags, num_amps)
            # quest.setDensityAmps(qureg, pointer_reals, pointer_imags)  --> not yet in the API


class setWeightedQureg(_PYQUEST):
    r"""Class setting a qureg as a weighted sum of two quregs

    Set the values of elements of the vector representation
    of the density matrix in a quantum register

    Args:
        fac1: prefactor of first qureg in sum
        qureg1: first qureg in sum
        fac2: prefactor of second qureg in sum
        qureg2: second qureg in sum
        facout: prefactor of output qureg
        quregout: output qureg

    """

    def call_interactive(self,
                         fac1: complex,
                         qureg1: tqureg,
                         fac2: complex,
                         qureg2: tqureg,
                         facout: complex,
                         quregout: tqureg
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            fac1: prefactor of first qureg in sum
            qureg1: first qureg in sum
            fac2: prefactor of second qureg in sum
            qureg2: second qureg in sum
            facout: prefactor of output qureg
            quregout: output qureg

        Raises:
            RuntimeError: Qureg has to be a wavefunction qureg but density matrix qureg was used
        """
        if qureg1.isDensityMatrix and qureg2.isDensityMatrix and quregout.isDensityMatrix:
            quest.setWeightedQureg((fac1.real, fac1.imag), qureg1,
                                   (fac2.real, fac2.imag), qureg2,
                                   (facout.real, facout.imag), quregout)
        elif (
            not qureg1.isDensityMatrix and not qureg2.isDensityMatrix and (
                not quregout.isDensityMatrix)):
            quest.setWeightedQureg((fac1.real, fac1.imag), qureg1,
                                   (fac2.real, fac2.imag), qureg2,
                                   (facout.real, facout.imag), quregout)
        else:
            raise RuntimeError("All three quregs need to be of the same type, so all three "
                               + "wavefunctions OR all three density matrices")
