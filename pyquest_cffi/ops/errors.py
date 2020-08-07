"""Error operation in PyQuest-cffi"""
# Copyright 2019 HQS Quantum Simulations GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pyquest_cffi.questlib import quest, _PYQUEST, tqureg, ffi_quest
import numpy as np
from typing import Tuple, Sequence, List
import warnings
from pyquest_cffi import cheat


class mixDensityMatrix(_PYQUEST):
    r"""Modifies qureg to become (1-probability)*qureg + probability*qureg_other

    Both registers must be equal-dimension density matrices, and prob must be in [0, 1].

    Args:
        qureg: quantum register to be modified
        probability: the probability of qureg_other in the modified qureg
        qureg_other: the quantum register to be mixed into qureg

    """

    def call_interactive(self, qureg: tqureg,
                         probability: float,
                         qureg_other: tqureg) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register to be modified
            probability: the probability of qureg_other in the modified qureg
            qureg_other: the quantum register to be mixed into qureg

        Raises:
            RuntimeError: Both quregs must be density matrix,
                but at least one of them is a wavefunction
            RuntimeError: Qureg and Qureg_other must be defined for the same number of qubits
        """
        if not qureg.isDensityMatrix or not qureg_other.isDensityMatrix:
            raise RuntimeError("Both quregs must be density matrix, "
                               + "but at least one of them is a wavefunction")
        elif not (cheat.getNumQubits()(qureg=qureg) == cheat.getNumQubits()(qureg=qureg_other)):
            raise RuntimeError("Qureg and Qureg_other must be defined "
                               + "for the same number of qubits")
        else:
            quest.mixDensityMatrix(qureg, probability, qureg_other)

    def Kraus_matrices(self, probability: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    def superoperator_matrix(self, probability: float, **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class mixDephasing(_PYQUEST):
    r"""One qubit dephasing error

    Apply the dephasing :math:`\sigma^z` operator to a qubit q with probability p
        Can also be expressed as a Kraus operator :math:`\mathcal{K}`

    .. math::
        \rho &= (1-p) \rho + p \sigma^z_q \rho \sigma^z_q \\
        \rho &= \mathcal{K} \rho \mathcal{K} \\
        \vec{\rho} &= \mathcal{L} \vec{\rho} \\
        \mathcal{L} &= \begin{pmatrix}
            1 & 0 & 0 & 0\\
             0 & 1-2p & 0 & 0\\
             0 & 0 & 1-2p & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probability: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit: The qubit dephasing
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2

        Raises:
            RuntimeError: Probability of mixDephasing needs to be smaller that 1/2
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if probability > 1 / 2:
            raise RuntimeError(
                "Probability of mixDephasing needs to be smaller that 1/2")
        if qureg.isDensityMatrix:
            quest.mixDephasing(qureg, qubit, probability)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

    def Kraus_matrices(self, probability: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        sqp = np.sqrt(probability)
        sqmp = np.sqrt(1 - probability)
        dephasing = np.array([[sqp, 0], [0, -sqp]], dtype=np.complex)
        residual = np.array([[sqmp, 0], [0, sqmp]], dtype=np.complex)
        return (residual, dephasing)

    def superoperator_matrix(self, probability: float, **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1 - 2 * probability, 0, 0],
                           [0, 0, 1 - 2 * probability, 0],
                           [0, 0, 0, 1]], dtype=np.complex)
        return matrix


class mixDepolarising(_PYQUEST):
    r"""One qubit depolarisation error

    Apply the depolarisation operators :math:`\sigma^x`, :math:`\sigma^y` and :math:`\sigma^z`
    to a qubit q with an evenly distributed probability p`

    .. math::
        \rho = (1-p) \rho + \frac{p}{3} \left( \sigma^x_q \rho \sigma^x_q
                          + \sigma^y_q \rho \sigma^y_q +  \sigma^z_q \rho \sigma^z_q \right)

    Args:
        qureg: a qureg containing a density matrix
        qubit: The qubit depolarising
        probability: The probability/ relative amplitude with which the dephasing occurs,
                     probability needs to be smaller than 3/4

    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probability: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit: The qubit depolarising
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2

        Raises:
            RuntimeError: Probability of mixDepolarising needs to be smaller that 3/4
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if probability > 3 / 4:
            raise RuntimeError(
                "Probability of mixDepolarising needs to be smaller that 3/4")
        if qureg.isDensityMatrix:
            quest.mixDepolarising(qureg, qubit, probability)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

    def Kraus_matrices(self,
                       probability: float,
                       **kwargs
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        sqp = np.sqrt(probability / 3)
        sqmp = np.sqrt(1 - probability)
        residual = np.array([[sqmp, 0],
                             [0, sqmp]], dtype=np.complex)
        depol1 = np.array([[0, sqp],
                           [sqp, 0]], dtype=np.complex)
        depol2 = np.array([[0, -1j * sqp],
                           [1j * sqp, 0]], dtype=np.complex)
        depol3 = np.array([[sqp, 0],
                           [0, -sqp]], dtype=np.complex)
        return (residual, depol1, depol2, depol3)

    def superoperator_matrix(self, probability: float, **kwargs) -> np.ndarray:
        r"""The definition of the Superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        one_plus = 1 - 2 / 3 * probability
        one_minus = 1 - 4 / 3 * probability
        two_three = 2 / 3 * probability
        matrix = np.array([[one_plus, 0, 0, two_three],
                           [0, one_minus, 0, 0],
                           [0, 0, one_minus, 0],
                           [two_three, 0, 0, one_plus]], dtype=np.complex)
        return matrix


class mixDamping(_PYQUEST):
    r"""One qubit damping error

    Apply a pure damping error corresponding to zero temperature environments

    .. math::
        \rho &= \mathcal{K} \rho \mathcal{K}\\

    Args:
        qureg: a qureg containing a density matrix
        qubit: The damped qubit
        probability: The probability/ relative amplitude with which the dephasing occurs

    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probability: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit: The damped qubit
            probability: The probability/relative amplitude with which the dephasing occurs

        Raises:
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if qureg.isDensityMatrix:
            quest.mixDamping(qureg, qubit, probability)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

    def Kraus_matrices(self, probability: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        sqp = np.sqrt(probability)
        sqmp = np.sqrt(1 - probability)
        damping = np.array([[0, sqp], [0, 0]], dtype=np.complex)
        residual = np.array([[1, 0], [0, sqmp]], dtype=np.complex)
        return (residual, damping)

    def superoperator_matrix(self, probability: float, **kwargs) -> np.ndarray:
        r"""The definition of the Superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        sqmp = np.sqrt(1 - probability)
        matrix = np.zeros((16, 16), dtype=np.complex)
        matrix = np.array([[1, 0, 0, probability],
                           [0, sqmp, 0, 0],
                           [0, 0, sqmp, 0],
                           [0, 0, 0, 1 - probability]], dtype=np.complex)
        return matrix


class mixTwoQubitDepolarising(_PYQUEST):
    r"""Two qubit depolarisation error

    Apply any tensor product of two operators :math:`U` :math:`\sigma^x`, :math:`\sigma^y`
    and :math:`\sigma^z`  to two qubits q1 and q2 with an evenly distributed probability p`

    .. math::
        \rho &= (1-p) \rho + \frac{p}{15} \sum_{A, B \in \{ I, \sigma^x, \sigma^y, \sigma^z\}}
                A_{q1}B_{q2} \rho B_{q2}A_{q1} \\
        \rho &= \mathcal{K} \rho \mathcal{K}

    Args:
        qureg: a qureg containing a density matrix
        qubit1: The first qubit dephasing
        qubit2: The second qubit dephasing
        probability: The probability/ relative amplitude with which the depolarisation occurs.
            Needs to be smaller than :math:`\frac{15}{16}`

    """

    def call_interactive(self, qureg: tqureg,
                         qubit1: int,
                         qubit2: int,
                         probability: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit1: The first qubit dephasing
            qubit2: The second qubit dephasing
            probability: The probability/ relative amplitude with which the depolarisation occurs.
                Needs to be smaller than :math:`\frac{15}{16}`

        Raises:
            RuntimeError: Probability of twoQubitDepolariseErrors needs to be smaller that 15/16
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if probability > 15 / 16:
            raise RuntimeError(
                "Probability of twoQubitDepolariseErrors needs to be smaller that 15/16")
        if qureg.isDensityMatrix:
            quest.mixTwoQubitDepolarising(qureg, qubit1, qubit2, probability)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

    def superoperator_matrix(self, probability: float, **kwargs) -> None:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class mixTwoQubitDephasing(_PYQUEST):
    r"""Two qubit dephasing error

    Apply the dephasing :math:`\sigma^z` operator to two qubits q1 and q2 with probability p
        Can also be expressed as a Kraus operator :math:`\mathcal{K}`

     .. math::
        \rho &= (1-p) \rho + \frac{p}{3} \left( \sigma^z_{q1} \rho \sigma^z_{q1}
              + \sigma^z_{q2} \rho \sigma^z_{q2}
              +  \sigma^z_{q1}\sigma^z_{q2} \rho \sigma^z_{q2} \sigma^z_{q1} \right)\\
        \rho &= \mathcal{K} \rho \mathcal{K}

    Args:
        qureg: a qureg containing a density matrix
        qubit1: The first qubit dephasing
        qubit2: The second qubit dephasing
        probability: The probability/ relative amplitude with which the dephasing occurs,
                     probability needs to be smaller than 3/4

    """

    def call_interactive(self, qureg: tqureg,
                         qubit1: int,
                         qubit2: int,
                         probability: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit1: The first qubit dephasing
            qubit2: The second qubit dephasing
            probability: The probability/ relative amplitude with which the dephasing occurs,
                        probability needs to be smaller than 3/4

        Raises:
            RuntimeError: Probability of twoQubitDepolariseErrors needs to be smaller that 3/4
            RuntimeError: Qureg has to be a density matrix qureg but wavefunction qureg was used
        """
        if probability > 3 / 4:
            raise RuntimeError(
                "Probability of twoQubitDepolariseErrors needs to be smaller that 3/4")
        if qureg.isDensityMatrix:
            quest.mixTwoQubitDephasing(qureg, qubit1, qubit2, probability)
        else:
            raise RuntimeError("Qureg has to be a density matrix qureg but "
                               + "wavefunction qureg was used")

    def Kraus_matrices(self,
                       probability: float,
                       **kwargs
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        sqp = np.sqrt(probability / 3)
        sqmp = np.sqrt(1 - probability)
        residual = np.array([[sqmp, 0, 0, 0],
                             [0, sqmp, 0, 0],
                             [0, 0, sqmp, 0],
                             [0, 0, 0, sqmp]], dtype=np.complex)
        dephasing1 = np.array([[sqp, 0, 0, 0],
                               [0, 0, sqp, 0],
                               [0, 0, -sqp, 0],
                               [0, 0, 0, -sqp]], dtype=np.complex)
        dephasing2 = np.array([[sqp, 0, 0, 0],
                               [0, 0, -sqp, 0],
                               [0, 0, sqp, 0],
                               [0, 0, 0, -sqp]], dtype=np.complex)
        dephasing3 = np.array([[-sqp, 0, 0, 0],
                               [0, 0, sqp, 0],
                               [0, 0, sqp, 0],
                               [0, 0, 0, -sqp]], dtype=np.complex)
        return (residual, dephasing1, dephasing2, dephasing3)

    def superoperator_matrix(self, probability: float, **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}

        where A and B are arbitrary matrices

        Args:
            probability: The probability/ relative amplitude with which the dephasing occurs,
                probability needs to be smaller than 1/2
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()
        matrix = np.zeros((16, 16), dtype=np.complex)
        for ci in range(0, 16):
            matrix[ci, ci] = 1 if (ci % 4) == 1 else 1 - 2 * (probability)
        return matrix


class applyOneQubitDephaseError(mixDephasing):
    r"""One Qubit Dephasing - deprecated"""

    def __init__(self, *args, **kwargs) -> None:
        r"""Initialisation

        Args:
            *args: Arguments
            **kwargs: Additional keyword arguments
        """
        warnings.warn(
            "applyOneQubitDephaseError will be removed in future versions, use mixDephasing",
            DeprecationWarning)
        super().__init__(*args, **kwargs)


class applyOneQubitDepolariseError(mixDepolarising):
    r"""One Qubit Depolarisation - deprecated"""

    def __init__(self, *args, **kwargs) -> None:
        r"""Initialisation

        Args:
            *args: Arguments
            **kwargs: Additional keyword arguments
        """
        warnings.warn(
            "applyOneQubitDepolariseError will be removed in future versions, use mixDepolarising",
            DeprecationWarning)
        super().__init__(*args, **kwargs)


class applyOneQubitDampingError(mixDamping):
    r"""One Qubit Damping - deprecated"""

    def __init__(self, *args, **kwargs) -> None:
        r"""Initialisation

        Args:
            *args: Arguments
            **kwargs: Additional keyword arguments
        """
        warnings.warn(
            "applyOneQubitDampingError will be removed in future versions, use mixDamping",
            DeprecationWarning)
        super().__init__(*args, **kwargs)


class applyTwoQubitDephaseError(mixTwoQubitDephasing):
    r"""Two Qubit Dephasing - deprecated"""

    def __init__(self, *args, **kwargs) -> None:
        r"""Initialisation

        Args:
            *args: Arguments
            **kwargs: Additional keyword arguments
        """
        warnings.warn(
            "applyTwoQubitDephaseError will be removed in future versions,"
            + " use mixTwoQubitDephasing",
            DeprecationWarning)
        super().__init__(*args, **kwargs)


class applyTwoQubitDepolariseError(mixTwoQubitDepolarising):
    r"""Two Qubit Depolarisation - deprecated"""

    def __init__(self, *args, **kwargs) -> None:
        r"""Initialisation

        Args:
            *args: Arguments
            **kwargs: Additional keyword arguments
        """
        warnings.warn(
            "applyTwoQubitDepolariseError will be removed in future versions,"
            + " use mixTwoQubitDepolarising",
            DeprecationWarning)
        super().__init__(*args, **kwargs)


class mixMultiQubitKrausMap(_PYQUEST):
    r"""Error affecting multiple qubits

    An error acting on multiple qubtis simultaneously is defined by a set of Kraus operators

    Args:
        qureg: a qureg containing a density matrix
        qubits: The qubits the Kraus operators are acting on
        operators: The Kraus operators

    """

    def call_interactive(self, qureg: tqureg,
                         qubits: Sequence[int],
                         operators: Sequence[np.ndarray],
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubits: The qubits the Kraus operators are acting on
            operators: The Kraus operators

        Raises:
            RuntimeError: Number of target qubits and dimension of Kraus operators mismatch
            RuntimeError: Not a valid Kraus map
        """
        for op in operators:
            if 2**len(qubits) != op.shape[0] or 2**len(qubits) != op.shape[1]:
                raise RuntimeError("Number of target qubits"
                                   + " and dimension of Kraus operators mismatch")
        operator_sum = np.sum([op.conjugate().T @ op for op in operators], axis=0)
        if not np.array_equal(operator_sum, np.eye(operators[0].shape[0])):
            raise RuntimeError("Not a valid Kraus map")
        operator_pointers = ffi_quest.new("ComplexMatrixN[{}]".format(len(operators)))
        for co, op in enumerate(operators):
            operator_pointers[co] = quest.createComplexMatrixN(qureg.numQubitsRepresented)
            for i in range(op.shape[0]):
                for j in range(op.shape[0]):
                    operator_pointers[co].real[i][j] = np.real(op[i, j])
                    operator_pointers[co].imag[i][j] = np.imag(op[i, j])
        pointer_q = ffi_quest.new("int[{}]".format(len(qubits)))
        for co, qubit in enumerate(qubits):
            pointer_q[co] = qubit
        quest.mixMultiQubitKrausMap(
            qureg,
            pointer_q,
            len(qubits),
            operator_pointers,
            len(operators))
        for p in operator_pointers:
            quest.destroyComplexMatrixN(p)

    def Kraus_matrices(self, operators: Sequence[np.ndarray], **kwargs) -> Sequence[np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        return operators

    def superoperator_matrix(self, operators: Sequence[np.ndarray], **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.zeros((2 * operators[0].shape[0], 2 * operators[0].shape[0]), dtype=np.complex)
        for op in operators:
            matrix += np.kron(op, op.conjugate().T)
        return matrix


class mixTwoQubitKrausMap(_PYQUEST):
    r"""Error affecting two qubits

    An error acting on two qubtis simultaneously is defined by a set of Kraus operators

    Args:
        qureg: a qureg containing a density matrix
        target_qubit_1: The least significant qubit the Kraus operators are acting on
        target_qubit_2: The most significant qubit the Kraus operators are acting on
        operators: The Kraus operators

    """

    def call_interactive(self, qureg: tqureg,
                         target_qubit_1: Sequence[int],
                         target_qubit_2: Sequence[int],
                         operators: Sequence[np.ndarray],
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            target_qubit_1: The least significant qubit the Kraus operators are acting on
            target_qubit_2: The most significant qubit the Kraus operators are acting on
            operators: The Kraus operators

        Raises:
            RuntimeError: Number of target qubits and dimension of Kraus operators mismatch
            RuntimeError: Not a valid Kraus map
        """
        for op in operators:
            if op.shape[0] != 4 or op.shape[1] != 4:
                raise RuntimeError("Number of target qubits"
                                   + " and dimension of Kraus operators mismatch")
        operator_sum = np.sum([op.conjugate().T @ op for op in operators], axis=0)
        if not np.array_equal(operator_sum, np.eye(4)):
            raise RuntimeError("Not a valid Kraus map")
        operator_pointers = ffi_quest.new("ComplexMatrix4[{}]".format(len(operators)))
        for co, op in enumerate(operators):
            for i in range(4):
                for j in range(4):
                    operator_pointers[co].real[i][j] = np.real(op[i, j])
                    operator_pointers[co].imag[i][j] = np.imag(op[i, j])
        quest.mixTwoQubitKrausMap(
            qureg,
            target_qubit_1,
            target_qubit_2,
            operator_pointers,
            len(operators))

    def Kraus_matrices(self, operators: Sequence[np.ndarray], **kwargs) -> Sequence[np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        return operators

    def superoperator_matrix(self, operators: Sequence[np.ndarray], **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.zeros((2 * operators[0].shape[0], 2 * operators[0].shape[0]), dtype=np.complex)
        for op in operators:
            matrix += np.kron(op, op.conjugate().T)
        return matrix


class mixKrausMap(_PYQUEST):
    r"""General error affecting one qubit

    An error acting on one qubit is defined by a set of Kraus operators

    Args:
        qureg: a qureg containing a density matrix
        qubit: The qubit the Kraus operators are acting on
        operators: The Kraus operators

    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         operators: Sequence[np.ndarray],
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit: The qubit the Kraus operators are acting on
            operators: The Kraus operators

        Raises:
            RuntimeError: Number of target qubits and dimension of Kraus operators mismatch
            RecursionError: Not a valid Kraus map
        """
        for op in operators:
            if op.shape[0] != 2 or op.shape[1] != 2:
                raise RuntimeError("Number of target qubits"
                                   + " and dimension of Kraus operators mismatch")
        operator_sum = np.sum([op.conjugate().T @ op for op in operators], axis=0)
        if not np.array_equal(operator_sum, np.eye(2)):
            raise RecursionError("Not a valid Kraus map")
        operator_pointers = ffi_quest.new("ComplexMatrix2[{}]".format(len(operators)))
        for co, op in enumerate(operators):
            for i in range(2):
                for j in range(2):
                    operator_pointers[co].real[i][j] = np.real(op[i, j])
                    operator_pointers[co].imag[i][j] = np.imag(op[i, j])
        quest.mixKrausMap(
            qureg,
            qubit,
            operator_pointers,
            len(operators))

    def Kraus_matrices(self, operators: Sequence[np.ndarray], **kwargs) -> Sequence[np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        return operators

    def superoperator_matrix(self, operators: Sequence[np.ndarray], **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        Args:
            operators: The Kraus operators
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.zeros((2 * operators[0].shape[0], 2 * operators[0].shape[0]), dtype=np.complex)
        for op in operators:
            matrix += np.kron(op, op.conjugate().T)
        return matrix


class mixPauli(_PYQUEST):
    r"""Error on qubit defined by three Pauli operator weights

    Args:
        qureg: a qureg containing a density matrix
        qubit: The qubit the Pauli operators are acting on
        probX: The probability that Pauli X is acting on qubit as Kraus operator
        probY: The probability that Pauli Y is acting on qubit as Kraus operator
        probZ: The probability that Pauli Z is acting on qubit as Kraus operator

    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probX: float,
                         probY: float,
                         probZ: float,
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: a qureg containing a density matrix
            qubit: The qubit the Pauli operators are acting on
            probX: The probability that Pauli X is acting on qubit as Kraus operator
            probY: The probability that Pauli Y is acting on qubit as Kraus operator
            probZ: The probability that Pauli Z is acting on qubit as Kraus operator
        """
        quest.mixPauli(
            qureg,
            qubit,
            probX,
            probY,
            probZ)

    def Kraus_matrices(self,
                       probX: float,
                       probY: float,
                       probZ: float,
                       **kwargs) -> List[np.ndarray]:
        r"""The definition of the Kraus Operator as a matrix

        Args:
            probX: The probability that Pauli X is acting on qubit as Kraus operator
            probY: The probability that Pauli Y is acting on qubit as Kraus operator
            probZ: The probability that Pauli Z is acting on qubit as Kraus operator
            **kwargs: Additional keyword arguments

        Returns:
            Tuple[np.ndarray]
        """
        operators = [None, None, None]
        operators[0] = probX * np.array([[0, 1], [1, 0]], dtype=complex)
        operators[1] = probY * np.array([[0, -1j], [1j, 0]], dtype=complex)
        operators[2] = probZ * np.array([[1, 0], [0, -1]], dtype=complex)
        return operators

    def superoperator_matrix(self,
                             probX: float,
                             probY: float,
                             probZ: float,
                             **kwargs) -> np.ndarray:
        r"""The definition of the superoperator acting on the density matrix written as a vector

        Args:
            probX: The probability that Pauli X is acting on qubit as Kraus operator
            probY: The probability that Pauli Y is acting on qubit as Kraus operator
            probZ: The probability that Pauli Z is acting on qubit as Kraus operator
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        operators = self.Kraus_matrices(probX, probY, probZ)
        matrix = np.zeros((2 * operators[0].shape[0], 2 * operators[0].shape[0]), dtype=np.complex)
        for op in operators:
            matrix += np.kron(op, op.conjugate().T)
        return matrix
