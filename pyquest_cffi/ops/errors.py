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

from pyquest_cffi.questlib import quest, _PYQUEST, ffi_quest, tqureg
import numpy as np
from typing import Sequence, Optional, Union, Tuple, List
from time import time
import warnings


class applyOneQubitDephaseError(_PYQUEST):
    r"""
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
    Args:
        qureg: a qureg containing a density matrix
        qubit: The qubit dephasing
        probability: The probability/ relative amplitude with which the dephasing occurs, probability needs to be smaller than 1/2
    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probability: float) -> None:
        if probability > 1/2:
            raise RuntimeError(
                "probability of twoQubitDepolariseErrors needs to be smaller that 1/2")
        if qureg.isDensityMatrix:
            return quest.applyOneQubitDephaseError(qureg, qubit, probability)
        else:
            warnings.warn('qureg1 has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: tqureg,
                    qubit: Union[int, str],
                    probability: Union[float, str]) -> List[str]:
        """
        Static call of oneQubitDephaseError

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            probability: probability of dephasing, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = " applyOneQubitDephaseError({qureg:s}, {qubit}, {probability});".format(
            qureg=qureg, qubit=qubit, probability=probability)
        return [call]

    def Kraus_matrices(self, probability, **kwargs) -> Tuple[np.ndarray]:
        """
        The definition of the Kraus Operator as a matrix
        """
        sqp = np.sqrt(probability)
        sqmp = np.sqrt(1-probability)
        dephasing = np.array([[sqp, 0], [0, -sqp]], dtype=np.complex)
        residual = np.array([[sqmp, 0], [0, sqmp]], dtype=np.complex)
        return (residual, dephasing)

    def superoperator_matrix(self, probability, **kwargs) -> np.ndarray:
        r"""
        The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}    

        where A and B are arbitrary matrices
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1-2*probability, 0, 0],
                           [0, 0, 1-2*probability, 0],
                           [0, 0, 0, 1]], dtype=np.complex)
        return matrix


class applyOneQubitDepolariseError(_PYQUEST):
    r"""
    Apply the depolarisation operators :math:`\sigma^x`, :math:`\sigma^y` and :math:`\sigma^z`  to a qubit q 
        with an evenly distributed probability p`

    .. math::
        \rho = (1-p) \rho + \frac{p}{3} \left( \sigma^x_q \rho \sigma^x_q + \sigma^y_q \rho \sigma^y_q +  \sigma^z_q \rho \sigma^z_q \right)

    Args:
        qureg: a qureg containing a density matrix
        qubit: The qubit depolarising
        probability: The probability/ relative amplitude with which the dephasing occurs, probability needs to be smaller than 3/4
    """

    def call_interactive(self, qureg: tqureg,
                         qubit: int,
                         probability: float) -> None:
        if probability > 3/4:
            raise RuntimeError(
                "probability of twoQubitDepolariseErrors needs to be smaller that 3/4")
        if qureg.isDensityMatrix:
            return quest.applyOneQubitDepolariseError(qureg, qubit, probability)
        else:
            warnings.warn('qureg1 has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: tqureg,
                    qubit: Union[int, str],
                    probability: Union[float, str]) -> List[str]:
        """
        Static call of applyOneQubitDepolariseError

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            probability: probability of depolarisation, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = " applyOneQubitDepolariseError({qureg:s}, {qubit}, {probability});".format(
            qureg=qureg, qubit=qubit, probability=probability)
        return [call]

    def Kraus_matrices(self, probability, **kwargs) -> Tuple[np.ndarray]:
        """
        The definition of the Kraus Operator as a matrix
        """
        sqp = np.sqrt(probability/3)
        sqmp = np.sqrt(1-probability)
        residual = np.array([[sqmp, 0],
                             [0, sqmp]], dtype=np.complex)
        depol1 = np.array([[0, sqp],
                           [sqp, 0]], dtype=np.complex)
        depol2 = np.array([[0, -1j*sqp],
                           [1j*sqp, 0]], dtype=np.complex)
        depol3 = np.array([[sqp, 0],
                           [0, -sqp]], dtype=np.complex)
        return (residual, depol1, depol2, depol3)

    def superoperator_matrix(self, probability, **kwargs) -> np.ndarray:
        r"""
        The definition of the Superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}    

        where A and B are arbitrary matrices
        """
        one_plus = 1-2/3*probability
        one_minus = 1-4/3*probability
        two_three = 2/3*probability
        matrix = np.array([[one_plus, 0, 0, two_three],
                           [0, one_minus, 0, 0],
                           [0, 0, one_minus, 0],
                           [two_three, 0, 0, one_plus]], dtype=np.complex)
        return matrix


class applyOneQubitDampingError(_PYQUEST):
    r"""
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
        if qureg.isDensityMatrix:
            return quest.applyOneQubitDampingError(qureg, qubit, probability)
        else:
            warnings.warn('qureg1 has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: tqureg,
                    qubit: Union[int, str],
                    probability: Union[float, str]) -> List[str]:
        """
        Static call of applyOneQubitDampingError

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            probability: probability of damping, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "applyOneQubitDampingError({qureg:s}, {qubit}, {probability});".format(
            qureg=qureg, qubit=qubit, probability=probability)
        return [call]

    def Kraus_matrices(self, probability, **kwargs) -> Tuple[np.ndarray]:
        """
        The definition of the Kraus Operator as a matrix
        """
        sqp = np.sqrt(probability)
        sqmp = np.sqrt(1-probability)
        damping = np.array([[0, sqp], [0, 0]], dtype=np.complex)
        residual = np.array([[1, 0], [0, sqmp]], dtype=np.complex)
        return (residual, damping)

    def superoperator_matrix(self, probability, **kwargs) -> np.ndarray:
        r"""
        The definition of the Superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}    

        where A and B are arbitrary matrices
        """
        sqmp = np.sqrt(1-probability)
        matrix = np.zeros((16, 16), dtype=np.complex)
        matrix = np.array([[1, 0, 0, probability],
                           [0, sqmp, 0, 0],
                           [0, 0, sqmp, 0],
                           [0, 0, 0, 1-probability]], dtype=np.complex)
        return matrix


class applyTwoQubitDepolariseError(_PYQUEST):
    r"""
    Apply any tensor product of two operators :math:`U` :math:`\sigma^x`, :math:`\sigma^y` and :math:`\sigma^z`  to two qubits q1 and q2 
        with an evenly distributed probability p`

    .. math::
        \rho &= (1-p) \rho + \frac{p}{15} \sum_{A, B \in \{ I, \sigma^x, \sigma^y, \sigma^z\}}  A_{q1}B_{q2} \rho B_{q2}A_{q1} \\
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
        if probability > 15/16:
            raise RuntimeError(
                "probability of twoQubitDepolariseErrors needs to be smaller that 15/16")
        if qureg.isDensityMatrix:
            return quest.applyTwoQubitDeolariseError(qureg, qubit1, qubit2, probability)
        else:
            warnings.warn('qureg has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: tqureg,
                    qubit1: Union[int, str],
                    qubit2: Union[int, str],
                    probability: Union[float, str]) -> List[str]:
        """
        Static call of applyTwoQubitDepolariseError

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit1: The first qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            qubit2: The second qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            probability: probability of depolarisation, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = " applyOneQubitDepolariseError({qureg:s}, {qubit1}, {qubit2}, {probability});".format(
            qureg=qureg, qubit1=qubit1, qubit2=qubit2, probability=probability)
        return [call]

    def superoperator_matrix(self, probability, **kwargs) -> np.ndarray:
        r"""
        The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}    

        where A and B are arbitrary matrices
        """
        one_plus = 1-2/3*probability
        one_minus = 1-4/3*probability
        two_three = 2/3*probability
        matrix = np.zeros((16, 16), dtype=np.complex)
        matrix = np.array([[one_plus, 0, 0, two_three],
                           [0, one_minus, 0, 0],
                           [0, 0, one_minus, 0],
                           [two_three, 0, 0, one_plus]], dtype=np.complex)
        return None


class applyTwoQubitDephaseError(_PYQUEST):
    r"""
    Apply the dephasing :math:`\sigma^z` operator to two qubits q1 and q2 with probability p
        Can also be expressed as a Kraus operator :math:`\mathcal{K}`

     .. math::
        \rho &= (1-p) \rho + \frac{p}{3} \left( \sigma^z_{q1} \rho \sigma^z_{q1} + \sigma^z_{q2} \rho \sigma^z_{q2} +  \sigma^z_{q1}\sigma^z_{q2} \rho \sigma^z_{q2} \sigma^z_{q1} \right)\\
        \rho &= \mathcal{K} \rho \mathcal{K} 

    Args:
        qureg: a qureg containing a density matrix
        qubit1: The first qubit dephasing
        qubit2: The second qubit dephasing
        probability: The probability/ relative amplitude with which the dephasing occurs, probability needs to be smaller than 3/4
    """

    def call_interactive(self, qureg: tqureg,
                         qubit1: int,
                         qubit2: int,
                         probability: float) -> None:
        if probability > 3/4:
            raise RuntimeError(
                "probability of twoQubitDepolariseErrors needs to be smaller that 3/4")
        if qureg.isDensityMatrix:
            return quest.applyTwoQubitDephaseError(qureg, qubit1, qubit2, probability)
        else:
            warnings.warn('qureg has to be a density matrix  qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None

    def call_static(self, qureg: tqureg,
                    qubit1: Union[int, str],
                    qubit2: Union[int, str],
                    probability: Union[float, str]) -> List[str]:
        """
        Static call of applyTwoQubitDephaseError

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit1: The first qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            qubit2: The second qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            probability: probability of dephasing, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = " applyTwoQubitDephaseError({qureg:s}, {qubit1}, {qubit2}, {probability});".format(
            qureg=qureg, qubit1=qubit1, qubit2=qubit2, probability=probability)
        return [call]

    def Kraus_matrices(self, probability, **kwargs) -> Tuple[np.ndarray]:
        """
        The definition of the Kraus Operator as a matrix
        """
        sqp = np.sqrt(probability/3)
        sqmp = np.sqrt(1-probability)
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

    def superoperator_matrix(self, probability, **kwargs) -> np.ndarray:
        r"""
        The definition of the superoperator acting on the density matrix written as a vector

        .. math::
            \rho = A \rho B \\
            \vec{\rho} = \mathcal{L} \vec{\rho}    

        where A and B are arbitrary matrices
        """

        matrix = np.zeros((16, 16), dtype=np.complex)
        for ci in range(0, 16):
            matrix[ci, ci] = 1 if (ci % 4) == 1 else 1-2(probability)
        return matrix
