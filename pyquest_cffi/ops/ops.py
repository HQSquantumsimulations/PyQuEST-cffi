"""Python classes for Quest functions"""
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

from pyquest_cffi.questlib import (
    quest, _PYQUEST, ffi_quest, qreal, tqureg, tquestenv, paulihamil
)
import numpy as np
from typing import Sequence, Optional, Tuple
from pyquest_cffi import cheat


class hadamard(_PYQUEST):
    r"""Implements Hadamard gate

    .. math::
        U = \frac{1}{\sqrt{2}} \begin{pmatrix}
        1 & 1\\
        1 & -1
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.hadamard(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        return matrix


class pauliX(_PYQUEST):
    r"""Implements Pauli X gate

    .. math::
        U =  \begin{pmatrix}
        0 & 1\\
        1 & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.pauliX(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return matrix


class pauliY(_PYQUEST):
    r"""Implements Pauli Y gate

    .. math::
        U =  \begin{pmatrix}
        0 & -i\\
        i & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.pauliY(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return matrix


class pauliZ(_PYQUEST):
    r"""Implements Pauli Z gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & -1
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.pauliZ(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        return matrix


class sGate(_PYQUEST):
    r"""Implements S gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & i
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.sGate(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        return matrix


class tGate(_PYQUEST):
    r"""Implements T gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & e^{i \frac{\pi}{4}}
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
        """
        quest.tGate(qureg, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        return matrix


class compactUnitary(_PYQUEST):
    r"""Implements general unitary gate U in compact notation

    .. math::
        U = \begin{pmatrix}
        \alpha & -\beta^{*}\\
        \beta & \alpha^{*}
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        alpha: complex parameter :math:`\alpha` of the unitary matrix
        beta: complex parameter :math:`\beta` of the unitary matrix

    """

    def call_interactive(self, qureg: tqureg, qubit: int, alpha: complex, beta: complex) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            alpha: complex parameter :math:`\alpha` of the unitary matrix
            beta: complex parameter :math:`\beta` of the unitary matrix

        Raises:
            RuntimeError: compactUnitary needs parameters |alpha|**2+|beta|**2 == 1
        """
        if not np.isclose(np.abs(alpha)**2 + np.abs(beta)**2, 1):
            raise RuntimeError("compactUnitary needs parameters |alpha|**2+|beta|**2 == 1")
        else:
            calpha = ffi_quest.new("Complex *")
            calpha.real = np.real(alpha)
            calpha.imag = np.imag(alpha)
            cbeta = ffi_quest.new("Complex *")
            cbeta.real = np.real(beta)
            cbeta.imag = np.imag(beta)
            quest.compactUnitary(qureg, qubit, calpha[0], cbeta[0])

    def matrix(self, alpha: complex, beta: complex, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            alpha: complex parameter :math:`\alpha` of the unitary matrix
            beta: complex parameter :math:`\beta` of the unitary matrix
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]], dtype=complex)
        return matrix


class phaseShift(_PYQUEST):
    r"""Implements pure :math:`\left|1 \right\rangle` phase shift gate

    .. math::
        U = \begin{pmatrix}
        1 & 0\\
        0 & e^{i \theta}
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the ro

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the ro
        """
        if not (0 <= theta and theta <= 2 * np.pi):
            theta = np.mod(theta, 2 * np.pi)
        quest.phaseShift(qureg, qubit, theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the ro
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
        return matrix


class rotateAroundAxis(_PYQUEST):
    r"""Implements rotation around arbitraty axis on Bloch sphere

    .. math::
        U = \begin{pmatrix}
             \cos(\frac{\theta}{2}) & 0\\
         0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            -i \sin(\frac{\theta}{2}) v_z  &  \sin(\frac{\theta}{2}) \left(-i v_x - v_y \right) \\
         \sin(\frac{\theta}{2}) \left(-i v_x + v_y \right) & i \sin(\frac{\theta}{2}) v_z)
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation
        vector: Direction of the rotation axis, unit-vector

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float, vector: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
            vector: Direction of the rotation axis, unit-vector

        Raises:
            RuntimeError: vector needs to be a three component numpy array and unit-vector
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        if not (vector.shape == (3,) and np.isclose(np.linalg.norm(vector), 1)):
            raise RuntimeError("vector needs to be a three component numpy array and unit-vector")
        else:
            vec = ffi_quest.new("Vector *")
            vec.x = vector[0]
            vec.y = vector[1]
            vec.z = vector[2]
            quest.rotateAroundAxis(qureg,
                                   qubit,
                                   theta,
                                   vec[0])

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the rotation
            vector: Direction of the rotation axis, unit-vector
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[c - 1j * s * vz, s * (-1j * vx - vy)],
                           [s * (-1j * vx + vy), c + 1j * s * vz]], dtype=complex)
        return matrix


class rotateAroundSphericalAxis(_PYQUEST):
    r"""Implements rotation around an axis given in spherical coordinates

    .. math::
        U &= \begin{pmatrix}
             \cos(\frac{\theta}{2}) & 0\\
         0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            -i \sin(\frac{\theta}{2}) v_z  &  \sin(\frac{\theta}{2}) \left(-i v_x - v_y \right) \\
         \sin(\frac{\theta}{2}) \left(-i v_x + v_y \right) & i \sin(\frac{\theta}{2}) v_z)
        \end{pmatrix}\\
        v_x &= \sin(\theta_{sph}) \cos(\phi_{sph})\\
        v_y &= \sin(\theta_{sph}) \sin(\phi_{sph})\\
        v_z &= \cos(\theta_{sph})

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation
        spherical_theta: Rotation axis, unit-vector spherical coordinates theta
        spherical_phi: Rotation axis, unit-vector spherical coordinates phi

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float,
                         spherical_theta: float, spherical_phi: float,) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
            spherical_theta: Rotation axis, unit-vector spherical coordinates theta
            spherical_phi: Rotation axis, unit-vector spherical coordinates phi
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)

        vec = ffi_quest.new("Vector *")
        vec.x = np.sin(spherical_theta) * np.cos(spherical_phi)
        vec.y = np.sin(spherical_theta) * np.sin(spherical_phi)
        vec.z = np.cos(spherical_theta)
        quest.rotateAroundAxis(qureg,
                               qubit,
                               theta,
                               vec[0])

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the rotation
            vector: Direction of the rotation axis, unit-vector
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[c - 1j * s * vz, s * (-1j * vx - vy)],
                           [s * (-1j * vx + vy), c + 1j * s * vz]], dtype=complex)
        return matrix


class rotateX(_PYQUEST):
    r"""Implements :math:`e^{-i \frac{\theta}{2} \sigma^x}` XPower gate

    .. math::
        U = \begin{pmatrix}
        \cos(\frac{\theta}{2}) & 0\\
         0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
        0  &  -i \sin(\frac{\theta}{2})   \\
         -i \sin(\frac{\theta}{2})  & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.rotateX(qureg,
                      qubit,
                      theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        return matrix


class rotateY(_PYQUEST):
    r"""Implements :math:`e^{-i \frac{\theta}{2} \sigma^y}` XPower gate

    .. math::
        U = \begin{pmatrix}
             \cos(\frac{\theta}{2}) & 0\\
         0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            0  &  - \sin(\frac{\theta}{2})   \\
          \sin(\frac{\theta}{2})  & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.rotateY(qureg,
                      qubit,
                      theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[c, -s], [s, c]], dtype=complex)
        return matrix


class rotateZ(_PYQUEST):
    r"""Implements :math:`e^{-i \frac{\theta}{2} \sigma^z}` XPower gate

    .. math::
        U = \begin{pmatrix}
             \cos(\frac{\theta}{2}) & 0\\
         0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
             - i \sin(\frac{\theta}{2}) & 0   \\
        0 &  i \sin(\frac{\theta}{2})
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`

    """

    def call_interactive(self, qureg: tqureg, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.rotateZ(qureg,
                      qubit,
                      theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle theta of the rotation, in interval 0 to 2 :math:`2 \pi`
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)
        return matrix


class unitary(_PYQUEST):
    r"""Implements an arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate

    """

    def call_interactive(self, qureg: tqureg, qubit: int, matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            matrix: Unitary matrix of the one qubit gate

        Raises:
            RuntimeError: matrix needs to be a (2, 2) unitary numpy array
        """
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("matrix needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            for i in range(2):
                for j in range(2):
                    mat.real[i][j] = np.real(matrix[i, j])
                    mat.imag[i][j] = np.imag(matrix[i, j])
            quest.unitary(qureg,
                          qubit,
                          mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: Unitary matrix of the one qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


# Controlled and other Two-Qubit Operations


class twoQubitUnitary(_PYQUEST):
    r"""General two qubit unitary gate

    Implements a general two-qubit gate defined by a matrix
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        target_qubit_1: least significant qubit
        target_qubit_2: most sifnificant qubit
        matrix: 4 by 4 matrix that defines the two qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         target_qubit_1: int,
                         target_qubit_2: int,
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            target_qubit_1: least significant qubit
            target_qubit_2: most sifnificant qubit
            matrix: 4 by 4 matrix that defines the two qubit gate
        """
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.twoQubitUnitary(qureg,
                              target_qubit_1,
                              target_qubit_2,
                              mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class controlledTwoQubitUnitary(_PYQUEST):
    r"""Controlled two qubit unitary gate

    Implements a general two-qubit gate defined by a matrix controlled by a third qubit
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        control: controll qubit
        target_qubit_1: least significant qubit
        target_qubit_2: most sifnificant qubit
        matrix: 4 by 4 matrix that defines the two qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg, control: int,
                         target_qubit_1: int,
                         target_qubit_2: int,
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: controll qubit
            target_qubit_1: least significant qubit
            target_qubit_2: most sifnificant qubit
            matrix: 4 by 4 matrix that defines the two qubit gate
        """
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat[0].real[i][j] = np.real(matrix[i, j])
                mat[0].imag[i][j] = np.imag(matrix[i, j])
        quest.controlledTwoQubitUnitary(qureg,
                                        control,
                                        target_qubit_1,
                                        target_qubit_2,
                                        mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|xy> |1xy> -> |1> U |xy>

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        dim = matrix.shape[0]
        return np.block([[np.eye(dim), np.zeros((dim, dim))],
                         [np.zeros((dim, dim)), matrix]])


class controlledCompactUnitary(_PYQUEST):
    r"""Implements a controlled general unitary gate U in compact notation

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \alpha & -\beta^{*}\\
        0 & 0 & \beta & \alpha^{*}
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        alpha: complex parameter :math:`\alpha` of the unitary matrix
        beta: complex parameter :math:`\beta` of the unitary matrix

    """

    def call_interactive(self,
                         qureg: tqureg,
                         control: int,
                         qubit: int,
                         alpha: complex,
                         beta: complex) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            alpha: complex parameter :math:`\alpha` of the unitary matrix
            beta: complex parameter :math:`\beta` of the unitary matrix

        Raises:
            RuntimeError: compactUnitary needs parameters |alpha|**2+|beta|**2 == 1
        """
        if not np.isclose(np.abs(alpha)**2 + np.abs(beta)**2, 1):
            raise RuntimeError("compactUnitary needs parameters |alpha|**2+|beta|**2 == 1")
        else:
            calpha = ffi_quest.new("Complex *")
            calpha.real = np.real(alpha)
            calpha.imag = np.imag(alpha)
            cbeta = ffi_quest.new("Complex *")
            cbeta.real = np.real(beta)
            cbeta.imag = np.imag(beta)
            quest.controlledCompactUnitary(qureg, control, qubit, calpha[0], cbeta[0])

    def matrix(self, alpha: complex, beta: complex, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            alpha: complex parameter :math:`\alpha` of the unitary matrix
            beta: complex parameter :math:`\beta` of the unitary matrix
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, alpha, -np.conj(beta)],
                           [0, 0, beta, np.conj(alpha)]], dtype=complex)
        return matrix


class controlledNot(_PYQUEST):
    r"""Implements a controlled NOT gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 0 & 1\\
        0 & 0 & 1 & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        quest.controlledNot(qureg, control, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0x> -> |0>|x> |1x> -> |1> NOT |x>

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=complex)
        return matrix


class controlledPauliY(_PYQUEST):
    r"""Implements a controlled PauliY gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 0 & -i\\
        0 & 0 & i & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        quest.controlledPauliY(qureg, control, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, -1j],
                           [0, 0, 1j, 0]], dtype=complex)
        return matrix


class controlledPhaseFlip(_PYQUEST):
    r"""Implements a controlled phase flip gate also known as controlled Z gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & -1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        quest.controlledPhaseFlip(qureg, control, qubit)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, -1]], dtype=complex)
        return matrix


class swapGate(_PYQUEST):
    r"""Implements a SWAP gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 0 & 1 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        quest.swapGate(qureg,
                       control,
                       qubit,
                       )

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
        return matrix


class sqrtSwapGate(_PYQUEST):
    r"""Implements a square root SWAP gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & \frac{1}{2}(1+i) & \frac{1}{2}(1-i) & 0\\
        0 & \frac{1}{2}(1-i) & \frac{1}{2}(1+i) & 0\\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        quest.sqrtSwapGate(qureg,
                           control,
                           qubit,
                           )

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                           [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
                           [0, 0, 0, 1]], dtype=complex)
        return matrix


class sqrtISwap(_PYQUEST):
    r"""Implements a square root ISwap gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & \frac{1}{\sqrt{2}} & \frac{i}{\sqrt{2}} & 0\\
        0 & \frac{i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0\\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        matrix = self.matrix()
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.twoQubitUnitary(qureg,
                              control,
                              qubit,
                              mat[0])

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                           [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                           [0, 0, 0, 1]], dtype=complex)
        return matrix


class invSqrtISwap(_PYQUEST):
    r"""Implements inverse square root ISwap gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & \frac{1}{\sqrt{2}} & \frac{-i}{\sqrt{2}} & 0\\
        0 & \frac{-i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0\\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        matrix = self.matrix()
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.twoQubitUnitary(qureg,
                              control,
                              qubit,
                              mat[0])

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                           [0, -1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                           [0, 0, 0, 1]], dtype=complex)
        return matrix


class controlledPhaseShift(_PYQUEST):
    r"""Implements a controlled phase flip shift also known as controlled Z power gate

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & e^{i\theta}
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: The angle of the controlled Z-rotation

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            theta: The angle of the controlled Z-rotation
        """
        if not (0 <= theta and theta <= 2 * np.pi):
            theta = np.mod(theta, 2 * np.pi)
        quest.controlledPhaseShift(qureg, control, qubit, theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            theta: The angle of the controlled Z-rotation
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, np.exp(1j * theta)]], dtype=complex)
        return matrix


class controlledRotateAroundAxis(_PYQUEST):
    r"""Rotation around a general axis.

    Implements a controlled rotation around a vector :math:`\vec{v}`
    :math:`e^{-i \frac{\theta}{2} \vec{v} \vec{\sigma}}`

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos(\frac{\theta}{2}) & 0\\
        0 & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0\\
        0 & 0 & -i \sin(\frac{\theta}{2}) v_z & \sin(\frac{\theta}{2}) \left(-i v_x - v_y \right)\\
        0 & 0 & \sin(\frac{\theta}{2}) \left(-i v_x + v_y \right) & i \sin(\frac{\theta}{2}) v_z)
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation
        vector: Direction of the rotation axis, unit-vector

    """

    def call_interactive(self,
                         qureg: tqureg,
                         control: int,
                         qubit: int,
                         theta: float,
                         vector: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
            vector: Direction of the rotation axis, unit-vector

        Raises:
            RuntimeError: vector needs to be a three component numpy array and unit-vector
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        if not (vector.shape == (3,) and np.isclose(np.linalg.norm(vector), 1)):
            raise RuntimeError("vector needs to be a three component numpy array and unit-vector")
        else:
            vec = ffi_quest.new("Vector *")
            vec.x = vector[0]
            vec.y = vector[1]
            vec.z = vector[2]
            quest.controlledRotateAroundAxis(qureg, control,
                                             qubit,
                                             theta,
                                             vec[0])

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            theta: Angle theta of the rotation
            vector: Direction of the rotation axis, unit-vector
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c - 1j * s * vz, s * (-1j * vx - vy)],
                           [0, 0, s * (-1j * vx + vy), c + 1j * s * vz]], dtype=complex)
        return matrix


class controlledRotateX(_PYQUEST):
    r"""Implements a controlled rotation around the X axis

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos(\frac{\theta}{2}) & 0\\
        0 & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0\\
        0 & 0 & 0  &   -i \sin(\frac{\theta}{2}) \\
        0 & 0 & -i \sin(\frac{\theta}{2})    & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.controlledRotateX(qureg, control,
                                qubit,
                                theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            theta: Angle theta of the rotation
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c, -1j * s],
                           [0, 0, -1j * s, c]], dtype=complex)
        return matrix


class controlledRotateY(_PYQUEST):
    r"""Implements a controlled rotation around the Y axis `

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos(\frac{\theta}{2}) & 0\\
        0 & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0\\
        0 & 0 & 0  &   - \sin(\frac{\theta}{2}) \\
        0 & 0 &  \sin(\frac{\theta}{2})    & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.controlledRotateY(qureg, control,
                                qubit,
                                theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            theta: Angle theta of the rotation
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c, -s],
                           [0, 0, s, c]], dtype=complex)
        return matrix


class controlledRotateZ(_PYQUEST):
    r"""Implements a controlled rotation around the Y axis `

    .. math::
        U = \begin{pmatrix}
            1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos(\frac{\theta}{2}) & 0\\
        0 & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}
        + \begin{pmatrix}
            0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0\\
        0 & 0 &    - i \sin(\frac{\theta}{2}) & 0 \\
        0 & 0 & 0 & i \sin(\frac{\theta}{2})
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int, theta: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
            theta: Angle theta of the rotation
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)
        quest.controlledRotateZ(qureg, control,
                                qubit,
                                theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            theta: Angle theta of the rotation
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c - 1j * s, 0],
                           [0, 0, 0, c + 1j * s]], dtype=complex)
        return matrix


class controlledUnitary(_PYQUEST):
    r"""Implements a controlled arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        control: qubit that controls the unitary
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int, matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the unitary
            qubit: qubit the unitary gate is applied to
            matrix: Unitary matrix of the one qubit gate

        Raises:
            RuntimeError: vector needs to be a (2, 2) unitary numpy array
        """
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("vector needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            for i in range(2):
                for j in range(2):
                    mat.real[i][j] = np.real(matrix[i, j])
                    mat.imag[i][j] = np.imag(matrix[i, j])
            quest.controlledUnitary(qureg, control,
                                    qubit,
                                    mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|x> |1x> -> |1> U |x>

        Args:
            matrix: Unitary matrix of the one qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, matrix[0, 0], matrix[0, 1]],
                        [0, 0, matrix[1, 0], matrix[1, 1]]], dtype=complex)
        return mat


# Multi-controlled and mutli-qubit Operations


class multiControlledTwoQubitUnitary(_PYQUEST):
    r"""Two qubit unitary gate controlled by multiple qubits

    Implements a general two-qubit gate defined by a matrix controlled by multipe qubits
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        control: controll qubit
        target_qubit_1: least significant qubit
        target_qubit_2: most sifnificant qubit
        matrix: 4 by 4 matrix that defines the two qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         controls: Sequence[int],
                         target_qubit_1: int,
                         target_qubit_2: int,
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: control qubits
            target_qubit_1: least significant qubit
            target_qubit_2: most sifnificant qubit
            matrix: 4 by 4 matrix that defines the two qubit gate
        """
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer = ffi_quest.new("int[{}]".format(len(controls)))
        number_controls = len(controls)
        for co, control in enumerate(controls):
            pointer[co] = control
        quest.multiControlledTwoQubitUnitary(qureg,
                                             pointer,
                                             number_controls,
                                             target_qubit_1,
                                             target_qubit_2,
                                             mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class multiControlledPhaseFlip(_PYQUEST):
    r"""Phase Flip controlled by multipe qubits

    Implements a multi controlled phase flip gate also known as controlled Z gate.
    If all qubits in the controls are :math:`\left|1\right\rangle` the sign is flipped.
    No change occurs otherwise

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        number_controls: number of the control qubits

    """

    def call_interactive(self,
                         qureg: tqureg,
                         controls: Sequence[int],
                         number_controls: Optional[int] = None) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: qubits that control the application of the unitary
            number_controls: number of the control qubits
        """
        pointer = ffi_quest.new("int[{}]".format(len(controls)))
        if number_controls is None:
            number_controls = len(controls)
        for co, control in enumerate(controls):
            pointer[co] = control
        quest.multiControlledPhaseFlip(qureg, pointer, number_controls)

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError


class multiControlledPhaseShift(_PYQUEST):
    r"""Phase Shift controlled by multiple qubits

    Implements a multi controlled phase flip gate also known as controlled Z power gate.
    If all qubits in the controls are :math:`\left|1\right\rangle` the phase is shifter by theta.
    No change occurs otherwise

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        number_controls: number of the control qubits
        theta: Angle of the rotation around Z-axis

    """

    def call_interactive(self, qureg: tqureg, controls: Sequence[int],
                         number_controls: Optional[int] = None, theta: float = 0) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: qubits that control the application of the unitary
            number_controls: number of the control qubits
            theta: Angle of the rotation around Z-axis
        """
        if not (0 <= theta and theta <= 4 * np.pi):
            theta = np.mod(theta, 4 * np.pi)

        pointer = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer[co] = control
        if number_controls is None:
            number_controls = len(controls)
        quest.multiControlledPhaseShift(qureg, pointer, number_controls, theta)

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            theta: Angle of the rotation around Z-axis
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError


class multiControlledUnitary(_PYQUEST):
    r"""Generic unitary gate controlled by multiple qubits

    Implements a multi-controlled arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate

    """

    def call_interactive(self, qureg: tqureg,
                         controls: Sequence[int],
                         qubit: int, matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: qubits that control the application of the unitary
            qubit: qubit the unitary gate is applied to
            matrix: Unitary matrix of the one qubit gate

        Raises:
            RuntimeError: vector needs to be a (2, 2) unitary numpy array
        """
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("vector needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            for i in range(2):
                for j in range(2):
                    mat.real[i][j] = np.real(matrix[i, j])
                    mat.imag[i][j] = np.imag(matrix[i, j])
            pointer = ffi_quest.new("int[{}]".format(len(controls)))
            for co, control in enumerate(controls):
                pointer[co] = control
            number_controls = len(controls)
            quest.multiControlledUnitary(qureg, pointer,
                                         number_controls,
                                         qubit,
                                         mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: Unitary matrix of the one qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError


class controlledMultiQubitUnitary(_PYQUEST):
    r"""Controlled general unitary gate acting on N qubits

    Implements a general N-qubit gate defined by a matrix and controlled by a third qubit
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        control: controll qubit
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg, control: int,
                         targets: Sequence[int],
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: controll qubit
            targets: list of target qubits of the N qubit gate
                    the first qubit in targets is treated as the least significant one
                    the second as the second least significant one etc.
            matrix: N by N matrix that defines the N qubit gate

        Raises:
            RuntimeError: Shape of matrix and length of targets are different
        """
        if 2**len(targets) != matrix.shape[0] or 2**len(targets) != matrix.shape[1]:
            raise RuntimeError("Shape of matrix and length of targets are different")
        dim = matrix.shape[0]
        mat = quest.createComplexMatrixN(len(targets))
        for i in range(dim):
            for j in range(dim):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer = ffi_quest.new("int[{}]".format(len(targets)))
        for co, target in enumerate(targets):
            pointer[co] = target
        quest.controlledMultiQubitUnitary(qureg,
                                          control,
                                          pointer,
                                          len(targets),
                                          mat)
        quest.destroyComplexMatrixN(mat)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        The control qubit is always assumed to be the most relevant
        qubit |0xy> -> |0>|xy> |1xy> -> |1> U |xy>

        Args:
            matrix: N by N matrix that defines the N qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        dim = matrix.shape[0]
        return np.block([[np.eye(dim), np.zeros((dim, dim))],
                         [np.zeros((dim, dim)), matrix]])


class multiControlledMultiQubitUnitary(_PYQUEST):
    r"""General N-qubit unitary gate controlled by multiple qubits

    Implements a general N-qubit gate defined by a matrix controlled by multipe qubits
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        controls: controll qubits
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         controls: Sequence[int],
                         targets: Sequence[int],
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: controll qubits
            targets: list of target qubits of the N qubit gate
                    the first qubit in targets is treated as the least significant one
                    the second as the second least significant one etc.
            matrix: N by N matrix that defines the N qubit gate

        Raises:
            RuntimeError: Shape of matrix and length of targets are different
        """
        if 2**len(targets) != matrix.shape[0] or 2**len(targets) != matrix.shape[1]:
            raise RuntimeError("Shape of matrix and length of targets are different")
        dim = matrix.shape[0]
        mat = quest.createComplexMatrixN(len(targets))
        for i in range(dim):
            for j in range(dim):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer_c = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer_c[co] = control
        number_controls = len(controls)
        pointer = ffi_quest.new("int[{}]".format(len(targets)))
        for co, t in enumerate(targets):
            pointer[co] = t
        quest.multiControlledMultiQubitUnitary(qureg,
                                               pointer_c,
                                               number_controls,
                                               pointer,
                                               len(targets),
                                               mat)
        quest.destroyComplexMatrixN(mat)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class multiQubitUnitary(_PYQUEST):
    r"""General unitary gate acting on N qubits

    Implements a general N-qubit gate defined by a matrix
    If the matrix basis states are given by 0=|00>  1=|01> 2=|10> 3=|11>
    the least significant qubit is the right qubit and the most
    significant qubit is the left qubit

    Args:
        qureg: quantum register
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    """

    def call_interactive(self, qureg: tqureg, targets: Sequence[int], matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            targets: list of target qubits of the N qubit gate
                    the first qubit in targets is treated as the least significant one
                    the second as the second least significant one etc.
            matrix: N by N matrix that defines the N qubit gate

        Raises:
            RuntimeError: Shape of matrix and length of targets are different
        """
        if 2**len(targets) != matrix.shape[0] or 2**len(targets) != matrix.shape[1]:
            raise RuntimeError("Shape of matrix and length of targets are different")
        dim = matrix.shape[0]
        mat = quest.createComplexMatrixN(len(targets))
        for i in range(dim):
            for j in range(dim):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer = ffi_quest.new("int[{}]".format(len(targets)))
        for co, target in enumerate(targets):
            pointer[co] = target
        quest.multiQubitUnitary(qureg,
                                pointer,
                                len(targets),
                                mat)
        quest.destroyComplexMatrixN(mat)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: N by N matrix that defines the N qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class multiRotateZ(_PYQUEST):
    r"""Applying a Z-Rotation to multiple qubits

    A Z-Rotation with a given angle is applyied to multiple qubits

    Args:
        qureg: quantum register
        qubits: target qubits
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         qubits: Sequence[int],
                         angle: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubits: target qubits
            angle: Angle of rotation of the RotateZ gate
        """
        number_qubits = len(qubits)
        pointer = ffi_quest.new("int[{}]".format(len(qubits)))
        for co, q in enumerate(qubits):
            pointer[co] = q
        quest.multiRotateZ(qureg,
                           pointer,
                           number_qubits,
                           angle)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class multiRotatePauli(_PYQUEST):
    r"""Applying a set of different Pauli rotations to multiple qubits

    A set of Pauli rotations with a given angle is applied to multiple qubits

    Args:
        qureg: quantum register
        qubits: target qubits
        paulis: Pauli operators encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
        matrix: N by N matrix that defines the N qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         qubits: Sequence[int],
                         paulis: Sequence[int],
                         angle: float) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubits: target qubits
            paulis: Pauli operators encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
            angle: Angle of rotation of paulis

        Raises:
            RuntimeError: Number of qubits different from number of applied Paulis
        """
        if len(qubits) != len(paulis):
            raise RuntimeError("Number of qubits different from number of applied Paulis")
        number_qubits = len(qubits)
        pointer = ffi_quest.new("int[{}]".format(len(qubits)))
        for co, q in enumerate(qubits):
            pointer[co] = q
        pointer_paulis = ffi_quest.new("enum pauliOpType[{}]".format(len(qubits)))
        for co, p in enumerate(paulis):
            pointer_paulis[co] = p
        quest.multiRotatePauli(qureg,
                               pointer,
                               pointer_paulis,
                               number_qubits,
                               angle)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class multiStateControlledUnitary(_PYQUEST):
    r"""One qubit unitary controlled by multiple states

    Implements a general one-qubit gate defined by a matrix controlled by
    the state of multiple qubits
    Contrary to the multiControlled function the unitary operation here can be executed
    either when the controlling qubit is in state |0> or in state |1> depending
    on the control_states

    Args:
        qureg: quantum register
        controls: controll qubits
        controll_states: list of ints defining if the controlling gate acts like a
                        a normal control or anti-control (unitary is applied when state is |0>)
                        For each entry: 1 -> normal controlled, 0 -> anti-controlled
        qubit: The qubit the unitary is acting on
        matrix: 2 by 2 matrix that defines the one qubit gate

    """

    def call_interactive(self,
                         qureg: tqureg,
                         controls: Sequence[int],
                         control_states: Sequence[int],
                         qubit: int,
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: controll qubits
            control_states: list of ints defining if the controlling gate acts like a
                            a normal control or anti-control (unitary is applied when state is |0>)
                            For each entry: 1 -> normal controlled, 0 -> anti-controlled
            qubit: The qubit the unitary is acting on
            matrix: 2 by 2 matrix that defines the one qubit gate

        Raises:
            RuntimeError: Different Number of controls and control states
        """
        if len(controls) != len(control_states):
            raise RuntimeError("Different Number of controls and control states")
        mat = ffi_quest.new("ComplexMatrix2 *")
        for i in range(2):
            for j in range(2):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer_controls = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer_controls[co] = control
        number_controls = len(controls)
        pointer_states = ffi_quest.new("int[{}]".format(len(control_states)))
        for co, state in enumerate(control_states):
            pointer_states[co] = state
        quest.multiStateControlledUnitary(qureg,
                                          pointer_controls,
                                          pointer_states,
                                          number_controls,
                                          qubit,
                                          mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


# Extra gates


class MolmerSorensenXX(_PYQUEST):
    r"""Molmer Sorensen gate

    Implements a fixed phase MolmerSorensen XX gate (http://arxiv.org/abs/1705.02771)
    Uses decomposition according to http://arxiv.org/abs/quant-ph/0507171

    .. math::
        U = \frac{1}{\sqrt{2}} \begin{pmatrix}
            1 & 0 & 0 & i\\
        0 & 1 & i & 0\\
        0 & i & 1 & 0\\
        i & 0 & 0 & 1
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to

    """

    def call_interactive(self, qureg: tqureg, control: int, qubit: int) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the unitary
            qubit: qubit the unitary gate is applied to
        """
        matrix = self.matrix()
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.twoQubitUnitary(qureg,
                              control,
                              qubit,
                              mat[0])

    def matrix(self, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        matrix = np.array([[1, 0, 0, 1j],
                           [0, 1, 1j, 0],
                           [0, 1j, 1, 0],
                           [1j, 0, 0, 1]], dtype=complex) * (1 - 1j) / 2
        return matrix


# Apply operations


class applyDiagonalOp(_PYQUEST):
    r"""Applying a diagonal operator to state

    Apply a diagonal complex operator, which is possibly non-unitary
    and non-Hermitian, on the entire quantum register.

    Args:
        qureg: quantum register input, is not changed
        operator: operator acting on a certain number of qubits (operator[0]: int)
            and in a certain QuEST environment (operator[1]: tquestenv)

    """

    def call_interactive(self,
                         qureg: tqureg,
                         operator: Tuple[int, tquestenv],
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            operator: operator acting on a certain number of qubits (operator[0]: int)
                and in a certain QuEST environment (operator[1]: tquestenv)

        Raises:
            RuntimeError: Qureg and DiagonalOp must be defined for the same number of qubits
        """
        diagonal_op = quest.createDiagonalOp(operator[0], operator[1])
        if not (cheat.getNumQubits()(qureg=qureg) == diagonal_op.numQubits):
            raise RuntimeError("Qureg and DiagonalOp must be defined for the "
                               + "same number of qubits")
        quest.applyDiagonalOp(qureg, diagonal_op)
        quest.destroyDiagonalOp(diagonal_op, operator[1])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class applyMatrix2(_PYQUEST):
    r"""Applying a general 2-by-2 matrix, which may be non-unitary

    The matrix is left-multiplied onto the state, for both
    state-vectors and density matrices. Hence, this function differs
    from unitary() by more than just permitting a non-unitary matrix.

    Args:
        qureg: quantum register input, is not changed
        qubit: qubit to operate the matrix upon
        matrix: matrix to apply

    Warning:
        After applyMatrix2 the quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         qubit: int,
                         matrix: np.ndarray,
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit to operate the matrix upon
            matrix: matrix to apply
        """
        mat = ffi_quest.new("ComplexMatrix2 *")
        for i in range(2):
            for j in range(2):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.applyMatrix2(qureg, qubit, mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 2 by 2 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class applyMatrix4(_PYQUEST):
    r"""Applying a general 4-by-4 matrix, which may be non-unitary

    The matrix is left-multiplied onto the state, for both
    state-vectors and density matrices. Hence, this function differs from
    twoQubitUnitary() by more than just permitting a non-unitary matrix.

    Args:
        qureg: quantum register input, is not changed
        control: qubit that controls the application of the matrix
        qubit: qubit to operate the matrix upon
        matrix: matrix to apply

    Warning:
        After applyMatrix2 the quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         control: int,
                         qubit: int,
                         matrix: np.ndarray,
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            control: qubit that controls the application of the matrix
            qubit: qubit to operate the matrix upon
            matrix: matrix to apply
        """
        mat = ffi_quest.new("ComplexMatrix4 *")
        for i in range(4):
            for j in range(4):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        quest.applyMatrix4(qureg, control, qubit, mat[0])

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class applyMatrixN(_PYQUEST):
    r"""Applying a general N-by-N matrix, which may be non-unitary, on any number of target qubits

    The matrix is left-multiplied onto the state, for both
    state-vectors and density matrices. Hence, this function differs
    from multiQubitUnitary() by more than just permitting a non-unitary matrix.

    Args:
        qureg: quantum register input, is not changed
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    Warning:
        After applyMatrixN the quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         targets: Sequence[int],
                         matrix: np.ndarray,
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            targets: list of target qubits of the N qubit gate
                    the first qubit in targets is treated as the least significant one
                    the second as the second least significant one etc.
            matrix: N by N matrix that defines the N qubit gate

        Raises:
            RuntimeError: Shape of matrix and length of targets are different
        """
        if 2**len(targets) != matrix.shape[0] or 2**len(targets) != matrix.shape[1]:
            raise RuntimeError("Shape of matrix and length of targets are different")
        dim = matrix.shape[0]
        mat = quest.createComplexMatrixN(len(targets))
        for i in range(dim):
            for j in range(dim):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer = ffi_quest.new("int[{}]".format(len(targets)))
        for co, target in enumerate(targets):
            pointer[co] = target
        quest.applyMatrixN(qureg,
                           pointer,
                           len(targets),
                           mat)
        quest.destroyComplexMatrixN(mat)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: N by N matrix that defines the N qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class applyMultiControlledMatrixN(_PYQUEST):
    r"""Apply a general N-by-N matrix, which may be non-unitary, with additional controlled qubits

    A sum of products of Pauli operators (including Identity) is applied to a state.
    The state is not changed but the corresponding copy with the Pauli sum applied is
    written to qureg_out
    For each qubit a Pauli operator must be given in each sum term (can be identity)

    Args:
        qureg: quantum register
        controls: controll qubits
        targets: list of target qubits of the N qubit gate
                 the first qubit in targets is treated as the least significant one
                 the second as the second least significant one etc.
        matrix: N by N matrix that defines the N qubit gate

    Warning:
        After applyMultiControlledMatrixN the quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         controls: Sequence[int],
                         targets: Sequence[int],
                         matrix: np.ndarray) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            controls: controll qubits
            targets: list of target qubits of the N qubit gate
                    the first qubit in targets is treated as the least significant one
                    the second as the second least significant one etc.
            matrix: N by N matrix that defines the N qubit gate

        Raises:
            RuntimeError: Shape of matrix and length of targets are different
        """
        if 2**len(targets) != matrix.shape[0] or 2**len(targets) != matrix.shape[1]:
            raise RuntimeError("Shape of matrix and length of targets are different")
        dim = matrix.shape[0]
        mat = quest.createComplexMatrixN(len(targets))
        for i in range(dim):
            for j in range(dim):
                mat.real[i][j] = np.real(matrix[i, j])
                mat.imag[i][j] = np.imag(matrix[i, j])
        pointer_c = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer_c[co] = control
        pointer = ffi_quest.new("int[{}]".format(len(targets)))
        for co, target in enumerate(targets):
            pointer[co] = target
        quest.applyMultiControlledMatrixN(qureg,
                                          pointer_c,
                                          len(controls),
                                          pointer,
                                          len(targets),
                                          mat)
        quest.destroyComplexMatrixN(mat)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: N by N matrix that defines the N qubit gate
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray
        """
        return matrix


class applyPauliHamil(_PYQUEST):
    r"""Applying PauliHamil (a Hermitian but not necessarily unitary operator) to state

    This is merely an encapsulation of applyPauliSum(), which can refer to for elaborated doc.
    Applies each Pauli product in pauli_hamil to qureg in turn, and adding the resulting
    state to the initially-blanked qureg_out. Ergo it should scale with the total number
    of Pauli operators specified (excluding identities), and the qureg dimension.

    Args:
        qureg: quantum register input, is not changed
        paulis: List of Lists of Pauli operators in each product
                encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
        matrix: N by N matrix that defines the N qubit gate
        qureg_out: quantum register after application of Pauli sum

    Warning:
        After applyPauliHamil the output quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         pauli_hamil: paulihamil,
                         qureg_out: tqureg
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            pauli_hamil: PauliHamil instance to be applied
            qureg_out: quantum register after application of Pauli sum

        Raises:
            RuntimeError: Qureg and PauliHamil must be defined for the same number of qubits
        """
        if not (cheat.getNumQubits()(qureg=qureg) == pauli_hamil.numQubits):
            raise RuntimeError("Qureg and PauliHamil must be defined for the "
                               + "same number of qubits")
        quest.applyPauliHamil(qureg,
                              pauli_hamil,
                              qureg_out)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class applyPauliSum(_PYQUEST):
    r"""Applying a sum of Products of Pauli operators to state

    A sum of products of Pauli operators (including Identity) is applied to a state.
    The state is not changed but the corresponding copy with the Pauli sum applied is
    written to qureg_out
    For each qubit a Pauli operator must be given in each sum term (can be identity)

    Args:
        qureg: quantum register input, is not changed
        paulis: List of Lists of Pauli operators in each product
                encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
        matrix: N by N matrix that defines the N qubit gate
        qureg_out: quantum register after application of Pauli sum

    Warning:
        After applyPauliSum the quantum register is in general no longer normalised
        and does no longer represent a physical valid state without normalisation.

    """

    def call_interactive(self,
                         qureg: tqureg,
                         paulis: Sequence[Sequence[int]],
                         coefficients: Sequence[float],
                         qureg_out: tqureg
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            paulis: List of Lists of Pauli operators in each product
                    encoded as int via IDENTITY=0, PAULI_X=1, PAULI_Y=2, PAULI_Z=3
            coefficients: coefficients of the paulis to be summed
            qureg_out: quantum register after application of Pauli sum

        Raises:
            RuntimeError: Size of Qureg and number of lenght of PauliProduct does not match
        """
        for product in paulis:
            if qureg.numQubitsRepresented != len(product):
                raise RuntimeError(
                    "Size of Qureg and number of lenght of PauliProduct does not match")
        if qureg.numQubitsRepresented != qureg_out.numQubitsRepresented:
            raise RuntimeError("Size of Qureg and output QuregS does not match")
        flat_list = [p for product in paulis for p in product]
        pointer_paulis = ffi_quest.new("enum pauliOpType[{}]".format(len(flat_list)))
        for co, p in enumerate(flat_list):
            pointer_paulis[co] = p
        pointer = ffi_quest.new("{}[{}]".format(qreal, len(coefficients)))
        for co, c in enumerate(coefficients):
            pointer[co] = float(c)
        quest.applyPauliSum(qureg,
                            pointer_paulis,
                            pointer,
                            len(coefficients),
                            qureg_out)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


class applyTrotterCircuit(_PYQUEST):
    r"""Applying a trotterisation of unitary evolution exp(-i*pauli_hamil*time) to qureg

    This is a sequence of unitary operators, effected by multiRotatePauli(), which together
    approximate the action of full unitary-time evolution under the given Hamiltonian.
    These formulations are taken from 'Finding Exponential Product Formulas of Higher Orders',
    Naomichi Hatano and Masuo Suzuki (2005).

    Args:
        qureg: the register to modify under the approximate unitary-time evolution
        pauli_hamil: PauliHamil under which to approxiamte unitary-time evolution
        time: the target evolution time, which is permitted to be both positive and negative
        order: the order of Trotter-Suzuki decomposition to use
        repetitions: the number of repetitions of the decomposition of the given order

    """

    def call_interactive(self,
                         qureg: tqureg,
                         pauli_hamil: paulihamil,
                         time: float,
                         order: int,
                         repetitions: int
                         ) -> None:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: the register to modify under the approximate unitary-time evolution
            pauli_hamil: PauliHamil under which to approxiamte unitary-time evolution
            time: the target evolution time, which is permitted to be both positive and negative
            order: the order of Trotter-Suzuki decomposition to use
            repetitions: the number of repetitions of the decomposition of the given order
        """
        quest.applyTrotterCircuit(qureg,
                                  pauli_hamil,
                                  time,
                                  order,
                                  repetitions)

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        r"""The definition of the gate as a unitary matrix

        Args:
            matrix: 4 by 4 matrix that defines the two qubit gate
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()


# Measurement


class measure(_PYQUEST):
    r"""Implements a one-qubit Measurement operation

    Args:
        qureg: quantum register
        qubit: the measured qubit
        readout: The readout register for static compilation
        readout_index: The index in the readout register for static compilation

    """

    def call_interactive(self, qureg: tqureg, qubit: int) -> int:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to

        Returns:
            int
        """
        return quest.measure(qureg, qubit)


class measureWithStats(_PYQUEST):
    r"""Measures a single qubit and gives the probability of that outcome.

    Args:
        qureg: quantum register
        qubit: the measured qubit
        outcome_proba: where to set the probability of the occurred outcome

    """

    def call_interactive(self, qureg: tqureg, qubit: int, outcome_proba: float) -> int:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            outcome_proba: where to set the probability of the occurred outcome

        Returns:
            int
        """
        outcome_pointer = ffi_quest.new("{}[{}]".format(qreal, 1))
        outcome_pointer[0] = outcome_proba
        return quest.measureWithStats(qureg, qubit, outcome_pointer)


class collapseToOutcome(_PYQUEST):
    r"""Updates qureg to be consistent with measuring measureQubit and returns the probability.

    Args:
        qureg: quantum register
        qubit: the measured qubit
        outcome: where to set the probability of the occurred outcome

    """

    def call_interactive(self, qureg: tqureg, qubit: int, outcome: int) -> float:
        r"""Interactive call of PyQuest-cffi

        Args:
            qureg: quantum register
            qubit: qubit the unitary gate is applied to
            outcome: where to set the probability of the occurred outcome

        Returns:
            float
        """
        return quest.collapseToOutcome(qureg, qubit, outcome)
