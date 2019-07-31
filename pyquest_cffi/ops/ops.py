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

from pyquest_cffi.questlib import quest, _PYQUEST, ffi_quest
import numpy as np
from typing import Sequence, Optional, Union, List
import uuid
import warnings


class PiModuloWarning(RuntimeWarning):
    pass


warnings.filterwarnings("ignore", category=PiModuloWarning)


class hadamard(_PYQUEST):
    r"""
    Implements Hadamard gate

    .. math::
        U = \frac{1}{\sqrt{2}} \begin{pmatrix}
        1 & 1\\
        1 & -1
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.hadamard(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of hadamard

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "hadamard({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]], dtype=np.complex)
        return matrix


class pauliX(_PYQUEST):
    r"""
    Implements Pauli X gate

    .. math::
        U =  \begin{pmatrix}
        0 & 1\\
        1 & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.pauliX(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of pauliX

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "pauliX({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[0, 1], [1, 0]], dtype=np.complex)
        return matrix


class pauliY(_PYQUEST):
    r"""
    Implements Pauli Y gate

    .. math::
        U =  \begin{pmatrix}
        0 & -i\\
        i & 0
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.pauliY(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of pauliY

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "pauliY({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
        return matrix


class pauliZ(_PYQUEST):
    r"""
    Implements Pauli Z gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & -1
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.pauliZ(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of pauliY

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "pauliZ({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0], [0, -1]], dtype=np.complex)
        return matrix


class sGate(_PYQUEST):
    r"""
    Implements S gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & i
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.sGate(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of sGate

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "sGate({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex)
        return matrix


class tGate(_PYQUEST):
    r"""
    Implements T gate

    .. math::
        U =  \begin{pmatrix}
        1 & 0\\
        0 & e^{i \frac{\pi}{4}}
        \end{pmatrix}

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, qubit: int):
        quest.tGate(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[str, int]) -> List[str]:
        """
        Static call of tGate

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "tGate({qureg:s}, {qubit});".format(
            qureg=qureg, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex)
        return matrix


class compactUnitary(_PYQUEST):
    r"""
    Implements general unitary gate U in compact notation

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

    def call_interactive(self, qureg, qubit: int, alpha: complex, beta: complex):
        if not np.isclose(np.abs(alpha)**2+np.abs(beta)**2, 1):
            raise RuntimeError("compactUnitary needs parameters |alpha|**2+|beta|**2 == 1")
        else:
            calpha = ffi_quest.new("Complex *")
            calpha.real = np.real(alpha)
            calpha.imag = np.imag(alpha)
            cbeta = ffi_quest.new("Complex *")
            cbeta.real = np.real(beta)
            cbeta.imag = np.imag(beta)
            quest.compactUnitary(qureg, qubit, calpha[0], cbeta[0])

    def call_static(self, qureg: str, qubit: Union[str, int],
                    alpha: Union[str, complex], beta: Union[str, complex]) -> List[str]:
        """
        Static call of compactUnitary

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            alpha: The variable alpha, if complex value is used directly,
                    if string must be the name of previously defined C-variable of type Complex
            beta: The variable beta, if complex value is used directly,
                    if string must be the name of previously defined C-variable of type Complex
        """
        if isinstance(alpha, str) and isinstance(beta, str):
            call = "compactUnitary({qureg:s}, {qubit}, {alpha}, {beta});".format(
                qureg=qureg, qubit=qubit, alpha=alpha, beta=beta)
        else:
            t = '{}'.format(uuid.uuid4().hex)
            call = ''
            for ab, ab_val in [('alpha', alpha), ('beta', beta)]:
                call += 'Complex {ab}_{t};'.format(ab=ab, t=t)
                call += '{ab}_{t}.real = {r}; {ab}_{t}.imag = {i};'.format(
                    ab=ab, t=t, r=np.real(ab_val), i=np.imag(ab_val))
            call += "compactUnitary({qureg:s}, {qubit}, {alpha}, {beta});".format(
                qureg=qureg, qubit=qubit,
                alpha='alpha_{t}'.format(t=t),
                beta='beta_{t}'.format(t=t))
        return [call]

    def matrix(self, alpha: complex, beta: complex, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]], dtype=np.complex)
        return matrix


class phaseShift(_PYQUEST):
    r"""
    Implements pure :math:`\left|1 \right\rangle` phase shift gate

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

    def call_interactive(self, qureg, qubit: int, theta: float):
        if not (0 <= theta and theta <= 2*np.pi):
            theta = np.mod(theta, 2*np.pi)
            warnings.warn('choose rotation angle between 0 and 2 pi '
                          + ' applying modulo 2*pi', PiModuloWarning)
        quest.phaseShift(qureg, qubit, theta)

    def call_static(self, qureg: str, qubit: Union[str, int], theta: Union[str, float]) -> List[str]:
        """
        Static call of phaseShift

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "phaseShift({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0], [0, np.exp(1j*theta)]], dtype=np.complex)
        return matrix


class rotateAroundAxis(_PYQUEST):
    r"""
    Implements rotation around arbitraty axis on Bloch sphere

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

    def call_interactive(self, qureg, qubit: int, theta: float, vector: np.ndarray):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4*pi (is devided by 2) '
                          + ' applying modulo 4*pi', PiModuloWarning)
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

    def call_static(self, qureg: str, qubit: Union[str, int],
                    theta: Union[str, float], vector: Union[str, np.ndarray]) -> List[str]:
        """
        Static call of rotateAroundAxis

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
            vector: The vector or rotation axis, if 3-element np.ndarray values are used directly,
                    if string must be the name of previously defined C-variable of type Vector
        """
        lines = []
        if not isinstance(vector, str):
            t = '{}'.format(uuid.uuid4().hex)

            lines.append('Vector vector_{t};'.format(t=t))
            lines.append('vector_{t}.x = {x}; vector_{t}.y = {y}; vector_{t}.z = {z};'.format(
                t=t, x=vector[0], y=vector[1], z=vector[2]))
            vector = 'vector_{t}'.format(t=t)
        lines.append("rotateAroundAxis({qureg:s}, {qubit}, {theta}, {vector});".format(
            qureg=qureg, qubit=qubit, theta=theta, vector=vector))
        return lines

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[c-1j*s*vz, s*(-1j*vx-vy)],
                           [s*(-1j*vx+vy), c+1j*s*vz]], dtype=np.complex)
        return matrix


class rotateAroundSphericalAxis(_PYQUEST):
    r"""
    Implements rotation around an axis given in spherical coordinates

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

    def call_interactive(self, qureg, qubit: int, theta: float,
                         spherical_theta: float, spherical_phi: float,):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4*pi (is devided by 2) '
                          + ' applying modulo 4*pi', PiModuloWarning)

        vec = ffi_quest.new("Vector *")
        vec.x = np.sin(spherical_theta)*np.cos(spherical_phi)
        vec.y = np.sin(spherical_theta)*np.sin(spherical_phi)
        vec.z = np.cos(spherical_theta)
        quest.rotateAroundAxis(qureg,
                               qubit,
                               theta,
                               vec[0])

    def call_static(self, qureg: str, qubit: Union[str, int],
                    theta: Union[str, float],
                    spherical_theta:  Union[str, float], spherical_phi:  Union[str, float],) -> List[str]:
        """
        Static call of rotateAroundSphericalAxis

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
            spherical_theta: The variable spherical_theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
            spherical_phi: The variable spherical_theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        lines = []
        if isinstance(spherical_phi, str) or isinstance(spherical_theta, str):
            raise NotImplementedError()
        else:
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('Vector vector_{t};'.format(t=t))
            lines.append('vector_{t}.x = {x}; vector_{t}.y = {y}; vector_{t}.z = {z};'.format(
                t=t, x=np.sin(spherical_theta)*np.cos(spherical_phi),
                y=np.sin(spherical_theta)*np.sin(spherical_phi),
                z=np.cos(spherical_theta)))
            lines.append("compactUnitary({qureg:s}, {qubit}, {vector});".format(
                qureg=qureg, qubit=qubit,
                vector='vector_{t}'.format(t=t),
            ))
        return lines

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[c-1j*s*vz, s*(-1j*vx-vy)],
                           [s*(-1j*vx+vy), c+1j*s*vz]], dtype=np.complex)
        return matrix


class rotateX(_PYQUEST):
    r"""
    Implements :math:`e^{-i \frac{\theta}{2} \sigma^x}` XPower gate

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

    def call_interactive(self, qureg, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.rotateX(qureg,
                      qubit,
                      theta)

    def call_static(self, qureg: str, qubit: Union[str, int], theta: Union[str, float]) -> List[str]:
        """
        Static call of rotateX

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex)
        return matrix


class rotateY(_PYQUEST):
    r"""
    Implements :math:`e^{-i \frac{\theta}{2} \sigma^y}` XPower gate

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

    def call_interactive(self, qureg, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.rotateY(qureg,
                      qubit,
                      theta)

    def call_static(self, qureg: str, qubit: Union[str, int], theta: Union[str, float]) -> List[str]:
        """
        Static call of rotateY

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[c, -s], [s, c]], dtype=np.complex)
        return matrix


class rotateZ(_PYQUEST):
    r"""
    Implements :math:`e^{-i \frac{\theta}{2} \sigma^z}` XPower gate

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

    def call_interactive(self, qureg, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.rotateZ(qureg,
                      qubit,
                      theta)

    def call_static(self, qureg: str, qubit: Union[str, int],
                    theta: Union[str, float]) -> List[str]:
        """
        Static call of rotateZ

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "rotateZ({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[c-1j*s, 0], [0, c+1j*s]], dtype=np.complex)
        return matrix


class unitary(_PYQUEST):
    r"""
    Implements an arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate
    """

    def call_interactive(self, qureg, qubit: int, matrix: np.ndarray):
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("matrix needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 0])
            cComplex.imag = np.imag(matrix[0, 0])
            mat.r0c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 1])
            cComplex.imag = np.imag(matrix[0, 1])
            mat.r0c1 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 0])
            cComplex.imag = np.imag(matrix[1, 0])
            mat.r1c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 1])
            cComplex.imag = np.imag(matrix[1, 1])
            mat.r1c1 = cComplex[0]
            quest.unitary(qureg,
                          qubit,
                          mat[0])

    def call_static(self, qureg: str, qubit: Union[str, int], matrix: Union[str, np.ndarray]) -> List[str]:
        """
        Static call of unitary

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            matrix: The unitary matrix, if np.ndarray, values are used directly
                if string must be the name of previously defined C-variable of type ComplexMatrix2
        """
        lines = []
        if isinstance(matrix, str):
            lines.append("controlledUnitary({qureg:s}, {qubit}, {matrix});".format(
                qureg=qureg, qubit=qubit, matrix=matrix))
        else:
            t = '{}'.format(uuid.uuid4().hex)

            lines.append('ComplexMatrix2 mat_{t};'.format(t=t))
            lines.append('Complex c_{t}_r0c0; c_{t}_r0c0.real = {x}; c_{t}_r0c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 0])))
            lines.append('mat_{t}.r0c0 = c_{t}_r0c0{x};'.format(t=t))
            lines.append('Complex c_{t}_r0c1; c_{t}_r0c1.real = {x}; c_{t}_r0c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 1])))
            lines.append('mat_{t}.r0c1 = c_{t}_r0c1{x};'.format(t=t))
            lines.append('Complex c_{t}_r1c0; c_{t}_r1c0.real = {x}; c_{t}_r1c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 0])))
            lines.append('mat_{t}.r1c0 = c_{t}_r1c0;'.format(t=t))
            lines.append('Complex c_{t}_r1c1; c_{t}_r1c1.real = {x}; c_{t}_r1c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 1])))
            lines.append('mat_{t}.r1c1 = c_{t}_r1c1;'.format(t=t))
            lines.append("unitary({qureg:s}, {qubit}, {matrix});".format(
                qureg=qureg, qubit=qubit,
                matrix='mat_{t}'.format(t=t),
            ))
        return lines

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        return matrix

# Controlled Operations


class controlledCompactUnitary(_PYQUEST):
    r"""
    Implements a controlled general unitary gate U in compact notation

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

    def call_interactive(self, qureg, control: int,  qubit: int, alpha: complex, beta: complex):
        if not np.isclose(np.abs(alpha)**2+np.abs(beta)**2, 1):
            raise RuntimeError("compactUnitary needs parameters |alpha|**2+|beta|**2 == 1")
        else:
            calpha = ffi_quest.new("Complex *")
            calpha.real = np.real(alpha)
            calpha.imag = np.imag(alpha)
            cbeta = ffi_quest.new("Complex *")
            cbeta.real = np.real(beta)
            cbeta.imag = np.imag(beta)
            quest.controlledCompactUnitary(qureg, control, qubit, calpha[0], cbeta[0])

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    alpha: Union[str, complex], beta: Union[str, complex]) -> List[str]:
        """
        Static call of controlledCompactUnitary

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            alpha: The variable alpha, if complex value is used directly,
                    if string must be the name of previously defined C-variable of type Complex
            beta: The variable beta, if complex value is used directly,
                    if string must be the name of previously defined C-variable of type Complex
        """
        if isinstance(alpha, str) and isinstance(beta, str):
            call = "controlledCompactUnitary(({qureg:s}, {control}, {qubit}, {alpha}, {beta})".format(
                qureg=qureg, control=control, qubit=qubit, alpha=alpha, beta=beta)
        else:
            t = '{}'.format(uuid.uuid4().hex)
            call = ''
            for ab, ab_val in [('alpha', alpha), ('beta', beta)]:
                call += 'Complex {ab}_{t};'.format(ab=ab, t=t)
                call += '{ab}_{t}.real = {r}; {ab}_{t}.imag = {i};'.format(
                    ab=ab, t=t, r=np.real(ab_val), i=np.imag(ab_val))
            call += "controlledCompactUnitary(({qureg:s}, {control} ,{qubit}, {alpha}, {beta})".format(
                qureg=qureg, control=control, qubit=qubit,
                alpha='alpha_{t}'.format(t=t),
                beta='beta_{t}'.format(t=t))
        return [call]

    def matrix(self, alpha: complex, beta: complex, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, alpha, -np.conj(beta)],
                           [0, 0, beta, np.conj(alpha)]], dtype=np.complex)
        return matrix


class controlledNot(_PYQUEST):
    r"""
    Implements a controlled NOT gate

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

    def call_interactive(self, qureg, control: int, qubit: int):

        quest.controlledNot(qureg, control, qubit)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        call = "controlledNot({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=np.complex)
        return matrix


class controlledPauliY(_PYQUEST):
    r"""
    Implements a controlled PauliY gate

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

    def call_interactive(self, qureg, control: int, qubit: int):

        quest.controlledPauliY(qureg, control, qubit)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        """
        Static call of controlledPauliY

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "controlledPauliY({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, -1j],
                           [0, 0, 1j, 0]], dtype=np.complex)
        return matrix


class controlledPhaseFlip(_PYQUEST):
    r"""
    Implements a controlled phase flip gate also known as controlled Z gate

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

    def call_interactive(self, qureg, control: int, qubit: int):

        quest.controlledPhaseFlip(qureg, control, qubit)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        """
        Static call of controlledPhaseFlip

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "controlledPhaseFlip({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, -1]], dtype=np.complex)
        return matrix


class sqrtISwap(_PYQUEST):
    r"""
    Implements a square root ISwap gate

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

    def call_interactive(self, qureg, control: int, qubit: int):

        quest.rotateY(qureg, qubit, -np.pi/2)
        quest.rotateX(qureg, control, np.pi/2)
        quest.controlledPhaseFlip(qureg, control, qubit)
        quest.rotateY(qureg, qubit, -np.pi/4)
        quest.rotateX(qureg, control, -np.pi/4)
        quest.controlledPhaseFlip(qureg, control, qubit)
        quest.rotateY(qureg, qubit, np.pi/2)
        quest.rotateX(qureg, control, -np.pi/2)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        call_list = list()
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=-np.pi/2))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=np.pi/2))
        call_list.append("controlledPhaseFlip({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit))
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=-np.pi/4))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=-np.pi/4))
        call_list.append("controlledPhaseFlip({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit))
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=np.pi/2))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=-np.pi/2))
        return call_list

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
                           [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
                           [0, 0, 0, 1]], dtype=np.complex)
        return matrix


class invSqrtISwap(_PYQUEST):
    r"""
    Implements inverse square root ISwap gate

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

    def call_interactive(self, qureg, control: int, qubit: int):

        quest.rotateY(qureg, qubit, -np.pi/2)
        quest.rotateX(qureg, control, np.pi/2)
        quest.controlledPhaseFlip(qureg, control, qubit)
        quest.rotateY(qureg, qubit, np.pi/4)
        quest.rotateX(qureg, control, np.pi/4)
        quest.controlledPhaseFlip(qureg, control, qubit)
        quest.rotateY(qureg, qubit, np.pi/2)
        quest.rotateX(qureg, control, -np.pi/2)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        call_list = list()
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=-np.pi/2))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=np.pi/2))
        call_list.append("controlledPhaseFlip({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit))
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=np.pi/4))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=np.pi/4))
        call_list.append("controlledPhaseFlip({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit))
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=np.pi/2))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=-np.pi/2))
        return call_list

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1/np.sqrt(2), -1j/np.sqrt(2), 0],
                           [0, -1j/np.sqrt(2), 1/np.sqrt(2), 0],
                           [0, 0, 0, 1]], dtype=np.complex)
        return matrix


class controlledPhaseShift(_PYQUEST):
    r"""
    Implements a controlled phase flip shift also known as controlled Z power gate

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
        theta: The angle of the controlled Z-rotation
        qubit: qubit the unitary gate is applied to
    """

    def call_interactive(self, qureg, control: int, qubit: int, theta: float):

        if not (0 <= theta and theta <= 2*np.pi):
            theta = np.mod(theta, 2*np.pi)
            warnings.warn('choose rotation angle between 0 and 2 pi '
                          + ' applying modulo 2*pi', PiModuloWarning)
        quest.controlledPhaseShift(qureg, control, qubit, theta)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    theta: Union[str, int]) -> List[str]:
        """
        Static call of controlledNot

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "controlledNot({qureg:s}, {control}, {qubit}, {theta});".format(
            qureg=qureg, control=control, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, np.exp(1j*theta)]], dtype=np.complex)
        return matrix


class controlledRotateAroundAxis(_PYQUEST):
    r"""
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
        0 & 0 & -i \sin(\frac{\theta}{2}) v_z  &  \sin(\frac{\theta}{2}) \left(-i v_x - v_y \right) \\
        0 & 0 & \sin(\frac{\theta}{2}) \left(-i v_x + v_y \right) & i \sin(\frac{\theta}{2}) v_z)
        \end{pmatrix}

    Args:
        qureg: quantum register
        control: qubit that controls the application of the unitary
        qubit: qubit the unitary gate is applied to
        theta: Angle theta of the rotation
        vector: Direction of the rotation axis, unit-vector
    """

    def call_interactive(self, qureg, control: int, qubit: int, theta: float, vector: np.ndarray):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2!)'
                          + ' applying modulo 4*pi', PiModuloWarning)
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

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    theta: Union[str, float], vector: Union[str, np.ndarray]) -> List[str]:
        """
        Static call of controlledRotateAroundAxis

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
            vector: The vector or rotation axis, if 3-element np.ndarray values are used directly,
                    if string must be the name of previously defined C-variable of type Vector
        """
        if isinstance(vector, str):
            call = "phaseShift({qureg:s}, {control}, {qubit}, {theta}, {vector});".format(
                qureg=qureg, control=control, qubit=qubit, theta=theta, vector=vector)
        else:
            t = '{}'.format(uuid.uuid4().hex)
            call = ''
            call += 'Vector vector_{t};'.format(t=t)
            call += 'vector_{t}.x = {x}; vector_{t}.y = {y}; vector_{t}.z = {z};'.format(
                t=t, x=vector[0], y=vector[1], z=vector[2])
            call += "compactUnitary({qureg:s}, {qubit}, {vector});".format(
                qureg=qureg, control=control, qubit=qubit,
                vector='vector_{t}'.format(t=t),
            )
        return [call]

    def matrix(self, theta: float, vector: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        vx = vector[0]
        vy = vector[1]
        vz = vector[2]
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c-1j*s*vz, s*(-1j*vx-vy)],
                           [0, 0, s*(-1j*vx+vy), c+1j*s*vz]], dtype=np.complex)
        return matrix


class controlledRotateX(_PYQUEST):
    r"""
    Implements a controlled rotation around the X axis `

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

    def call_interactive(self, qureg, control: int, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.controlledRotateX(qureg, control,
                                qubit,
                                theta)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    theta: Union[str, float]) -> List[str]:
        """
        Static call of controlledRotateX

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "controlledRotateX({qureg:s}, {control}, {qubit}, {theta});".format(
            qureg=qureg, control=control, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c, -1j*s],
                           [0, 0, -1j*s, c]], dtype=np.complex)
        return matrix


class controlledRotateY(_PYQUEST):
    r"""
    Implements a controlled rotation around the Y axis `

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

    def call_interactive(self, qureg, control: int, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.controlledRotateY(qureg, control,
                                qubit,
                                theta)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    theta: Union[str, float]) -> List[str]:
        """
        Static call of controlledRotateY

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "controlledRotateY({qureg:s}, {control}, {qubit}, {theta});".format(
            qureg=qureg, control=control, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c, -s],
                           [0, 0, s, c]], dtype=np.complex)
        return matrix


class controlledRotateZ(_PYQUEST):
    r"""
    Implements a controlled rotation around the Y axis `

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

    def call_interactive(self, qureg, control: int, qubit: int, theta: float):
        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi (is devided by 2)'
                          + ' applying modulo 4*pi', PiModuloWarning)
        quest.controlledRotateZ(qureg, control,
                                qubit,
                                theta)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    theta: Union[str, float]) -> List[str]:
        """
        Static call of controlledRotateZ

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "controlledRotateZ({qureg:s}, {control}, {qubit}, {theta});".format(
            qureg=qureg, control=control, qubit=qubit, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        c = np.cos(theta/2)
        s = np.sin(theta/2)
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, c-1j*s, 0],
                           [0, 0, 0, c+1j*s]], dtype=np.complex)
        return matrix


class controlledUnitary(_PYQUEST):
    r"""
    Implements a controlled arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        control: qubit that controls the unitary
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate
    """

    def call_interactive(self, qureg, control, qubit: int, matrix: np.ndarray):
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("vector needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 0])
            cComplex.imag = np.imag(matrix[0, 0])
            mat.r0c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 1])
            cComplex.imag = np.imag(matrix[0, 1])
            mat.r0c1 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 0])
            cComplex.imag = np.imag(matrix[1, 0])
            mat.r1c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 1])
            cComplex.imag = np.imag(matrix[1, 1])
            mat.r1c1 = cComplex[0]
            quest.controlledUnitary(qureg, control,
                                    qubit,
                                    mat[0])

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    matrix: Union[str, np.ndarray]) -> List[str]:
        """
        Static call of controlledUnitary

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            control: The control in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            matrix: The unitary matrix, if np.ndarray, values are used directly
                if string must be the name of previously defined C-variable of type ComplexMatrix2
        """
        lines = []
        if isinstance(matrix, str):
            call = "controlledUnitary({qureg:s}, {control}, {qubit}, {matrix});".format(
                qureg=qureg, control=control, qubit=qubit, matrix=matrix)
        else:
            t = '{}'.format(uuid.uuid4().hex)

            lines.append('ComplexMatrix2 mat_{t};'.format(t=t))
            lines.append('Complex c_{t}_r0c0; c_{t}_r0c0.real = {x}; c_{t}_r0c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 0])))
            lines.append('mat_{t}.r0c0 = c_{t}_r0c0{x};'.format(t=t))
            lines.append('Complex c_{t}_r0c1; c_{t}_r0c1.real = {x}; c_{t}_r0c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 1])))
            lines.append('mat_{t}.r0c1 = c_{t}_r0c1{x};'.format(t=t))
            lines.append('Complex c_{t}_r1c0; c_{t}_r1c0.real = {x}; c_{t}_r1c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 0])))
            lines.append('mat_{t}.r1c0 = c_{t}_r1c0;'.format(t=t))
            lines.append('Complex c_{t}_r1c1; c_{t}_r1c1.real = {x}; c_{t}_r1c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 1])))
            lines.append('mat_{t}.r1c1 = c_{t}_r1c1;'.format(t=t))
            lines.append("controlledUnitary({qureg:s}, {control}, {qubit}, {matrix});".format(
                qureg=qureg, control=control, qubit=qubit,
                matrix='mat_{t}'.format(t=t),
            ))
        return lines

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, matrix[0, 0], matrix[0, 1]],
                        [0, 0, matrix[1, 0], matrix[1, 1]]], dtype=np.complex)
        return mat

# Multi-controlled Operations


class multiControlledPhaseFlip(_PYQUEST):
    r"""
    Implements a multi controlled phase flip gate also known as controlled Z gate.
    If all qubits in the controls are :math:`\left|1\right\rangle` the sign is flipped.
    No change occurs otherwise

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        number_controls: number of the control qubits
    """

    def call_interactive(self, qureg, controls: Sequence[int],  number_controls: int):
        pointer = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer[co] = control
        quest.multiControlledPhaseFlip(qureg, pointer,  number_controls)

    def call_static(self, qureg: str, controls: str, number_controls: Union[str, int],
                    ) -> List[str]:
        """
        Static call of mulitControlledPhaseFlip

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            controls: the control in the quantum register,
                     must be the name of previously defined C-point to array of type int
            number_controls: The variable numer_controls, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
        """
        call = "multiControlledPhaseFlip({qureg:s}, {controls}, { number_controls});".format(
            qureg=qureg, controls=controls,  number_controls=number_controls)
        return [call]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        raise NotImplementedError


class multiControlledPhaseShift(_PYQUEST):
    r"""
    Implements a mulit controlled phase flip gate also known as controlled Z power gate.
    If all qubits in the controls are :math:`\left|1\right\rangle` the phase is shifter by theta.
    No change occurs otherwise

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        number_controls: number of the control qubits
        theta: Angle of the rotation around Z-axis
    """

    def call_interactive(self, qureg, controls: Sequence[int], number_controls: int, theta: float):

        if not (0 <= theta and theta <= 4*np.pi):
            theta = np.mod(theta, 4*np.pi)
            warnings.warn('choose rotation angle between 0 and 4 pi '
                          + ' applying modulo 4*pi', PiModuloWarning)
        pointer = ffi_quest.new("int[{}]".format(len(controls)))
        for co, control in enumerate(controls):
            pointer[co] = control
        quest.multiControlledPhaseShift(qureg, pointer, number_controls, theta)

    def call_static(self, qureg: str, controls: Union[str], number_controls: Union[str, int],
                    theta: Union[str, float]) -> List[str]:
        """
        Static call of mulitControlledPhaseShift

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            controls: the control in the quantum register,
                     must be the name of previously defined C-pointer of to array of int
            numer_controls: The variable numer_controls, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            theta: The variable theta, if float value is used directly,
                    if string must be the name of previously defined C-variable of type qreal
        """
        call = "multiControlledPhaseShift({qureg:s}, {controls}, { number_controls}, {theta});".format(
            qureg=qureg, controls=controls,  number_controls=number_controls, theta=theta)
        return [call]

    def matrix(self, theta: float, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        raise NotImplementedError


class multiControlledUnitary(_PYQUEST):
    r"""
    Implements a mulit-controlled arbitraty one-qubit gate given by a unitary matrix

    Args:
        qureg: quantum register
        controls: qubits that control the application of the unitary
        qubit: qubit the unitary gate is applied to
        matrix: Unitary matrix of the one qubit gate
    """

    def call_interactive(self, qureg,
                         controls: Sequence[int], number_controls: int,
                         qubit: int, matrix: np.ndarray):
        if not (matrix.shape == (2, 2) and np.all(np.isclose(matrix.conj().T @ matrix, np.eye(2)))):
            raise RuntimeError("vector needs to be a (2, 2) unitary numpy array")
        else:
            mat = ffi_quest.new("ComplexMatrix2 *")
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 0])
            cComplex.imag = np.imag(matrix[0, 0])
            mat.r0c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[0, 1])
            cComplex.imag = np.imag(matrix[0, 1])
            mat.r0c1 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 0])
            cComplex.imag = np.imag(matrix[1, 0])
            mat.r1c0 = cComplex[0]
            cComplex = ffi_quest.new("Complex *")
            cComplex.real = np.real(matrix[1, 1])
            cComplex.imag = np.imag(matrix[1, 1])
            mat.r1c1 = cComplex[0]
            pointer = ffi_quest.new("int[{}]".format(len(controls)))
            for co, control in enumerate(controls):
                pointer[co] = control
            quest.multiControlledUnitary(qureg, pointer,
                                         number_controls,
                                         qubit,
                                         mat[0])

    def call_static(self, qureg: str, controls: Union[str, Sequence[int]], number_controls: Union[str, int], qubit: Union[str, int],
                    matrix: Union[str, float]):
        """
        Static call of mulitControlledUnitary

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            controls: the control in the quantum register,
                     must be the name of previously defined C-pointer to array of type int of
            number_controls: The variable numer_controls, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            matrix: The unitary matrix, if np.ndarray, values are used directly
                if string must be the name of previously defined C-variable of type ComplexMatrix2
        """
        lines = []
        if isinstance(matrix, str):
            lines.append("controlledUnitary({qureg:s}, {control}, {qubit}, {matrix});".format(
                qureg=qureg, controls=controls, qubit=qubit, matrix=matrix))
        else:
            t = '{}'.format(uuid.uuid4().hex)

            lines.append('ComplexMatrix2 mat_{t};'.format(t=t))
            lines.append('Complex c_{t}_r0c0; c_{t}_r0c0.real = {x}; c_{t}_r0c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 0])))
            lines.append('mat_{t}.r0c0 = c_{t}_r0c0{x};'.format(t=t))
            lines.append('Complex c_{t}_r0c1; c_{t}_r0c1.real = {x}; c_{t}_r0c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[0, 1])))
            lines.append('mat_{t}.r0c1 = c_{t}_r0c1{x};'.format(t=t))
            lines.append('Complex c_{t}_r1c0; c_{t}_r1c0.real = {x}; c_{t}_r1c0.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 0])))
            lines.append('mat_{t}.r1c0 = c_{t}_r1c0;'.format(t=t))
            lines.append('Complex c_{t}_r1c1; c_{t}_r1c1.real = {x}; c_{t}_r1c1.imag={y};'.format(
                t=t, x=np.real(matrix[0, 0]), y=np.imag(matrix[1, 1])))
            lines.append('mat_{t}.r1c1 = c_{t}_r1c1;'.format(t=t))
            lines.append("multiControlledUnitary({qureg:s}, {controls}, {qubit}, {matrix});".format(
                qureg=qureg, controls=control, qubit=qubit,
                matrix='mat_{t}'.format(t=t),
            ))
        return lines

    def matrix(self, matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        raise NotImplementedError

# measurement


class measure(_PYQUEST):
    r"""
    Implements a one-qubit Measurement operation

    Args:
        qureg: quantum register
        qubit: the measured qubit
        readout: The readout register for static compilation
        readout_index: The index in the readout register for static compilation
    """

    def call_interactive(self, qureg, qubit: int) -> int:
        return quest.measure(qureg, qubit)

    def call_static(self, qureg: str, qubit: Union[int, str],
                    readout: Optional[str] = None, readout_index: Optional[Union[int, str]] = None) -> List[str]:
        """
        Static call of measure

        Args:
            qureg: The name of the previously created quantum register as a string
            qubit: The qubit in the quantum register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            readout_index: Index in the readout register, if int value is used directly,
                    if string must be the name of previously defined C-variable of type int
            readout: The name of the previously created C-variable of type qreal
        """
        if readout is None:
            raise RuntimeError(
                'Static measuremente needs a readout register name (readout)'
                + 'to know where to save the measurement result')
        else:
            if readout_index is None:
                call = "{readout:s} = measure({qureg:s}, {qubit})".format(
                    readout=readout, qureg=qureg, qubit=qubit)
            else:
                call = "{readout:s}[{readout_index}] = measure({qureg:s}, {qubit})".format(
                    readout=readout, readout_index=readout_index, qureg=qureg, qubit=qubit)
        return [call]

# Extra gates:


class MolmerSorensenXX(_PYQUEST):
    r"""
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

    def call_interactive(self, qureg, control: int, qubit: int):
        quest.rotateY(qureg, control, np.pi/2)
        quest.controlledNot(qureg, control, qubit)
        quest.rotateZ(qureg, control, np.pi/2)
        quest.rotateX(qureg, qubit, np.pi/2)
        quest.rotateY(qureg, control, -np.pi/2)

    def call_static(self, qureg: str, control: Union[str, int], qubit: Union[str, int],
                    ) -> List[str]:
        call_list = list()
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=np.pi/4))
        call_list.append("controlledNot({qureg:s}, {control}, {qubit});".format(
            qureg=qureg, control=control, qubit=qubit))
        call_list.append("rotateZ({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=np.pi/4))
        call_list.append("rotateX({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=qubit, theta=np.pi/4))
        call_list.append("rotateY({qureg:s}, {qubit}, {theta});".format(
            qureg=qureg, qubit=control, theta=-np.pi/4))
        return call_list

    def matrix(self, **kwargs) -> np.ndarray:
        """
        The definition of the gate as a unitary matrix
        """
        matrix = np.array([[1, 0, 0, 1j],
                           [0, 1, 1j, 0],
                           [0, 1j, 1, 0],
                           [1j, 0, 0, 1]], dtype=np.complex)*(1-1j)/2
        return matrix
