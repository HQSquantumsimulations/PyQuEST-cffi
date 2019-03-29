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

import typing
import numpy as np
from pyquest_cffi.questlib import quest, _PYQUEST, tqureg
import warnings
from typing import Union, List
import uuid


class initZeroState(_PYQUEST):
    """
    Initialise zero state in quantum register

    Args:
        qureg: quantum register
    """

    def call_interactive(self, qureg: tqureg) -> None:
        return quest.initZeroState(qureg)

    def call_static(self, qureg:  str):
        """
        Static call of initZeroState

        Args:
            qureg: The name of the previously created quantum register as a string
        """
        call = "initZeroState({qureg});".format(
            qureg=qureg)
        return [call]


class initPlusState(_PYQUEST):
    """
    Initialise plus state in quantum register

    Args:
        qureg: quantum register
    """

    def call_interactive(self, qureg: tqureg) -> None:
        return quest.initPlusState(qureg)

    def call_static(self, qureg:  str):
        """
        Static call of initPlusState

        Args:
            qureg: The name of the previously created quantum register as a string
        """
        call = "initPlusState({qureg});".format(
            qureg=qureg)
        return [call]


class initClassicalState(_PYQUEST):
    """
    Initialise classic state, a classic integer in binary representation in the quantum register

    Args:
        qureg: The quantum register
        int: The integer that is initialised in binary representation in the quantum register
    """

    def call_interactive(self, qureg: tqureg, state: int) -> None:
        return quest.initClassicalState(qureg, state)

    def call_static(self, qureg:  str, state: Union[int, str]) -> List[str]:
        """
        Static call of initClassicalState

        Args:
            qureg: The name of the previously created quantum register as a string
            state: If state is of int type state will be initialised on the quantum register
                    if state is a string, it must be the name of a previously defined C-variable of type int
        """
        call = "initClassicalState({qureg}, {state});".format(
            qureg=qureg, state=state)
        return [call]


class initPureState(_PYQUEST):
    """
    Initialises a pure state in one wavefunction quantum register based on the pure state of another quantum register

    Args:
        qureg_a: Quantum register that is initialised
        qureg_b: Quantum register that contains the reference pure state
    """

    def call_interactive(self, qureg_a: tqureg, qureg_b: tqureg) -> None:
        return quest.initPureState(qureg_a, qureg_b)

    def call_static(self, qureg_a:  str, qureg_b:  str,) -> List[str]:
        """
        Static call of initPureState

        Args:
            qureg_a: The name of a previously created quantum register as a string
            qureg_b: The name of a previously created quantum register as a string
        """
        call = "initPureState({qureg_a}, {qureg_b});".format(
            qureg_a=qureg_a, qureg_b=qureg_b)
        return [call]


class initStateFromAmps(_PYQUEST):
    """
    Initialise a wavefunction in a quantum register based on the real and imaginary parts of the statevector

    Args:
        qureg: the quantum register
        reals: The real parts of the statevector
        imags: The imaginary parts of the statevector
    """

    def call_interactive(self,
                         qureg: tqureg,
                         reals: typing.Union[np.ndarray, typing.List[float]],
                         imags: typing.Union[np.ndarray, typing.List[float]]
                         ) -> None:
        reals = list(reals)
        imags = list(imags)
        assert len(reals) == np.max(np.shape(reals))
        assert len(imags) == np.max(np.shape(imags))
        if qureg.isDensityMatrix:
            warnings.warn('qureg has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        else:
            return quest.initStateFromAmps(qureg, reals, imags)

    def call_static(self, qureg:  str, reals:  Union[str, np.ndarray], imags: Union[str, np.ndarray]) -> List[str]:
        """
        Static call of initStateFromAmps

        Args:
            qureg: The name of the previously created quantum register as a string
            reals: If reals is of np.ndarray type, the real numbers in reals will be initialised on the quantum register
                    if reals is a string, it must be the name of a previously defined qreal array in C
            imags: If imags is of np.ndarray type, the real numbers in imags will be initialised on the quantum register
                    if imags is a string, it must be the name of a previously defined qreal array in C
        """
        lines = []
        if not isinstance(reals, str):
            print(reals)
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal reals_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(reals)):
                lines.append('reals_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=reals[i]))
            reals = 'reals_{t}'.format(t=t)
        if not isinstance(imags, str):
            print(imags)
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal imags_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(imags)):
                lines.append('imags_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=imags[i]))
            imags = 'imags_{t}'.format(t=t)
        lines.append("initStateFromAmps({qureg}, {reals}, {imags});".format(
            qureg=qureg, reals=reals, imags=imags))
        return lines


class initStateDebug(_PYQUEST):
    def call_interactive(self, qureg: tqureg) -> None:
        return quest.initStateDebug(qureg)

    def call_static(self, qureg:  str) -> List[str]:
        call = "initStateDebug({qureg});".format(
            qureg=qureg)
        return [call]


class setAmps(_PYQUEST):
    """
    Set the values of elements of the statvector in a quantum register

    Args:
        qureg: The quantum register
        startind: The index of the first element of the statevector that is set
        reals: the new real values of the elements of the statevector between startind and startind+numamps
        imgas: the new imaginary values of the elements of the statevector between startind and startind+numamps
        numaps: the number of new values that are set in the statevector
    """

    def call_interactive(self,
                         qureg: tqureg,
                         startind: int,
                         reals: typing.Union[np.ndarray, typing.List[float]],
                         imags: typing.Union[np.ndarray, typing.List[float]],
                         numamps: int
                         ) -> None:
        reals = list(reals)
        imags = list(imags)
        assert len(reals) == np.max(np.shape(reals))
        assert len(imags) == np.max(np.shape(imags))
        assert len(reals) == numamps
        assert len(reals) == numamps
        if qureg.isDensityMatrix:
            warnings.warn('qureg has to be a wavefunction qureg'
                          + ' but density matrix qureg was used', RuntimeWarning)
            return None
        else:
            return quest.setAmps(qureg, startind, reals, imags, numamps)

    def call_static(self, qureg:  str,
                    startind: Union[int, str],
                    reals: 'str',
                    imags: 'str',
                    numamps: Union[int, str]) -> List[str]:
        """
        Static call of setAmps

        Args:
            qureg: The name of the previously created quantum register as a string
            startind: if startind is int, the number is used directly as startindex, 
                    if it is a string it must be the name of a previously defined C-Variable of type int
            reals: If reals is of np.ndarray type, the real numbers in reals will be initialised on the quantum register
                    if reals is a string, it must be the name of a previously defined qreal array in C
            imags: If imags is of np.ndarray type, the real numbers in imags will be initialised on the quantum register
                    if imags is a string, it must be the name of a previously defined qreal array in C
            numamps: if numamps is int, the number is used directly as number of amplitudes, 
                    if it is a string it must be the name of a previously defined C-Variable of type int
        """
        lines = []
        if not isinstance(reals, str):
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal reals_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(reals)):
                lines.append('reals_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=reals[i]))
            reals = 'reals_{t}'.format(t=t)
        if not isinstance(imags, str):
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal imags_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(imags)):
                lines.append('imags_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=imags[i]))
            imags = 'imags_{t}'.format(t=t)
        lines.append("setAmps({qureg}, {startind},{reals},{imags},{numamps});".format(
            qureg=qureg,
            startind=startind,
            reals=reals,
            imags=imags,
            numamps=numamps))
        return lines

# cant find it in the API


class setDensityAmps(_PYQUEST):
    """
    Set the values of elements of the vector representation of the density matrix in a quantum register

    Args:
        qureg: The quantum register of a density matrix
        startind: The index of the first element of the density matrix that is set
        reals: the new real values of the elements of the density matrix between startind and startind+numamps
        imgas: the new imaginary values of the elements of the density matrix between startind and startind+numamps
        numaps: the number of new values that are set in the density matrix
    """

    def call_interactive(self,
                         qureg: tqureg,
                         startind: int,
                         reals: typing.Union[np.ndarray, typing.List[float]],
                         imags: typing.Union[np.ndarray, typing.List[float]],
                         numamps: int
                         ) -> None:
        reals = list(reals)
        imags = list(imags)
        assert len(reals) == np.max(np.shape(reals))
        assert len(imags) == np.max(np.shape(imags))
        assert len(reals) == numamps
        assert len(reals) == numamps
        if not qureg.isDensityMatrix:
            warnings.warn('qureg has to be a density matrix qureg'
                          + ' but wavefunction qureg was used', RuntimeWarning)
            return None
        else:
            return quest.statevec_setAmps(qureg, startind, reals, imags, numamps)

    def call_static(self, qureg:  str,
                    startind: Union[int, str],
                    reals: 'str',
                    imags: 'str',
                    numamps: Union[int, str]) -> List[str]:
        """
        Static call of setDensityAmps

        Args:
            qureg: The name of the previously created quantum register as a string
            startind: if startind is int, the number is used directly as startindex, 
                    if it is a string it must be the name of a previously defined C-Variable of type int
            reals: If reals is of np.ndarray type, the real numbers in reals will be initialised on the quantum register
                    if reals is a string, it must be the name of a previously defined qreal array in C
            imags: If imags is of np.ndarray type, the real numbers in imags will be initialised on the quantum register
                    if imags is a string, it must be the name of a previously defined qreal array in C
            numamps: if numamps is int, the number is used directly as number of amplitudes, 
                    if it is a string it must be the name of a previously defined C-Variable of type int
        """
        lines = []
        if not isinstance(reals, str):
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal reals_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(reals)):
                lines.append('reals_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=reals[i]))
            reals = 'reals_{t}'.format(t=t)
        if not isinstance(imags, str):
            t = '{}'.format(uuid.uuid4().hex)
            lines.append('qreal imags_{t}[{length}];'.format(t=t, length=len(reals),))
            for i in range(len(imags)):
                lines.append('imags_{t}[{index}] = {val};'.format(
                    t=t, index=i, val=imags[i]))
            imags = 'imags_{t}'.format(t=t)
        lines.append("statevec_setAmps({qureg}, {startind},{reals},{imags},{numamps});".format(
            qureg=qureg,
            startind=startind,
            reals=reals,
            imags=imags,
            numamps=numamps))
        return lines
