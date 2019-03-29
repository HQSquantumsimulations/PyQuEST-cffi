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

"""Provides access to the wavefunction or density matrix.

.. autosummary::
    :toctree: generated/

    pyquest_cffi.cheat.measurement
    pyquest_cffi.cheat.initialization

"""

from pyquest_cffi.cheat.measurement import (
    calcFidelity,
    calcInnerProduct,
    calcProbOfOutcome,
    calcPurity,
    calcTotalProb,
    getAbsoluteValSquaredatIndex,
    getDensityMatrixatRowColumn,
    getStateVectoratIndex,
    getAmp,
    getDensityAmp,
    getProbAmp,
    getRealAmp,
    getImagAmp,
    basis_state_to_index,
    getStateVector,
    getDensityMatrix,
    getOccupationProbability,
    getExpectationValue,
    getRepeatedMeasurement)
from pyquest_cffi.cheat.initialization import (
    initClassicalState,
    initPlusState,
    initPureState,
    initStateDebug,
    initStateFromAmps,
    initZeroState,
    setAmps,
    setDensityAmps
)
