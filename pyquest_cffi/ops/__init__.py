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

"""Provides operators for unitary and error prone time evolution on digital quantum computers.

.. autosummary::
    :toctree: generated/

    pyquest_cffi.ops.ops
    pyquest_cffi.ops.errors

"""

from pyquest_cffi.ops.ops import (
    hadamard,
    sGate,
    tGate,
    pauliX,
    pauliY,
    pauliZ,
    phaseShift,
    rotateX,
    rotateY,
    rotateZ,
    rotateAroundAxis,
    rotateAroundSphericalAxis,
    compactUnitary,
    unitary)
from pyquest_cffi.ops.ops import (
    controlledPhaseFlip,
    controlledPhaseShift,
    controlledNot,
    controlledPauliY,
    controlledRotateX,
    controlledRotateY,
    controlledRotateZ,
    controlledRotateAroundAxis,
    controlledCompactUnitary,
    controlledUnitary,
    PiModuloWarning,
    MolmerSorensenXX,
    sqrtISwap,
    invSqrtISwap)
from pyquest_cffi.ops.ops import (
    multiControlledPhaseFlip,
    multiControlledPhaseShift,
    multiControlledUnitary)
from pyquest_cffi.ops.ops import measure
from pyquest_cffi.ops.errors import(
    applyOneQubitDephaseError,
    applyOneQubitDepolariseError,
    applyOneQubitDampingError,
    applyTwoQubitDephaseError,
    applyTwoQubitDepolariseError
)
