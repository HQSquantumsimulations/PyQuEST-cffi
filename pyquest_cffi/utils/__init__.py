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

"""Utilities to create and destroy environment etc.

.. autosummary::
    :toctree: generated/

    pyquest_cffi.utils.dataoperation
    pyquest_cffi.utils.reporting
    pyquest_cffi.utils.compilation_utils

"""

from pyquest_cffi.utils.dataoperation import (
    createQureg,
    createDensityQureg,
    createQuestEnv,
    destroyQureg,
    destroyQuestEnv
)
from pyquest_cffi.utils.reporting import (
    reportQuESTEnv,
    reportQuregParams,
    reportStateToScreen
)
from pyquest_cffi.utils.compilation_utils import (
    defineVariable,
    createProgrammPreamble,
    createProgrammEnd,
    QuESTCompiler,
    write_code_to_disk,
    callCompiledQuestProgramm
)