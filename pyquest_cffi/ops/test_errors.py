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

import pytest
import sys
import numpy as np
import numpy.testing as npt
from pyquest_cffi import ops
from pyquest_cffi import cheat
from pyquest_cffi import utils

@pytest.mark.parametrize("prob", list(np.arange(0, 1, 0.05)))
@pytest.mark.parametrize("gate_def", [(ops.applyOneQubitDampingError, 1),
                                        (ops.applyOneQubitDephaseError,1/2),
                                        (ops.applyOneQubitDepolariseError,3/4)])
def test_one_qubit_errors(prob, gate_def) -> None:

    op = gate_def[0]
    prob = prob*gate_def[1]
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(1, env)
    state = 1/2*np.array([1, 1, 1, 1])
    cheat.initPlusState()(dm)
    # cheat.initStateFromAmps()(dm, np.real(state), np.imag(state))
    # cheat.setDensityAmps()(qureg=dm, startind=0, reals=np.real(state), imags=np.imag(state), numamps=4)

    op()(qureg=dm, qubit=0, probability=prob)
    superop = op().superoperator_matrix(probability=prob)
    end_matrix = (superop @ state).reshape((2, 2))
    matrix = cheat.getDensityMatrix()(dm)
    npt.assert_array_almost_equal(matrix, end_matrix)


if __name__ == '__main__':
    pytest.main(sys.argv)
