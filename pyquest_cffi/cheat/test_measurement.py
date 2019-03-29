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


def test_calcPurity():
    env = utils.createQuestEnv()()
    qureg = utils.createQureg()(2, env)
    cheat.initZeroState()(qureg)
    with npt.assert_warns(RuntimeWarning):
        purity = cheat.measurement.calcPurity()(qureg)
    assert(purity is None)
    qureg = utils.createDensityQureg()(2,env)
    cheat.initZeroState()(qureg)
    purity = cheat.measurement.calcPurity()(qureg)
    npt.assert_equal(purity, 1)

def test_basis_state_to_index():
    basis_state = [0, 0, 1, 0, 1]
    index = cheat.basis_state_to_index(basis_state, endianness='little')
    npt.assert_array_equal(index, 20)
    index = cheat.basis_state_to_index(basis_state, endianness='big')
    npt.assert_array_equal(index, 5)


if __name__ == '__main__':
    pytest.main(sys.argv)
