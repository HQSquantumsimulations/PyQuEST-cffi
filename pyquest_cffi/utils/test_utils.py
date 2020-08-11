"""Testing utils (dataoperators and reporting) in PyQuest-cffi"""
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
from pyquest_cffi import cheat
from pyquest_cffi import utils


def test_QuestEnv() -> None:
    """Testing the creation, destruction and reporting of a QuestEnv"""
    env = utils.createQuestEnv()()
    dm = utils.createDensityQureg()(1, env)
    utils.reportQuESTEnv()(env=env)
    result_type = utils.reportQuESTEnv().restype
    argument_type = utils.reportQuESTEnv().argtype

    assert result_type == 'void'
    assert argument_type == ['QuESTEnv']

    utils.destroyQuestEnv()(env=env)


def test_QuReg() -> None:
    """Testing the creation, cloning, destruction and reporting of a QuReg"""
    env = utils.createQuestEnv()()
    wave_qureg = utils.createQureg()(num_qubits=2, env=env)
    cheat.initZeroState()(qureg=wave_qureg)
    density_qureg = utils.createDensityQureg()(num_qubits=2, env=env)
    cheat.initZeroState()(qureg=density_qureg)
    cloned_qureg = utils.createCloneQureg()(qureg=density_qureg, env=env)
    cheat.initZeroState()(qureg=cloned_qureg)

    try:
        [wave_string, density_string, cloned_string] = ['', '', '']
        cheat.getEnvironmentString()(env=env, qureg=wave_qureg, string=wave_string)
        cheat.getEnvironmentString()(env=env, qureg=density_qureg, string=density_string)
        cheat.getEnvironmentString()(env=env, qureg=cloned_qureg, string=cloned_string)
        assert cloned_string == density_string
        cheat.getEnvironmentString()(env=env, qureg=cloned_qureg, string=cloned_string)
        assert cloned_string == wave_string
    except NotImplementedError:
        pass  # getEnvironmentString unittest not implemented

    assert wave_qureg.isDensityMatrix == False
    assert density_qureg.isDensityMatrix == True
    assert cloned_qureg.isDensityMatrix == True
    assert np.all(cheat.getDensityMatrix()(qureg=density_qureg)
                  == cheat.getDensityMatrix()(qureg=cloned_qureg))
    assert not np.all(cheat.getStateVector()(qureg=wave_qureg)
                      == cheat.getDensityMatrix()(qureg=cloned_qureg))

    npt.assert_raises(TypeError, utils.cloneQureg(), cloned_qureg, wave_qureg)

    to_be_cloned = utils.createDensityQureg()(num_qubits=3, env=env)
    cheat.initZeroState()(qureg=to_be_cloned)
    clone_into = utils.createDensityQureg()(num_qubits=3, env=env)
    cheat.initZeroState()(qureg=clone_into)
    utils.cloneQureg()(clone_into, to_be_cloned)
    assert clone_into.isDensityMatrix == True

    result_type_list = ['', '', '']
    argument_type_list = [[''], [''], ['']]
    for qureg in [wave_qureg, density_qureg, cloned_qureg]:
        utils.reportQuregParams()(qureg=qureg)
        utils.reportState()(qureg=qureg)
        utils.reportStateToScreen()(qureg=qureg, env=env)
        result_type_list[0] = utils.reportQuregParams().restype
        argument_type_list[0] = utils.reportQuregParams().argtype
        result_type_list[1] = utils.reportState().restype
        argument_type_list[1] = utils.reportState().argtype
        result_type_list[2] = utils.reportStateToScreen().restype
        argument_type_list[2] = utils.reportStateToScreen().argtype

    npt.assert_array_equal(result_type_list, ["void", "void", "void"])
    npt.assert_array_equal(argument_type_list, [["Qureg"], ["Qureg"], ["Qureg", "QuESTEnv", "int"]])

    for qureg in [wave_qureg, density_qureg, cloned_qureg]:
        utils.destroyQureg()(env=env, qubits=qureg)


def test_PauliHamil() -> None:
    """Testing the creation, destruction and reporting of a PauliHamil"""
    env = utils.createQuestEnv()()
    pauli_hamil = utils.createPauliHamil()(number_qubits=5, number_pauliprods=3)
    cheat.initPauliHamil()(pauli_hamil=pauli_hamil,
                           coeffs=[0.1, 0.2, 0.3, 0.2, 0.4],
                           codes=[[2, 1, 3], [1, 2, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0]])
    utils.reportPauliHamil()(pauli_hamil=pauli_hamil)
    result_type = utils.reportPauliHamil().restype
    argument_type = utils.reportPauliHamil().argtype

    assert result_type == "floats/integers"
    assert argument_type == ["Coefficient", "PauliMatrix"]

    utils.destroyPauliHamil()(pauli_hamil=pauli_hamil)



if __name__ == '__main__':
    pytest.main(sys.argv)
