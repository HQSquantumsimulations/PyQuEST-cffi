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
from pyquest_cffi.questlib import quest, _PYQUEST, tquestenv, tqureg


class reportQuESTEnv(_PYQUEST):
    """
    Report the properties of the QuEST simulation environment to stdout

    Args:
        env: QuEST environment for which the parameter are reported
    """
    def call_interactive(self, env: tquestenv) -> None:
        return quest.reportQuESTEnv(env)

    def call_static(self, env: str):
        call = "reportQuESTEnv({env});".format(
            env=env)
        return [call]

    @property
    def restype(self) -> str:
        return "void"

    @property
    def argtype(self) -> typing.List[str]:
        return ["QuESTEnv"]


class reportQuregParams(_PYQUEST):
    """
    Reports the parameters of a quantum register to stdout

    Args:
        qureg: Quantum register for which the parameter are reported
    """
    def call_interactive(self, qureg: tqureg) -> None:
        return quest.reportQuregParams(qureg)

    def call_static(self, qureg: str):
        call = "reportQuregParams({qureg});".format(
            qureg=qureg)
        return [call]

    @property
    def restype(self) -> str:
        return "void"

    @property
    def argtype(self) -> typing.List[str]:
        return ["Qureg"]


class reportState(_PYQUEST):
    def call_interactive(self, qureg: tqureg) -> None:
        return quest.reportState(qureg)

    def call_static(self):
        raise NotImplementedError

    @property
    def restype(self) -> str:
        return "void"

    @property
    def argtype(self) -> typing.List[str]:
        return ["Qureg"]


class reportStateToScreen(_PYQUEST):
    """
    Report statevector or density matrix in a qureg to stdout

    Args:
        qureg: the quantum register
        env: the environment of the quantum register
    """
    def call_interactive(self, qureg: tqureg, env: tquestenv, a: int= 0) -> None:
        return quest.reportStateToScreen(qureg, env, a)

    def call_static(self, qureg: str, env: str, *args, **kwargs) -> str:
        call = "reportStateToScreen({qureg}, {env}, 0);".format(
            qureg=qureg, env=env)
        return [call]

    @property
    def restype(self) -> str:
        return "void"

    @property
    def argtype(self) -> typing.List[str]:
        return ["Qureg", "QuESTEnv", "int"]
