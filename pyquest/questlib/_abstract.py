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

import abc
import typing


tquestenv = typing.TypeVar("QuESTEnv")
tqureg = typing.TypeVar("Qureg")


class _PYQUEST(abc.ABC):

    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    def __call__(self, *args, **kwargs):
        if self.interactive:
            return self.call_interactive(*args, **kwargs)
        else:
            return self.call_static(*args, **kwargs)

    @abc.abstractmethod
    def call_interactive(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def call_static(self, *args, **kwargs):
        pass

    @property
    def interactive(self) -> bool:
        return self._interactive
