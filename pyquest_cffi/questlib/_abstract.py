"""Abstract classes of PyQuest-cffi"""
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
from typing import TypeVar, Any

tquestenv = TypeVar("QuESTEnv")
tqureg = TypeVar("Qureg")
paulihamil = TypeVar("PauliHamil")


class _PYQUEST(abc.ABC):

    def __init__(self, interactive: bool = True) -> None:
        self._interactive = True

    def __call__(self, *args, **kwargs) -> Any:
        return self.call_interactive(*args, **kwargs)

    @abc.abstractmethod
    def call_interactive(self, *args, **kwargs) -> Any:
        pass

    @property
    def interactive(self) -> bool:
        return self._interactive
