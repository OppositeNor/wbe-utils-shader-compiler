# Copyright 2026 OppositeNor
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
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelSystemValue:
    """A kernel parameter with a system-value semantic, forwarded from the entry point's parameter of the same type."""

    name: str
    type_name: str
    semantic: str


@dataclass(frozen=True)
class KernelBinding:
    """
    A kernel parameter bound to a descriptor-heap resource with `[wbe::bind(resource_id, access?, desc_type?)]`.

    `desc_type` and `access` are resolved: inferred from the parameter type unless the attribute states them.
    """

    name: str
    type_name: str
    resource_id: str
    desc_type: str
    access: str


@dataclass(frozen=True)
class KernelPush:
    """A four-byte scalar kernel parameter supplied as push data with `[wbe::push(value)]`."""

    name: str
    type_name: str
    value: str


@dataclass(frozen=True)
class KernelPushData:
    """
    A kernel parameter of a struct type, or an array of one, embedded in the push data as-is with `[wbe::push_data]`.

    The struct declares its own `wbe::desc` handles and `wbe::push_constant` scalars, as hand-written push data does, so
    helpers that take it work unchanged. `type_name` is the struct; `element_count` is the array length, or 0.
    """

    name: str
    type_name: str
    element_count: int


KernelParameter = KernelSystemValue | KernelBinding | KernelPush | KernelPushData


@dataclass(frozen=True)
class Kernel:
    """
    A validated `[wbe::kernel(x, y, z)]` function: its source, workgroup size, and parameters in declaration order.
    """

    name: str
    source_path: Path
    thread_group_size: tuple[int, int, int]
    parameters: tuple[KernelParameter, ...]

    @property
    def bindings(self) -> list[KernelBinding]:
        return [parameter for parameter in self.parameters if isinstance(parameter, KernelBinding)]

    @property
    def pushes(self) -> list[KernelPush]:
        return [parameter for parameter in self.parameters if isinstance(parameter, KernelPush)]

    @property
    def push_data(self) -> list[KernelPushData]:
        return [parameter for parameter in self.parameters if isinstance(parameter, KernelPushData)]

    @property
    def system_values(self) -> list[KernelSystemValue]:
        return [parameter for parameter in self.parameters if isinstance(parameter, KernelSystemValue)]
