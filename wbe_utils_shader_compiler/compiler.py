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

from pathlib import Path
from typing import Any

from . import _native
from .kernel import Kernel, KernelBinding, KernelParameter, KernelPush, KernelSystemValue


DEFAULT_PROFILE = "spirv_1_6"

KERNEL_ATTRIBUTE = "wbe_kernel"
BIND_ATTRIBUTE = "wbe_bind"
PUSH_ATTRIBUTE = "wbe_push"

ACCESS_READ = "read"
ACCESS_WRITE = "write"
ACCESS_READ_WRITE = "read_write"

PUSH_SCALAR_TYPES = ("uint32", "int32", "float32")

# The generated entry point's push-data struct and its variable.
PUSH_DATA_TYPE = "WBEKernelPushData"
PUSH_DATA_VARIABLE = "wbe_kernel_push_data"


class WBEUtilsShaderCompiler:
    """
    Reflects `[wbe::kernel]` Slang functions and generates the entry points that call them.

    A kernel is an ordinary function marked `[wbe::kernel(x, y, z)]` with its workgroup size. Every parameter is either a
    system value (it has an `SV_` semantic), a resource bound with `[wbe::bind(resource_id, access?, desc_type?)]`, or a
    four-byte scalar supplied as push data with `[wbe::push(value)]`. The engine entry point declares one push-data struct
    of `DescriptorHandle` fields annotated `wbe::desc` and scalars annotated `wbe::push_constant`, the contract the White
    Bird Engine pipeline compiler reads; the test entry point binds plain global resources and uniforms instead. Both
    include the kernel's source, which must declare the `wbe::kernel`, `wbe::bind`, `wbe::push`, `wbe::desc`, and
    `wbe::push_constant` attributes (the engine's `utils/utils.slang` does).

    The compiler owns one Slang global session; it is not thread safe.
    """

    def __init__(self, profile: str = DEFAULT_PROFILE) -> None:
        """
        :param profile: The target profile reflection checks kernels with, optionally followed by `+`-separated
            capabilities, as slangc's -profile.
        """
        self._profile = profile
        self._reflector = _native.KernelReflector()

    def reflect_kernel(self, source_path: Path, kernel_name: str, include_dirs: list[Path] | None = None,
            macro_definitions: list[tuple[str, str]] | None = None) -> Kernel:
        """
        Reflect and validate one kernel.

        :param source_path: The source module declaring the kernel.
        :param kernel_name: The kernel function; it must not be generic or overloaded.
        :param include_dirs: Include directories, in search order.
        :param macro_definitions: The (name, value) macro definitions to preprocess with.
        :return: The kernel, with every binding's descriptor type and access resolved.
        :raises RuntimeError: If the module fails to compile, or the function is not a valid kernel.
        """
        resolved_source_path = source_path.resolve()
        raw_kernel = self._reflector.reflect_kernel(
            resolved_source_path.as_posix(),
            kernel_name,
            [include_dir.as_posix() for include_dir in include_dirs or []],
            list(macro_definitions or []),
            self._profile,
        )
        return _parse_kernel(raw_kernel, resolved_source_path)

    def generate_entry_source(self, kernel: Kernel, entry_name: str = "main") -> str:
        """
        Generate the engine entry point of a kernel.

        :param kernel: The kernel.
        :param entry_name: The name of the generated compute entry point.
        :return: Slang source that includes the kernel's source and declares the entry point.
        """
        lines = _source_header(kernel)
        has_push_data = bool(kernel.bindings or kernel.pushes)
        if has_push_data:
            lines.append(f"struct {PUSH_DATA_TYPE} {{")
            for parameter in kernel.parameters:
                if isinstance(parameter, KernelBinding):
                    lines.append(f'    [wbe::desc("{parameter.desc_type}", "{parameter.resource_id}", "{parameter.access}")]')
                    lines.append(f"    DescriptorHandle<{parameter.type_name}> {parameter.name};")
                elif isinstance(parameter, KernelPush):
                    lines.append(f'    [wbe::push_constant("{parameter.value}")]')
                    lines.append(f"    {parameter.type_name} {parameter.name};")
            lines.append("};")
            lines.append("")
            lines.append("[vk::push_constant]")
            lines.append(f"ConstantBuffer<{PUSH_DATA_TYPE}> {PUSH_DATA_VARIABLE};")
            lines.append("")
        arguments = [
            parameter.name if isinstance(parameter, KernelSystemValue) else f"{PUSH_DATA_VARIABLE}.{parameter.name}"
            for parameter in kernel.parameters
        ]
        lines.extend(_entry_point(kernel, entry_name, arguments))
        return "\n".join(lines)

    def generate_test_source(self, kernel: Kernel, entry_name: str = "main") -> str:
        """
        Generate a test entry point of a kernel, which binds plain global resources and uniforms.

        Each binding and push parameter becomes a global of its type, named as the parameter without a leading `p_`,
        so a test binds them by those names.

        :param kernel: The kernel.
        :param entry_name: The name of the generated compute entry point.
        :return: Slang source that includes the kernel's source and declares the globals and the entry point.
        :raises RuntimeError: If two parameters map to the same global name.
        """
        lines = _source_header(kernel)
        arguments: list[str] = []
        global_names: set[str] = set()
        for parameter in kernel.parameters:
            if isinstance(parameter, KernelSystemValue):
                arguments.append(parameter.name)
                continue
            global_name = kernel_test_global_name(parameter.name)
            if global_name in global_names:
                raise RuntimeError(
                    f"Kernel '{kernel.name}' in {kernel.source_path.as_posix()}: parameters map to the test global "
                    f"'{global_name}' more than once.")
            global_names.add(global_name)
            uniform = "uniform " if isinstance(parameter, KernelPush) else ""
            lines.append(f"{uniform}{parameter.type_name} {global_name};")
            arguments.append(global_name)
        if global_names:
            lines.append("")
        lines.extend(_entry_point(kernel, entry_name, arguments))
        return "\n".join(lines)


def kernel_test_global_name(parameter_name: str) -> str:
    """The global a test entry point declares for a kernel parameter: its name without a leading `p_`."""
    return parameter_name.removeprefix("p_") if parameter_name != "p_" else parameter_name


def _source_header(kernel: Kernel) -> list[str]:
    return [
        f"// Generated by wbe-utils-shader-compiler from kernel '{kernel.name}' in {kernel.source_path.as_posix()}.",
        "// Do not edit.",
        f'#include "{kernel.source_path.as_posix()}"',
        "",
    ]


def _entry_point(kernel: Kernel, entry_name: str, arguments: list[str]) -> list[str]:
    x, y, z = kernel.thread_group_size
    parameters = ", ".join(
        f"{parameter.type_name} {parameter.name} : {parameter.semantic}" for parameter in kernel.system_values)
    return [
        '[shader("compute")]',
        f"[numthreads({x}, {y}, {z})]",
        f"void {entry_name}({parameters}) {{",
        f"    {kernel.name}({', '.join(arguments)});",
        "}",
        "",
    ]


def _kernel_error(kernel_name: str, source_path: Path, message: str) -> RuntimeError:
    return RuntimeError(f"Kernel '{kernel_name}' in {source_path.as_posix()}: {message}")


def _find_attributes(raw_attributes: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [attribute for attribute in raw_attributes if attribute["name"] == name]


def _parse_kernel(raw_kernel: dict[str, Any], source_path: Path) -> Kernel:
    name: str = raw_kernel["name"]
    kernel_attributes = _find_attributes(raw_kernel["attributes"], KERNEL_ATTRIBUTE)
    if len(kernel_attributes) != 1:
        raise _kernel_error(name, source_path, "the function must be marked with exactly one [wbe::kernel(x, y, z)].")
    thread_group_size = kernel_attributes[0]["arguments"]
    if (len(thread_group_size) != 3
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in thread_group_size)):
        raise _kernel_error(name, source_path, f"[wbe::kernel] needs three positive integers, not {thread_group_size}.")
    parameters = tuple(
        _parse_parameter(raw_parameter, name, source_path) for raw_parameter in raw_kernel["parameters"])
    return Kernel(name, source_path, (thread_group_size[0], thread_group_size[1], thread_group_size[2]), parameters)


def _parse_parameter(raw_parameter: dict[str, Any], kernel_name: str, source_path: Path) -> KernelParameter:
    name: str = raw_parameter["name"]
    binds = _find_attributes(raw_parameter["attributes"], BIND_ATTRIBUTE)
    pushes = _find_attributes(raw_parameter["attributes"], PUSH_ATTRIBUTE)
    semantic = raw_parameter["semantic"]
    if semantic is not None:
        if not semantic.upper().startswith("SV_"):
            raise _kernel_error(kernel_name, source_path,
                f"parameter '{name}' has the semantic '{semantic}'; only system-value (SV_) semantics are supported.")
        if binds or pushes:
            raise _kernel_error(kernel_name, source_path,
                f"system-value parameter '{name}' must not be marked [wbe::bind] or [wbe::push].")
        return KernelSystemValue(name, raw_parameter["type_name"], semantic)
    if len(binds) + len(pushes) != 1:
        raise _kernel_error(kernel_name, source_path,
            f"parameter '{name}' must have a system-value semantic, or exactly one [wbe::bind] or [wbe::push].")
    if binds:
        return _parse_binding(raw_parameter, binds[0]["arguments"], kernel_name, source_path)
    return _parse_push(raw_parameter, pushes[0]["arguments"], kernel_name, source_path)


def _inferred_binding(raw_parameter: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    """The descriptor type of a parameter and its allowed accesses, the default first; None for unsupported types."""
    kind = raw_parameter["kind"]
    if kind == "constant_buffer":
        return "uniform_buffer", (ACCESS_READ,)
    if kind == "sampler_state":
        return "sampler", (ACCESS_READ,)
    if kind != "resource":
        return None
    shape = raw_parameter["resource_shape"]
    access = raw_parameter["resource_access"]
    if access == ACCESS_READ:
        accesses: tuple[str, ...] = (ACCESS_READ,)
    elif access == ACCESS_READ_WRITE:
        accesses = (ACCESS_READ_WRITE, ACCESS_READ, ACCESS_WRITE)
    else:
        return None
    if shape in ("structured_buffer", "byte_address_buffer"):
        return "storage_buffer", accesses
    if shape in ("texture_1d", "texture_2d", "texture_3d", "texture_cube"):
        return ("sampled_image" if access == ACCESS_READ else "storage_image"), accesses
    if shape == "acceleration_structure" and access == ACCESS_READ:
        return "acceleration_structure", accesses
    return None


def _parse_binding(raw_parameter: dict[str, Any], arguments: list[Any], kernel_name: str,
        source_path: Path) -> KernelBinding:
    name: str = raw_parameter["name"]
    type_name: str = raw_parameter["type_name"]
    if len(arguments) != 3 or any(not isinstance(argument, str) for argument in arguments):
        raise _kernel_error(kernel_name, source_path,
            f"[wbe::bind] of parameter '{name}' must have string arguments (resource_id, access, desc_type).")
    resource_id, access, desc_type = arguments
    if not resource_id:
        raise _kernel_error(kernel_name, source_path, f"[wbe::bind] of parameter '{name}' needs a resource id.")
    inferred = _inferred_binding(raw_parameter)
    if inferred is None:
        raise _kernel_error(kernel_name, source_path,
            f"parameter '{name}' of type {type_name} cannot be bound; bind a constant buffer, structured or byte "
            "address buffer, texture, sampler, or acceleration structure.")
    inferred_desc_type, accesses = inferred
    if desc_type and desc_type != inferred_desc_type:
        raise _kernel_error(kernel_name, source_path,
            f"[wbe::bind] of parameter '{name}' declares '{desc_type}', but its type {type_name} is a "
            f"'{inferred_desc_type}'.")
    if access and access not in accesses:
        raise _kernel_error(kernel_name, source_path,
            f"[wbe::bind] of parameter '{name}' declares the access '{access}'; its type {type_name} allows "
            f"{', '.join(repr(allowed) for allowed in accesses)}.")
    return KernelBinding(name, type_name, resource_id, inferred_desc_type, access or accesses[0])


def _parse_push(raw_parameter: dict[str, Any], arguments: list[Any], kernel_name: str, source_path: Path) -> KernelPush:
    name: str = raw_parameter["name"]
    type_name: str = raw_parameter["type_name"]
    if len(arguments) != 1 or not isinstance(arguments[0], str) or not arguments[0]:
        raise _kernel_error(kernel_name, source_path, f"[wbe::push] of parameter '{name}' needs a non-empty value.")
    if raw_parameter["kind"] != "scalar" or raw_parameter["scalar_type"] not in PUSH_SCALAR_TYPES:
        raise _kernel_error(kernel_name, source_path,
            f"push parameter '{name}' of type {type_name} must be a uint, int, or float.")
    return KernelPush(name, type_name, arguments[0])
