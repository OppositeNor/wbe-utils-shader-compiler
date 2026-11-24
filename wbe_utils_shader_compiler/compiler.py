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
from .kernel import Kernel, KernelBinding, KernelParameter, KernelPush, KernelPushData, KernelSystemValue
from .slang_compiler import SlangCompiler


ManifestResource = dict[str, Any]


DEFAULT_PROFILE = "spirv_1_6"

KERNEL_ATTRIBUTE = "wbe_kernel"
BIND_ATTRIBUTE = "wbe_bind"
PUSH_ATTRIBUTE = "wbe_push"
PUSH_DATA_ATTRIBUTE = "wbe_push_data"

ACCESS_READ = "read"
ACCESS_WRITE = "write"
ACCESS_READ_WRITE = "read_write"

PUSH_SCALAR_TYPES = ("uint32", "int32", "float32")

# The generated entry point's push-data struct and its variable.
PUSH_DATA_TYPE = "WBEKernelPushData"
PUSH_DATA_VARIABLE = "wbe_kernel_push_data"


class WBEUtilsShaderCompiler:
    """
    Compiles White Bird Engine shader resources, and reflects `[wbe::kernel]` Slang functions and generates the entry
    points that call them.

    `compile()` turns one `shader` manifest resource into its runtime resource: every variant is compiled to SPIR-V and
    normalized metadata with SlangCompiler, and a kernel resource compiles the entry point generated for its kernel.

    A kernel is an ordinary function marked `[wbe::kernel(x, y, z)]` with its workgroup size. Every parameter is either a
    system value (it has an `SV_` semantic), a resource bound with `[wbe::bind(resource_id, access?, desc_type?)]`, or a
    four-byte scalar supplied as push data with `[wbe::push(value)]`, or a struct (or array of one) of `wbe::desc` and
    `wbe::push_constant` fields embedded in the push data with `[wbe::push_data]`. The engine entry point declares one push-data struct
    of `DescriptorHandle` fields annotated `wbe::desc` and scalars annotated `wbe::push_constant`, the contract the White
    Bird Engine pipeline compiler reads; the test entry point binds plain global resources and uniforms instead. Both
    include the kernel's source, which must declare the `wbe::kernel`, `wbe::bind`, `wbe::push`, `wbe::desc`, and
    `wbe::push_constant` attributes (the engine's `utils/utils.slang` does).

    The compiler owns one Slang global session; it is not thread safe.
    """

    def __init__(self, profile: str = DEFAULT_PROFILE, include_dirs: list[Path] | None = None,
            macro_definitions: list[tuple[str, str]] | None = None, generated_source_dir: Path | None = None,
            debug_info: bool = False) -> None:
        """
        :param profile: The target profile shaders compile and kernels are reflected with, optionally followed by
            `+`-separated capabilities, as slangc's -profile.
        :param include_dirs: Include directories, in search order.
        :param macro_definitions: The (name, value) macro definitions every shader compiles and every kernel is
            reflected with, besides a variant's own; a variant must not define them again.
        :param generated_source_dir: The directory receiving the entry points generated for kernel resources; required
            to compile them.
        :param debug_info: Whether compiled shaders carry full debug information.
        """
        self._profile = profile
        self._include_dirs = list(include_dirs or [])
        self._macro_definitions = list(macro_definitions or [])
        self._generated_source_dir = generated_source_dir
        self._slang_compiler = SlangCompiler(profile, self._include_dirs, debug_info)
        self._reflector = _native.KernelReflector()

    def get_supported_resource_types(self) -> list[str]:
        return ["shader"]

    def compile(self, resource: ManifestResource, manifest_path: Path, res_dir: Path,
            res_output_dir: Path) -> list[ManifestResource]:
        """
        Compile one shader resource.

        The resource names its source with `src`, relative to `res_dir`, and compiles either every `[shader(...)]` entry
        point of it, one `stage` and `entry`, or the entry point generated for its `kernel`. Each of its `variants`
        (`{ name, values = { { name, value } }, default? }`; one `default` variant without macros when absent) compiles
        to `<source stem>.<id>[.<variant>].spv` and `.json` next to the source under `res_output_dir`, with the variant
        name only when there are several; a kernel variant's generated entry point is written at the same path, with a
        `.slang` extension, under the generated source directory.

        :param resource: The shader resource.
        :param manifest_path: The manifest declaring it, for diagnostics.
        :param res_dir: The resource directory `src` is relative to.
        :param res_output_dir: The output resource directory.
        :return: The runtime resource: the declaration without `src` and `kernel`, with `variants` replaced by the
            compiled variants (`{ name, path, metadata, is_default }`, paths relative to `res_output_dir`).
        :raises RuntimeError: If the resource is invalid or a variant fails to compile.
        """
        result = dict(resource)
        source = result.get("src")
        if not isinstance(source, str) or not source:
            raise RuntimeError(f"Shader resource must provide a src field in {manifest_path}: {resource}.")
        resource_id = result.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            raise RuntimeError(f"Shader resource must provide a non-empty id in {manifest_path}: {resource}.")
        if Path(resource_id).name != resource_id:
            raise RuntimeError(f"Shader resource id must be usable as a file name in {manifest_path}: {resource}.")
        stage = result.get("stage")
        if stage is not None and not isinstance(stage, str):
            raise RuntimeError(f"Shader resource stage must be a string in {manifest_path}: {resource}.")
        entry = result.get("entry")
        if entry is not None and not isinstance(entry, str):
            raise RuntimeError(f"Shader resource entry must be a string in {manifest_path}: {resource}.")
        # A kernel resource compiles the entry point generated for one [wbe::kernel] function of its source.
        kernel = result.get("kernel")
        if kernel is not None:
            if not isinstance(kernel, str) or not kernel:
                raise RuntimeError(f"Shader resource kernel must be a non-empty string in {manifest_path}: {resource}.")
            if stage is not None or entry is not None:
                raise RuntimeError(
                    f"Shader resource must declare either a kernel, or a stage and entry, in {manifest_path}: {resource}.")
            if self._generated_source_dir is None:
                raise RuntimeError(
                    f"Shader resource declares a kernel, but the compiler has no generated source directory, in "
                    f"{manifest_path}: {resource}.")
        source_rel = Path(source)
        source_path = res_dir / source_rel
        if source_path.suffix != ".slang":
            raise RuntimeError(f"Unsupported shader file type: {source_path.suffix}.")
        if not source_path.is_file():
            raise RuntimeError(f"Shader source file not found: {source_path.as_posix()}.")
        declared_variants = _read_variants(result, manifest_path)
        compiler_macros = {name for name, _ in self._macro_definitions}
        for name, values, _ in declared_variants:
            redefined = sorted(compiler_macros.intersection(macro for macro, _ in values))
            if redefined:
                raise RuntimeError(
                    f"Shader resource variant '{name}' defines the compiler's macros {redefined} in {manifest_path}: "
                    f"{resource}.")
        variants: list[dict[str, Any]] = []
        for name, values, is_default in declared_variants:
            output_rel, metadata_rel = _get_output_files(
                source_rel, resource_id, name if len(declared_variants) > 1 else None)
            compiled_source_path = source_path
            if kernel is not None:
                compiled_source_path = self._write_kernel_entry(
                    source_path, output_rel.with_suffix(".slang"), kernel, values)
            self._slang_compiler.compile(compiled_source_path, res_output_dir / output_rel,
                res_output_dir / metadata_rel, stage, entry, [*self._macro_definitions, *values])
            variants.append({
                "name": name,
                "path": output_rel.as_posix(),
                "metadata": metadata_rel.as_posix(),
                "is_default": is_default,
            })
        result["variants"] = variants
        result.pop("src", None)
        result.pop("kernel", None)
        return [result]

    def reflect_kernel(self, source_path: Path, kernel_name: str, include_dirs: list[Path] | None = None,
            macro_definitions: list[tuple[str, str]] | None = None) -> Kernel:
        """
        Reflect and validate one kernel.

        :param source_path: The source module declaring the kernel.
        :param kernel_name: The kernel function; it must not be generic or overloaded.
        :param include_dirs: Include directories, in search order; the compiler's when None.
        :param macro_definitions: The (name, value) macro definitions to preprocess with, besides the compiler's.
        :return: The kernel, with every binding's descriptor type and access resolved.
        :raises RuntimeError: If the module fails to compile, or the function is not a valid kernel.
        """
        resolved_source_path = source_path.resolve()
        raw_kernel = self._reflector.reflect_kernel(
            resolved_source_path.as_posix(),
            kernel_name,
            [include_dir.as_posix() for include_dir in (self._include_dirs if include_dirs is None else include_dirs)],
            [*self._macro_definitions, *(macro_definitions or [])],
            self._profile,
        )
        return _parse_kernel(raw_kernel, resolved_source_path)

    def _write_kernel_entry(self, source_path: Path, generated_rel_path: Path, kernel_name: str,
            macro_definitions: list[tuple[str, str]]) -> Path:
        """Reflect a kernel with a variant's definitions and write its entry point; return the generated source."""
        assert self._generated_source_dir is not None
        source = self.generate_entry_source(self.reflect_kernel(source_path, kernel_name, None, macro_definitions))
        generated_path = self._generated_source_dir / generated_rel_path
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        # Rewriting identical content would only update the timestamp.
        if not generated_path.is_file() or generated_path.read_text(encoding="utf-8") != source:
            generated_path.write_text(source, encoding="utf-8")
        return generated_path

    def generate_entry_source(self, kernel: Kernel, entry_name: str = "main") -> str:
        """
        Generate the engine entry point of a kernel.

        :param kernel: The kernel.
        :param entry_name: The name of the generated compute entry point.
        :return: Slang source that includes the kernel's source and declares the entry point.
        """
        lines = _source_header(kernel)
        has_push_data = bool(kernel.bindings or kernel.pushes or kernel.push_data)
        if has_push_data:
            lines.append(f"struct {PUSH_DATA_TYPE} {{")
            for parameter in kernel.parameters:
                if isinstance(parameter, KernelBinding):
                    lines.append(f'    [wbe::desc("{parameter.desc_type}", "{parameter.resource_id}", "{parameter.access}")]')
                    lines.append(f"    DescriptorHandle<{parameter.type_name}> {parameter.name};")
                elif isinstance(parameter, KernelPush):
                    lines.append(f'    [wbe::push_constant("{parameter.value}")]')
                    lines.append(f"    {parameter.type_name} {parameter.name};")
                elif isinstance(parameter, KernelPushData):
                    lines.append(f"    {_declaration(parameter)};")
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
            if isinstance(parameter, KernelBinding):
                lines.append(f"{parameter.type_name} {global_name};")
            elif isinstance(parameter, KernelPushData):
                lines.append(f"uniform {_declaration(parameter, global_name)};")
            else:
                lines.append(f"uniform {parameter.type_name} {global_name};")
            arguments.append(global_name)
        if global_names:
            lines.append("")
        lines.extend(_entry_point(kernel, entry_name, arguments))
        return "\n".join(lines)


def kernel_test_global_name(parameter_name: str) -> str:
    """The global a test entry point declares for a kernel parameter: its name without a leading `p_`."""
    return parameter_name.removeprefix("p_") if parameter_name != "p_" else parameter_name


def _get_output_files(source_rel: Path, resource_id: str, variant_name: str | None) -> tuple[Path, Path]:
    """
    The SPIR-V and metadata files of one variant, relative to the output resource directory.

    Both sit next to the source as `<source stem>.<resource id>`, followed by `.<variant name>` when the resource has
    several variants, so resources sharing a source, which differ in id, and the variants of one resource never share an
    output.
    """
    output_name = f"{source_rel.stem}.{resource_id}"
    if variant_name is not None:
        output_name = f"{output_name}.{variant_name}"
    output_base = source_rel.with_name(output_name)
    return output_base.with_name(f"{output_base.name}.spv"), output_base.with_name(f"{output_base.name}.json")


def _read_variants(resource: ManifestResource, manifest_path: Path) -> list[tuple[str, list[tuple[str, str]], bool]]:
    """
    The (name, macro definitions, is default) of each variant of a shader resource, in declaration order.

    Each variant is `{ name, values, default }`: `values` lists the `{ name, value }` macro definitions it compiles with,
    and `default` marks the variant an empty selection names. Without a default, the first variant is the default. A
    resource without variants has one, named `default`, defining no macros.
    """
    raw_variants = resource.get("variants")
    if raw_variants is None:
        return [("default", [], True)]
    variants: list[tuple[str, list[tuple[str, str]], bool]] = []
    names: set[str] = set()
    default_index: int | None = None
    for raw_variant in raw_variants:
        name = raw_variant.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"Shader resource variant name must be a non-empty string in {manifest_path}: {resource}.")
        if name in names:
            raise RuntimeError(
                f"Shader resource variant '{name}' is declared more than once in {manifest_path}: {resource}.")
        names.add(name)
        if raw_variant.get("default", False):
            if default_index is not None:
                raise RuntimeError(
                    f"Shader resource declares more than one default variant in {manifest_path}: {resource}.")
            default_index = len(variants)
        values: list[tuple[str, str]] = []
        for raw_value in raw_variant.get("values", []):
            macro = raw_value["name"]
            if any(macro == defined for defined, _ in values):
                raise RuntimeError(
                    f"Shader resource variant '{name}' defines macro '{macro}' more than once in "
                    f"{manifest_path}: {resource}.")
            values.append((macro, raw_value["value"]))
        variants.append((name, values, False))
    if default_index is None:
        default_index = 0
    return [(name, values, index == default_index) for index, (name, values, _) in enumerate(variants)]


def _declaration(parameter: KernelPushData, name: str | None = None) -> str:
    """The declaration of a push-data parameter: `<type> <name>`, followed by `[<count>]` for an array."""
    declaration = f"{parameter.type_name} {name or parameter.name}"
    return f"{declaration}[{parameter.element_count}]" if parameter.element_count else declaration


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
    push_data = _find_attributes(raw_parameter["attributes"], PUSH_DATA_ATTRIBUTE)
    semantic = raw_parameter["semantic"]
    if semantic is not None:
        if not semantic.upper().startswith("SV_"):
            raise _kernel_error(kernel_name, source_path,
                f"parameter '{name}' has the semantic '{semantic}'; only system-value (SV_) semantics are supported.")
        if binds or pushes or push_data:
            raise _kernel_error(kernel_name, source_path,
                f"system-value parameter '{name}' must not be marked [wbe::bind], [wbe::push], or [wbe::push_data].")
        return KernelSystemValue(name, raw_parameter["type_name"], semantic)
    if len(binds) + len(pushes) + len(push_data) != 1:
        raise _kernel_error(kernel_name, source_path,
            f"parameter '{name}' must have a system-value semantic, or exactly one [wbe::bind], [wbe::push], or "
            "[wbe::push_data].")
    if binds:
        return _parse_binding(raw_parameter, binds[0]["arguments"], kernel_name, source_path)
    if push_data:
        return _parse_push_data(raw_parameter, kernel_name, source_path)
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


def _parse_push_data(raw_parameter: dict[str, Any], kernel_name: str, source_path: Path) -> KernelPushData:
    name: str = raw_parameter["name"]
    if raw_parameter["kind"] == "struct":
        return KernelPushData(name, raw_parameter["type_name"], 0)
    if (raw_parameter["kind"] == "array" and raw_parameter["element_kind"] == "struct"
            and raw_parameter["element_count"] > 0):
        return KernelPushData(name, raw_parameter["element_type_name"], raw_parameter["element_count"])
    raise _kernel_error(kernel_name, source_path,
        f"push-data parameter '{name}' of type {raw_parameter['type_name']} must be a struct or a sized array of one.")
