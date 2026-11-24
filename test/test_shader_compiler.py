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

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest
from wbe_utils_shader_compiler import (Kernel, KernelBinding, KernelPush, KernelPushData, KernelSystemValue,
    WBEUtilsShaderCompiler, kernel_test_global_name)


SHADER_DIR = Path(__file__).resolve().parent / "shaders"
KERNELS_PATH = SHADER_DIR / "kernels.slang"
# The profile the White Bird Engine compiles shaders with.
ENGINE_PROFILE = "spirv_1_6+spvDescriptorHeapEXT+vk_mem_model"


@pytest.fixture(scope="module")
def compiler() -> WBEUtilsShaderCompiler:
    return WBEUtilsShaderCompiler(ENGINE_PROFILE)


@pytest.fixture
def resource_compiler(tmp_path: Path) -> WBEUtilsShaderCompiler:
    """A compiler of shader resources whose sources are under test/, generating kernel entry points into tmp_path."""
    if shutil.which("slangc") is None or shutil.which("spirv-val") is None:
        pytest.skip("slangc and spirv-val must be on PATH.")
    return WBEUtilsShaderCompiler(ENGINE_PROFILE, [SHADER_DIR], generated_source_dir=tmp_path / "generated")


def compile_resource(compiler: WBEUtilsShaderCompiler, resource: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    results = compiler.compile(resource, Path("shaders/test.manifest.lua"), SHADER_DIR.parent, output_dir)
    assert len(results) == 1
    return results[0]


def read_metadata(output_dir: Path, variant: dict[str, Any]) -> dict[str, Any]:
    assert (output_dir / variant["path"]).is_file()
    return json.loads((output_dir / variant["metadata"]).read_text(encoding="utf-8"))


def push_fields(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    push_data = [parameter for parameter in metadata["parameters"] if parameter["binding"]["kind"] == "pushConstantBuffer"]
    assert len(push_data) == 1
    return push_data[0]["type"]["fields"]


def reflect(compiler: WBEUtilsShaderCompiler, kernel_name: str,
        macro_definitions: list[tuple[str, str]] | None = None) -> Kernel:
    return compiler.reflect_kernel(KERNELS_PATH, kernel_name, [SHADER_DIR], macro_definitions)


def compile_source(source: str, directory: Path, name: str) -> dict[str, Any]:
    """Compile generated source to SPIR-V with slangc and return its reflection."""
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not on PATH.")
    source_path = directory / f"{name}.slang"
    source_path.write_text(source, encoding="utf-8")
    reflection_path = directory / f"{name}.json"
    result = subprocess.run([slangc, "-target", "spirv", "-profile", ENGINE_PROFILE, "-I", SHADER_DIR.as_posix(),
        "-o", (directory / f"{name}.spv").as_posix(), "-reflection-json", reflection_path.as_posix(),
        source_path.as_posix()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(reflection_path.read_text(encoding="utf-8"))


def test_reflects_parameters_in_declaration_order(compiler: WBEUtilsShaderCompiler) -> None:
    kernel = reflect(compiler, "copy_payload")
    assert kernel.name == "copy_payload"
    assert kernel.source_path == KERNELS_PATH
    assert kernel.thread_group_size == (64, 1, 1)
    assert kernel.parameters == (
        KernelSystemValue("p_dispatch_thread_id", "vector<uint,3>", "SV_DISPATCHTHREADID"),
        KernelBinding("p_info", "ConstantBuffer<Info, DefaultDataLayout>", "info", "uniform_buffer", "read"),
        KernelBinding("p_source", "StructuredBuffer<Payload, DefaultDataLayout>", "source", "storage_buffer", "read"),
        KernelBinding("p_destination", "RWStructuredBuffer<float, DefaultDataLayout>", "destination", "storage_buffer",
            "write"),
        KernelPush("p_offset", "float", "offset"),
    )


def test_infers_descriptor_types_and_default_accesses(compiler: WBEUtilsShaderCompiler) -> None:
    kernel = reflect(compiler, "every_binding")
    assert kernel.thread_group_size == (8, 4, 2)
    assert [(binding.resource_id, binding.desc_type, binding.access) for binding in kernel.bindings] == [
        ("wrapped", "storage_buffer", "read_write"),
        ("bytes", "storage_buffer", "read"),
        ("sampled", "sampled_image", "read"),
        ("storage", "storage_image", "read_write"),
        ("sampler", "sampler", "read"),
    ]
    assert kernel.bindings[0].type_name == "RWStructuredBuffer<Wrapper<matrix<float,3,3>>, DefaultDataLayout>"
    assert [(push.name, push.type_name, push.value) for push in kernel.pushes] == [
        ("p_frame_slot", "uint", "frame_slot"), ("p_level", "int", "level")]
    assert [value.semantic for value in kernel.system_values] == ["SV_GROUPINDEX", "SV_GROUPID"]


def test_reflects_push_data_structs_and_arrays(compiler: WBEUtilsShaderCompiler) -> None:
    kernel = reflect(compiler, "shade")
    assert kernel.push_data == [KernelPushData("p_lights", "LightDescIndices", 3), KernelPushData("p_more_lights", "Lights", 0)]


def test_generated_entry_point_embeds_push_data(compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    kernel = reflect(compiler, "shade")
    source = compiler.generate_entry_source(kernel)
    assert "    LightDescIndices p_lights[3];" in source
    assert "    Lights p_more_lights;" in source
    reflection = compile_source(source, tmp_path, "shade")
    push_data = [parameter for parameter in reflection["parameters"]
        if parameter["binding"]["kind"] == "pushConstantBuffer"]
    fields = push_data[0]["type"]["elementType"]["fields"]
    assert [field["name"] for field in fields] == ["p_lights", "p_more_lights", "p_output"]
    # The embedded structs keep their own attributes.
    light_fields = fields[0]["type"]["elementType"]["fields"]
    assert [(field["name"], field["userAttribs"][0]["name"]) for field in light_fields] == [
        ("shadow", "wbe_desc"), ("index", "wbe_push_constant")]
    test_reflection = compile_source(compiler.generate_test_source(kernel), tmp_path, "shade_test")
    assert [parameter["name"] for parameter in test_reflection["parameters"]] == ["lights", "more_lights", "output"]


def test_reflects_kernels_without_parameters(compiler: WBEUtilsShaderCompiler) -> None:
    assert reflect(compiler, "no_parameters").parameters == ()


@pytest.mark.parametrize(("kernel_name", "message"), [
    ("unannotated_parameter", "must have a system-value semantic, or exactly one"),
    ("vector_push", "must be a uint, int, or float"),
    ("scalar_push_data", "must be a struct or a sized array of one"),
    ("desc_type_mismatch", "is a 'storage_buffer'"),
    ("write_to_read_only", "declares the access 'write'"),
    ("bound_system_value", "must not be marked"),
    ("bound_scalar", "cannot be bound"),
    ("bound_and_pushed", "exactly one [wbe::bind], [wbe::push], or [wbe::push_data]"),
    ("empty_workgroup", "three positive integers"),
    ("not_a_kernel", "exactly one [wbe::kernel(x, y, z)]"),
])
def test_rejects_invalid_kernels(compiler: WBEUtilsShaderCompiler, kernel_name: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=f"Kernel '{kernel_name}'.*{_escape(message)}"):
        reflect(compiler, kernel_name)


def test_rejects_generic_and_missing_kernels(compiler: WBEUtilsShaderCompiler) -> None:
    with pytest.raises(RuntimeError, match="generic_kernel"):
        reflect(compiler, "generic_kernel")
    with pytest.raises(RuntimeError, match="missing_kernel"):
        reflect(compiler, "missing_kernel")


def test_reports_module_errors(compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    source_path = tmp_path / "broken.slang"
    source_path.write_text("void broken() { undefined_function(); }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="undefined_function"):
        compiler.reflect_kernel(source_path, "broken")


def test_rejects_unknown_profiles() -> None:
    with pytest.raises(RuntimeError, match="Unknown Slang capability 'not_a_capability'"):
        WBEUtilsShaderCompiler("spirv_1_6+not_a_capability").reflect_kernel(KERNELS_PATH, "copy_payload", [SHADER_DIR])


def test_preprocesses_with_macro_definitions(compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    kernel = reflect(compiler, "copy_payload", [("SCALE", "2")])
    source = compiler.generate_entry_source(kernel)
    # The generated entry point includes the kernel's source, so it is compiled with the same definitions.
    assert f'#include "{KERNELS_PATH.as_posix()}"' in source


def test_generated_entry_point_declares_the_push_data_contract(compiler: WBEUtilsShaderCompiler,
        tmp_path: Path) -> None:
    kernel = reflect(compiler, "every_binding")
    reflection = compile_source(compiler.generate_entry_source(kernel), tmp_path, "every_binding")
    entry_point = reflection["entryPoints"][0]
    assert entry_point["name"] == "main"
    assert entry_point["threadGroupSize"] == [8, 4, 2]
    assert [parameter.get("semanticName") for parameter in entry_point["parameters"]] == ["SV_GROUPINDEX", "SV_GROUPID"]
    push_data = [parameter for parameter in reflection["parameters"]
        if parameter["binding"]["kind"] == "pushConstantBuffer"]
    assert len(push_data) == 1
    fields = push_data[0]["type"]["elementType"]["fields"]
    assert [(field["name"], field["userAttribs"]) for field in fields] == [
        ("p_wrapped", [{"name": "wbe_desc", "arguments": ["storage_buffer", "wrapped", "read_write"]}]),
        ("p_bytes", [{"name": "wbe_desc", "arguments": ["storage_buffer", "bytes", "read"]}]),
        ("p_sampled", [{"name": "wbe_desc", "arguments": ["sampled_image", "sampled", "read"]}]),
        ("p_storage", [{"name": "wbe_desc", "arguments": ["storage_image", "storage", "read_write"]}]),
        ("p_sampler", [{"name": "wbe_desc", "arguments": ["sampler", "sampler", "read"]}]),
        ("p_frame_slot", [{"name": "wbe_push_constant", "arguments": ["frame_slot"]}]),
        ("p_level", [{"name": "wbe_push_constant", "arguments": ["level"]}]),
    ]
    # Descriptor handles are aligned uint2 values, followed by the four-byte scalars.
    assert [field["binding"]["offset"] for field in fields] == [0, 8, 16, 24, 32, 40, 44]


def test_generated_entry_point_without_parameters_declares_no_push_data(compiler: WBEUtilsShaderCompiler,
        tmp_path: Path) -> None:
    kernel = reflect(compiler, "no_parameters")
    source = compiler.generate_entry_source(kernel, "run")
    assert "push_constant" not in source
    reflection = compile_source(source, tmp_path, "no_parameters")
    assert [entry_point["name"] for entry_point in reflection["entryPoints"]] == ["run"]
    assert reflection["parameters"] == []


def test_generated_test_entry_point_binds_globals(compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    kernel = reflect(compiler, "copy_payload")
    reflection = compile_source(compiler.generate_test_source(kernel), tmp_path, "copy_payload_test")
    assert [parameter["name"] for parameter in reflection["parameters"]] == ["info", "source", "destination", "offset"]
    assert reflection["entryPoints"][0]["threadGroupSize"] == [64, 1, 1]


def test_generated_test_entry_point_rejects_colliding_globals(compiler: WBEUtilsShaderCompiler) -> None:
    kernel = reflect(compiler, "colliding_test_globals")
    with pytest.raises(RuntimeError, match="test global 'values'"):
        compiler.generate_test_source(kernel)


@pytest.mark.parametrize(("parameter_name", "global_name"), [("p_values", "values"), ("values", "values"), ("p_", "p_")])
def test_test_global_names_drop_the_parameter_prefix(parameter_name: str, global_name: str) -> None:
    assert kernel_test_global_name(parameter_name) == global_name


def test_compiles_every_entry_point_of_a_shader(resource_compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    resource = {"id": "test_shader", "type": "shader", "src": "shaders/shaders.slang", "label": "kept"}
    result = compile_resource(resource_compiler, resource, tmp_path)
    assert result == {"id": "test_shader", "type": "shader", "label": "kept", "variants": [{"name": "default",
        "path": "shaders/shaders.test_shader.spv", "metadata": "shaders/shaders.test_shader.json", "is_default": True}]}
    metadata = read_metadata(tmp_path, result["variants"][0])
    assert sorted(entry_point["name"] for entry_point in metadata["entry_points"]) == ["clear", "main"]
    assert [(field["name"], field["user_attributes"]) for field in push_fields(metadata)] == [
        ("values", [{"name": "wbe_desc", "arguments": [{"kind": "string", "string_value": "storage_buffer"},
            {"kind": "string", "string_value": "values"}, {"kind": "string", "string_value": ""}]}]),
        ("frame_slot", [{"name": "wbe_push_constant", "arguments": [{"kind": "string", "string_value": "frame_slot"}]}]),
    ]
    # The metadata comes from the build defining WBE_REFLECTION_METADATA; the SPIR-V from the one that does not.
    assert "metadata_only" in [parameter["name"] for parameter in metadata["parameters"]]
    assert not (tmp_path / "shaders/shaders.test_shader.spv.reflection").exists()


def test_compiles_one_stage_and_entry(resource_compiler: WBEUtilsShaderCompiler, tmp_path: Path) -> None:
    resource = {"id": "clear", "src": "shaders/shaders.slang", "stage": "compute", "entry": "clear"}
    metadata = read_metadata(tmp_path, compile_resource(resource_compiler, resource, tmp_path)["variants"][0])
    assert [entry_point["name"] for entry_point in metadata["entry_points"]] == ["clear"]


def test_compiles_each_variant_with_its_macro_definitions(resource_compiler: WBEUtilsShaderCompiler,
        tmp_path: Path) -> None:
    resource = {"id": "offset", "src": "shaders/shaders.slang", "variants": [
        {"name": "zero"},
        {"name": "one", "values": [{"name": "OFFSET", "value": "1"}], "default": True},
    ]}
    result = compile_resource(resource_compiler, resource, tmp_path)
    assert result["variants"] == [
        {"name": "zero", "path": "shaders/shaders.offset.zero.spv", "metadata": "shaders/shaders.offset.zero.json",
            "is_default": False},
        {"name": "one", "path": "shaders/shaders.offset.one.spv", "metadata": "shaders/shaders.offset.one.json",
            "is_default": True},
    ]
    spirv = [(tmp_path / variant["path"]).read_bytes() for variant in result["variants"]]
    assert spirv[0] != spirv[1]


def test_compiles_the_generated_entry_point_of_a_kernel(resource_compiler: WBEUtilsShaderCompiler,
        tmp_path: Path) -> None:
    resource = {"id": "copy", "src": "shaders/kernels.slang", "kernel": "copy_payload"}
    output_dir = tmp_path / "res"
    result = compile_resource(resource_compiler, resource, output_dir)
    assert "kernel" not in result and "src" not in result
    metadata = read_metadata(output_dir, result["variants"][0])
    assert [entry_point["name"] for entry_point in metadata["entry_points"]] == ["main"]
    assert [field["name"] for field in push_fields(metadata)] == ["p_info", "p_source", "p_destination", "p_offset"]
    generated_path = tmp_path / "generated" / "shaders" / "kernels.copy.slang"
    assert f'#include "{KERNELS_PATH.as_posix()}"' in generated_path.read_text(encoding="utf-8")
    # An unchanged kernel leaves the generated entry point untouched.
    modified_time = generated_path.stat().st_mtime_ns
    compile_resource(resource_compiler, resource, output_dir)
    assert generated_path.stat().st_mtime_ns == modified_time


def test_compiles_kernels_with_the_compiler_macro_definitions(tmp_path: Path) -> None:
    if shutil.which("slangc") is None or shutil.which("spirv-val") is None:
        pytest.skip("slangc and spirv-val must be on PATH.")
    resource = {"id": "copy", "src": "shaders/kernels.slang", "kernel": "copy_payload"}
    spirv: list[bytes] = []
    for macro_definitions in [[], [("SCALE", "3")]]:
        compiler = WBEUtilsShaderCompiler(ENGINE_PROFILE, [SHADER_DIR], macro_definitions, tmp_path / "generated")
        result = compile_resource(compiler, resource, tmp_path / "res")
        spirv.append((tmp_path / "res" / result["variants"][0]["path"]).read_bytes())
    assert spirv[0] != spirv[1]
    # A variant cannot define the compiler's macros again.
    redefining = {**resource, "variants": [{"name": "doubled", "values": [{"name": "SCALE", "value": "2"}]}]}
    with pytest.raises(RuntimeError, match=_escape("defines the compiler's macros ['SCALE']")):
        compile_resource(compiler, redefining, tmp_path / "res")


@pytest.mark.parametrize(("resource", "message"), [
    ({"id": "missing_source"}, "must provide a src field"),
    ({"id": "", "src": "shaders/shaders.slang"}, "non-empty id"),
    ({"id": "a/b", "src": "shaders/shaders.slang"}, "usable as a file name"),
    ({"id": "stage", "src": "shaders/shaders.slang", "stage": 1}, "stage must be a string"),
    ({"id": "entry", "src": "shaders/shaders.slang", "entry": 1}, "entry must be a string"),
    ({"id": "kernel", "src": "shaders/kernels.slang", "kernel": ""}, "kernel must be a non-empty string"),
    ({"id": "kernel", "src": "shaders/kernels.slang", "kernel": "copy_payload", "entry": "main"},
        "either a kernel, or a stage and entry"),
    ({"id": "half", "src": "shaders/shaders.slang", "entry": "main"}, "both stage and entry"),
    ({"id": "suffix", "src": "test_shader_compiler.py"}, "Unsupported shader file type"),
    ({"id": "absent", "src": "shaders/absent.slang"}, "not found"),
    ({"id": "variants", "src": "shaders/shaders.slang", "variants": [{"name": "a"}, {"name": "a"}]},
        "declared more than once"),
    ({"id": "variants", "src": "shaders/shaders.slang", "variants": [{"name": ""}]}, "non-empty string"),
    ({"id": "variants", "src": "shaders/shaders.slang",
        "variants": [{"name": "a", "default": True}, {"name": "b", "default": True}]}, "more than one default"),
    ({"id": "variants", "src": "shaders/shaders.slang", "variants": [
        {"name": "a", "values": [{"name": "OFFSET", "value": "1"}, {"name": "OFFSET", "value": "2"}]}]},
        "defines macro 'OFFSET' more than once"),
])
def test_rejects_invalid_shader_resources(resource_compiler: WBEUtilsShaderCompiler, tmp_path: Path,
        resource: dict[str, Any], message: str) -> None:
    with pytest.raises(RuntimeError, match=_escape(message)):
        compile_resource(resource_compiler, resource, tmp_path)


def test_kernel_resources_need_a_generated_source_directory(tmp_path: Path) -> None:
    compiler = WBEUtilsShaderCompiler(ENGINE_PROFILE, [SHADER_DIR])
    with pytest.raises(RuntimeError, match="no generated source directory"):
        compile_resource(compiler, {"id": "copy", "src": "shaders/kernels.slang", "kernel": "copy_payload"}, tmp_path)


def test_reports_compile_errors_with_the_compiler_output(resource_compiler: WBEUtilsShaderCompiler,
        tmp_path: Path) -> None:
    resource = {"id": "broken", "src": "shaders/shaders.slang", "variants": [
        {"name": "broken", "values": [{"name": "OFFSET", "value": "undefined_identifier"}]}]}
    with pytest.raises(RuntimeError, match="Failed to compile shader(.|\\n)*undefined_identifier"):
        compile_resource(resource_compiler, resource, tmp_path)


def _escape(text: str) -> str:
    return "".join(f"\\{character}" if character in "[]().*+?^$|{}" else character for character in text)
