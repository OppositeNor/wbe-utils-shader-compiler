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
from wbe_utils_shader_compiler import (Kernel, KernelBinding, KernelPush, KernelSystemValue, WBEUtilsShaderCompiler,
    kernel_test_global_name)


SHADER_DIR = Path(__file__).resolve().parent / "shaders"
KERNELS_PATH = SHADER_DIR / "kernels.slang"
# The profile the White Bird Engine compiles shaders with.
ENGINE_PROFILE = "spirv_1_6+spvDescriptorHeapEXT+vk_mem_model"


@pytest.fixture(scope="module")
def compiler() -> WBEUtilsShaderCompiler:
    return WBEUtilsShaderCompiler(ENGINE_PROFILE)


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


def test_reflects_kernels_without_parameters(compiler: WBEUtilsShaderCompiler) -> None:
    assert reflect(compiler, "no_parameters").parameters == ()


@pytest.mark.parametrize(("kernel_name", "message"), [
    ("unannotated_parameter", "must have a system-value semantic, or exactly one"),
    ("vector_push", "must be a uint, int, or float"),
    ("desc_type_mismatch", "is a 'storage_buffer'"),
    ("write_to_read_only", "declares the access 'write'"),
    ("bound_system_value", "must not be marked"),
    ("bound_scalar", "cannot be bound"),
    ("bound_and_pushed", "exactly one [wbe::bind] or [wbe::push]"),
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


def _escape(text: str) -> str:
    return "".join(f"\\{character}" if character in "[]().*+?^$|{}" else character for character in text)
