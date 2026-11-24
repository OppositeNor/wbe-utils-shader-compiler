# wbe-utils-shader-compiler

`wbe-utils-shader-compiler` compiles White Bird Engine shader resources to SPIR-V and normalized shader metadata, and
turns Slang kernel functions into compute entry points.

A kernel is an ordinary Slang function marked `[wbe::kernel(x, y, z)]` whose parameters say where their values come from.
The compiler reflects the kernel and generates the entry point that fetches those values and calls it, so shaders do not
declare push-data structs by hand, and tests can call the same kernel with plain buffers.

The public API is a Python package. Kernel reflection is implemented in a C++23 native extension built with CMake and
pybind11 against the Slang compilation API.

## Kernels

```slang
[wbe::kernel(64)]
void copy_payload(uint3 p_dispatch_thread_id : SV_DispatchThreadID,
    [wbe::bind("info")] ConstantBuffer<Info> p_info,
    [wbe::bind("source")] StructuredBuffer<Payload> p_source,
    [wbe::bind("destination", "write")] RWStructuredBuffer<float> p_destination,
    [wbe::push("frame_slot")] uint p_frame_slot) {
    ...
}
```

- `[wbe::kernel(x, y = 1, z = 1)]` gives the workgroup size. The kernel must not be generic.
- A parameter with an `SV_` semantic is forwarded from the entry point's parameter of the same type and semantic.
- `[wbe::bind(resource_id, access = "", desc_type = "")]` binds a constant buffer, structured or byte address buffer,
  texture, sampler, or acceleration structure. The descriptor type is inferred from the parameter type; a stated
  `desc_type` must match it. The access defaults to the type's (`read_write` for RW types, `read` otherwise); an RW type
  may narrow it to `read` or `write`.
- `[wbe::push(value)]` supplies a four-byte `uint`, `int`, or `float` as push data.
- `[wbe::push_data]` embeds a struct, or a sized array of one, in the push data as-is. The struct declares its own
  `wbe::desc` handles and `wbe::push_constant` scalars, as hand-written push data does, so helpers that take such a
  struct (for example the engine's `LightParamDescIndices`) work unchanged.
- Any other parameter is an error.

| Parameter type | Descriptor type | Default access |
|---|---|---|
| `ConstantBuffer<T>` | `uniform_buffer` | `read` |
| `StructuredBuffer<T>`, `ByteAddressBuffer` | `storage_buffer` | `read` |
| `RWStructuredBuffer<T>`, `RWByteAddressBuffer` | `storage_buffer` | `read_write` |
| `Texture*` | `sampled_image` | `read` |
| `RWTexture*` | `storage_image` | `read_write` |
| `SamplerState` | `sampler` | `read` |
| `RaytracingAccelerationStructure` | `acceleration_structure` | `read` |

The kernel's source must declare the `wbe_kernelAttribute`, `wbe_bindAttribute`, `wbe_pushAttribute`, and
`wbe_push_dataAttribute` attributes it uses, and the `wbe_descAttribute` and `wbe_push_constantAttribute` attributes the
generated entry point uses. The White Bird Engine declares them in `utils/utils.slang`; `test/shaders/attributes.slang` declares them for this package's tests.

## Generated Entry Points

`generate_entry_source()` declares one `[vk::push_constant]` struct with a `DescriptorHandle<T>` field annotated
`wbe::desc(desc_type, resource_id, access)` per binding, a scalar field annotated `wbe::push_constant(value)` per push,
and a field of the parameter's struct (or array) type per push-data parameter, in parameter order, and a compute entry point that passes the fields to the kernel. This is the push-data contract the
engine's pipeline compiler reads, so generated shaders need no engine support beyond it.

`generate_test_source()` declares each binding, push, and push-data parameter as a global (pushes and push data as
`uniform`), named as the parameter without
a leading `p_`, and an entry point that passes them to the kernel, so a test binds plain buffers and values by name.

Both include the kernel's source by absolute path and must be compiled with the same include directories and macro
definitions the kernel was reflected with.

## Shader Resources

`compile()` turns one `shader` manifest resource into its runtime resource:

```python
{
    "id": str,                 # usable as a file name
    "src": str,                # the .slang source, relative to res_dir
    "stage": str,              # optional, with entry: compile only that entry point
    "entry": str,
    "kernel": str,             # optional, instead of stage and entry: compile the generated entry point of this kernel
    "variants": [              # optional; one "default" variant without macros when absent
        {"name": str, "values": [{"name": str, "value": str}], "default": bool},
    ],
}
```

Without `stage` and `entry`, every `[shader(...)]` entry point of the source is compiled. Each variant compiles to
`<source stem>.<id>[.<variant>].spv` and `.json` next to the source under `res_output_dir`, with the variant name only
when there are several. A kernel variant's generated entry point is written to the same path, with a `.slang` extension,
under the compiler's `generated_source_dir`, and rewritten only when it changes. The returned resource drops `src` and
`kernel`, keeps every other field, and replaces `variants` with `{ name, path, metadata, is_default }` records whose paths
are relative to `res_output_dir`.

`SlangCompiler` runs `slangc` twice per variant: once for the SPIR-V and reflection, and once more with
`WBE_REFLECTION_METADATA=1`, whose reflection replaces the first so metadata-only declarations reach it. Both outputs are
validated with `spirv-val`, and `ShaderMetadataNormalizer` rewrites the reflection in the engine's snake-case metadata
format. The compiler's own macro definitions apply to every variant, which must not define them again.

## Python Usage

```python
from pathlib import Path

from wbe_utils_shader_compiler import WBEUtilsShaderCompiler

compiler = WBEUtilsShaderCompiler(
    profile="spirv_1_6+spvDescriptorHeapEXT+vk_mem_model",
    include_dirs=[Path("res/assets/shaders"), Path("res/assets/shaders/utils")],
    macro_definitions=[("WBE_SHADER_DEBUG", "1")],
    generated_source_dir=Path("build/debug/generated/shaders"),
    debug_info=True,
)

resources = compiler.compile(
    resource={"id": "svogi_occupancy_injection", "src": "assets/shaders/svogi/svogi_kernel.slang",
        "kernel": "inject_svo_occupancy"},
    manifest_path=Path("assets/shaders/svogi/svogi.manifest.lua"),
    res_dir=Path("res"),
    res_output_dir=Path("build/debug/bin/res"),
)

kernel = compiler.reflect_kernel(Path("res/assets/shaders/svogi/svogi_kernel.slang"), "inject_svo_occupancy")
entry_source = compiler.generate_entry_source(kernel)
test_source = compiler.generate_test_source(kernel)
```

`compile()` raises `RuntimeError` for an invalid resource, and with the tool output when `slangc` or `spirv-val` fails.
`reflect_kernel()` raises `RuntimeError` with the Slang diagnostics when the module does not compile, and with a
description of the problem when the function is not a valid kernel; it uses the compiler's include directories unless
others are given, and its macro definitions besides the given ones. A `WBEUtilsShaderCompiler` owns one Slang global
session and is not thread safe.

## Repository Layout

```text
.
├── CMakeLists.txt
├── build.py
├── pyproject.toml
├── setup.py
├── include/
├── src/
├── test/
└── wbe_utils_shader_compiler/
```

## Requirements

- Python 3.10 or newer
- CMake 3.22 or newer
- A C++23 compiler
- Python build and test packages listed in `pyproject.toml`
- A Slang installation: `bin/slangc`, `slang.h` in `include/` or `include/slang/`, and `lib/libslang-compiler`
- `spirv-val` on `PATH` to compile shader resources

The Slang installation is the one whose `slangc` is first on `PATH`, so kernels are reflected by the compiler that
builds the shaders. Set `SLANG_DIR` (or `-DWBE_SLANG_ROOT_DIR=...`) to use another one.

## Build Commands

The build wrapper resolves paths relative to `build.py`, so the current working directory does not affect build output.

```sh
python build.py configure
python build.py build
python build.py test
python build.py clean
```

## Install

From the repository root:

```sh
pip install .
```
