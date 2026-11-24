# wbe-utils-shader-compiler

`wbe-utils-shader-compiler` turns Slang kernel functions into White Bird Engine compute entry points.

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

The kernel's source must declare the `wbe_kernelAttribute`, `wbe_bindAttribute`, and `wbe_pushAttribute` attributes it
uses, and the `wbe_descAttribute` and `wbe_push_constantAttribute` attributes the generated entry point uses. The White
Bird Engine declares them in `utils/utils.slang`; `test/shaders/attributes.slang` declares them for this package's tests.

## Generated Entry Points

`generate_entry_source()` declares one `[vk::push_constant]` struct with a `DescriptorHandle<T>` field annotated
`wbe::desc(desc_type, resource_id, access)` per binding and a scalar field annotated `wbe::push_constant(value)` per push,
in parameter order, and a compute entry point that passes the fields to the kernel. This is the push-data contract the
engine's pipeline compiler reads, so generated shaders need no engine support beyond it.

`generate_test_source()` declares each binding and push as a global (pushes as `uniform`), named as the parameter without
a leading `p_`, and an entry point that passes them to the kernel, so a test binds plain buffers and values by name.

Both include the kernel's source by absolute path and must be compiled with the same include directories and macro
definitions the kernel was reflected with.

## Python Usage

```python
from pathlib import Path

from wbe_utils_shader_compiler import WBEUtilsShaderCompiler

compiler = WBEUtilsShaderCompiler(profile="spirv_1_6+spvDescriptorHeapEXT+vk_mem_model")
kernel = compiler.reflect_kernel(
    source_path=Path("res/assets/shaders/svogi/svogi_kernel.slang"),
    kernel_name="inject_svo_occupancy",
    include_dirs=[Path("res/assets/shaders"), Path("res/assets/shaders/utils")],
    macro_definitions=[("WBE_SHADER_DEBUG", "1")],
)
entry_source = compiler.generate_entry_source(kernel)
test_source = compiler.generate_test_source(kernel)
```

`reflect_kernel()` raises `RuntimeError` with the Slang diagnostics when the module does not compile, and with a
description of the problem when the function is not a valid kernel. A `WBEUtilsShaderCompiler` owns one Slang global
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
