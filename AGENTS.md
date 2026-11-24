# White Bird Engine — Conventions

Compact reference for AI agents. High info density, low fluff.

## Change Discipline

- Make the smallest change that solves the problem.
- Do not refactor unrelated code.
- Do not rename existing APIs unless explicitly requested.
- Do not modify generated files.
- Do not modify dependencies.
- Before editing multiple files, verify that each file is required.

## Agent Workflow

For non-trivial tasks:

1. Inspect relevant files only.
2. Identify the owner layer of the change.
3. Check existing patterns before introducing new ones.
4. Make the minimal implementation.
5. Run the relevant build/test target.
6. Report modified files and validation results.

Do not provide large explanations unless requested.

## Validation

Use `build.py` for normal configure, build, test, and clean operations.

```sh
python build.py configure
python build.py build
python build.py test
python build.py clean
```

Do not include lint in validation. The user performs lint checks manually.

## Build

Always use the Python build script — never invoke `cmake` directly.

```sh
python build.py configure                 # configure the default Debug build
python build.py build                     # build the default Debug target
python build.py test                      # build, then run pytest tests
python build.py clean                     # remove build, dist, cache, and native outputs
python build.py build --build-type Release
```

Build outputs land in `build/<build-type>/`, e.g. `build/debug/`.

## File / Directory Layout

- `include/wbe_utils_shader_compiler/...` — native headers (mirrors `src/`).
- `src/...` — native implementation (`.cpp`); `python_module.cpp` holds the pybind11 bindings.
- `wbe_utils_shader_compiler/` — the Python package: `compiler.py` (`WBEUtilsShaderCompiler`: shader resources, kernel
  validation and generation), `kernel.py` (the reflected kernel types), `slang_compiler.py` (`SlangCompiler`: the slangc
  and spirv-val runs), `shader_metadata.py` (`ShaderMetadataNormalizer`: the engine metadata format).
- `test/` — pytest tests; `test/shaders/` holds the Slang sources they reflect.

Header / source naming: `snake_case.hh` / `snake_case.cpp`. Header guards: `WBE_FILE_<UPPER_SNAKE>_HH`. One primary class per file; file name matches the class in `snake_case`.

## Special Member Functions (Rule of 6)

Classes that manage resources, ownership, or non-trivial state must explicitly follow the Rule of 6:

- Default constructor
- Destructor
- Copy constructor
- Copy assignment operator
- Move constructor
- Move assignment operator

Rules:
- If a class owns a resource, define or delete copy/move operations explicitly.
- Do not rely on compiler-generated copy/move operations for resource-owning types unless the behavior is verified to be correct.
- Prefer move semantics for ownership transfer.
- Copy operations must preserve ownership invariants and perform deep copies when required.
- Move operations must leave the moved-from object in a valid destructible state.
- Use `= delete` for unsupported operations instead of leaving accidental behavior.
- Use `= default` only when the generated implementation matches the intended ownership semantics.

## Naming

- `snake_case` for files, variables, functions, members. `PascalCase` for types. `UPPER_SNAKE` for macros and constants of macro-like nature.
- Function parameters: prefix `p_` (e.g. `p_allocator`, `p_buffer_size`). Local variables: no prefix. Member fields: no prefix.
- **No abbreviations** unless they are universally well-known. Examples:
  - Forbidden: `ci` (use `create_info`), `mgr` (use `manager`), `tex` (use `texture`), `cfg` (use `config` only if widely understood, else `configuration`).
  - Allowed well-known: `MPSC` (multiple producers, single consumer), `SPSC` (single producer, single consumer), `i`/`j`/`k` as for-loop indices, `id`, `uuid`, `gpu`, `cpu`, `os`, `vk` (Vulkan), `rma` (roughness-metallic-ao texture channel pack).
- If a name is too long, each word may be shortened to its first **≥4** letters. Examples: `initiate` → `init`, `information` → `info`, `allocator` → `alloc`; use shortening only when it remains unambiguous (`init`, `info`, `config`, `descr`).
- Preserve domain spellings already used in the codebase (e.g. test labels) even if unusual.

## Style

- C++23. `.clang-format` is authoritative; key rules:
  - `PointerAlignment: Left`, `ReferenceAlignment: Left` (`T* p`, `T& r`).
  - `ColumnLimit: 125`.
  - No short single-line forms: functions, lambdas, blocks, `if`, loops, `case`, `enum` all break to next line.
  - `AlwaysBreakTemplateDeclarations: Yes`.
  - `BreakStringLiterals: false` — long error-message string literals stay on one line.

## Header / Source Separation

Non-trivial method implementations must be placed in `.cpp` files.

Allowed inline definitions in headers:
- Simple getters/setters.
- Trivial methods with a short, obvious implementation.
- Functions that require header definitions (e.g. templates).

Avoid placing complex logic, control flow, or dependency-heavy code in headers. Public headers should expose interfaces and declarations rather than implementation details.

## API Design

- **Prefer exposing classes and interfaces over free-function wrappers.** When a feature has internal state, a constructable context, or a small set of related operations, declare the class in the public header and expose its members. Do not hide it behind a file-local impl class wrapped by free functions in the .cpp. Internal helpers, anonymous-namespace utilities, and detail types stay in the `.cpp`.
- **Name derived implementations with the interface (or its abbreviation) as a prefix.** For example, `RenderObject` → `ROImage`, `ROGraphicsPipelineVK`; `Renderer` → `RendererVK`; `RenderTask` → `RTRunGraphicsPipelineVK`. Pick a short, consistent abbreviation per interface and use it for all derived types.

## Documentation

Use Doxygen for C++ and Slang.

Required:
- Public/protected classes, structs, fields, functions.

Not required:
- Private members.
- Overrides unless behavior differs.

Document contracts, ownership, threading, and errors; do not restate signatures.

## Python Scripts

- Do not create Python environments manually. Assume the current environment is capable for development; if it is not, pause, notify the user, and let the user resolve it.
- All function parameters and return values must have type annotations. Use `-> None` explicitly for procedures.
- Prefer concrete generics (`list[str]`, `dict[str, Any]`) over bare `list` / `dict`. Use `from typing import Any, Callable` etc. when needed.
- Annotate non-trivial local variables whose type cannot be obviously inferred (e.g. empty containers: `result: list[str] = []`).
- Use PEP 604 unions (`X | None`) rather than `Optional[X]`.

## Tests

Run tests with `python build.py test`.

## Shader Resource Contract

- `WBEUtilsShaderCompiler.compile()` owns shader resource validation, variants, output naming, and compilation; hosts such
  as the engine ACP adapter only resolve `src` relative to `res_dir` and configure the compiler. Keep the resource format
  and output names in `README.md` in sync.
- The normalized metadata format is read by the engine's `include/function/render/shader_metadata.hh`; change it only
  with the engine. Metadata comes from the `WBE_REFLECTION_METADATA=1` compile.
- Tool failures raise `RuntimeError` carrying the tool's output; do not print.

## Kernel Contract

- Kernels are concrete Slang functions marked `[wbe::kernel(x, y, z)]`. Every parameter is a system value (an `SV_`
  semantic), a `[wbe::bind(resource_id, access, desc_type)]` resource, a `[wbe::push(value)]` four-byte scalar, or a
  `[wbe::push_data]` struct (or sized array of one) embedded in the push data as-is; any other parameter, a generic
  kernel, or a kernel without exactly one `wbe::kernel` is an error.
- Binding descriptor types and accesses are inferred from the parameter type (`_inferred_binding()` in `compiler.py`);
  explicit attribute values may only match the descriptor type or narrow an RW access. Keep the table in `README.md` in
  sync.
- The generated engine entry point emits the White Bird Engine push-data contract: one `[vk::push_constant]` struct of
  `wbe::desc` `DescriptorHandle<T>` fields, `wbe::push_constant` scalars, and embedded push-data structs in parameter order, with every access stated.
  Do not change it without the engine's pipeline compiler.
- The generated test entry point names each binding and push global as its parameter without a leading `p_`.
- The native layer only reflects; validation, inference, and generation belong to the Python package. It returns plain
  Python objects and does not expose Slang types.
- Reflection uses the Slang installation whose `slangc` is first on `PATH` (or `SLANG_DIR` / `WBE_SLANG_ROOT_DIR`), so it
  matches the compiler that builds the shaders.
- The native target is `wbe_utils_shader_compiler_native` with output name `_native`, so the engine can build it beside
  other packages' `_native` targets in one CMake build.
