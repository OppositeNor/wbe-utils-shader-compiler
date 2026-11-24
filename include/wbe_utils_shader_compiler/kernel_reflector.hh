/* Copyright 2026 OppositeNor

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef WBE_FILE_KERNEL_REFLECTOR_HH
#define WBE_FILE_KERNEL_REFLECTOR_HH

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <slang-com-ptr.h>
#include <slang.h>

namespace wbe::shader_compiler
{
/**
 * @class KernelReflector
 * @brief Reflects Slang kernel functions through the Slang compilation API.
 * @details Owns one Slang global session, created at construction and reused by every reflection. Each reflection loads
 * the source module in a new session, checks the named function as a compute entry point, and returns what the command
 * line reflection omits: full type names, parameter user attributes, and system-value semantics. The reflector is
 * neither copyable nor movable, and it is not thread safe.
 */
class KernelReflector
{
public:
    /**
     * @brief Create the Slang global session.
     * @throws std::runtime_error If the global session cannot be created.
     */
    KernelReflector();
    ~KernelReflector() = default;
    KernelReflector(const KernelReflector& p_other) = delete;
    KernelReflector& operator=(const KernelReflector& p_other) = delete;
    KernelReflector(KernelReflector&& p_other) = delete;
    KernelReflector& operator=(KernelReflector&& p_other) = delete;

    /**
     * @brief Reflect one kernel function of a source module.
     * @details The returned dictionary holds the function's `name` and `attributes`, and its `parameters` in declaration
     * order. Each parameter holds `name`, `semantic` (None without a system-value semantic), `type_name` (the full type
     * name, usable in generated source), `kind`, `resource_shape`, `resource_access`, `scalar_type`, `element_count`,
     * `element_type_name`, and `element_kind` (0, None, and "none" except for arrays), and `attributes`.
     * Each attribute holds `name` and its `arguments`, defaults included, as Python int, float, or str values.
     * @param p_source_path The source module.
     * @param p_kernel_name The function to reflect; it is checked as a compute entry point, so it must not be generic.
     * @param p_include_dirs Include directories, in search order.
     * @param p_macro_definitions The (name, value) macro definitions to preprocess with.
     * @param p_profile The target profile, optionally followed by `+`-separated capabilities, as slangc's -profile.
     * @throws std::runtime_error With the Slang diagnostics if the module or kernel fails to load, check, or link, if the
     * kernel is generic, or if the profile or a capability is unknown.
     */
    pybind11::dict reflect_kernel(const std::filesystem::path& p_source_path,
        const std::string& p_kernel_name,
        const std::vector<std::filesystem::path>& p_include_dirs,
        const std::vector<std::pair<std::string, std::string>>& p_macro_definitions,
        const std::string& p_profile) const;

private:
    Slang::ComPtr<slang::IGlobalSession> global_session;
};
} // namespace wbe::shader_compiler

#endif
