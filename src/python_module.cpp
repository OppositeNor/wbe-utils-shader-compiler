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
#include "wbe_utils_shader_compiler/kernel_reflector.hh"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, p_module)
{
    p_module.doc() = "Native backend for the White Bird Engine shader compiler.";

    py::class_<wbe::shader_compiler::KernelReflector>(p_module, "KernelReflector")
        .def(py::init<>())
        .def(
            "reflect_kernel",
            [](const wbe::shader_compiler::KernelReflector& p_reflector,
               const std::string& p_source_path,
               const std::string& p_kernel_name,
               const std::vector<std::string>& p_include_dirs,
               const std::vector<std::pair<std::string, std::string>>& p_macro_definitions,
               const std::string& p_profile) -> py::dict {
                std::vector<std::filesystem::path> include_dirs(p_include_dirs.begin(), p_include_dirs.end());
                return p_reflector.reflect_kernel(
                    std::filesystem::path(p_source_path), p_kernel_name, include_dirs, p_macro_definitions, p_profile);
            },
            py::arg("source_path"),
            py::arg("kernel_name"),
            py::arg("include_dirs"),
            py::arg("macro_definitions"),
            py::arg("profile"));
}
