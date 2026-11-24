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
cmake_minimum_required(VERSION 3.22)

project(wbe_utils_shader_compiler LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE WBE_PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE WBE_PYBIND11_RESULT
)
if(NOT WBE_PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 is required. Install dependencies from pyproject.toml before configuring.")
endif()
set(pybind11_DIR "${WBE_PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake package directory")
find_package(pybind11 CONFIG REQUIRED)

# The Slang installation is the one whose slangc is first on PATH, so reflection uses the same compiler as the shader
# builds: <root>/bin/slangc, slang.h in <root>/include or <root>/include/slang (the Vulkan SDK layout), and
# <root>/lib/libslang-compiler. WBE_SLANG_ROOT_DIR, or the SLANG_DIR environment variable, selects another installation.
if(NOT DEFINED WBE_SLANG_ROOT_DIR AND DEFINED ENV{SLANG_DIR})
    set(WBE_SLANG_ROOT_DIR "$ENV{SLANG_DIR}")
endif()
if(NOT DEFINED WBE_SLANG_ROOT_DIR)
    # Search PATH only, as the shader builds do; CMake's prefix paths could find a different installation first.
    find_program(WBE_SLANGC_PROGRAM slangc REQUIRED NO_CACHE NO_PACKAGE_ROOT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH NO_CMAKE_INSTALL_PREFIX)
    get_filename_component(WBE_SLANGC_DIR "${WBE_SLANGC_PROGRAM}" DIRECTORY)
    get_filename_component(WBE_SLANG_ROOT_DIR "${WBE_SLANGC_DIR}" DIRECTORY)
    if(NOT EXISTS "${WBE_SLANG_ROOT_DIR}/include/slang.h" AND NOT EXISTS "${WBE_SLANG_ROOT_DIR}/include/slang/slang.h")
        get_filename_component(WBE_SLANGC_REAL_PROGRAM "${WBE_SLANGC_PROGRAM}" REALPATH)
        get_filename_component(WBE_SLANGC_DIR "${WBE_SLANGC_REAL_PROGRAM}" DIRECTORY)
        get_filename_component(WBE_SLANG_ROOT_DIR "${WBE_SLANGC_DIR}" DIRECTORY)
    endif()
endif()
set(WBE_SLANG_ROOT_DIR "${WBE_SLANG_ROOT_DIR}" CACHE PATH "Slang installation directory")
if(EXISTS "${WBE_SLANG_ROOT_DIR}/include/slang.h")
    set(WBE_SLANG_INCLUDE_DIR "${WBE_SLANG_ROOT_DIR}/include")
elseif(EXISTS "${WBE_SLANG_ROOT_DIR}/include/slang/slang.h")
    set(WBE_SLANG_INCLUDE_DIR "${WBE_SLANG_ROOT_DIR}/include/slang")
else()
    message(FATAL_ERROR "slang.h was not found under '${WBE_SLANG_ROOT_DIR}/include'. Put slangc on PATH, or set SLANG_DIR or -DWBE_SLANG_ROOT_DIR=... to a Slang installation.")
endif()
find_library(WBE_SLANG_COMPILER_LIBRARY NAMES slang-compiler slang PATHS "${WBE_SLANG_ROOT_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
get_filename_component(WBE_SLANG_LIBRARY_DIR "${WBE_SLANG_COMPILER_LIBRARY}" DIRECTORY)

if(NOT DEFINED PYTHON_EXTENSION_OUTPUT_DIRECTORY)
    set(PYTHON_EXTENSION_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/wbe_utils_shader_compiler")
endif()

# The module is named _native; the target name is unique so the engine can build it beside other compilers' _native.
pybind11_add_module(wbe_utils_shader_compiler_native
    src/kernel_reflector.cpp
    src/python_module.cpp
)

target_include_directories(wbe_utils_shader_compiler_native PRIVATE include "${WBE_SLANG_INCLUDE_DIR}")
target_link_libraries(wbe_utils_shader_compiler_native PRIVATE "${WBE_SLANG_COMPILER_LIBRARY}")

set_target_properties(wbe_utils_shader_compiler_native PROPERTIES
    OUTPUT_NAME _native
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_EXTENSION_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${PYTHON_EXTENSION_OUTPUT_DIRECTORY}"
    BUILD_RPATH "${WBE_SLANG_LIBRARY_DIR}"
)
