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

#include <array>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <slang-com-ptr.h>
#include <slang.h>

namespace py = pybind11;

namespace wbe::shader_compiler
{
namespace
{
std::string diagnostics_text(slang::IBlob* p_diagnostics)
{
    if (p_diagnostics == nullptr)
    {
        return {};
    }
    return {static_cast<const char*>(p_diagnostics->getBufferPointer()), p_diagnostics->getBufferSize()};
}

void check_slang_result(SlangResult p_result, slang::IBlob* p_diagnostics, std::string_view p_operation)
{
    if (SLANG_FAILED(p_result))
    {
        throw std::runtime_error(std::format("Failed to {}:\n{}", p_operation, diagnostics_text(p_diagnostics)));
    }
}

const char* type_kind_name(slang::TypeReflection::Kind p_kind)
{
    switch (p_kind)
    {
    case slang::TypeReflection::Kind::Struct:
        return "struct";
    case slang::TypeReflection::Kind::Array:
        return "array";
    case slang::TypeReflection::Kind::Matrix:
        return "matrix";
    case slang::TypeReflection::Kind::Vector:
        return "vector";
    case slang::TypeReflection::Kind::Scalar:
        return "scalar";
    case slang::TypeReflection::Kind::ConstantBuffer:
        return "constant_buffer";
    case slang::TypeReflection::Kind::Resource:
        return "resource";
    case slang::TypeReflection::Kind::SamplerState:
        return "sampler_state";
    case slang::TypeReflection::Kind::TextureBuffer:
        return "texture_buffer";
    case slang::TypeReflection::Kind::ShaderStorageBuffer:
        return "shader_storage_buffer";
    case slang::TypeReflection::Kind::ParameterBlock:
        return "parameter_block";
    case slang::TypeReflection::Kind::Pointer:
        return "pointer";
    case slang::TypeReflection::Kind::Enum:
        return "enum";
    default:
        return "other";
    }
}

const char* resource_shape_name(SlangResourceShape p_shape)
{
    switch (p_shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
    {
    case SLANG_TEXTURE_1D:
        return "texture_1d";
    case SLANG_TEXTURE_2D:
        return "texture_2d";
    case SLANG_TEXTURE_3D:
        return "texture_3d";
    case SLANG_TEXTURE_CUBE:
        return "texture_cube";
    case SLANG_TEXTURE_BUFFER:
        return "texture_buffer";
    case SLANG_STRUCTURED_BUFFER:
        return "structured_buffer";
    case SLANG_BYTE_ADDRESS_BUFFER:
        return "byte_address_buffer";
    case SLANG_ACCELERATION_STRUCTURE:
        return "acceleration_structure";
    case SLANG_RESOURCE_NONE:
        return "none";
    default:
        return "other";
    }
}

const char* resource_access_name(SlangResourceAccess p_access)
{
    switch (p_access)
    {
    case SLANG_RESOURCE_ACCESS_NONE:
        return "none";
    case SLANG_RESOURCE_ACCESS_READ:
        return "read";
    case SLANG_RESOURCE_ACCESS_READ_WRITE:
        return "read_write";
    case SLANG_RESOURCE_ACCESS_WRITE:
        return "write";
    default:
        return "other";
    }
}

const char* scalar_type_name(slang::TypeReflection::ScalarType p_scalar_type)
{
    switch (p_scalar_type)
    {
    case slang::TypeReflection::ScalarType::Bool:
        return "bool";
    case slang::TypeReflection::ScalarType::Int32:
        return "int32";
    case slang::TypeReflection::ScalarType::UInt32:
        return "uint32";
    case slang::TypeReflection::ScalarType::Int64:
        return "int64";
    case slang::TypeReflection::ScalarType::UInt64:
        return "uint64";
    case slang::TypeReflection::ScalarType::Float16:
        return "float16";
    case slang::TypeReflection::ScalarType::Float32:
        return "float32";
    case slang::TypeReflection::ScalarType::Float64:
        return "float64";
    case slang::TypeReflection::ScalarType::None:
        return "none";
    default:
        return "other";
    }
}

py::object attribute_argument(slang::UserAttribute* p_attribute, uint32_t p_index)
{
    slang::TypeReflection* argument_type = p_attribute->getArgumentType(p_index);
    if (argument_type != nullptr && argument_type->getKind() == slang::TypeReflection::Kind::Scalar)
    {
        if (argument_type->getScalarType() == slang::TypeReflection::ScalarType::Float32)
        {
            float value = 0.0F;
            if (SLANG_SUCCEEDED(p_attribute->getArgumentValueFloat(p_index, &value)))
            {
                return py::float_(value);
            }
        }
        int value = 0;
        if (SLANG_SUCCEEDED(p_attribute->getArgumentValueInt(p_index, &value)))
        {
            return py::int_(value);
        }
    }
    size_t size = 0;
    const char* value = p_attribute->getArgumentValueString(p_index, &size);
    if (value == nullptr)
    {
        throw std::runtime_error(
            std::format("Failed to read argument {} of attribute '{}'.", p_index, p_attribute->getName()));
    }
    return py::str(value, size);
}

py::dict attribute_dict(slang::UserAttribute* p_attribute)
{
    py::list arguments;
    for (uint32_t i = 0; i < p_attribute->getArgumentCount(); ++i)
    {
        arguments.append(attribute_argument(p_attribute, i));
    }
    py::dict result;
    result["name"] = py::str(p_attribute->getName());
    result["arguments"] = arguments;
    return result;
}

py::list function_attributes(slang::FunctionReflection* p_function)
{
    py::list result;
    for (unsigned int i = 0; i < p_function->getUserAttributeCount(); ++i)
    {
        result.append(attribute_dict(p_function->getUserAttributeByIndex(i)));
    }
    return result;
}

py::list variable_attributes(slang::VariableReflection* p_variable)
{
    py::list result;
    for (unsigned int i = 0; i < p_variable->getUserAttributeCount(); ++i)
    {
        result.append(attribute_dict(p_variable->getUserAttributeByIndex(i)));
    }
    return result;
}

std::string type_full_name(slang::TypeReflection* p_type)
{
    Slang::ComPtr<slang::IBlob> name;
    if (SLANG_FAILED(p_type->getFullName(name.writeRef())) || name == nullptr)
    {
        throw std::runtime_error("Failed to read the full name of a kernel parameter type.");
    }
    return {static_cast<const char*>(name->getBufferPointer()), name->getBufferSize()};
}

py::dict parameter_dict(slang::VariableLayoutReflection* p_parameter)
{
    slang::TypeLayoutReflection* type_layout = p_parameter->getTypeLayout();
    slang::TypeReflection* type = type_layout->getType();
    const slang::TypeReflection::Kind kind = type->getKind();
    const bool is_resource = kind == slang::TypeReflection::Kind::Resource;
    py::dict result;
    result["name"] = py::str(p_parameter->getName());
    const char* semantic = p_parameter->getSemanticName();
    result["semantic"] = semantic != nullptr ? py::object(py::str(semantic)) : py::object(py::none());
    result["type_name"] = type_full_name(type);
    result["kind"] = type_kind_name(kind);
    result["resource_shape"] = is_resource ? resource_shape_name(type->getResourceShape()) : "none";
    result["resource_access"] = is_resource ? resource_access_name(type->getResourceAccess()) : "none";
    result["scalar_type"] = kind == slang::TypeReflection::Kind::Scalar ? scalar_type_name(type->getScalarType()) : "none";
    // Arrays also report their element type, which generated declarations need: `<element type> <name>[<count>]`.
    const bool is_array = kind == slang::TypeReflection::Kind::Array;
    slang::TypeReflection* element_type = is_array ? type->getElementType() : nullptr;
    result["element_count"] = is_array ? type->getElementCount() : 0;
    result["element_type_name"] =
        element_type != nullptr ? py::object(py::str(type_full_name(element_type))) : py::object(py::none());
    result["element_kind"] = element_type != nullptr ? type_kind_name(element_type->getKind()) : "none";
    result["attributes"] = variable_attributes(p_parameter->getVariable());
    return result;
}

/**
 * Split a slangc-style profile into its profile and capabilities.
 */
std::pair<std::string, std::vector<std::string>> split_profile(const std::string& p_profile)
{
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= p_profile.size())
    {
        const size_t end = p_profile.find('+', start);
        parts.push_back(p_profile.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
    std::string profile = parts.front();
    parts.erase(parts.begin());
    return {profile, parts};
}
} // namespace

KernelReflector::KernelReflector()
{
    if (SLANG_FAILED(slang::createGlobalSession(global_session.writeRef())) || global_session == nullptr)
    {
        throw std::runtime_error("Failed to create the Slang global session.");
    }
}

py::dict KernelReflector::reflect_kernel(const std::filesystem::path& p_source_path,
    const std::string& p_kernel_name,
    const std::vector<std::filesystem::path>& p_include_dirs,
    const std::vector<std::pair<std::string, std::string>>& p_macro_definitions,
    const std::string& p_profile) const
{
    const auto [profile_name, capability_names] = split_profile(p_profile);
    const SlangProfileID profile = global_session->findProfile(profile_name.c_str());
    if (profile == SLANG_PROFILE_UNKNOWN)
    {
        throw std::runtime_error(std::format("Unknown Slang profile '{}'.", profile_name));
    }
    std::vector<slang::CompilerOptionEntry> target_options;
    for (const std::string& capability_name : capability_names)
    {
        const SlangCapabilityID capability = global_session->findCapability(capability_name.c_str());
        if (capability == SLANG_CAPABILITY_UNKNOWN)
        {
            throw std::runtime_error(std::format("Unknown Slang capability '{}'.", capability_name));
        }
        slang::CompilerOptionEntry entry{};
        entry.name = slang::CompilerOptionName::Capability;
        entry.value.kind = slang::CompilerOptionValueKind::Int;
        entry.value.intValue0 = static_cast<int32_t>(capability);
        target_options.push_back(entry);
    }
    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.profile = profile;
    target.compilerOptionEntries = target_options.data();
    target.compilerOptionEntryCount = static_cast<uint32_t>(target_options.size());

    std::vector<std::string> include_dirs;
    include_dirs.reserve(p_include_dirs.size());
    for (const std::filesystem::path& include_dir : p_include_dirs)
    {
        include_dirs.push_back(include_dir.string());
    }
    std::vector<const char*> search_paths;
    search_paths.reserve(include_dirs.size());
    for (const std::string& include_dir : include_dirs)
    {
        search_paths.push_back(include_dir.c_str());
    }
    std::vector<slang::PreprocessorMacroDesc> macros;
    macros.reserve(p_macro_definitions.size());
    for (const auto& [name, value] : p_macro_definitions)
    {
        macros.push_back({name.c_str(), value.c_str()});
    }

    slang::SessionDesc session_desc{};
    session_desc.targets = &target;
    session_desc.targetCount = 1;
    session_desc.searchPaths = search_paths.data();
    session_desc.searchPathCount = static_cast<SlangInt>(search_paths.size());
    session_desc.preprocessorMacros = macros.data();
    session_desc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());
    Slang::ComPtr<slang::ISession> session;
    check_slang_result(global_session->createSession(session_desc, session.writeRef()), nullptr, "create a Slang session");

    Slang::ComPtr<slang::IBlob> diagnostics;
    const std::string source_path = p_source_path.string();
    slang::IModule* module = session->loadModule(source_path.c_str(), diagnostics.writeRef());
    if (module == nullptr)
    {
        throw std::runtime_error(
            std::format("Failed to load shader module '{}':\n{}", source_path, diagnostics_text(diagnostics.get())));
    }
    Slang::ComPtr<slang::IEntryPoint> entry_point;
    diagnostics = nullptr;
    check_slang_result(
        module->findAndCheckEntryPoint(p_kernel_name.c_str(), SLANG_STAGE_COMPUTE, entry_point.writeRef(),
            diagnostics.writeRef()),
        diagnostics.get(),
        std::format("check kernel '{}' in '{}' as a compute entry point", p_kernel_name, source_path));
    if (entry_point->getSpecializationParamCount() != 0)
    {
        throw std::runtime_error(std::format(
            "Kernel '{}' in '{}' is generic; kernels must be concrete functions.", p_kernel_name, source_path));
    }

    std::array<slang::IComponentType*, 2> components = {module, entry_point.get()};
    Slang::ComPtr<slang::IComponentType> composite;
    diagnostics = nullptr;
    check_slang_result(session->createCompositeComponentType(components.data(), static_cast<SlangInt>(components.size()),
                           composite.writeRef(), diagnostics.writeRef()),
        diagnostics.get(),
        std::format("compose kernel '{}'", p_kernel_name));
    Slang::ComPtr<slang::IComponentType> linked;
    diagnostics = nullptr;
    check_slang_result(composite->link(linked.writeRef(), diagnostics.writeRef()), diagnostics.get(),
        std::format("link kernel '{}'", p_kernel_name));
    diagnostics = nullptr;
    slang::ProgramLayout* layout = linked->getLayout(0, diagnostics.writeRef());
    if (layout == nullptr || layout->getEntryPointCount() != 1)
    {
        throw std::runtime_error(std::format(
            "Failed to reflect kernel '{}':\n{}", p_kernel_name, diagnostics_text(diagnostics.get())));
    }

    slang::EntryPointReflection* reflected_entry_point = layout->getEntryPointByIndex(0);
    py::list parameters;
    for (unsigned int i = 0; i < reflected_entry_point->getParameterCount(); ++i)
    {
        parameters.append(parameter_dict(reflected_entry_point->getParameterByIndex(i)));
    }
    py::dict result;
    result["name"] = p_kernel_name;
    result["attributes"] = function_attributes(reflected_entry_point->getFunction());
    result["parameters"] = parameters;
    return result;
}
} // namespace wbe::shader_compiler
