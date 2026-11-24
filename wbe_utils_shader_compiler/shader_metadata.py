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

from typing import Any


class ShaderMetadataNormalizer:
    """
    Normalizes slangc reflection JSON into the White Bird Engine shader metadata format.

    The format names every member in snake case, fills absent members with empty defaults, converts user attribute
    arguments to `{ kind, <kind>_value }` records, and flattens the descriptor-table members of parameter-block arrays of
    structs into one array field per member. The normalizer holds no state.
    """

    def normalize(self, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize slangc's -reflection-json output.

        :param raw_metadata: The parsed reflection JSON.
        :return: The metadata: its global `parameters` and its `entry_points`.
        :raises RuntimeError: If a user attribute has a boolean or unsupported argument.
        """
        return {
            "parameters": [self._transform_shader_parameter(parameter) for parameter in raw_metadata.get("parameters", [])],
            "entry_points": [self._transform_shader_entry_point(entry_point) for entry_point in raw_metadata.get("entryPoints", [])],
        }

    def _transform_shader_user_attribute_argument(self, raw_argument: Any) -> dict[str, Any]:
        if isinstance(raw_argument, bool):
            raise RuntimeError("Unsupported boolean Slang user attribute argument.")
        if isinstance(raw_argument, int):
            return {"kind": "int", "integer_value": raw_argument}
        if isinstance(raw_argument, float):
            return {"kind": "float", "floating_point_value": raw_argument}
        if isinstance(raw_argument, str):
            return {"kind": "string", "string_value": raw_argument}
        raise RuntimeError(f"Unsupported Slang user attribute argument: {raw_argument!r}.")

    def _transform_shader_user_attributes(self, raw_declaration: dict[str, Any]) -> list[dict[str, Any]]:
        raw_attributes = raw_declaration.get("userAttribs", [])
        if not isinstance(raw_attributes, list):
            return []
        attributes: list[dict[str, Any]] = []
        for raw_attribute in raw_attributes:
            if not isinstance(raw_attribute, dict):
                continue
            raw_arguments = raw_attribute.get("arguments", [])
            if not isinstance(raw_arguments, list):
                raw_arguments = []
            attributes.append({
                "name": str(raw_attribute.get("name", "")),
                "arguments": [self._transform_shader_user_attribute_argument(argument) for argument in raw_arguments],
            })
        return attributes

    def _transform_shader_binding(self, raw_binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": str(raw_binding.get("kind", "")),
            "index": int(raw_binding.get("index", 0)),
            "count": int(raw_binding.get("count", raw_binding.get("used", 0))),
            "space": int(raw_binding.get("space", 0)),
            "offset": int(raw_binding.get("offset", 0)),
            "size": int(raw_binding.get("size", 0)),
            "element_stride": int(raw_binding.get("elementStride", 0)),
        }

    def _shader_type_shape(self, raw_type: dict[str, Any]) -> str:
        kind = raw_type.get("kind", "")
        if kind == "array":
            element_type = raw_type.get("elementType", {})
            if isinstance(element_type, dict):
                return self._shader_type_shape(element_type)
        scalar_type = raw_type.get("scalarType")
        if isinstance(scalar_type, str):
            return scalar_type
        base_shape = raw_type.get("baseShape")
        if isinstance(base_shape, str):
            return base_shape
        return str(kind)

    def _shader_buffer_size(self, raw_type: dict[str, Any]) -> int:
        if raw_type.get("kind") == "array":
            element_type = raw_type.get("elementType", {})
            if isinstance(element_type, dict):
                return self._shader_buffer_size(element_type)
        element_var_layout = raw_type.get("elementVarLayout", {})
        if isinstance(element_var_layout, dict):
            binding = element_var_layout.get("binding", {})
            if isinstance(binding, dict):
                return int(binding.get("size", 0))
        binding = raw_type.get("binding", {})
        if isinstance(binding, dict):
            return int(binding.get("size", 0))
        return 0

    def _transform_shader_type_atom(self, raw_type: dict[str, Any]) -> dict[str, Any]:
        element_type = raw_type.get("elementType", {})
        if not isinstance(element_type, dict):
            element_type = {}
        result_type = raw_type.get("resultType", {})
        if not isinstance(result_type, dict):
            result_type = {}
        reflected_decl_type = raw_type
        kind = str(raw_type.get("kind", ""))
        if kind in {"constantBuffer", "parameterBlock"} and element_type:
            reflected_decl_type = element_type
        elif kind == "resource" and result_type:
            reflected_decl_type = result_type
        scalar_type = self._shader_type_shape(raw_type)
        element_scalar_type = self._shader_type_shape(element_type) if element_type else self._shader_type_shape(result_type)
        raw_name = raw_type.get("name", "")
        if isinstance(raw_name, str) and raw_name:
            normalized_name = raw_name
        else:
            normalized_name = str(reflected_decl_type.get("name", ""))
        user_attributes = self._transform_shader_user_attributes(raw_type)
        if not user_attributes and reflected_decl_type is not raw_type:
            user_attributes = self._transform_shader_user_attributes(reflected_decl_type)
        return {
            "kind": kind,
            "scalar_type": str(scalar_type),
            "access": str(raw_type.get("access", "")),
            "name": normalized_name,
            "element_count": int(raw_type.get("elementCount", 0)),
            "element_scalar_type": str(element_scalar_type),
            "element_access": str(element_type.get("access", result_type.get("access", ""))),
            "user_attributes": user_attributes,
        }

    def _transform_shader_type_field(self, raw_field: dict[str, Any]) -> dict[str, Any]:
        raw_type = raw_field.get("type", {})
        if not isinstance(raw_type, dict):
            raw_type = {}
        nested_type = raw_type
        while nested_type.get("kind") == "array":
            element_type = nested_type.get("elementType", {})
            if not isinstance(element_type, dict):
                nested_type = {}
                break
            nested_type = element_type
        if nested_type.get("kind") == "resource":
            result_type = nested_type.get("resultType", {})
            if isinstance(result_type, dict):
                nested_type = result_type
        nested_fields = nested_type.get("fields", [])
        if not isinstance(nested_fields, list):
            nested_fields = []
        return {
            "name": str(raw_field.get("name", "")),
            "type": self._transform_shader_type_atom(raw_type),
            "stage": str(raw_field.get("stage", "")),
            "binding": self._transform_shader_binding(raw_field.get("binding", {})),
            "buffer_size": self._shader_buffer_size(raw_type),
            "semantic_name": str(raw_field.get("semanticName", "")),
            "semantic_index": int(raw_field.get("semanticIndex", 0)),
            "user_attributes": self._transform_shader_user_attributes(raw_field),
            "fields": [self._transform_shader_type_field(field) for field in nested_fields if isinstance(field, dict)],
        }

    def _flatten_shader_type_fields(self, raw_fields: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw_field in raw_fields:
            if not isinstance(raw_field, dict):
                continue
            raw_type = raw_field.get("type", {})
            raw_binding = raw_field.get("binding", {})
            if not isinstance(raw_type, dict) or not isinstance(raw_binding, dict):
                result.append(raw_field)
                continue
            element_type = raw_type.get("elementType", {})
            element_fields = element_type.get("fields", []) if isinstance(element_type, dict) else []
            if raw_type.get("kind") != "array" or element_type.get("kind") != "struct" or not isinstance(element_fields, list):
                result.append(raw_field)
                continue
            base_binding_index = int(raw_binding.get("index", 0))
            element_count = int(raw_type.get("elementCount", 0))
            flattened_any = False
            for element_field in element_fields:
                if not isinstance(element_field, dict):
                    continue
                element_binding = element_field.get("binding", {})
                element_field_type = element_field.get("type", {})
                if not isinstance(element_binding, dict) or element_binding.get("kind") != "descriptorTableSlot":
                    continue
                if not isinstance(element_field_type, dict):
                    element_field_type = {}
                flattened_field = dict(element_field)
                flattened_field["name"] = f"{raw_field.get('name', '')}.{element_field.get('name', '')}"
                flattened_field["type"] = {
                    "kind": "array",
                    "elementCount": element_count,
                    "elementType": element_field_type,
                }
                flattened_field["binding"] = {
                    **element_binding,
                    "index": base_binding_index + int(element_binding.get("index", 0)),
                    "count": element_count,
                }
                result.append(flattened_field)
                flattened_any = True
            if not flattened_any:
                result.append(raw_field)
        return result

    def _transform_shader_type(self, raw_type: dict[str, Any]) -> dict[str, Any]:
        kind = str(raw_type.get("kind", ""))
        element_type = raw_type.get("elementType", {})
        if not isinstance(element_type, dict):
            element_type = {}
        fields = raw_type.get("fields", [])
        if not isinstance(fields, list):
            fields = []
        element_var_layout = raw_type.get("elementVarLayout", {})
        element_layout_type = {}
        if isinstance(element_var_layout, dict) and isinstance(element_var_layout.get("type"), dict):
            element_layout_type = element_var_layout["type"]
        if kind in {"constantBuffer", "parameterBlock"}:
            field_source = element_layout_type if element_layout_type else element_type
            fields = field_source.get("fields", []) if isinstance(field_source.get("fields", []), list) else []
            if kind == "parameterBlock":
                fields = self._flatten_shader_type_fields(fields)
        return {
            "kind": kind,
            "name": str(raw_type.get("name", "")),
            "fields": [self._transform_shader_type_field(field) for field in fields if isinstance(field, dict)],
            "element_count": int(raw_type.get("elementCount", 0)),
            "element_type": self._transform_shader_type_atom(element_type),
            "scalar_type": str(raw_type.get("scalarType", raw_type.get("baseShape", ""))),
            "access": str(raw_type.get("access", "")),
            "element_access": str(element_type.get("access", "")),
            "user_attributes": self._transform_shader_user_attributes(raw_type),
        }

    def _transform_shader_parameter(self, raw_parameter: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(raw_parameter.get("name", "")),
            "stage": str(raw_parameter.get("stage", "")),
            "binding": self._transform_shader_binding(raw_parameter.get("binding", {})),
            "type": self._transform_shader_type(raw_parameter.get("type", {})),
            "semantic_name": str(raw_parameter.get("semanticName", "")),
            "semantic_index": int(raw_parameter.get("semanticIndex", 0)),
            "user_attributes": self._transform_shader_user_attributes(raw_parameter),
        }

    def _transform_shader_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "stage": str(raw_result.get("stage", "")),
            "binding": self._transform_shader_binding(raw_result.get("binding", {})),
            "semantic_name": str(raw_result.get("semanticName", "")),
            "semantic_index": int(raw_result.get("semanticIndex", 0)),
            "type": self._transform_shader_type(raw_result.get("type", {})),
        }

    def _transform_shader_entry_point(self, raw_entry_point: dict[str, Any]) -> dict[str, Any]:
        bindings = raw_entry_point.get("bindings", [])
        if not isinstance(bindings, list):
            bindings = []
        return {
            "name": str(raw_entry_point.get("name", "")),
            "stage": str(raw_entry_point.get("stage", "")),
            "parameters": [self._transform_shader_parameter(parameter) for parameter in raw_entry_point.get("parameters", []) if isinstance(parameter, dict)],
            "result": self._transform_shader_result(raw_entry_point.get("result", {})),
            "bindings": [self._transform_shader_binding(binding.get("binding", binding)) for binding in bindings if isinstance(binding, dict)],
            "user_attributes": self._transform_shader_user_attributes(raw_entry_point),
        }
