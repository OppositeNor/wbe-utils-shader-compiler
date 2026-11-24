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
/**
 * @file attributes.slang
 * @brief The attributes the shader compiler reads and generates, declared as the White Bird Engine declares them.
 */
#ifndef WBE_FILE_ATTRIBUTES_SLANG
#define WBE_FILE_ATTRIBUTES_SLANG

[__AttributeUsage(_AttributeTargets.Function)]
struct wbe_kernelAttribute {
    int x;
    int y = 1;
    int z = 1;
};

[__AttributeUsage(_AttributeTargets.Param)]
struct wbe_bindAttribute {
    string resource_id;
    string access = "";
    string desc_type = "";
};

[__AttributeUsage(_AttributeTargets.Param)]
struct wbe_pushAttribute {
    string value;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct wbe_descAttribute {
    string desc_type;
    string resource_id;
    string access = "";
};

[__AttributeUsage(_AttributeTargets.Var)]
struct wbe_push_constantAttribute {
    string value;
};

#endif
