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
 * @file shaders.slang
 * @brief Hand-written entry points the shader compiler tests compile as shader resources.
 * @details OFFSET selects the value main adds; it defaults to 0. Builds that define WBE_REFLECTION_METADATA also declare a
 * metadata-only constant buffer.
 */
#include "attributes.slang"

#ifndef OFFSET
#define OFFSET 0
#endif

struct Info {
    uint count;
};

#ifdef WBE_REFLECTION_METADATA
ConstantBuffer<Info> metadata_only;
#endif

struct DescIndices {
    [wbe::desc("storage_buffer", "values")]
    DescriptorHandle<RWStructuredBuffer<float>> values;
    [wbe::push_constant("frame_slot")]
    uint frame_slot;
};

[vk::push_constant]
ConstantBuffer<DescIndices> di;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 p_dispatch_thread_id : SV_DispatchThreadID) {
    RWStructuredBuffer<float> values = di.values;
    values[p_dispatch_thread_id.x] += float(OFFSET + di.frame_slot);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void clear(uint3 p_dispatch_thread_id : SV_DispatchThreadID) {
    RWStructuredBuffer<float> values = di.values;
    values[p_dispatch_thread_id.x] = 0.0;
}
