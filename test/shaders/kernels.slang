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
 * @file kernels.slang
 * @brief Kernels the shader compiler tests reflect and generate entry points for, valid and invalid.
 * @details SCALE selects the factor copy_payload applies; it defaults to 1.
 */
#include "attributes.slang"

#ifndef SCALE
#define SCALE 1
#endif

struct Payload {
    float value;
    uint flag;
};

struct Wrapper<T> {
    T value;
    uint flag;
};

struct Info {
    uint count;
};

[wbe::kernel(64)]
void copy_payload(uint3 p_dispatch_thread_id : SV_DispatchThreadID,
    [wbe::bind("info")] ConstantBuffer<Info> p_info,
    [wbe::bind("source", "read", "storage_buffer")] StructuredBuffer<Payload> p_source,
    [wbe::bind("destination", "write")] RWStructuredBuffer<float> p_destination,
    [wbe::push("offset")] float p_offset) {
    uint index = p_dispatch_thread_id.x;
    if (index < p_info.count) {
        p_destination[index] = p_source[index].value * SCALE + p_offset;
    }
}

[wbe::kernel(8, 4, 2)]
void every_binding(uint p_group_index : SV_GroupIndex, uint3 p_group_id : SV_GroupID,
    [wbe::bind("wrapped")] RWStructuredBuffer<Wrapper<float3x3>> p_wrapped,
    [wbe::bind("bytes")] ByteAddressBuffer p_bytes,
    [wbe::bind("sampled")] Texture2D<float4> p_sampled,
    [wbe::bind("storage")] RWTexture3D<float4> p_storage,
    [wbe::bind("sampler")] SamplerState p_sampler,
    [wbe::push("frame_slot")] uint p_frame_slot,
    [wbe::push("level")] int p_level) {
    float4 color = p_sampled.SampleLevel(p_sampler, float2(0.5, 0.5), float(p_level));
    p_storage[p_group_id] = color + float(p_bytes.Load(p_group_index * 4) + p_frame_slot);
    p_wrapped[p_group_index].flag = p_frame_slot;
}

[wbe::kernel(1)]
void unannotated_parameter(StructuredBuffer<float> p_values) {
}

[wbe::kernel(1)]
void vector_push([wbe::push("offset")] float2 p_offset) {
}

[wbe::kernel(1)]
void desc_type_mismatch([wbe::bind("values", "", "uniform_buffer")] StructuredBuffer<float> p_values) {
}

[wbe::kernel(1)]
void write_to_read_only([wbe::bind("values", "write")] StructuredBuffer<float> p_values) {
}

[wbe::kernel(1)]
void bound_system_value([wbe::bind("thread")] uint3 p_dispatch_thread_id : SV_DispatchThreadID) {
}

[wbe::kernel(1)]
void bound_scalar([wbe::bind("count")] uint p_count) {
}

[wbe::kernel(1)]
void bound_and_pushed([wbe::bind("count")] [wbe::push("count")] uint p_count) {
}

[wbe::kernel(0)]
void empty_workgroup(uint3 p_dispatch_thread_id : SV_DispatchThreadID) {
}

void not_a_kernel(uint3 p_dispatch_thread_id : SV_DispatchThreadID) {
}

[wbe::kernel(1)]
void generic_kernel<T>([wbe::bind("values")] StructuredBuffer<T> p_values) {
}

[wbe::kernel(1)]
void no_parameters() {
}

[wbe::kernel(1)]
void colliding_test_globals([wbe::bind("first")] StructuredBuffer<float> p_values,
    [wbe::bind("second")] StructuredBuffer<float> values) {
}
