#include <metal_stdlib>
using namespace metal;

struct CodebaseVertexOut {
    float4 position [[position]];
    float2 local;
    float group;
    float weight;
    float selected;
};

vertex CodebaseVertexOut codebase_particles_pipeline_vertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    // View reads are bound in canonical name order: group, position, selected, weight.
    const device float *groups [[buffer(0)]],
    const device packed_float2 *positions [[buffer(1)]],
    const device float *selected [[buffer(2)]],
    const device float *weights [[buffer(3)]])
{
    constexpr float2 corners[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    const float2 local = corners[vertex_id];
    const float weight = weights[instance_id];
    const float is_selected = selected[instance_id];
    const float radius = (0.010 + weight * 0.007) * mix(1.0, 1.42, is_selected);

    CodebaseVertexOut out;
    out.position = float4(float2(positions[instance_id]) + local * radius, 0.0, 1.0);
    out.local = local;
    out.group = groups[instance_id];
    out.weight = weight;
    out.selected = is_selected;
    return out;
}

fragment float4 codebase_particles_pipeline_fragment(CodebaseVertexOut in [[stage_in]])
{
    constexpr float3 palette[5] = {
        float3(0.18, 0.78, 1.00), // crates: cyan
        float3(0.72, 0.42, 1.00), // docs: violet
        float3(0.25, 1.00, 0.58), // examples: green
        float3(1.00, 0.43, 0.28), // baseline: coral
        float3(1.00, 0.78, 0.24)  // support/root: gold
    };
    const float distance_squared = dot(in.local, in.local);
    if (distance_squared > 1.0) discard_fragment();

    const uint group = uint(clamp(in.group, 0.0, 4.0));
    const float3 base = palette[group];
    const float core = pow(max(1.0 - distance_squared, 0.0), 2.4);
    const float rim = smoothstep(1.0, 0.35, distance_squared);
    const float highlight = smoothstep(0.34, 0.0, distance(in.local, float2(-0.28, 0.30)));
    const float selection_ring = in.selected * smoothstep(0.36, 0.58, distance_squared);
    const float brightness = 0.28 + core * 0.92 + highlight * 0.55 + in.weight * 0.06;
    const float3 color = base * brightness + highlight * 0.20 + selection_ring * float3(0.85);
    return float4(color, max(rim, selection_ring));
}
