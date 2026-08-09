#include <metal_stdlib>
using namespace metal;

struct PointVertexOut {
    float4 position [[position]];
    float2 local;
    float4 color;
    float depth;
};

vertex PointVertexOut stentor_breach_pipeline_vertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    const device float4* colors [[buffer(0)]],
    const device packed_float3* positions [[buffer(1)]],
    const device float* radii [[buffer(2)]])
{
    constexpr float2 corners[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    const float3 p = float3(positions[instance_id]);
    const float2 local = corners[vertex_id];
    const float size = radii[instance_id];
    PointVertexOut out;
    out.position = float4(p.xy + local * size, clamp(p.z / 8.0f, 0.0f, 1.0f), 1.0f);
    out.local = local;
    out.color = colors[instance_id];
    out.depth = p.z;
    return out;
}

fragment float4 stentor_breach_pipeline_fragment(PointVertexOut in [[stage_in]])
{
    const float d2 = dot(in.local, in.local);
    if (d2 > 1.0f) discard_fragment();
    const float core = exp(-d2 * 5.5f);
    const float halo = exp(-d2 * 1.7f) * 0.28f;
    const float glint = smoothstep(0.20f, 0.0f, distance(in.local, float2(-0.25f, 0.28f)));
    const float depth_fade = smoothstep(7.0f, 2.2f, in.depth);
    const float3 rgb = in.color.rgb * (0.28f + core * 1.35f + halo * 0.7f) + glint * 0.32f;
    const float alpha = in.color.a * (core + halo) * depth_fade;
    return float4(rgb, alpha);
}
