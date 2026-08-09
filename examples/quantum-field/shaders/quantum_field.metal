#include <metal_stdlib>
using namespace metal;

struct QuantumVertexOut {
    float4 position [[position]];
    float2 local;
    float intensity;
    float phase;
};

vertex QuantumVertexOut quantum_field_pipeline_vertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    const device packed_float2 *positions [[buffer(0)]],
    const device packed_float2 *trail_positions [[buffer(1)]],
    const device float *intensities [[buffer(2)]],
    const device float *phases [[buffer(3)]])
{
    constexpr float2 corners[6] = {
        float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0),
        float2(0.0,  1.0), float2(1.0, -1.0), float2(1.0, 1.0)
    };

    const float2 head = float2(positions[instance_id]);
    float2 tail = float2(trail_positions[instance_id]);
    float2 segment = head - tail;
    float segment_length = length(segment);
    if (segment_length < 0.0015) {
        segment = float2(0.0015, 0.0);
        tail = head - segment;
        segment_length = 0.0015;
    }

    const float2 direction = segment / segment_length;
    const float2 normal = float2(-direction.y, direction.x);
    const float along = corners[vertex_id].x;
    const float across = corners[vertex_id].y;
    const float intensity = intensities[instance_id];
    const float width = mix(0.0012, 0.0042, intensity) * mix(0.55, 1.0, along);
    const float2 world = mix(tail, head, along) + normal * across * width;

    QuantumVertexOut out;
    out.position = float4(world, 0.0, 1.0);
    out.local = float2(along, across);
    out.intensity = intensity;
    out.phase = phases[instance_id];
    return out;
}

fragment float4 quantum_field_pipeline_fragment(QuantumVertexOut in [[stage_in]])
{
    const float phase01 = fract(in.phase / 6.28318530718 + 1.0);
    const float3 phase_color = 0.55 + 0.45 * cos(
        6.28318530718 * (phase01 + float3(0.00, 0.34, 0.67))
    );
    const float transverse = exp(-3.8 * in.local.y * in.local.y);
    const float tail_fade = smoothstep(0.0, 0.82, in.local.x);
    const float head = smoothstep(0.70, 1.0, in.local.x);
    const float glow = transverse * (0.06 + 0.36 * tail_fade) * in.intensity;
    const float core = pow(transverse, 5.0) * head * in.intensity;
    const float3 color =
        phase_color * (glow + core * 0.72)
        + float3(0.72, 0.90, 1.0) * core * 0.34;
    return float4(color, clamp(glow + core, 0.0, 1.0));
}
