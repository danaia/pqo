#include <metal_stdlib>
using namespace metal;

inline float repository_hash(uint value)
{
    return fract(sin((float(value) + 1.0) * 12.9898) * 43758.5453);
}

inline uint repository_group(uint index)
{
    // Exact file counts from the repository snapshot used to create this view:
    // crates 44, docs 34, examples 42, baseline 30, supporting/root 31.
    if (index < 44) return 0;
    if (index < 78) return 1;
    if (index < 120) return 2;
    if (index < 150) return 3;
    return 4;
}

inline float2 repository_center(uint group)
{
    constexpr float2 centers[5] = {
        float2(-0.48,  0.18), // compiler/runtime crates
        float2( 0.42,  0.46), // documentation
        float2( 0.50, -0.38), // examples
        float2(-0.46, -0.43), // baseline application
        float2( 0.00,  0.00)  // shaders, kernels, scripts, and root files
    };
    return centers[group];
}

kernel void codebase_particles_organize(
    device packed_float2 *positions [[buffer(0)]],
    device packed_float2 *anchors [[buffer(1)]],
    device packed_float2 *velocities [[buffer(2)]],
    device float *groups [[buffer(3)]],
    device float *weights [[buffer(4)]],
    device float *initialized [[buffer(5)]],
    constant uint &file_count [[buffer(6)]],
    constant float &spring [[buffer(7)]],
    constant float &orbit [[buffer(8)]],
    constant float &drag [[buffer(9)]],
    constant float &motion [[buffer(10)]],
    constant float &maximum_speed [[buffer(11)]],
    constant float &dt [[buffer(12)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= file_count) return;

    const uint group = repository_group(index);
    const float2 center = repository_center(group);
    const float seed_a = repository_hash(index * 2);
    const float seed_b = repository_hash(index * 2 + 1);

    // Device-private buffers have no guaranteed initial contents. Use an exact
    // sentinel rather than assuming a newly allocated f32 starts at zero.
    if (initialized[index] != 17.0) {
        const float angle = seed_a * 6.28318530718;
        const float radius = 0.045 + 0.19 * sqrt(seed_b);
        const float2 radial = float2(cos(angle), sin(angle));
        const float2 seeded_position = center + radial * radius;
        positions[index] = packed_float2(seeded_position);
        anchors[index] = packed_float2(seeded_position);
        velocities[index] = packed_float2(float2(-radial.y, radial.x) * (0.015 + seed_b * 0.025));
        groups[index] = float(group);
        // A stable visual proxy for source weight: mostly small files with a
        // sparse long tail, matching the shape of real codebase file sizes.
        weights[index] = 0.35 + pow(seed_a, 3.0) * 1.65;
        initialized[index] = 17.0;
        return;
    }

    float2 position = float2(positions[index]);
    float2 velocity = float2(velocities[index]);
    const float2 offset = position - float2(anchors[index]);
    const float phase = seed_a * 6.28318530718 + float(index) * 0.071;
    const float2 current = float2(cos(phase), sin(phase)) * motion;
    const float2 tangent = float2(-offset.y, offset.x);
    const float2 acceleration = -offset * spring + tangent * orbit + current - velocity * drag;

    velocity += acceleration * dt;
    const float speed = length(velocity);
    if (speed > maximum_speed) velocity *= maximum_speed / speed;
    position += velocity * dt;

    positions[index] = packed_float2(position);
    velocities[index] = packed_float2(velocity);
}

inline float2 repository_rotate(float2 value, float angle)
{
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float2(
        value.x * cosine - value.y * sine,
        value.x * sine + value.y * cosine
    );
}

inline float2 repository_project_point(
    float2 point,
    float2 camera_center,
    float zoom,
    float aspect)
{
    float2 projected = (point - camera_center) * zoom;
    projected.x /= max(aspect, 0.1);
    return projected;
}

kernel void codebase_particles_select(
    const device packed_float2 *positions [[buffer(0)]],
    device float *selected [[buffer(1)]],
    device float *click_seen [[buffer(2)]],
    constant float &pointer_x [[buffer(3)]],
    constant float &pointer_y [[buffer(4)]],
    constant float &click_generation [[buffer(5)]],
    constant float &center_x [[buffer(6)]],
    constant float &center_y [[buffer(7)]],
    constant float &zoom [[buffer(8)]],
    constant float &orbit_angle [[buffer(9)]],
    constant float &orbit_selected [[buffer(10)]],
    constant float &aspect [[buffer(11)]],
    constant float &selection_radius [[buffer(12)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0 || click_seen[0] == click_generation) return;

    constexpr uint file_count = 181;
    const uint selected_index = min(uint(max(selected[0], 0.0)), file_count - 1);
    const bool orbiting = orbit_selected > 0.5;
    const float2 pivot = float2(positions[selected_index]);
    const float2 camera_center = orbiting ? pivot : float2(center_x, center_y);
    const float2 pointer = float2(pointer_x, pointer_y);
    float closest_distance = selection_radius;
    int closest = -1;

    for (uint candidate = 0; candidate < file_count; ++candidate) {
        float2 point = float2(positions[candidate]);
        if (orbiting) point = pivot + repository_rotate(point - pivot, orbit_angle);
        const float distance_to_pointer = distance(
            repository_project_point(point, camera_center, zoom, aspect),
            pointer
        );
        if (distance_to_pointer < closest_distance) {
            closest_distance = distance_to_pointer;
            closest = int(candidate);
        }
    }

    if (closest >= 0) selected[0] = float(closest);
    click_seen[0] = click_generation;
}

kernel void codebase_particles_project(
    const device packed_float2 *world_positions [[buffer(0)]],
    const device float *groups [[buffer(1)]],
    const device float *weights [[buffer(2)]],
    const device float *selected [[buffer(3)]],
    device packed_float2 *render_positions [[buffer(4)]],
    device float *render_groups [[buffer(5)]],
    device float *render_weights [[buffer(6)]],
    device float *render_selected [[buffer(7)]],
    constant float &center_x [[buffer(8)]],
    constant float &center_y [[buffer(9)]],
    constant float &zoom [[buffer(10)]],
    constant float &orbit_angle [[buffer(11)]],
    constant float &orbit_selected [[buffer(12)]],
    constant float &aspect [[buffer(13)]],
    uint index [[thread_position_in_grid]])
{
    constexpr uint file_count = 181;
    if (index >= file_count) return;
    const uint selected_index = min(uint(max(selected[0], 0.0)), file_count - 1);
    const bool orbiting = orbit_selected > 0.5;
    const float2 pivot = float2(world_positions[selected_index]);
    float2 point = float2(world_positions[index]);
    if (orbiting) point = pivot + repository_rotate(point - pivot, orbit_angle);
    const float2 camera_center = orbiting ? pivot : float2(center_x, center_y);

    render_positions[index] = packed_float2(
        repository_project_point(point, camera_center, zoom, aspect)
    );
    render_groups[index] = groups[index];
    render_weights[index] = weights[index];
    render_selected[index] = index == selected_index ? 1.0 : 0.0;
}
