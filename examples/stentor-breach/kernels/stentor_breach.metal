#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

inline float hash11(float p) {
    return fract(sin(p * 127.1f + 311.7f) * 43758.5453123f);
}

inline float3 hash31(float p) {
    return fract(sin(float3(p, p + 19.19f, p + 47.47f) *
        float3(127.1f, 269.5f, 419.2f)) * 43758.5453f) * 2.0f - 1.0f;
}

inline float3 safe_normalize(float3 value) {
    const float magnitude = length(value);
    return magnitude > 0.00001f ? value / magnitude : float3(1.0f, 0.0f, 0.0f);
}

inline float3 stentor_surface(float a, float b) {
    const float theta = 2.0f * PI * a;
    const float y_unit = 2.0f * b - 1.0f;
    const float ring = sqrt(max(1.0f - y_unit * y_unit, 0.0f));
    const float pear = 0.66f + 0.24f * (0.5f - 0.5f * y_unit);
    float3 p = float3(
        ring * cos(theta) * pear,
        y_unit * 1.13f,
        ring * sin(theta) * pear * 0.72f
    );
    p.x += 0.34f;
    p.y -= 0.03f;
    return p;
}

inline float3 inside_stentor(float index, float scale) {
    const float3 random = hash31(index * 1.771f);
    const float3 direction = safe_normalize(random);
    const float radius = pow(hash11(index * 2.311f + 4.0f), 0.333333f) * scale;
    const float y = direction.y * radius;
    const float pear = 0.82f + 0.18f * (-direction.y);
    return float3(
        0.34f + direction.x * radius * 0.74f * pear,
        -0.03f + y * 1.08f,
        direction.z * radius * 0.54f * pear
    );
}

inline float3 holophrya_center(float stage, float progress, float chemotaxis) {
    const float approach = smoothstep(0.0f, 1.0f, clamp((stage + progress - 0.2f) / 1.5f, 0.0f, 1.0f));
    const float feed = smoothstep(2.0f, 3.5f, stage + progress);
    const float attraction = approach * (0.52f + 0.48f * chemotaxis);
    return float3(
        mix(-1.30f, -0.63f, attraction) + 0.035f * sin(progress * PI * 2.0f),
        mix(-0.24f, 0.06f, attraction) + 0.05f * sin(feed * 5.0f),
        -0.06f + 0.035f * cos(progress * PI * 2.0f)
    );
}

kernel void stentor_advance_clock(
    device packed_float4* state [[buffer(0)]],
    constant float& playing [[buffer(1)]],
    constant float& selected_phase [[buffer(2)]],
    constant float& reset_scene [[buffer(3)]],
    constant float& dt [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index > 0u) return;
    float4 value = float4(state[0]);
    float elapsed = value.x;
    float generation = value.w;
    const float requested = clamp(round(selected_phase), 0.0f, 4.0f);
    if (reset_scene > 0.5f) {
        elapsed = requested * 8.0f;
        generation += 1.0f;
    } else if (playing > 0.5f) {
        elapsed += dt;
        if (elapsed >= 40.0f) {
            elapsed -= 40.0f;
            generation += 1.0f;
        }
    } else if (abs(floor(elapsed / 8.0f) - requested) > 0.1f) {
        elapsed = requested * 8.0f;
        generation += 1.0f;
    }
    const float stage = min(floor(elapsed / 8.0f), 4.0f);
    const float progress = fract(elapsed / 8.0f);
    state[0] = packed_float4(elapsed, stage, progress, generation);
}

kernel void stentor_evolve_particles(
    device packed_float3* position [[buffer(0)]],
    device packed_float3* velocity [[buffer(1)]],
    device packed_float3* home [[buffer(2)]],
    device float* role [[buffer(3)]],
    device float* seed [[buffer(4)]],
    device float* generation [[buffer(5)]],
    device float4* color [[buffer(6)]],
    device float* radius [[buffer(7)]],
    const device packed_float4* clock_state [[buffer(8)]],
    constant float& particle_density [[buffer(9)]],
    constant float& membrane_tension [[buffer(10)]],
    constant float& brownian_motion [[buffer(11)]],
    constant float& chemotaxis [[buffer(12)]],
    constant float& repair_rate [[buffer(13)]],
    constant float& dt [[buffer(14)]],
    uint index [[thread_position_in_grid]])
{
    const float4 clock = float4(clock_state[0]);
    const float stage = clock.y;
    const float progress = clock.z;
    const float cycle_generation = clock.w;
    const float fi = float(index);
    const float s = hash11(fi + 1.0f);
    const bool needs_reset = generation[index] != cycle_generation;

    float particle_role;
    if (index < 17600u) particle_role = 0.0f;
    else if (index < 31600u) particle_role = 1.0f;
    else if (index < 38000u) particle_role = 2.0f;
    else if (index < 45200u) particle_role = 3.0f;
    else if (index < 46800u) particle_role = 4.0f;
    else particle_role = 5.0f;

    // A stable stochastic mask keeps each biological population represented at
    // every density. Inactive particles retain no visible radius and are marked
    // for clean initialization if the user raises the density later.
    const bool active = hash11(fi * 5.313f + 17.0f) < clamp(particle_density, 0.16f, 1.0f);
    if (!active) {
        color[index] = float4(0.0f);
        radius[index] = 0.0f;
        generation[index] = -1.0f;
        return;
    }

    float3 target = float3(0.0f);
    float3 initial = float3(0.0f);
    float point_radius = 0.006f;
    float4 point_color = float4(0.85f, 0.93f, 0.73f, 0.72f);

    if (particle_role == 0.0f) {
        initial = stentor_surface(hash11(fi * 1.117f), hash11(fi * 1.913f));
        target = initial;
        point_radius = mix(0.0052f, 0.0088f, s);
        point_color = mix(float4(0.80f, 1.00f, 0.82f, 0.78f),
                          float4(1.00f, 0.82f, 0.36f, 0.98f), hash11(fi * 3.1f));
    } else if (particle_role == 1.0f) {
        initial = inside_stentor(fi, 0.93f);
        target = initial;
        point_radius = mix(0.0035f, 0.0062f, s);
        point_color = float4(0.82f, 0.94f, 0.68f, mix(0.12f, 0.34f, s));
    } else if (particle_role == 2.0f) {
        const uint cluster = (index - 31600u) / 400u;
        const float3 center = inside_stentor(float(cluster) * 41.0f + 9.0f, 0.63f);
        const float3 local = safe_normalize(hash31(fi * 1.37f)) *
            pow(hash11(fi * 1.71f), 0.333f) * (0.07f + 0.016f * float(cluster % 4u));
        initial = center + local;
        target = initial;
        point_radius = mix(0.005f, 0.009f, s);
        point_color = cluster % 3u == 0u
            ? float4(0.60f, 0.88f, 0.12f, 0.68f)
            : float4(0.98f, 0.70f, 0.18f, 0.58f);
    } else if (particle_role == 3.0f) {
        const float3 center = holophrya_center(stage, progress, chemotaxis);
        const float3 direction = safe_normalize(hash31(fi * 2.07f));
        const float shell = mix(0.10f, 0.29f, pow(hash11(fi * 0.81f), 0.33f));
        initial = float3(-1.30f, -0.24f, -0.06f) + direction * shell;
        target = center + direction * shell;
        point_radius = mix(0.0045f, 0.0085f, s);
        point_color = mix(float4(0.10f, 0.92f, 0.88f, 0.78f),
                          float4(0.25f, 0.72f, 0.61f, 0.44f), shell / 0.29f);
    } else if (particle_role == 4.0f) {
        const uint filament = (index - 45200u) / 400u;
        const float along = float((index - 45200u) % 400u) / 399.0f;
        const float3 center = holophrya_center(stage, progress, chemotaxis);
        const float angle = (float(filament) - 1.5f) * 0.24f;
        const float3 start = center + float3(0.24f, angle * 0.34f, sin(angle) * 0.18f);
        const float deployed = smoothstep(0.05f, 0.78f, stage + progress);
        const float3 wound = float3(-0.42f, 0.10f + angle * 0.28f, angle * 0.11f);
        const float3 end = mix(start + float3(0.10f, 0.0f, 0.0f), wound, deployed);
        const float wave = sin(along * PI * 3.0f + clock.x * 2.0f + float(filament));
        initial = mix(start, end, along) + float3(0.0f, wave * 0.012f, wave * 0.009f);
        target = initial;
        point_radius = 0.0044f;
        point_color = float4(0.22f, 1.00f, 0.96f, 0.94f);
    } else {
        initial = inside_stentor(fi, 0.62f);
        target = initial;
        point_radius = mix(0.006f, 0.012f, s);
        point_color = hash11(fi * 2.7f) > 0.62f
            ? float4(0.63f, 0.91f, 0.12f, 0.82f)
            : float4(0.96f, 0.74f, 0.30f, 0.72f);
    }

    if (needs_reset) {
        position[index] = packed_float3(initial);
        velocity[index] = packed_float3(float3(0.0f));
        home[index] = packed_float3(initial);
        role[index] = particle_role;
        seed[index] = s;
        generation[index] = cycle_generation;
    }

    float3 p = float3(position[index]);
    float3 v = float3(velocity[index]);
    const float3 base_home = float3(home[index]);
    const float time = clock.x;
    const float3 thermal = hash31(fi * 0.37f + floor(time * 18.0f)) *
        brownian_motion * (particle_role == 0.0f ? 0.10f : 0.23f);
    float3 acceleration = thermal;

    if (particle_role == 0.0f) {
        const float wound_distance = length(float2(base_home.y - 0.10f, base_home.z));
        const float wound_side = smoothstep(-0.22f, -0.70f, base_home.x);
        const float wound_mask = wound_side * (1.0f - smoothstep(0.16f, 0.46f, wound_distance));
        const float rupture = stage < 2.0f ? 0.0f :
            (stage < 4.0f ? smoothstep(0.0f, 0.7f, stage + progress - 2.0f) :
             1.0f - smoothstep(0.0f, 1.0f, progress * repair_rate * 1.5f));
        const float3 opened = base_home + float3(-0.30f, base_home.y - 0.10f, base_home.z) * wound_mask * rupture;
        const float ripple = sin(fi * 0.21f + time * 4.3f) * 0.012f;
        const float3 membrane_target = opened + safe_normalize(base_home - float3(0.34f, -0.03f, 0.0f)) * ripple;
        acceleration += (membrane_target - p) * mix(5.0f, 18.0f, membrane_tension);
        acceleration -= v * mix(2.4f, 5.5f, membrane_tension);
    } else if (particle_role == 1.0f || particle_role == 2.0f) {
        const float mobility = particle_role == 2.0f ? 0.55f : 1.0f;
        acceleration += (target - p) * (0.5f + membrane_tension * 0.8f) * mobility;
        acceleration -= v * 2.2f;
        acceleration += float3(
            sin(time * 0.61f + fi * 0.013f),
            cos(time * 0.48f + fi * 0.017f),
            sin(time * 0.55f + fi * 0.011f)
        ) * 0.025f;
    } else if (particle_role == 3.0f) {
        acceleration += (target - p) * 13.0f;
        acceleration -= v * 5.8f;
    } else if (particle_role == 4.0f) {
        acceleration += (target - p) * 28.0f;
        acceleration -= v * 7.0f;
    } else {
        const float rupture_age = clamp(stage + progress - 2.0f, 0.0f, 2.0f);
        if (stage >= 2.0f && stage < 4.0f) {
            const float delay = s * 0.78f;
            const float released = smoothstep(delay, delay + 0.18f, rupture_age);
            const float3 plume = float3(-1.0f, (s - 0.5f) * 0.42f, (hash11(fi * 4.1f) - 0.5f) * 0.34f);
            acceleration += plume * released * (1.2f + 1.4f * s);
            const float3 holo = holophrya_center(stage, progress, chemotaxis);
            acceleration += safe_normalize(holo - p) * chemotaxis * smoothstep(2.8f, 3.8f, stage + progress) * 1.2f;
            acceleration -= v * 0.9f;
        } else if (stage >= 4.0f) {
            acceleration += (base_home - p) * repair_rate * 2.4f;
            acceleration -= v * 2.0f;
        } else {
            acceleration += (base_home - p) * 1.2f - v * 2.4f;
        }
    }

    v += acceleration * dt;
    const float max_speed = particle_role == 5.0f ? 1.6f : 0.72f;
    if (length(v) > max_speed) v = safe_normalize(v) * max_speed;
    p += v * dt;
    position[index] = packed_float3(p);
    velocity[index] = packed_float3(v);
    color[index] = point_color;
    radius[index] = point_radius;
}

kernel void stentor_project_particles(
    const device packed_float3* world_position [[buffer(0)]],
    const device float4* world_color [[buffer(1)]],
    const device float* world_radius [[buffer(2)]],
    device packed_float3* render_position [[buffer(3)]],
    device float4* render_color [[buffer(4)]],
    device float* render_radius [[buffer(5)]],
    const device packed_float4* clock_state [[buffer(6)]],
    constant float& aspect [[buffer(7)]],
    constant float& yaw [[buffer(8)]],
    constant float& pitch [[buffer(9)]],
    constant float& distance [[buffer(10)]],
    uint index [[thread_position_in_grid]])
{
    float3 p = float3(world_position[index]);
    const float time = float4(clock_state[0]).x;
    const float orbit = yaw + sin(time * 0.08f) * 0.11f;
    const float cy = cos(orbit), sy = sin(orbit);
    const float cp = cos(pitch), sp = sin(pitch);
    p.xz = float2(cy * p.x - sy * p.z, sy * p.x + cy * p.z);
    p.yz = float2(cp * p.y - sp * p.z, sp * p.y + cp * p.z);
    const float depth = max(distance - p.z, 0.25f);
    // Fill the portrait-biased specimen window while preserving enough lateral
    // room for Holophrya and the rupture plume.
    const float focal = 3.85f;
    const float scale = focal / depth;
    render_position[index] = packed_float3(p.x * scale / aspect, p.y * scale, depth);
    render_color[index] = world_color[index];
    render_radius[index] = world_radius[index] * scale;
}
