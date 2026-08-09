#include <metal_stdlib>
using namespace metal;

struct FieldSample {
    float2 value;
    float2 gradient_x;
    float2 gradient_y;
};

inline float hash11(float value)
{
    return fract(sin(value * 127.1 + 311.7) * 43758.5453123);
}

inline float2 complex_multiply(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void add_packet(
    thread FieldSample &sample,
    float2 point,
    float2 center,
    float2 wave_vector,
    float sigma,
    float phase_offset,
    float amplitude)
{
    const float2 relative = point - center;
    const float inverse_sigma_squared = 1.0 / max(sigma * sigma, 1e-5);
    const float envelope = amplitude
        * exp(-0.5 * dot(relative, relative) * inverse_sigma_squared);
    const float angle = dot(wave_vector, relative) + phase_offset;
    const float2 wave = envelope * float2(cos(angle), sin(angle));
    const float2 logarithmic_x = float2(
        -relative.x * inverse_sigma_squared,
        wave_vector.x
    );
    const float2 logarithmic_y = float2(
        -relative.y * inverse_sigma_squared,
        wave_vector.y
    );
    sample.value += wave;
    sample.gradient_x += complex_multiply(wave, logarithmic_x);
    sample.gradient_y += complex_multiply(wave, logarithmic_y);
}

inline FieldSample evaluate_field(float2 point, float time, float sigma)
{
    FieldSample sample = { float2(0.0), float2(0.0), float2(0.0) };

    // Two counter-propagating packets collide through the center while a
    // slower transverse packet breaks symmetry and creates moving nodes.
    const float sweep = 0.30 * sin(time * 0.34);
    add_packet(
        sample,
        point,
        float2(-0.43 + sweep, 0.18 * sin(time * 0.47)),
        float2(8.2, 1.35),
        sigma,
        -3.65 * time,
        1.0
    );
    add_packet(
        sample,
        point,
        float2(0.43 - sweep, -0.18 * sin(time * 0.47)),
        float2(-8.2, -1.35),
        sigma,
        -3.65 * time + 0.42,
        1.0
    );
    add_packet(
        sample,
        point,
        float2(0.10 * sin(time * 0.29), 0.38 * cos(time * 0.31)),
        float2(-1.7, 6.4),
        sigma * 0.86,
        -2.15 * time + 1.1,
        0.72
    );
    return sample;
}

inline float probability_density(FieldSample sample)
{
    return dot(sample.value, sample.value);
}

inline float2 probability_velocity(FieldSample sample, float flow_gain)
{
    const float density = max(probability_density(sample), 1e-5);
    // Im(conj(psi) grad(psi)) / |psi|^2, in visualization units.
    const float current_x =
        sample.value.x * sample.gradient_x.y
        - sample.value.y * sample.gradient_x.x;
    const float current_y =
        sample.value.x * sample.gradient_y.y
        - sample.value.y * sample.gradient_y.x;
    return float2(current_x, current_y) * (flow_gain / density);
}

inline float2 spawn_position(
    uint index,
    uint generation,
    float time,
    float half_extent,
    float sigma,
    uint candidate_count)
{
    float2 best_position = float2(0.0);
    float best_score = -1.0;
    const uint count = clamp(candidate_count, 1u, 16u);
    for (uint candidate = 0; candidate < count; ++candidate) {
        const float seed =
            float(index + 1u) * 19.19
            + float(generation + 1u) * 73.73
            + float(candidate + 1u) * 151.51;
        const float2 proposal = (float2(
            hash11(seed + 0.17),
            hash11(seed + 9.31)
        ) * 2.0 - 1.0) * half_extent;
        const float density = probability_density(
            evaluate_field(proposal, time, sigma)
        );
        // Jittering the score avoids visible lock-in to a small set of maxima.
        const float score = density * mix(0.72, 1.0, hash11(seed + 27.4));
        if (score > best_score) {
            best_score = score;
            best_position = proposal;
        }
    }
    return best_position;
}

kernel void quantum_field_sample_main(
    device packed_float2 *positions [[buffer(0)]],
    device packed_float2 *velocities [[buffer(1)]],
    device packed_float2 *trail_positions [[buffer(2)]],
    device float *ages [[buffer(3)]],
    device float *lifetimes [[buffer(4)]],
    device float *intensities [[buffer(5)]],
    device float *phases [[buffer(6)]],
    const device float *times [[buffer(7)]],
    constant uint &population [[buffer(8)]],
    constant float &half_extent [[buffer(9)]],
    constant float &packet_sigma [[buffer(10)]],
    constant float &flow_gain [[buffer(11)]],
    constant float &velocity_response [[buffer(12)]],
    constant float &maximum_speed [[buffer(13)]],
    constant float &density_floor [[buffer(14)]],
    constant uint &spawn_candidates [[buffer(15)]],
    constant float &minimum_lifetime [[buffer(16)]],
    constant float &lifetime_span [[buffer(17)]],
    constant float &dt [[buffer(18)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= population) return;

    const float time = times[0];
    float2 position = float2(positions[index]);
    float2 velocity = float2(velocities[index]);
    float age = ages[index];
    float lifetime = lifetimes[index];
    FieldSample sample = evaluate_field(position, time, packet_sigma);
    float density = probability_density(sample);
    const bool invalid = !isfinite(position.x) || !isfinite(position.y);
    const bool escaped = any(abs(position) > half_extent);
    const bool depleted = density < density_floor;
    const bool expired = lifetime <= 0.0 || age >= lifetime;

    if (invalid || escaped || depleted || expired) {
        const uint generation = uint(max(age, 0.0) * 19.0 + time * 7.0);
        position = spawn_position(
            index,
            generation,
            time,
            half_extent * 0.94,
            packet_sigma,
            spawn_candidates
        );
        sample = evaluate_field(position, time, packet_sigma);
        density = probability_density(sample);
        velocity = probability_velocity(sample, flow_gain);
        age = 0.0;
        lifetime = minimum_lifetime
            + lifetime_span * hash11(float(index + 1u) * 41.3 + time * 3.1);
        positions[index] = packed_float2(position);
        trail_positions[index] = packed_float2(position);
        ages[index] = age;
        lifetimes[index] = lifetime;
    }

    float2 target_velocity = probability_velocity(sample, flow_gain);
    const float target_speed = length(target_velocity);
    if (target_speed > maximum_speed) {
        target_velocity *= maximum_speed / target_speed;
    }
    const float response = 1.0 - exp(-velocity_response * dt);
    velocity = mix(velocity, target_velocity, response);

    const float phase = atan2(sample.value.y, sample.value.x);
    const float normalized_density = density / (1.0 + density);
    velocities[index] = packed_float2(velocity);
    intensities[index] = clamp(
        0.10 + normalized_density * 1.25,
        0.0,
        1.0
    );
    phases[index] = phase;
}
