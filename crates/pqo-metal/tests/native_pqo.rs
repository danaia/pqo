#![cfg(target_os = "macos")]

use pqo_metal::{BenchmarkConfig, BenchmarkMode, BenchmarkRunner, MetalRuntime};
use pqo_syntax::parse;
use pqo_validator::Validator;

const SOURCE: &str = include_str!("../../../examples/hello-particle/hello-particle.agent.pqo");
const CRYSTAL_SOURCE: &str = include_str!("../../../examples/hello-crystal/crystal.pqo");
const NEON_FLOCK_SOURCE: &str = include_str!("../../../examples/neon-flock/neon-flock.pqo");
const MARBLE_WATER_SOURCE: &str = include_str!("../../../examples/marble-water/marble-water.pqo");
const QUANTUM_FIELD_SOURCE: &str =
    include_str!("../../../examples/quantum-field/quantum-field.pqo");

#[test]
fn native_pqo_kernel_generates_compiles_and_executes_metal() {
    let graph = parse(SOURCE).expect("agent-native source must parse");
    let integrate = graph
        .kernels
        .iter()
        .find(|kernel| kernel.name == "integrate")
        .expect("integration kernel");
    let implementation = &integrate.implementations[0];
    let generated = implementation
        .source_text
        .as_deref()
        .expect("native Pqo kernel must package generated Metal");
    assert!(implementation.source.starts_with("pqo://generated/"));
    assert!(generated.contains("kernel void integrate_main"));
    assert!(generated.contains("device packed_float3 *position [[buffer(0)]]"));

    let validated = Validator::validate(&graph)
        .validated
        .expect("generated graph must validate");
    let result = MetalRuntime::benchmark(
        validated,
        BenchmarkConfig {
            mode: BenchmarkMode::Headless,
            runner: BenchmarkRunner::PqoPlan,
            warmup_ticks: 0,
            sample_ticks: 1,
            ..BenchmarkConfig::default()
        },
    )
    .expect("generated Metal must compile and execute");

    assert_eq!(result.sample_ticks, 1);
    assert!(result.runtime.shader_hashes.iter().any(|shader| {
        shader.source_path == "pqo://generated/hello_particle_agent/integrate.metal"
    }));
}

#[test]
fn crystal_pqo_source_compiles_and_executes_packaged_metal() {
    let graph = parse(CRYSTAL_SOURCE).expect("crystal source must parse");
    let validated = Validator::validate(&graph)
        .validated
        .expect("crystal graph must validate");
    let result = MetalRuntime::benchmark(
        validated,
        BenchmarkConfig {
            mode: BenchmarkMode::Headless,
            runner: BenchmarkRunner::PqoPlan,
            warmup_ticks: 0,
            sample_ticks: 1,
            ..BenchmarkConfig::default()
        },
    )
    .expect("crystal Metal must compile and execute");

    assert_eq!(result.sample_ticks, 1);
    assert!(
        result
            .runtime
            .shader_hashes
            .iter()
            .any(|shader| shader.source_path == "kernels/crystal.metal")
    );
    assert!(
        result
            .runtime
            .shader_hashes
            .iter()
            .any(|shader| shader.source_path == "shaders/crystal.metal")
    );
}

#[test]
fn neon_flock_compiles_and_executes_native_and_external_metal() {
    let graph = parse(NEON_FLOCK_SOURCE).expect("neon flock source must parse");
    let validated = Validator::validate(&graph)
        .validated
        .expect("neon flock graph must validate");
    let result = MetalRuntime::benchmark(
        validated,
        BenchmarkConfig {
            mode: BenchmarkMode::Rendered,
            runner: BenchmarkRunner::PqoPlan,
            warmup_ticks: 1,
            sample_ticks: 1,
            ..BenchmarkConfig::default()
        },
    )
    .expect("neon flock Metal must compile, execute, and render");

    assert_eq!(result.sample_ticks, 1);
    for source in [
        "kernels/neon_flock.metal",
        "pqo://generated/neon_flock/advance_agents.metal",
        "pqo://generated/neon_flock/evolve_trails.metal",
        "shaders/neon_flock.metal",
    ] {
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| shader.source_path == source),
            "missing shader identity for {source}"
        );
    }
}

#[test]
fn marble_water_compiles_and_executes_the_particle_simulation() {
    let graph = parse(MARBLE_WATER_SOURCE).expect("marble water source must parse");
    let validated = Validator::validate(&graph)
        .validated
        .expect("marble water graph must validate");
    let result = MetalRuntime::benchmark_project(
        validated,
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/marble-water"),
        BenchmarkConfig {
            mode: BenchmarkMode::Headless,
            runner: BenchmarkRunner::PqoPlan,
            warmup_ticks: 1,
            sample_ticks: 2,
            ..BenchmarkConfig::default()
        },
    )
    .expect("marble water Metal must compile and execute");

    assert_eq!(result.sample_ticks, 2);
    for source in [
        "kernels/marble_water.metal",
        "pqo://generated/marble_water/integrate_water.metal",
        "shaders/marble_water.metal",
    ] {
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| shader.source_path == source),
            "missing shader identity for {source}"
        );
    }
}

#[test]
fn quantum_field_compiles_and_executes_native_and_external_metal() {
    let graph = parse(QUANTUM_FIELD_SOURCE).expect("quantum field source must parse");
    let validated = Validator::validate(&graph)
        .validated
        .expect("quantum field graph must validate");
    let result = MetalRuntime::benchmark_project(
        validated,
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/quantum-field"),
        BenchmarkConfig {
            mode: BenchmarkMode::Rendered,
            runner: BenchmarkRunner::PqoPlan,
            warmup_ticks: 1,
            sample_ticks: 1,
            ..BenchmarkConfig::default()
        },
    )
    .expect("quantum field Metal must compile, execute, and render");

    assert_eq!(result.sample_ticks, 1);
    for source in [
        "pqo://generated/quantum_field/advance_clock.metal",
        "kernels/quantum_field.metal",
        "pqo://generated/quantum_field/advance_quanta.metal",
        "pqo://generated/quantum_field/evolve_trails.metal",
        "shaders/quantum_field.metal",
    ] {
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| shader.source_path == source),
            "missing shader identity for {source}"
        );
    }
}
