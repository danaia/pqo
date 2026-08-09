# Quantum Field

Quantum Field visualizes a two-dimensional complex scalar wave as 32,768
GPU-resident tracer particles. Three moving Gaussian wave packets interfere;
their amplitude controls where particles are generated, their phase controls
color, and their probability current controls motion.

This is a physically inspired quantum-field visualization, not a numerical
prediction from a full relativistic quantum field theory. In particular,
particle creation and annihilation are represented by importance-sampling and
recycling a fixed GPU particle pool.

## Run

From the repository root:

```text
cargo build --package pqo-cli
target/debug/pqo check examples/quantum-field/quantum-field.pqo
target/debug/pqo explain examples/quantum-field/quantum-field.pqo
target/debug/pqo examples/quantum-field/quantum-field.pqo
```

Close the Metal window to stop the program. You can also build a portable Pqo
package that includes both Metal files:

```text
target/debug/pqo build examples/quantum-field/quantum-field.pqo
target/debug/pqo examples/quantum-field/quantum-field.lmp
```

## What is "smart generation" here?

Each expired tracer tests eight deterministic candidate positions entirely on
the GPU and chooses a high-density sample. This bounded importance sampler
concentrates visual detail where `|psi|^2` is large, while staggered lifetimes
continually refresh coverage as the field changes. There is no per-particle CPU
spawn loop, readback, or random-number state.

## Implementation boundary

- **Native Pqo:** the shared simulation clock, typed position integration,
  lifetime evolution, and trail memory.
- **External Metal compute:** complex wave evaluation, analytic gradient,
  probability current, bounded importance sampling, and respawn policy.
- **External Metal render:** phase color, luminous trails, and additive GPU
  composition.
