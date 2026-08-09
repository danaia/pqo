# Codebase Particles

A native Metal view of this Pqo repository as a living constellation. The
current snapshot contains 181 relevant source, configuration, and documentation
files. Each file becomes one GPU particle.

- Cyan: compiler and runtime crates (44 files)
- Violet: documentation (34 files)
- Green: runnable examples (42 files)
- Coral: the full baseline application (30 files)
- Gold: root files, benchmarks, scripts, kernels, and shaders (31 files)

Particle scale is a deterministic approximation of the repository's long-tail
file-size distribution. The cluster counts come from the repository snapshot;
this first experiment does not yet regenerate metadata when files change.

Run from the repository root:

```sh
target/debug/pqo check examples/codebase-particles/codebase-particles.pqo
target/debug/pqo explain examples/codebase-particles/codebase-particles.pqo
target/debug/pqo examples/codebase-particles/codebase-particles.pqo
```
