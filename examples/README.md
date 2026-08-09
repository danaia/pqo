# Runnable Pqo examples

Every `.pqo` entry point in this directory can be launched directly with the
Pqo CLI.

From the repository root:

```text
pqo examples/hello-particle/hello-particle.pqo
pqo examples/neon-flock/neon-flock.pqo
pqo examples/codebase-particles/codebase-particles.pqo
pqo examples/hello-crystal/crystal.pqo
pqo build examples/marble-water/marble-water.pqo
pqo examples/marble-water/marble-water.lmp
```

The Crystal example is interactive: drag across the crystal to slice it, drag
the black background to spin it, and scroll to zoom. The cut heals
automatically.

Codebase Particles renders the repository's source, configuration, and
documentation files as five animated subsystem constellations.

From inside an example directory:

```text
cd examples/hello-particle
pqo hello-particle.pqo

cd ../hello-crystal
pqo crystal.pqo
```

The installed distribution places the same programs at:

```text
pqo ~/.pqo/examples/hello-particle.pqo
pqo ~/.pqo/examples/neon-flock.pqo
pqo ~/.pqo/examples/crystal.pqo
pqo ~/.pqo/examples/marble-water.lmp
```

`pqo run FILE` is the explicit equivalent of `pqo FILE`. Use `pqo check FILE`
to validate without opening a window and `pqo explain FILE` to inspect the
canonical graph, execution plan, and generated or packaged Metal.

Update the installed compiler and runtime with:

```text
pqo update
```
