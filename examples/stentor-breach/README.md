# Stentor Breach

An 8,000–49,600-particle, GPU-resident study of a **Holophrya** piercing a **Stentor**, feeding on the expelled cytoplasm, and the Stentor repairing its membrane. The default remains 12,400 particles.

The simulation is biologically inspired rather than a calibrated laboratory model. Its forces are explicit and tunable:

- membrane particles use a damped elastic shell with adjustable tension;
- cytoplasm and organelles use overdamped confinement plus Langevin-style Brownian forcing;
- rupture particles are pressure-advected through a localized wound;
- Holophrya follows the plume with a tunable chemotactic response;
- four harpoon filaments deploy into the wound;
- the repair phase contracts the wound and restores displaced material.

## Run

```sh
cargo build -p pqo-cli
target/debug/pqo check examples/stentor-breach/stentor-breach.pqo
target/debug/pqo examples/stentor-breach/stentor-breach.pqo
```

The source-run command uses the checked-in `ui/dist` panel and does not require
local Node dependencies. To edit and rebuild the Vue panel itself:

```sh
cd examples/stentor-breach/ui
npm ci
npm run build:ui
```

The packaged `stentor-breach.lmp` is also self-contained and can be run with an
installed Pqo binary from any working directory.

The five phases loop automatically every 40 seconds. Use the companion panel to pause, jump to a phase, increase the visible particle population, tune the four biophysical parameters, or reset the specimen.

## Particle populations

| Population | Default | Maximum |
| --- | ---: | ---: |
| Stentor membrane | 4,400 | 17,600 |
| Cytoplasm | 3,500 | 14,000 |
| Organelle clusters | 1,600 | 6,400 |
| Holophrya | 1,800 | 7,200 |
| Harpoon filaments | 400 | 1,600 |
| Expelled material | 700 | 2,800 |
| **Total** | **12,400** | **49,600** |
