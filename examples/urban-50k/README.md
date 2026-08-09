# URBAN / 50K

An interactive, deterministic human-population mobility study rendered as a 3D city particle field.

The UI initializes 50,000 agents once, then evolves their walking, running/cycling, car, rail, and air routes without runtime randomness. Agent affect can be inspected independently from transport mode, and Manhattan, Barcelona, and Tokyo-derived street layouts can be compared from the same control surface.

## Run the research UI

```sh
cd examples/urban-50k/ui
npm install
npm run dev
```

This first version is a browser-based visual and interaction prototype intended to establish the simulation grammar, interface, and performance envelope before moving agent state into native PQO/Metal streams.
