# ciftty

`ciftty` is a terminal CIF viewer written in Rust (`ratatui` + `crossterm`).

## Quick Start

```bash
cargo run -- path/to/structure.cif
```

## Core Controls

- `h/j/k/l` rotate, `u/o` roll, `w/a/s/d` pan
- `+/-` or mouse wheel zoom, `,/.` or `Ctrl+wheel` FOV, `z` FOV-size lock
- `b` bonds, `c` cell, `x` cell-on-top overlay
- `r` boundary-repeat atoms, `t` bonded outside-cell atoms
- `n/m` max bond length (`Shift+N/M` larger step)
- `i` isometric, `Shift+A/B/C` axis views
- `v` toggle orientation gizmo (`x/y/z` RGB axes)
- `g` cycle render theme (`dense`/`classic`/`orbital`/`neon`, default: `orbital`)
- `Tab` select atom, `Shift+L` labels (default: on), `q` quit
