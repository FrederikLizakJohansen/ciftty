# ciftty

`ciftty` is a terminal CIF viewer written in Rust (`ratatui` + `crossterm`).

## Quick Start

```bash
cargo run -- path/to/structure.cif
```

## Core Controls

- `h/j/k/l` rotate, `u/o` roll, `w/a/s/d` pan
- `+/-` zoom, `,/.` FOV, `z` FOV-size lock
- `b` bonds, `c` cell, `x` cell-on-top overlay
- `r` boundary-repeat atoms, `t` bonded outside-cell atoms
- `n/m` max bond length (`Shift+N/M` larger step)
- `i` isometric, `Shift+A/B/C` axis views
- `Tab` select atom, `Shift+L` labels, `q` quit
