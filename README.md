# ciftty

`ciftty` is a terminal CIF crystal structure viewer written in Rust.

Renders atomic structures as shaded ASCII spheres with bonds, periodic images,
and a unit-cell wireframe — all inside your terminal.

## Install

```bash
cargo install ciftty
```

Requires Rust 1.85+ (edition 2024).

## Usage

```bash
# open a structure directly
ciftty path/to/structure.cif

# start without a file and use the built-in browser to open one
ciftty
```

Press `Shift+O` at any time to open the CIF browser dialog.

## Controls

### Navigation
| Key | Action |
|-----|--------|
| `h` / `l` | Rotate left / right |
| `j` / `k` | Rotate down / up |
| `u` / `o` | Roll |
| `w` / `a` / `s` / `d` | Pan |
| `i` | Isometric view |
| `Shift+A` / `B` / `C` | Snap to lattice axis a / b / c |

### Camera
| Key | Action |
|-----|--------|
| `+` / `-` or scroll wheel | Zoom in / out |
| `,` / `.` or `Ctrl+scroll` | Decrease / increase FOV |
| `z` | Toggle FOV-size lock (keeps apparent size constant while changing FOV) |

### Display
| Key | Action |
|-----|--------|
| `b` | Toggle bonds |
| `c` | Toggle unit-cell wireframe |
| `x` | Toggle cell-on-top overlay |
| `r` | Toggle boundary-repeat atom images |
| `t` | Toggle bonded outside-cell atom images |
| `n` / `m` | Decrease / increase max bond length |
| `Shift+N` / `Shift+M` | Larger bond length step |
| `[` / `]` | Decrease / increase sphere size |
| `g` | Cycle render theme (`orbital` → `neon` → `wild` → `dense` → `classic`) |
| `v` | Toggle orientation gizmo |
| `Shift+L` | Toggle atom labels |

### Spin lock
| Key | Action |
|-----|--------|
| `Shift+R` | Toggle spin-lock mode |
| `h` / `j` / `k` / `l` / `u` / `o` | Set spin direction (while in spin-lock mode) |
| `<` / `>` | Decrease / increase spin speed |

### Other
| Key | Action |
|-----|--------|
| `Shift+O` | Open CIF browser dialog |
| `Tab` | Cycle selected atom |
| `q` | Quit |

Mouse drag rotates the view; scroll wheel zooms (`Ctrl+scroll` changes FOV).

## Features

- Perspective projection with configurable FOV and zoom
- Lambert-shaded spheres with five render themes
- Bond detection with configurable max distance
- Symmetry expansion from `_symmetry_equiv_pos_as_xyz` / `_space_group_symop` loops
- Periodic boundary images and bonded outside-cell images
- Orientation gizmo (RGB x/y/z axes)
- Kitty keyboard protocol for smooth key-repeat

## License

MIT — see [LICENSE](LICENSE).
