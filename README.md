# ciftty

`ciftty` is a terminal CIF crystal structure viewer and editor written in Rust.

Renders atomic structures as shaded ASCII spheres with bonds, periodic images,
and a unit-cell wireframe. It can also simulate powder XRD patterns and edit/save
structures directly in the TUI.

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

# browse a sampled ensemble (optionally with a target CIF for ranking)
ciftty ensemble path/to/samples_dir [path/to/target.cif]
```

Press `Shift+O` at any time to open the CIF browser dialog.
Press `Shift+E` to edit the current structure, or `Shift+F` to start a new one.

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

### Diffraction
| Key | Action |
|-----|--------|
| `Shift+X` | Toggle XRD panel |
| `Shift+W` | Cycle X-ray source (`Cu Kα` / `Mo Kα` / `Co Kα` / `Ag Kα`) |
| `Shift+Y` | Toggle target reference details (ensemble mode) |

### Ensemble (when started with `ciftty ensemble ...`)
| Key | Action |
|-----|--------|
| `↑` / `↓` | Previous / next sampled structure |
| `PageUp` / `PageDown` | Jump 5 samples |
| `Shift+P` | Cycle ranking mode (`target` / `novelty` / `name`) |
| `Shift+G` | Toggle 3×3 ensemble grid view (up to 9 samples/page) |

In ensemble mode, camera/display controls apply only to the focused sample.

### Editor
| Key | Action |
|-----|--------|
| `Shift+E` | Toggle editor for current structure |
| `Shift+F` | Open editor with a new empty structure |

### Editor (while editor is open)
| Key | Action |
|-----|--------|
| `Enter` | Enter/leave field edit mode |
| `Tab` / `Shift+Tab` | Next / previous field |
| Arrow keys | Move focus (nav mode) or cursor/focus (edit mode) |
| `A` | Add atom row |
| `D` / `Delete` | Delete focused atom row |
| `Ctrl+A` | Apply edits to the live structure |
| `Ctrl+S` | Save CIF (existing path or `<title>.cif`) |
| `Esc` | Exit edit mode (or close editor from nav mode) |

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
- Simulated powder XRD panel with wavelength presets and top-peak labels
- Ensemble mode for browsing multiple sampled CIFs with target-distance ranking
- Built-in structure editor for title, cell, space group, and asymmetric-unit atoms
- Apply edits live and save back to CIF
- Kitty keyboard protocol for smooth key-repeat

## License

MIT — see [LICENSE](LICENSE).
