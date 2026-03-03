# Terminal-Native 3D Crystal Structure Viewer (TUI)

## Technical Design Specification

------------------------------------------------------------------------

## 1. Objective

Design and implement a terminal-native 3D crystal structure viewer
capable of:

-   Loading CIF (and optionally POSCAR/XYZ) files
-   Rendering atoms in 3D directly in a terminal (TTY)
-   Supporting interactive rotation, zoom, and toggling of visual
    elements
-   Performing real-time depth-aware rendering (z-buffer)
-   Operating without OpenGL, Qt, or GUI frameworks

The tool should feel similar to `htop` or `lazygit`, but for
crystallography.

------------------------------------------------------------------------

## 2. High-Level Architecture

Input Layer (CIF / POSCAR) ↓ Scene Model (Atoms, Bonds, Cell) ↓ Camera +
Controls (Rotation / Zoom) ↓ Render Pipeline (Transform → Project →
Rasterize → Z-Buffer) ↓ Terminal Backend (ANSI + Key Input)

------------------------------------------------------------------------

## 3. Technology Stack Options

### Option A: Python

-   pymatgen or ase for CIF parsing
-   textual or urwid for TUI
-   NumPy for vector math

Pros: - Fast development - Easy CIF integration

Cons: - Performance limits for large systems

### Option B: Rust

-   ratatui for TUI
-   crossterm for terminal control
-   Custom math implementation

Pros: - Faster rendering - Better scaling to large structures

Cons: - Slower iteration initially

------------------------------------------------------------------------

## 4. Data Model

Atom: - position_cart (3D vector) - element - radius - color_index

Bond: - atom_i - atom_j

Cell: - lattice matrix (3x3) - precomputed 12 edge segments

------------------------------------------------------------------------

## 5. Camera Model

Maintain: - Rotation matrix R (3x3) - Translation vector t - Zoom scalar
s - Orthographic projection (initially)

Transform: p_cam = R @ (p - center) + t

Projection: x_screen = s \* p_cam.x y_screen = s \* p_cam.y z_depth =
p_cam.z

------------------------------------------------------------------------

## 6. Rendering Pipeline

Framebuffer: - Width × Height (terminal size) - z-buffer (float grid) -
char buffer (character grid) - color buffer (ANSI codes)

Atom Rasterization: 1. Project atom center 2. Compute projected radius
3. For each pixel in bounding box: - Check inside circle - Compute
sphere depth: z_pixel = z_center - sqrt(r\^2 - dx\^2 - dy\^2) - Compare
with z-buffer - Apply Lambert shading - Write glyph + color

Shading: Light direction L = normalize(\[0.3, 0.4, 1.0\]) Intensity I =
ambient + diffuse \* max(0, dot(n, L))

Glyph ramp: " .:-=+\*#%@"

------------------------------------------------------------------------

## 7. Controls

h/j/k/l → Rotate\
u/o → Roll\
+/- → Zoom\
w/a/s/d → Pan\
b → Toggle bonds\
c → Toggle cell\
l → Toggle labels\
tab → Cycle selection\
q → Quit

------------------------------------------------------------------------

## 8. UI Layout

Main viewport (3D render) Side panel (structure info + selected atom)
Bottom bar (key help)

------------------------------------------------------------------------

## 9. Development Milestones

M0 --- Skeleton (CIF load + TUI loop)\
M1 --- 2D projection (points only)\
M2 --- Z-buffer + disks\
M3 --- Spherical shading\
M4 --- Cell + bonds\
M5 --- Selection\
M6 --- Optimization

------------------------------------------------------------------------

## 10. Minimal Viable Implementation

-   Orthographic projection
-   Z-buffer
-   Filled sphere rasterization
-   10-character shading ramp
-   Keyboard rotation

------------------------------------------------------------------------

End of specification.
