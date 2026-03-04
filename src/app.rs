use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::io::Stdout;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::model::{Cell, Structure};

const ROTATION_STEP: f32 = 0.1;
const ROLL_STEP: f32 = 0.1;
const PAN_STEP: f32 = 0.08;
const MIN_ZOOM: f32 = 0.05;
const MIN_FOV_DEG: f32 = 10.0;
const MAX_FOV_DEG: f32 = 120.0;
const DEFAULT_FOV_DEG: f32 = 45.0;
const CAMERA_DISTANCE: f32 = 2.5;
const MAX_SCREEN_SCALE: f32 = 1_000.0;
const ISO_PITCH: f32 = 0.615_479_7; // atan(1/sqrt(2))
const ISO_YAW: f32 = std::f32::consts::FRAC_PI_4;
const BOUNDARY_EPSILON: f32 = 0.02;
// Keep cell edges slightly behind equal-depth atom surfaces without forcing
// the entire cell wireframe behind all atoms.
const CELL_LINE_DEPTH_BIAS: f32 = 1e-3;
const BOND_LINE_DEPTH_BIAS: f32 = 900.0;
const MOUSE_SENSITIVITY: f32 = 0.03; // radians per terminal column/row dragged
const MOUSE_WHEEL_ZOOM_FACTOR: f32 = 1.1;
const MOUSE_WHEEL_FOV_STEP_DEG: f32 = 2.0;
// Terminal character cells are roughly twice as tall as wide in pixels.
const CHAR_ASPECT: f32 = 2.0;
const DEFAULT_BOND_MAX_DISTANCE: f32 = 2.2;
const MIN_BOND_MAX_DISTANCE: f32 = 0.0;
const MAX_BOND_MAX_DISTANCE: f32 = 12.0;
const BOND_MAX_DISTANCE_STEP: f32 = 0.10;
// Classic ASCII shading ramp from dark to bright (spec §6).
const SHADE_RAMP_CLASSIC: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
// Dense shading ramp from dark to bright; this fills terminal rows better.
const SHADE_RAMP_DENSE: &[char] = &[' ', '░', '▒', '▓', '█'];
// Orbital theme: circular shades for a softer, stylized look.
const SHADE_RAMP_ORBITAL: &[char] = &[' ', '·', '∘', '○', '◍', '●'];
// Neon theme: high-contrast punctuated ramp.
const SHADE_RAMP_NEON: &[char] = &[' ', '.', ':', '*', 'o', 'O', '@'];
// Lambert light direction (normalized below in sphere_glyph).
const LIGHT: [f32; 3] = [0.268, 0.358, 0.894]; // normalize([0.3, 0.4, 1.0])
const CELL_LINE_COLOR: Color = Color::White;
const BOND_LINE_COLOR: Color = Color::Gray;
const LABEL_COLOR: Color = Color::White;
const AXIS_X_COLOR: Color = Color::Red;
const AXIS_Y_COLOR: Color = Color::Green;
const AXIS_Z_COLOR: Color = Color::Blue;
const ORIENTATION_GIZMO_RADIUS_COLS: f32 = 5.0;
const ORIENTATION_GIZMO_MARGIN_COLS: f32 = 2.0;
const ORIENTATION_GIZMO_MARGIN_ROWS: f32 = 2.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderTheme {
    Dense,
    Classic,
    Orbital,
    Neon,
}

impl RenderTheme {
    fn next(self) -> Self {
        match self {
            Self::Dense => Self::Classic,
            Self::Classic => Self::Orbital,
            Self::Orbital => Self::Neon,
            Self::Neon => Self::Dense,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Classic => "classic",
            Self::Orbital => "orbital",
            Self::Neon => "neon",
        }
    }

    fn shade_ramp(self) -> &'static [char] {
        match self {
            Self::Dense => SHADE_RAMP_DENSE,
            Self::Classic => SHADE_RAMP_CLASSIC,
            Self::Orbital => SHADE_RAMP_ORBITAL,
            Self::Neon => SHADE_RAMP_NEON,
        }
    }

    fn cell_line_glyph(self) -> char {
        match self {
            Self::Dense => '▒',
            Self::Classic => ':',
            Self::Orbital => '·',
            Self::Neon => '=',
        }
    }

    fn bond_line_glyph(self) -> char {
        match self {
            Self::Dense => '▓',
            Self::Classic => '-',
            Self::Orbital => '•',
            Self::Neon => '~',
        }
    }

    fn cell_line_color(self) -> Color {
        match self {
            Self::Orbital => Color::Cyan,
            Self::Neon => Color::LightCyan,
            _ => CELL_LINE_COLOR,
        }
    }

    fn bond_line_color(self) -> Color {
        match self {
            Self::Orbital => Color::LightMagenta,
            Self::Neon => Color::LightYellow,
            _ => BOND_LINE_COLOR,
        }
    }
}

pub struct App {
    structure: Structure,
    scene: SceneGeometry,
    base_scale: f32, // rotation-invariant fit scale, computed once from bounding sphere
    /// Current view rotation as a 3×3 matrix (Rz·Ry·Rx convention).
    /// Each key press / mouse drag left-multiplies by an elementary camera-space
    /// rotation, keeping 'k' == "tilt top toward viewer" at any orientation.
    rot_mat: [[f32; 3]; 3],
    zoom: f32,
    fov_deg: f32,
    lock_fov_zoom: bool,
    pan: [f32; 2],
    sphere_scale: f32, // multiplier on atom display radii (1.0 = full covalent radius)
    show_bonds: bool,
    show_cell: bool,
    cell_on_top: bool,
    show_boundary_images: bool,
    show_bonded_images: bool,
    bond_max_distance: f32, // absolute maximum bond length (angstrom)
    show_labels: bool,
    show_orientation_gizmo: bool,
    render_theme: RenderTheme,
    selected_atom: usize,
    should_quit: bool,
    /// Column/row of the last mouse-drag position, for delta calculation.
    drag_last: Option<(u16, u16)>,
    /// Cached element → count map, derived from structure.atoms (immutable).
    cached_element_counts: BTreeMap<String, usize>,
    /// Cached Hill-order empirical formula derived from cached_element_counts.
    cached_formula: String,
}

impl App {
    fn new(structure: Structure) -> Self {
        let show_boundary_images = true;
        let show_bonded_images = true;
        let bond_max_distance = DEFAULT_BOND_MAX_DISTANCE;
        let scene = build_scene(
            &structure,
            show_boundary_images,
            show_bonded_images,
            bond_max_distance,
        );
        let base_scale = bounding_sphere_scale(&scene);
        // Compute once — structure.atoms is immutable after construction.
        let cached_element_counts = element_counts(&structure.atoms);
        let cached_formula = empirical_formula(&cached_element_counts);
        Self {
            structure,
            scene,
            base_scale,
            // Start in an oblique view so periodic images do not collapse onto each other.
            rot_mat: mat_from_euler(0.35, 0.45, 0.0),
            zoom: 1.0,
            fov_deg: DEFAULT_FOV_DEG,
            lock_fov_zoom: true,
            pan: [0.0, 0.0],
            sphere_scale: 0.45,
            show_bonds: true,
            show_cell: true,
            cell_on_top: false,
            show_boundary_images,
            show_bonded_images,
            bond_max_distance,
            show_labels: true,
            show_orientation_gizmo: true,
            render_theme: RenderTheme::Orbital,
            selected_atom: 0,
            should_quit: false,
            drag_last: None,
            cached_element_counts,
            cached_formula,
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match key.kind {
            KeyEventKind::Press => self.handle_key_press(key.code),
            KeyEventKind::Repeat => self.handle_key_repeat(key.code),
            KeyEventKind::Release => {}
        }
    }

    /// Actions that fire once per key-down (toggles, discrete steps).
    fn handle_key_press(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('+') | KeyCode::Char('=') => self.set_zoom(self.zoom * 1.1),
            KeyCode::Char('-') => self.set_zoom(self.zoom / 1.1),
            KeyCode::Char(',') => self.set_fov(self.fov_deg - 2.0),
            KeyCode::Char('.') => self.set_fov(self.fov_deg + 2.0),
            KeyCode::Char('z') => self.toggle_fov_zoom_lock(),
            KeyCode::Char('[') => self.sphere_scale = (self.sphere_scale / 1.2).max(0.05),
            KeyCode::Char(']') => self.sphere_scale = (self.sphere_scale * 1.2).min(4.0),
            KeyCode::Char('b') => self.show_bonds = !self.show_bonds,
            KeyCode::Char('c') => self.show_cell = !self.show_cell,
            KeyCode::Char('x') => self.toggle_cell_on_top(),
            KeyCode::Char('r') => self.toggle_boundary_images(),
            KeyCode::Char('t') => self.toggle_bonded_images(),
            KeyCode::Char('n') => self.adjust_bond_max_distance(-BOND_MAX_DISTANCE_STEP),
            KeyCode::Char('m') => self.adjust_bond_max_distance(BOND_MAX_DISTANCE_STEP),
            KeyCode::Char('N') => self.adjust_bond_max_distance(-4.0 * BOND_MAX_DISTANCE_STEP),
            KeyCode::Char('M') => self.adjust_bond_max_distance(4.0 * BOND_MAX_DISTANCE_STEP),
            KeyCode::Char('i') => self.snap_view_isometric(),
            KeyCode::Char('v') => self.toggle_orientation_gizmo(),
            KeyCode::Char('g') => self.toggle_render_theme(),
            KeyCode::Char('L') => self.show_labels = !self.show_labels,
            KeyCode::Char('A') => self.snap_view_to_lattice_axis(0),
            KeyCode::Char('B') => self.snap_view_to_lattice_axis(1),
            KeyCode::Char('C') => self.snap_view_to_lattice_axis(2),
            KeyCode::Tab => {
                if !self.structure.atoms.is_empty() {
                    self.selected_atom = (self.selected_atom + 1) % self.structure.atoms.len();
                }
            }
            _ => self.apply_continuous_key(code),
        }
    }

    fn handle_key_repeat(&mut self, code: KeyCode) {
        self.apply_continuous_key(code);
    }

    fn apply_continuous_key(&mut self, code: KeyCode) {
        // Rotations left-multiply the current matrix so they act in CAMERA
        // space: 'k' always tilts the top toward the viewer, 'h' always
        // rotates around the current up axis, regardless of prior orientation.
        match code {
            KeyCode::Char('h') => {
                self.rot_mat = mat_mul_3x3(mat_rot_y(-ROTATION_STEP), self.rot_mat)
            }
            KeyCode::Char('l') => {
                self.rot_mat = mat_mul_3x3(mat_rot_y(ROTATION_STEP), self.rot_mat)
            }
            KeyCode::Char('j') => {
                self.rot_mat = mat_mul_3x3(mat_rot_x(ROTATION_STEP), self.rot_mat)
            }
            KeyCode::Char('k') => {
                self.rot_mat = mat_mul_3x3(mat_rot_x(-ROTATION_STEP), self.rot_mat)
            }
            KeyCode::Char('u') => {
                self.rot_mat = mat_mul_3x3(mat_rot_z(-ROLL_STEP), self.rot_mat)
            }
            KeyCode::Char('o') => {
                self.rot_mat = mat_mul_3x3(mat_rot_z(ROLL_STEP), self.rot_mat)
            }
            KeyCode::Char('w') => self.pan[1] += PAN_STEP,
            KeyCode::Char('s') => self.pan[1] -= PAN_STEP,
            KeyCode::Char('a') => self.pan[0] -= PAN_STEP,
            KeyCode::Char('d') => self.pan[0] += PAN_STEP,
            _ => {}
        }
    }

    fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.max(MIN_ZOOM);
    }

    fn set_fov(&mut self, fov_deg: f32) {
        let new_fov = fov_deg.clamp(MIN_FOV_DEG, MAX_FOV_DEG);
        if (new_fov - self.fov_deg).abs() <= f32::EPSILON {
            return;
        }
        if self.lock_fov_zoom {
            if let Some(target_extent) = self.projected_reference_extent(self.zoom, self.fov_deg) {
                if let Some(adjusted_zoom) =
                    self.solve_zoom_for_extent(target_extent, new_fov, self.zoom)
                {
                    self.zoom = adjusted_zoom;
                }
            }
        }
        self.fov_deg = new_fov;
    }

    fn toggle_fov_zoom_lock(&mut self) {
        self.lock_fov_zoom = !self.lock_fov_zoom;
    }

    fn toggle_boundary_images(&mut self) {
        self.show_boundary_images = !self.show_boundary_images;
        self.rebuild_scene();
    }

    fn toggle_bonded_images(&mut self) {
        self.show_bonded_images = !self.show_bonded_images;
        self.rebuild_scene();
    }

    fn toggle_cell_on_top(&mut self) {
        self.cell_on_top = !self.cell_on_top;
    }

    fn toggle_render_theme(&mut self) {
        self.render_theme = self.render_theme.next();
    }

    fn toggle_orientation_gizmo(&mut self) {
        self.show_orientation_gizmo = !self.show_orientation_gizmo;
    }

    fn adjust_bond_max_distance(&mut self, delta: f32) {
        let previous = self.bond_max_distance;
        self.bond_max_distance =
            (self.bond_max_distance + delta).clamp(MIN_BOND_MAX_DISTANCE, MAX_BOND_MAX_DISTANCE);
        if (self.bond_max_distance - previous).abs() <= f32::EPSILON {
            return;
        }
        self.rebuild_scene();
    }

    fn rebuild_scene(&mut self) {
        self.scene = build_scene(
            &self.structure,
            self.show_boundary_images,
            self.show_bonded_images,
            self.bond_max_distance,
        );
        self.base_scale = bounding_sphere_scale(&self.scene);
    }

    fn projected_reference_extent(&self, zoom: f32, fov_deg: f32) -> Option<f32> {
        let scale = self.base_scale * zoom.max(MIN_ZOOM);
        // Precompute rotation matrix and focal once — this function is called
        // repeatedly inside the binary search in solve_zoom_for_extent.
        let camera = CameraParams::from_mat(self.scene.center, scale, self.rot_mat, fov_deg, [
            0.0, 0.0,
        ]);
        let mut max_extent = 0.0f32;
        let mut saw_point = false;

        if !self.scene.cell_edges.is_empty() {
            for (start, end) in &self.scene.cell_edges {
                for p in [*start, *end] {
                    let Some(pr) = camera.project(p) else {
                        continue;
                    };
                    if !pr.x.is_finite() || !pr.y.is_finite() {
                        continue;
                    }
                    max_extent = max_extent.max(pr.x.abs().max(pr.y.abs()));
                    saw_point = true;
                }
            }
        } else {
            for atom in &self.scene.atoms {
                let Some(pr) = camera.project(atom.position) else {
                    continue;
                };
                if !pr.x.is_finite() || !pr.y.is_finite() {
                    continue;
                }
                max_extent = max_extent.max(pr.x.abs().max(pr.y.abs()));
                saw_point = true;
            }
        }

        if saw_point { Some(max_extent) } else { None }
    }

    fn solve_zoom_for_extent(
        &self,
        target_extent: f32,
        fov_deg: f32,
        seed_zoom: f32,
    ) -> Option<f32> {
        if !target_extent.is_finite() || target_extent <= 0.0 {
            return None;
        }

        let mut low = MIN_ZOOM;
        let low_extent = self.projected_reference_extent(low, fov_deg)?;
        if low_extent >= target_extent {
            return Some(low);
        }

        let mut high = seed_zoom.max(1.0).max(MIN_ZOOM);
        let mut high_extent = self
            .projected_reference_extent(high, fov_deg)
            .unwrap_or(low_extent);
        let max_zoom = 512.0f32;
        while high_extent < target_extent && high < max_zoom {
            high = (high * 2.0).min(max_zoom);
            high_extent = self
                .projected_reference_extent(high, fov_deg)
                .unwrap_or(high_extent);
        }

        if high_extent < target_extent {
            return Some(high);
        }

        for _ in 0..24 {
            let mid = 0.5 * (low + high);
            let mid_extent = self
                .projected_reference_extent(mid, fov_deg)
                .unwrap_or(high_extent);
            if mid_extent < target_extent {
                low = mid;
            } else {
                high = mid;
            }
        }
        Some(0.5 * (low + high))
    }

    fn snap_view_to_lattice_axis(&mut self, axis_index: usize) {
        let axis = if let Some(cell) = self.structure.cell {
            cell.lattice[axis_index]
        } else {
            match axis_index {
                0 => [1.0, 0.0, 0.0],
                1 => [0.0, 1.0, 0.0],
                _ => [0.0, 0.0, 1.0],
            }
        };

        let axis_len2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if axis_len2 <= f32::EPSILON {
            return;
        }

        let pitch = axis[1].atan2(axis[2]);
        let yz_len = (axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let yaw = (-axis[0]).atan2(yz_len);
        self.rot_mat = mat_from_euler(pitch, yaw, 0.0);
        self.pan = [0.0, 0.0];
    }

    fn snap_view_isometric(&mut self) {
        self.rot_mat = mat_from_euler(ISO_PITCH, ISO_YAW, 0.0);
        self.pan = [0.0, 0.0];
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.drag_last = Some((mouse.column, mouse.row));
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((last_col, last_row)) = self.drag_last {
                    let dcol = -(mouse.column as f32 - last_col as f32) / CHAR_ASPECT;
                    let drow = -(mouse.row as f32 - last_row as f32);
                    // Camera-space: apply pitch around current X axis, then yaw
                    // around current Y axis, both via left-multiply.
                    let dpitch = drow * MOUSE_SENSITIVITY;
                    let dyaw = dcol * MOUSE_SENSITIVITY;
                    self.rot_mat = mat_mul_3x3(mat_rot_x(dpitch), self.rot_mat);
                    self.rot_mat = mat_mul_3x3(mat_rot_y(dyaw), self.rot_mat);
                }
                self.drag_last = Some((mouse.column, mouse.row));
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.drag_last = None;
            }
            MouseEventKind::ScrollUp => self.handle_scroll_wheel(true, mouse.modifiers),
            MouseEventKind::ScrollDown => self.handle_scroll_wheel(false, mouse.modifiers),
            _ => {}
        }
    }

    fn handle_scroll_wheel(&mut self, scroll_up: bool, modifiers: KeyModifiers) {
        if modifiers.contains(KeyModifiers::CONTROL) {
            let delta = if scroll_up {
                -MOUSE_WHEEL_FOV_STEP_DEG
            } else {
                MOUSE_WHEEL_FOV_STEP_DEG
            };
            self.set_fov(self.fov_deg + delta);
            return;
        }

        let factor = if scroll_up {
            MOUSE_WHEEL_ZOOM_FACTOR
        } else {
            1.0 / MOUSE_WHEEL_ZOOM_FACTOR
        };
        self.set_zoom(self.zoom * factor);
    }
}

pub fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, structure: Structure) -> Result<()> {
    let mut app = App::new(structure);
    let tick_rate = Duration::from_millis(16);

    while !app.should_quit {
        // Wait up to one tick for the first event, then drain any further
        // events that arrived during the same tick window without blocking.
        let deadline = std::time::Instant::now() + tick_rate;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if !event::poll(remaining)? {
                break; // tick elapsed with no more events
            }
            match event::read()? {
                Event::Key(key) => app.handle_key(key),
                Event::Mouse(mouse) => app.handle_mouse(mouse),
                _ => {}
            }
            if deadline <= std::time::Instant::now() {
                break; // don't overrun into the next frame
            }
        }

        terminal.draw(|frame| {
            draw(frame, &app);
        })?;
    }

    Ok(())
}

fn draw(frame: &mut ratatui::Frame, app: &App) {
    let area = frame.area();
    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(24), Constraint::Length(44)])
        .split(area);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(root[1]);

    let viewport = render_viewport_text(
        app,
        root[0].width.saturating_sub(2) as usize,
        root[0].height.saturating_sub(2) as usize,
    );
    let viewport_paragraph =
        Paragraph::new(viewport).block(Block::default().borders(Borders::ALL).title("3D View"));
    frame.render_widget(viewport_paragraph, root[0]);

    let side = Paragraph::new(structure_panel_lines(app))
        .block(Block::default().borders(Borders::ALL).title("Structure"))
        .wrap(Wrap { trim: true });
    frame.render_widget(side, right[0]);

    let view_panel = Paragraph::new(controls_keys_panel_lines(app))
        .block(Block::default().borders(Borders::ALL).title("Controls"))
        .wrap(Wrap { trim: true });
    frame.render_widget(view_panel, right[1]);
}

fn structure_panel_lines(app: &App) -> Vec<Line<'static>> {
    let selected = app.structure.atoms.get(app.selected_atom);
    let total_atoms = app.structure.atoms.len();
    let element_counts = &app.cached_element_counts;
    let formula = app.cached_formula.clone();

    let mut lines = vec![
        kv_line(
            "Title",
            app.structure.title.clone(),
            Color::Cyan,
            Color::White,
        ),
        kv_line("Formula", formula, Color::Cyan, Color::LightYellow),
        elements_line(element_counts),
        kv_line(
            "Atom sites",
            total_atoms.to_string(),
            Color::Cyan,
            Color::LightGreen,
        ),
        kv_line(
            "Space group",
            app.structure
                .space_group
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            Color::Cyan,
            Color::LightBlue,
        ),
        Line::default(),
    ];

    if let Some(cell) = app.structure.cell {
        lines.push(Line::from(vec![
            Span::styled("a ", Style::default().fg(Color::Red)),
            Span::styled(format!("{:.3}", cell.a), Style::default().fg(Color::White)),
            Span::styled("  b ", Style::default().fg(Color::Green)),
            Span::styled(format!("{:.3}", cell.b), Style::default().fg(Color::White)),
            Span::styled("  c ", Style::default().fg(Color::Blue)),
            Span::styled(format!("{:.3}", cell.c), Style::default().fg(Color::White)),
            Span::styled(" A", Style::default().fg(Color::Gray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("alpha ", Style::default().fg(Color::LightRed)),
            Span::styled(
                format!("{:.1}", cell.alpha_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled("  beta ", Style::default().fg(Color::LightGreen)),
            Span::styled(
                format!("{:.1}", cell.beta_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled("  gamma ", Style::default().fg(Color::LightBlue)),
            Span::styled(
                format!("{:.1}", cell.gamma_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled(" deg", Style::default().fg(Color::Gray)),
        ]));
        lines.push(kv_line(
            "Volume",
            format!("{:.3} A^3", cell_volume(cell)),
            Color::Cyan,
            Color::LightCyan,
        ));
    } else {
        lines.push(kv_line(
            "Lattice",
            "unavailable".to_string(),
            Color::Cyan,
            Color::DarkGray,
        ));
    }

    lines.push(Line::default());

    if let Some(atom) = selected {
        let atom_col = atom_color(&atom.element, false);
        lines.push(kv_line(
            "Index",
            format!("{}/{}", app.selected_atom + 1, app.structure.atoms.len()),
            Color::Cyan,
            Color::LightYellow,
        ));
        lines.push(Line::from(vec![
            Span::styled("Atom: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                atom.label.clone(),
                Style::default().fg(atom_col).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Element: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                atom.element.clone(),
                Style::default().fg(atom_col).add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("x ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("{:.3}", atom.position[0]),
                Style::default().fg(Color::White),
            ),
            Span::styled("  y ", Style::default().fg(Color::Green)),
            Span::styled(
                format!("{:.3}", atom.position[1]),
                Style::default().fg(Color::White),
            ),
            Span::styled("  z ", Style::default().fg(Color::Blue)),
            Span::styled(
                format!("{:.3}", atom.position[2]),
                Style::default().fg(Color::White),
            ),
            Span::styled(" A", Style::default().fg(Color::Gray)),
        ]));
        if let Some(frac) = atom.fractional {
            lines.push(Line::from(vec![
                Span::styled("Frac ", Style::default().fg(Color::LightMagenta)),
                Span::styled("x ", Style::default().fg(Color::Red)),
                Span::styled(format!("{:.3}", frac[0]), Style::default().fg(Color::White)),
                Span::styled("  y ", Style::default().fg(Color::Green)),
                Span::styled(format!("{:.3}", frac[1]), Style::default().fg(Color::White)),
                Span::styled("  z ", Style::default().fg(Color::Blue)),
                Span::styled(format!("{:.3}", frac[2]), Style::default().fg(Color::White)),
            ]));
        }
    } else {
        lines.push(Line::from(vec![
            Span::styled("Selected: ", Style::default().fg(Color::Cyan)),
            Span::styled("none", Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines
}

fn controls_keys_panel_lines(app: &App) -> Vec<Line<'static>> {
    let boundary_color = match (app.show_boundary_images, app.scene.boundary_image_count > 0) {
        (true, true) => Color::LightGreen,
        (true, false) => Color::LightYellow,
        (false, _) => Color::LightRed,
    };
    let bonded_color = match (app.show_bonded_images, app.scene.bonded_image_count > 0) {
        (true, true) => Color::LightGreen,
        (true, false) => Color::LightYellow,
        (false, _) => Color::LightRed,
    };

    vec![
        control_short_value_line(
            "g",
            "Theme",
            app.render_theme.label().to_string(),
            theme_color(app.render_theme),
        ),
        control_short_bool_line("v", "Gizmo", app.show_orientation_gizmo),
        control_short_bool_line("b", "Bonds", app.show_bonds),
        control_short_bool_line("c", "Cell", app.show_cell),
        control_short_bool_line("x", "Cell top", app.cell_on_top),
        control_short_bool_line("Shift+L", "Labels", app.show_labels),
        control_short_value_line(
            "r",
            "Boundary imgs",
            if app.show_boundary_images {
                "on".to_string()
            } else {
                "off".to_string()
            },
            boundary_color,
        ),
        control_short_value_line(
            "t",
            "Bonded imgs",
            if app.show_bonded_images {
                "on".to_string()
            } else {
                "off".to_string()
            },
            bonded_color,
        ),
        control_short_bool_line("z", "FOV lock", app.lock_fov_zoom),
        control_short_value_line(
            "+/-, wheel",
            "Zoom",
            format!("{:.2}", app.zoom),
            Color::LightYellow,
        ),
        control_short_value_line(
            ",/., Ctrl+wheel",
            "FOV",
            format!("{:.1} deg", app.fov_deg),
            Color::LightYellow,
        ),
        control_short_value_line(
            "w/a/s/d",
            "Pan",
            format!("x {:.2} y {:.2}", app.pan[0], app.pan[1]),
            Color::White,
        ),
        control_short_value_line(
            "h/j/k/l,u/o",
            "Rot",
            {
                // Show the camera's look direction (row 2 of the rotation matrix).
                let d = app.rot_mat[2];
                format!("d {:.2} {:.2} {:.2}", d[0], d[1], d[2])
            },
            Color::White,
        ),
        control_short_value_line(
            "n/m/N/M",
            "Bond max",
            format!("{:.2} A", app.bond_max_distance),
            Color::LightYellow,
        ),
        control_short_value_line(
            "[/]",
            "Sphere",
            format!("{:.2}", app.sphere_scale),
            Color::LightYellow,
        ),
        control_short_value_line(
            "Tab",
            "Atom",
            format!("{}/{}", app.selected_atom + 1, app.structure.atoms.len()),
            Color::LightGreen,
        ),
        control_short_value_line("q", "Quit", "exit".to_string(), Color::LightRed),
    ]
}

fn kv_line(label: &str, value: String, label_color: Color, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label}: "), Style::default().fg(label_color)),
        Span::styled(value, Style::default().fg(value_color)),
    ])
}

fn elements_line(counts: &BTreeMap<String, usize>) -> Line<'static> {
    let mut spans = vec![Span::styled(
        format!("Elements ({}): ", counts.len()),
        Style::default().fg(Color::Cyan),
    )];
    if counts.is_empty() {
        spans.push(Span::styled("-", Style::default().fg(Color::DarkGray)));
        return Line::from(spans);
    }
    for (idx, element) in counts.keys().enumerate() {
        if idx > 0 {
            spans.push(Span::styled(", ", Style::default().fg(Color::Gray)));
        }
        spans.push(Span::styled(
            element.clone(),
            Style::default()
                .fg(atom_color(element, false))
                .add_modifier(Modifier::BOLD),
        ));
    }
    Line::from(spans)
}

fn control_short_prefix(key: &str, label: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!("[{key}]"),
            Style::default()
                .fg(Color::LightYellow)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", Style::default().fg(Color::Gray)),
        Span::styled(format!("{label} "), Style::default().fg(Color::Cyan)),
    ]
}

fn control_short_value_line(
    key: &str,
    label: &str,
    value: String,
    value_color: Color,
) -> Line<'static> {
    let mut spans = control_short_prefix(key, label);
    spans.push(Span::styled(value, Style::default().fg(value_color)));
    Line::from(spans)
}

fn control_short_bool_line(key: &str, label: &str, value: bool) -> Line<'static> {
    let mut spans = control_short_prefix(key, label);
    spans.push(bool_span(value));
    Line::from(spans)
}

fn bool_span(value: bool) -> Span<'static> {
    if value {
        Span::styled(
            "on",
            Style::default()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            "off",
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        )
    }
}

fn theme_color(theme: RenderTheme) -> Color {
    match theme {
        RenderTheme::Dense => Color::White,
        RenderTheme::Classic => Color::LightBlue,
        RenderTheme::Orbital => Color::Cyan,
        RenderTheme::Neon => Color::LightMagenta,
    }
}

fn element_counts(atoms: &[crate::model::Atom]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for atom in atoms {
        *counts.entry(atom.element.clone()).or_insert(0) += 1;
    }
    counts
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

fn empirical_formula(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "-".to_string();
    }

    let divisor = counts
        .values()
        .copied()
        .reduce(gcd_usize)
        .unwrap_or(1)
        .max(1);
    let mut order: Vec<String> = Vec::with_capacity(counts.len());
    if counts.contains_key("C") {
        order.push("C".to_string());
        if counts.contains_key("H") {
            order.push("H".to_string());
        }
        order.extend(
            counts
                .keys()
                .filter(|k| k.as_str() != "C" && k.as_str() != "H")
                .cloned(),
        );
    } else {
        order.extend(counts.keys().cloned());
    }

    let mut out = String::new();
    for element in order {
        let n = counts.get(&element).copied().unwrap_or(0) / divisor;
        if n == 0 {
            continue;
        }
        out.push_str(&element);
        if n > 1 {
            out.push_str(&n.to_string());
        }
    }
    if out.is_empty() { "-".to_string() } else { out }
}

fn cell_volume(cell: Cell) -> f32 {
    let a = cell.lattice[0];
    let b = cell.lattice[1];
    let c = cell.lattice[2];
    let cross_bc = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    (a[0] * cross_bc[0] + a[1] * cross_bc[1] + a[2] * cross_bc[2]).abs()
}

#[cfg(test)]
fn render_viewport(app: &App, width: usize, height: usize) -> String {
    let viewport = render_viewport_buffer(app, width, height);
    to_text_grid(&viewport.chars, width, height)
}

fn render_viewport_text(app: &App, width: usize, height: usize) -> Text<'static> {
    let viewport = render_viewport_buffer(app, width, height);
    to_colored_text(&viewport.chars, &viewport.colors, width, height)
}

struct ViewportBuffer {
    chars: Vec<char>,
    colors: Vec<Color>,
}

#[derive(Clone, Copy, Debug)]
struct ViewportTransform {
    center_col: f32,
    center_row: f32,
    // Physical scale in "column-width pixels" for one projected unit.
    units_to_cols: f32,
}

impl ViewportTransform {
    fn for_size(width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let center_col = (width.saturating_sub(1)) as f32 * 0.5;
        let center_row = (height.saturating_sub(1)) as f32 * 0.5;
        let half_w = center_col;
        let half_h_phys = center_row * CHAR_ASPECT;
        let units_to_cols = half_w.min(half_h_phys);
        Some(Self {
            center_col,
            center_row,
            units_to_cols,
        })
    }

    fn screen_position(self, x: f32, y: f32) -> (f32, f32) {
        (
            self.center_col + x * self.units_to_cols,
            self.center_row - y * self.units_to_cols / CHAR_ASPECT,
        )
    }
}

/// Precomputed camera state for a single frame.
///
/// Building this once per frame avoids recomputing trig values (sin/cos, tan)
/// for every projected point. The rotation matrix combines Rx(pitch), Ry(yaw),
/// and Rz(roll) into a single 3×3 matrix so each point transform is 9 muls +
/// 6 adds instead of 3 × sin_cos + 12 muls.
#[derive(Clone, Copy)]
struct CameraParams {
    center: [f32; 3],
    scale: f32,
    /// Combined rotation matrix: Rz(roll) · Ry(yaw) · Rx(pitch)
    rot: [[f32; 3]; 3],
    /// Focal length: 1 / tan(fov / 2)
    focal: f32,
    pan: [f32; 2],
}

impl CameraParams {
    /// Euler-angle constructor kept for use by the test-only `project_world` wrapper.
    #[cfg(test)]
    fn new(
        center: [f32; 3],
        scale: f32,
        rotation: [f32; 2],
        roll: f32,
        fov_deg: f32,
        pan: [f32; 2],
    ) -> Self {
        let (spitch, cpitch) = rotation[0].sin_cos();
        let (syaw, cyaw) = rotation[1].sin_cos();
        let (sroll, croll) = roll.sin_cos();
        // Rz(roll) · Ry(yaw) · Rx(pitch) — derived by multiplying the three
        // elementary rotation matrices in application order.
        let rot = [
            [
                cyaw * croll,
                croll * syaw * spitch - sroll * cpitch,
                croll * syaw * cpitch + sroll * spitch,
            ],
            [
                cyaw * sroll,
                sroll * syaw * spitch + croll * cpitch,
                sroll * syaw * cpitch - croll * spitch,
            ],
            [-syaw, cyaw * spitch, cyaw * cpitch],
        ];
        Self {
            center,
            scale,
            rot,
            focal: 1.0 / (0.5 * fov_deg.to_radians()).tan(),
            pan,
        }
    }

    /// Construct from an already-computed rotation matrix (the normal hot path).
    fn from_mat(
        center: [f32; 3],
        scale: f32,
        rot: [[f32; 3]; 3],
        fov_deg: f32,
        pan: [f32; 2],
    ) -> Self {
        Self {
            center,
            scale,
            rot,
            focal: 1.0 / (0.5 * fov_deg.to_radians()).tan(),
            pan,
        }
    }

    fn project(&self, position: [f32; 3]) -> Option<ProjectedPoint> {
        let lx = position[0] - self.center[0];
        let ly = position[1] - self.center[1];
        let lz = position[2] - self.center[2];
        let m = &self.rot;
        let rx = m[0][0] * lx + m[0][1] * ly + m[0][2] * lz;
        let ry = m[1][0] * lx + m[1][1] * ly + m[1][2] * lz;
        let rz = m[2][0] * lx + m[2][1] * ly + m[2][2] * lz;

        let sx = rx * self.scale;
        let sy = ry * self.scale;
        let sz = rz * self.scale;

        let depth = CAMERA_DISTANCE + sz;
        if depth <= 1e-3 {
            return None;
        }
        let screen_scale = self.focal / depth;
        if !screen_scale.is_finite() || screen_scale > MAX_SCREEN_SCALE {
            return None;
        }
        Some(ProjectedPoint {
            x: sx * screen_scale + self.pan[0],
            y: sy * screen_scale + self.pan[1],
            z: depth,
            screen_scale,
        })
    }
}

fn render_viewport_buffer(app: &App, width: usize, height: usize) -> ViewportBuffer {
    if width == 0 || height == 0 {
        return ViewportBuffer {
            chars: Vec::new(),
            colors: Vec::new(),
        };
    }
    if app.scene.atoms.is_empty() {
        let message = "No atoms loaded";
        let mut chars = vec![' '; width * height];
        let mut colors = vec![Color::Reset; width * height];
        for (i, ch) in message.chars().enumerate().take(width) {
            chars[i] = ch;
            colors[i] = LABEL_COLOR;
        }
        return ViewportBuffer { chars, colors };
    }

    let mut chars = vec![' '; width * height];
    let mut colors = vec![Color::Reset; width * height];
    let mut z_buffer = vec![f32::INFINITY; width * height];
    let scale = app.base_scale * app.zoom;
    let Some(viewport) = ViewportTransform::for_size(width, height) else {
        return ViewportBuffer { chars, colors };
    };
    // Build camera params once — avoids recomputing trig values per point.
    let camera = CameraParams::from_mat(app.scene.center, scale, app.rot_mat, app.fov_deg, app.pan);

    if app.show_cell && !app.cell_on_top {
        for (start, end) in &app.scene.cell_edges {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                *start,
                *end,
                app.render_theme.cell_line_glyph(),
                app.render_theme.cell_line_color(),
                &camera,
                CELL_LINE_DEPTH_BIAS,
                false,
            );
        }
    }

    if app.show_bonds {
        for bond in &app.scene.bonds {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                bond.start,
                bond.end,
                app.render_theme.bond_line_glyph(),
                app.render_theme.bond_line_color(),
                &camera,
                BOND_LINE_DEPTH_BIAS,
                false,
            );
        }
    }

    // Rasterize atoms as sphere disks, back-to-front so the z-buffer correctly
    // resolves overlaps even at equal depth (painter's order as tie-breaker).
    let mut atoms_projected: Vec<(&RenderAtom, ProjectedPoint)> = app
        .scene
        .atoms
        .iter()
        .filter_map(|atom| Some((atom, camera.project(atom.position)?)))
        .collect();

    // Sort back-to-front (largest z = furthest away drawn first).
    atoms_projected.sort_by(|a, b| b.1.z.partial_cmp(&a.1.z).unwrap_or(Ordering::Equal));

    for (atom, p) in &atoms_projected {
        if !p.x.is_finite() || !p.y.is_finite() || !p.screen_scale.is_finite() {
            continue;
        }
        let element = app
            .structure
            .atoms
            .get(atom.base_index)
            .map(|a| a.element.as_str())
            .unwrap_or("X");
        let r_world = display_radius(element) * app.sphere_scale;
        // Projected radius in column units (x screen axis).
        let r_depth = r_world * scale;
        let r_screen = (r_depth * p.screen_scale * viewport.units_to_cols).max(0.5);
        if !r_screen.is_finite() {
            continue;
        }

        // Sphere center in screen pixel coords.
        let (cx, cy) = viewport.screen_position(p.x, p.y);
        if !cx.is_finite() || !cy.is_finite() {
            continue;
        }

        // Bounding box: y extent is compressed by CHAR_ASPECT.
        let max_row = (height.saturating_sub(1)) as f32;
        let max_col = (width.saturating_sub(1)) as f32;
        let row0 = (cy - r_screen / CHAR_ASPECT).floor().clamp(0.0, max_row) as usize;
        let row1 = (cy + r_screen / CHAR_ASPECT).ceil().clamp(0.0, max_row) as usize;
        let col0 = (cx - r_screen).floor().clamp(0.0, max_col) as usize;
        let col1 = (cx + r_screen).ceil().clamp(0.0, max_col) as usize;
        if row0 > row1 || col0 > col1 {
            continue;
        }

        let selected = atom.base_index == app.selected_atom && !atom.is_image;
        let color = atom_color(element, selected);

        for row in row0..=row1 {
            for col in col0..=col1 {
                let dx = col as f32 - cx;
                // Scale dy to physical pixels for a round circle test.
                let dy_phys = (row as f32 - cy) * CHAR_ASPECT;
                if dx * dx + dy_phys * dy_phys > r_screen * r_screen {
                    continue;
                }

                // Surface normal in camera space (z points toward viewer).
                let nx = dx / r_screen;
                let ny = -dy_phys / r_screen; // screen-y is flipped vs world-y
                let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();

                // Sphere depth: centre_z minus the depth of this surface point.
                let z_pixel = p.z - nz * r_depth;

                let idx = row * width + col;
                if z_pixel > z_buffer[idx] {
                    continue;
                }
                z_buffer[idx] = z_pixel;
                chars[idx] = sphere_glyph(nx, ny, nz, selected, app.render_theme);
                colors[idx] = color;
            }
        }
    }

    if app.show_labels {
        if let Some(selected) = app.structure.atoms.get(app.selected_atom) {
            let label = format!("Selected: {} ({})", selected.label, selected.element);
            for (i, ch) in label.chars().enumerate().take(width) {
                chars[i] = ch;
                colors[i] = LABEL_COLOR;
            }
        }
    }

    if app.show_cell && app.cell_on_top {
        for (start, end) in &app.scene.cell_edges {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                *start,
                *end,
                app.render_theme.cell_line_glyph(),
                app.render_theme.cell_line_color(),
                &camera,
                CELL_LINE_DEPTH_BIAS,
                true,
            );
        }
    }

    if app.show_orientation_gizmo {
        draw_orientation_gizmo(&mut chars, &mut colors, width, height, app.rot_mat);
    }

    ViewportBuffer { chars, colors }
}

/// Lambert-shaded glyph for a sphere surface point.
/// (nx, ny, nz) is the outward surface normal in camera space (nz > 0 faces viewer).
fn sphere_glyph(nx: f32, ny: f32, nz: f32, selected: bool, theme: RenderTheme) -> char {
    let diffuse = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]).max(0.0);
    let intensity = 0.25 + 0.75 * diffuse;
    let ramp = theme.shade_ramp();
    let n = ramp.len() - 1;
    let idx = if selected {
        // Selected atoms are forced into the bright half of the ramp.
        ((0.5 + 0.5 * diffuse) * n as f32) as usize
    } else {
        (intensity * n as f32) as usize
    };
    ramp[idx.min(n)]
}

/// Visual display radius (Å) used for sphere rasterization.
fn display_radius(element: &str) -> f32 {
    // Scaled covalent radii — large enough to look solid at typical zoom levels.
    match element {
        "H" => 0.53,
        "B" => 1.00,
        "C" => 0.91,
        "N" => 0.87,
        "O" => 0.84,
        "F" => 0.78,
        "Na" => 1.54,
        "Mg" => 1.36,
        "Al" => 1.18,
        "Si" => 1.11,
        "P" => 1.07,
        "S" => 1.05,
        "Cl" => 1.02,
        "K" => 1.96,
        "Ca" => 1.74,
        "Ti" => 1.54,
        "V" => 1.47,
        "Cr" => 1.35,
        "Mn" => 1.35,
        "Fe" => 1.32,
        "Co" => 1.26,
        "Ni" => 1.24,
        "Cu" => 1.32,
        "Zn" => 1.22,
        _ => 1.00,
    }
}

fn atom_color(element: &str, selected: bool) -> Color {
    let mut rgb = match element {
        "H" => (255, 255, 255),
        "C" => (235, 235, 235),
        "N" => (120, 180, 255),
        "O" => (255, 96, 96),
        "F" | "Cl" => (112, 255, 144),
        "P" => (255, 180, 96),
        "S" => (255, 232, 112),
        "Na" => (160, 128, 255),
        "Mg" => (144, 255, 144),
        "Al" => (224, 224, 224),
        "Si" => (255, 212, 160),
        "K" => (208, 144, 255),
        "Ca" => (144, 255, 168),
        "Ti" => (216, 216, 240),
        "V" => (208, 208, 240),
        "Cr" => (184, 208, 255),
        "Mn" => (200, 168, 255),
        "Fe" => (255, 176, 96),
        "Co" => (168, 168, 255),
        "Ni" => (144, 255, 144),
        "Cu" => (255, 184, 136),
        "Zn" => (152, 216, 255),
        _ => (224, 224, 224),
    };
    if selected {
        rgb = blend_rgb(rgb, (255, 255, 255), 0.35);
    }
    Color::Rgb(rgb.0, rgb.1, rgb.2)
}

fn blend_rgb(base: (u8, u8, u8), tint: (u8, u8, u8), tint_ratio: f32) -> (u8, u8, u8) {
    let t = tint_ratio.clamp(0.0, 1.0);
    let b = 1.0 - t;
    (
        (base.0 as f32 * b + tint.0 as f32 * t).round() as u8,
        (base.1 as f32 * b + tint.1 as f32 * t).round() as u8,
        (base.2 as f32 * b + tint.2 as f32 * t).round() as u8,
    )
}

fn draw_orientation_gizmo(
    chars: &mut [char],
    colors: &mut [Color],
    width: usize,
    height: usize,
    rot_mat: [[f32; 3]; 3],
) {
    if width < 8 || height < 6 {
        return;
    }

    let radius = ORIENTATION_GIZMO_RADIUS_COLS;
    let origin_col = ORIENTATION_GIZMO_MARGIN_COLS + radius;
    let origin_row = ORIENTATION_GIZMO_MARGIN_ROWS + radius / CHAR_ASPECT;
    if origin_col >= (width.saturating_sub(1)) as f32
        || origin_row >= (height.saturating_sub(1)) as f32
    {
        return;
    }

    // Each world axis in camera space is a column of the rotation matrix:
    //   R · e_x = column 0 = [rot_mat[0][0], rot_mat[1][0], rot_mat[2][0]]
    let col0 = [rot_mat[0][0], rot_mat[1][0], rot_mat[2][0]];
    let col1 = [rot_mat[0][1], rot_mat[1][1], rot_mat[2][1]];
    let col2 = [rot_mat[0][2], rot_mat[1][2], rot_mat[2][2]];
    let mut axes = [
        ('x', AXIS_X_COLOR, col0),
        ('y', AXIS_Y_COLOR, col1),
        ('z', AXIS_Z_COLOR, col2),
    ];
    // Draw farther axes first so nearer axes remain readable.
    axes.sort_by(|a, b| b.2[2].partial_cmp(&a.2[2]).unwrap_or(Ordering::Equal));

    for (label, color, axis) in axes {
        draw_orientation_axis(
            chars, colors, width, height, origin_col, origin_row, radius, label, color, axis,
        );
    }
    put_colored_char(
        chars,
        colors,
        width,
        height,
        origin_col.round() as isize,
        origin_row.round() as isize,
        '+',
        Color::Gray,
    );
}

fn draw_orientation_axis(
    chars: &mut [char],
    colors: &mut [Color],
    width: usize,
    height: usize,
    origin_col: f32,
    origin_row: f32,
    radius: f32,
    label: char,
    color: Color,
    axis_cam: [f32; 3],
) {
    let dx = axis_cam[0] * radius;
    let dy = -axis_cam[1] * radius / CHAR_ASPECT;
    let physical_len = dx.abs().max(dy.abs() * CHAR_ASPECT);
    let steps = (physical_len.ceil() as usize).clamp(1, 48);

    for step in 1..=steps {
        let t = step as f32 / steps as f32;
        let col = origin_col + dx * t;
        let row = origin_row + dy * t;
        put_colored_char(
            chars,
            colors,
            width,
            height,
            col.round() as isize,
            row.round() as isize,
            '.',
            color,
        );
    }

    let (label_dx, label_dy) = if physical_len > 1e-3 {
        (dx * 1.2, dy * 1.2)
    } else {
        match label {
            'x' => (radius * 0.7, 0.0),
            'y' => (0.0, -radius * 0.7 / CHAR_ASPECT),
            _ => (-radius * 0.5, -radius * 0.5 / CHAR_ASPECT),
        }
    };

    put_colored_char(
        chars,
        colors,
        width,
        height,
        (origin_col + label_dx).round() as isize,
        (origin_row + label_dy).round() as isize,
        label,
        color,
    );
}

fn put_colored_char(
    chars: &mut [char],
    colors: &mut [Color],
    width: usize,
    height: usize,
    col: isize,
    row: isize,
    glyph: char,
    color: Color,
) {
    if col < 0 || row < 0 || col >= width as isize || row >= height as isize {
        return;
    }
    let idx = row as usize * width + col as usize;
    chars[idx] = glyph;
    colors[idx] = color;
}

#[cfg(test)]
fn to_text_grid(chars: &[char], width: usize, height: usize) -> String {
    let mut out = String::with_capacity((width + 1) * height);
    for y in 0..height {
        for x in 0..width {
            out.push(chars[y * width + x]);
        }
        if y + 1 < height {
            out.push('\n');
        }
    }
    out
}

fn to_colored_text(chars: &[char], colors: &[Color], width: usize, height: usize) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(height);
    for row in 0..height {
        if width == 0 {
            lines.push(Line::default());
            continue;
        }
        let row_start = row * width;
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut run_color = colors[row_start];
        let mut run = String::new();
        for col in 0..width {
            let idx = row_start + col;
            let color = colors[idx];
            if color != run_color {
                spans.push(Span::styled(
                    std::mem::take(&mut run),
                    Style::default().fg(run_color),
                ));
                run_color = color;
            }
            run.push(chars[idx]);
        }
        spans.push(Span::styled(run, Style::default().fg(run_color)));
        lines.push(Line::from(spans));
    }
    Text::from(lines)
}

fn rasterize_segment(
    chars: &mut [char],
    colors: &mut [Color],
    z_buffer: &mut [f32],
    width: usize,
    height: usize,
    viewport: ViewportTransform,
    start: [f32; 3],
    end: [f32; 3],
    glyph: char,
    color: Color,
    camera: &CameraParams,
    depth_bias: f32,
    overlay: bool,
) {
    let Some(p0) = camera.project(start) else {
        return;
    };
    let Some(p1) = camera.project(end) else {
        return;
    };

    let (x0, y0) = viewport.screen_position(p0.x, p0.y);
    let (x1, y1) = viewport.screen_position(p1.x, p1.y);
    // Approximate segment length in physical "column-width pixels".
    let segment_len = (x1 - x0).abs().max((y1 - y0).abs() * CHAR_ASPECT);
    let samples = (segment_len.ceil() as usize).clamp(2, 600);

    let max_row = height.saturating_sub(1) as f32;
    let max_col = width.saturating_sub(1) as f32;

    for step in 0..=samples {
        let t = step as f32 / samples as f32;
        // Interpolate directly in screen space — avoids a second
        // screen_position() call per sample that would be redundant.
        let sx_f = (x0 + (x1 - x0) * t).round();
        let sy_f = (y0 + (y1 - y0) * t).round();
        if !sx_f.is_finite() || !sy_f.is_finite() {
            continue;
        }
        if sx_f < 0.0 || sy_f < 0.0 || sx_f > max_col || sy_f > max_row {
            continue;
        }
        // Keep helper geometry behind atom points at identical depth.
        let z = p0.z + (p1.z - p0.z) * t + depth_bias;
        let idx = sy_f as usize * width + sx_f as usize;
        if overlay {
            chars[idx] = glyph;
            colors[idx] = color;
            z_buffer[idx] = f32::NEG_INFINITY;
            continue;
        }
        if z < z_buffer[idx] {
            z_buffer[idx] = z;
            chars[idx] = glyph;
            colors[idx] = color;
        }
    }
}

/// Convenience wrapper used by tests.  Hot paths build a `CameraParams` once
/// and call `camera.project()` directly to avoid per-point trig.
#[cfg(test)]
fn project_world(
    position: [f32; 3],
    center: [f32; 3],
    scale: f32,
    rotation: [f32; 2],
    roll: f32,
    fov_deg: f32,
    pan: [f32; 2],
) -> Option<ProjectedPoint> {
    CameraParams::new(center, scale, rotation, roll, fov_deg, pan).project(position)
}

/// Compute a rotation-invariant fit scale from the bounding-sphere radius.
///
/// Using the maximum distance from the scene centre to any geometry point means
/// the scale is the same for every orientation, so the apparent centre of
/// rotation stays fixed on screen as the user rotates.
fn bounding_sphere_scale(scene: &SceneGeometry) -> f32 {
    // Compare squared distances to avoid a sqrt per point; one sqrt at the end.
    let mut max_r_sq = (1e-3f32).powi(2);

    for atom in &scene.atoms {
        max_r_sq = max_r_sq.max(dist3_sq(atom.position, scene.center));
    }
    for (s, e) in &scene.cell_edges {
        max_r_sq = max_r_sq.max(dist3_sq(*s, scene.center));
        max_r_sq = max_r_sq.max(dist3_sq(*e, scene.center));
    }
    for bond in &scene.bonds {
        max_r_sq = max_r_sq.max(dist3_sq(bond.start, scene.center));
        max_r_sq = max_r_sq.max(dist3_sq(bond.end, scene.center));
    }

    0.92 / max_r_sq.sqrt()
}

fn dist3_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}


#[derive(Debug, Clone, Copy)]
struct ProjectedPoint {
    x: f32,
    y: f32,
    z: f32,
    screen_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct RenderAtom {
    base_index: usize,
    position: [f32; 3],
    is_image: bool,
}

#[derive(Debug, Clone, Copy)]
struct BondSegment {
    start: [f32; 3],
    end: [f32; 3],
}

#[derive(Debug, Clone)]
struct SceneGeometry {
    atoms: Vec<RenderAtom>,
    bonds: Vec<BondSegment>,
    cell_edges: Vec<([f32; 3], [f32; 3])>,
    center: [f32; 3],
    boundary_image_count: usize,
    bonded_image_count: usize,
}

fn build_scene(
    structure: &Structure,
    include_boundary_images: bool,
    include_bonded_images: bool,
    bond_max_distance: f32,
) -> SceneGeometry {
    let mut render_atoms: Vec<RenderAtom> = structure
        .atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| RenderAtom {
            base_index: index,
            position: atom.position,
            is_image: false,
        })
        .collect();

    let mut bonds = Vec::new();
    let mut cell_edges = Vec::new();
    let mut boundary_image_keys: HashSet<(usize, i32, i32, i32)> = HashSet::new();
    let mut bonded_image_keys: HashSet<(usize, i32, i32, i32)> = HashSet::new();

    if let Some(cell) = structure.cell {
        cell_edges = cell.edge_segments();
        if structure.atoms.iter().all(|a| a.fractional.is_some()) {
            if include_boundary_images {
                add_boundary_repeat_images(structure, &mut boundary_image_keys);
            }
            add_bonds_and_periodic_images(
                structure,
                cell,
                &boundary_image_keys,
                &mut bonds,
                &mut bonded_image_keys,
                include_bonded_images,
                bond_max_distance,
            );
            let mut image_keys = boundary_image_keys.clone();
            image_keys.extend(bonded_image_keys.iter().copied());
            for (atom_index, sx, sy, sz) in image_keys {
                let shift = cell.shift_to_cart([sx, sy, sz]);
                let atom = &structure.atoms[atom_index];
                render_atoms.push(RenderAtom {
                    base_index: atom_index,
                    position: [
                        atom.position[0] + shift[0],
                        atom.position[1] + shift[1],
                        atom.position[2] + shift[2],
                    ],
                    is_image: true,
                });
            }
        }
    }

    let center = if let Some(cell) = structure.cell {
        cell.frac_to_cart([0.5, 0.5, 0.5])
    } else {
        structure.center()
    };
    let boundary_image_count = boundary_image_keys.len();
    let bonded_image_count = bonded_image_keys.len();

    SceneGeometry {
        atoms: render_atoms,
        bonds,
        cell_edges,
        center,
        boundary_image_count,
        bonded_image_count,
    }
}

fn add_boundary_repeat_images(
    structure: &Structure,
    image_keys: &mut HashSet<(usize, i32, i32, i32)>,
) {
    for (atom_index, atom) in structure.atoms.iter().enumerate() {
        let Some(frac) = atom.fractional else {
            continue;
        };
        let axis_shifts = boundary_axis_shifts(frac);
        for sx in &axis_shifts[0] {
            for sy in &axis_shifts[1] {
                for sz in &axis_shifts[2] {
                    if *sx == 0 && *sy == 0 && *sz == 0 {
                        continue;
                    }
                    image_keys.insert((atom_index, *sx, *sy, *sz));
                }
            }
        }
    }
}

fn add_bonds_and_periodic_images(
    structure: &Structure,
    cell: Cell,
    boundary_image_keys: &HashSet<(usize, i32, i32, i32)>,
    bonds: &mut Vec<BondSegment>,
    image_keys: &mut HashSet<(usize, i32, i32, i32)>,
    include_bonded_images: bool,
    bond_max_distance: f32,
) {
    if bond_max_distance <= 0.0 {
        return;
    }

    let mut source_sites: Vec<(usize, [i32; 3])> = structure
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_index, _)| (atom_index, [0, 0, 0]))
        .collect();
    for &(atom_index, sx, sy, sz) in boundary_image_keys {
        source_sites.push((atom_index, [sx, sy, sz]));
    }

    let mut seen_bonds: HashSet<((usize, i32, i32, i32), (usize, i32, i32, i32))> = HashSet::new();

    for (i, source_shift) in source_sites {
        let atom_i = &structure.atoms[i];
        let Some(frac_i) = atom_i.fractional else {
            continue;
        };
        let source_key = (i, source_shift[0], source_shift[1], source_shift[2]);
        let source_cart_shift = cell.shift_to_cart(source_shift);
        let start = [
            atom_i.position[0] + source_cart_shift[0],
            atom_i.position[1] + source_cart_shift[1],
            atom_i.position[2] + source_cart_shift[2],
        ];

        for (j, atom_j) in structure.atoms.iter().enumerate() {
            let Some(frac_j) = atom_j.fractional else {
                continue;
            };
            for sx in -1..=1 {
                for sy in -1..=1 {
                    for sz in -1..=1 {
                        let target_shift = [
                            source_shift[0] + sx,
                            source_shift[1] + sy,
                            source_shift[2] + sz,
                        ];
                        let target_key = (j, target_shift[0], target_shift[1], target_shift[2]);
                        if source_key == target_key {
                            continue;
                        }
                        let bond_key = if source_key <= target_key {
                            (source_key, target_key)
                        } else {
                            (target_key, source_key)
                        };
                        if !seen_bonds.insert(bond_key) {
                            continue;
                        }

                        let delta_frac = [
                            (frac_j[0] + sx as f32) - frac_i[0],
                            (frac_j[1] + sy as f32) - frac_i[1],
                            (frac_j[2] + sz as f32) - frac_i[2],
                        ];
                        let distance = vec_norm(cell.frac_to_cart(delta_frac));
                        if distance > bond_max_distance {
                            continue;
                        }

                        let target_cart_shift = cell.shift_to_cart(target_shift);
                        let end = [
                            atom_j.position[0] + target_cart_shift[0],
                            atom_j.position[1] + target_cart_shift[1],
                            atom_j.position[2] + target_cart_shift[2],
                        ];
                        bonds.push(BondSegment { start, end });

                        if include_bonded_images && target_shift != [0, 0, 0] {
                            image_keys.insert(target_key);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
fn best_image_shift_and_distance(
    frac_i: [f32; 3],
    frac_j: [f32; 3],
    cell: Cell,
) -> ([i32; 3], f32) {
    let mut best_shift = [0, 0, 0];
    let mut best_distance = f32::INFINITY;

    for sx in -1..=1 {
        for sy in -1..=1 {
            for sz in -1..=1 {
                let delta_frac = [
                    (frac_j[0] + sx as f32) - frac_i[0],
                    (frac_j[1] + sy as f32) - frac_i[1],
                    (frac_j[2] + sz as f32) - frac_i[2],
                ];
                let delta_cart = cell.frac_to_cart(delta_frac);
                let distance = vec_norm(delta_cart);
                if distance < best_distance {
                    best_distance = distance;
                    best_shift = [sx, sy, sz];
                }
            }
        }
    }

    (best_shift, best_distance)
}

fn boundary_axis_shifts(frac: [f32; 3]) -> [Vec<i32>; 3] {
    let mut out = [vec![0], vec![0], vec![0]];
    for axis in 0..3 {
        let wrapped = wrap_fractional(frac[axis]);
        if wrapped <= BOUNDARY_EPSILON {
            out[axis].push(1);
        }
        if wrapped >= 1.0 - BOUNDARY_EPSILON {
            out[axis].push(-1);
        }
    }
    out
}

fn wrap_fractional(value: f32) -> f32 {
    let mut out = value - value.floor();
    if out < 0.0 {
        out += 1.0;
    }
    out
}

fn vec_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ── 3×3 rotation matrix helpers ──────────────────────────────────────────────

fn mat_rot_x(angle: f32) -> [[f32; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn mat_rot_y(angle: f32) -> [[f32; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn mat_rot_z(angle: f32) -> [[f32; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn mat_mul_3x3(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Build R = Rz(roll) · Ry(yaw) · Rx(pitch) — the same convention used by
/// the original Euler-angle code, kept for snap-view initialization.
fn mat_from_euler(pitch: f32, yaw: f32, roll: f32) -> [[f32; 3]; 3] {
    mat_mul_3x3(mat_rot_z(roll), mat_mul_3x3(mat_rot_y(yaw), mat_rot_x(pitch)))
}

#[cfg(test)]
mod tests {
    use super::{
        App, CHAR_ASPECT, DEFAULT_BOND_MAX_DISTANCE, ISO_PITCH, ISO_YAW, MIN_FOV_DEG,
        MOUSE_SENSITIVITY, RenderTheme, ViewportTransform, best_image_shift_and_distance,
        boundary_axis_shifts, build_scene, mat_from_euler, project_world, render_viewport,
    };
    use crate::cif::parse_cif_str;
    use crate::model::{Atom, Cell, Structure};
    use crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    #[test]
    fn detects_boundary_repeat_shifts() {
        let shifts = boundary_axis_shifts([0.01, 0.5, 0.99]);
        assert!(shifts[0].contains(&1));
        assert!(shifts[1].contains(&0));
        assert!(shifts[2].contains(&-1));
    }

    #[test]
    fn picks_nearest_periodic_image_for_bond_distance() {
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let (shift, distance) =
            best_image_shift_and_distance([0.95, 0.0, 0.0], [0.05, 0.0, 0.0], cell);
        assert_eq!(shift, [1, 0, 0]);
        assert!((distance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn perspective_projection_scales_with_depth() {
        let center = [0.0, 0.0, 0.0];
        let near = project_world(
            [0.5, 0.0, -0.8],
            center,
            1.0,
            [0.0, 0.0],
            0.0,
            45.0,
            [0.0, 0.0],
        )
        .expect("near point should project");
        let far = project_world(
            [0.5, 0.0, 0.8],
            center,
            1.0,
            [0.0, 0.0],
            0.0,
            45.0,
            [0.0, 0.0],
        )
        .expect("far point should project");

        assert!(near.x.abs() > far.x.abs());
        assert!(near.z < far.z);
    }

    #[test]
    fn fov_size_lock_keeps_extent_for_fov_changes_only() {
        let cif = include_str!("../Fe.cif");
        let structure = parse_cif_str(cif, "fe_fixture").expect("Fe.cif should parse");
        let mut app = App::new(structure);
        assert!(app.lock_fov_zoom);

        let old_fov = app.fov_deg;
        let old_zoom = app.zoom;
        let target_extent = app
            .projected_reference_extent(old_zoom, old_fov)
            .expect("extent should be computable");

        app.handle_key_press(KeyCode::Char('.'));

        assert!(app.fov_deg > old_fov);
        assert!((app.zoom - old_zoom).abs() > 1e-6);
        let new_extent = app
            .projected_reference_extent(app.zoom, app.fov_deg)
            .expect("extent should be computable");
        assert!((new_extent - target_extent).abs() < 2e-2);

        let fov_before_zoom_key = app.fov_deg;
        let zoom_before_zoom_key = app.zoom;
        app.handle_key_press(KeyCode::Char('+'));
        assert!(app.zoom > zoom_before_zoom_key);
        assert!((app.fov_deg - fov_before_zoom_key).abs() < 1e-6);
    }

    #[test]
    fn mouse_drag_rotates_with_configured_sensitivity() {
        let structure = Structure {
            title: "mouse drag sensitivity test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        // Start from identity so the expected result is unambiguous.
        app.rot_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 20,
            row: 10,
            modifiers: KeyModifiers::NONE,
        });
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 24,
            row: 7,
            modifiers: KeyModifiers::NONE,
        });

        let dpitch = (10.0 - 7.0_f32) * MOUSE_SENSITIVITY;
        let dyaw = -((24.0 - 20.0_f32) / CHAR_ASPECT) * MOUSE_SENSITIVITY;
        // Camera-space: pitch applied first, then yaw → mat_from_euler(dpitch, dyaw, 0)
        // starting from identity.
        let expected = mat_from_euler(dpitch, dyaw, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (app.rot_mat[i][j] - expected[i][j]).abs() < 1e-5,
                    "rot_mat[{i}][{j}]: got {} expected {}",
                    app.rot_mat[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn mouse_wheel_zooms_and_ctrl_wheel_changes_fov() {
        let structure = Structure {
            title: "mouse wheel test".to_string(),
            atoms: vec![Atom {
                label: "A".to_string(),
                element: "C".to_string(),
                position: [1.0, 0.0, 0.0],
                fractional: None,
            }],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        app.lock_fov_zoom = false;

        let zoom_before_scroll = app.zoom;
        let fov_before_scroll = app.fov_deg;
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        assert!(app.zoom > zoom_before_scroll);
        assert!((app.fov_deg - fov_before_scroll).abs() < 1e-6);

        let zoom_before_ctrl_scroll = app.zoom;
        let fov_before_ctrl_scroll = app.fov_deg;
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::CONTROL,
        });
        assert!((app.zoom - zoom_before_ctrl_scroll).abs() < 1e-6);
        assert!(app.fov_deg < fov_before_ctrl_scroll);
    }

    #[test]
    fn extreme_perspective_settings_do_not_panic() {
        let cif = include_str!("../Fe.cif");
        let structure = parse_cif_str(cif, "fe_fixture").expect("Fe.cif should parse");
        let mut app = App::new(structure);
        app.fov_deg = MIN_FOV_DEG;
        app.zoom = 25.0;
        app.rot_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // identity
        app.pan = [0.0, 0.0];

        let _ = render_viewport(&app, 80, 30);
    }

    #[test]
    fn builds_periodic_images_and_cross_boundary_bonds() {
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "test".to_string(),
            atoms: vec![
                Atom {
                    label: "A".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.95, 0.5, 0.5]),
                    fractional: Some([0.95, 0.5, 0.5]),
                },
                Atom {
                    label: "B".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.05, 0.5, 0.5]),
                    fractional: Some([0.05, 0.5, 0.5]),
                },
            ],
            cell: Some(cell),
            space_group: None,
        };

        let scene = build_scene(&structure, true, true, DEFAULT_BOND_MAX_DISTANCE);
        assert!(scene.boundary_image_count + scene.bonded_image_count > 0);
        assert!(!scene.bonds.is_empty());
    }

    #[test]
    fn bond_max_distance_controls_cross_boundary_bond_detection() {
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "bond max test".to_string(),
            atoms: vec![
                Atom {
                    label: "A".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.95, 0.5, 0.5]),
                    fractional: Some([0.95, 0.5, 0.5]),
                },
                Atom {
                    label: "B".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.16, 0.5, 0.5]),
                    fractional: Some([0.16, 0.5, 0.5]),
                },
            ],
            cell: Some(cell),
            space_group: None,
        };

        let strict = build_scene(&structure, false, true, 2.0);
        assert_eq!(strict.bonds.len(), 0);
        assert_eq!(strict.bonded_image_count, 0);

        let relaxed = build_scene(&structure, false, true, 2.2);
        assert!(!relaxed.bonds.is_empty());
        assert!(relaxed.bonded_image_count > 0);
    }

    #[test]
    fn boundary_repeat_atoms_seed_additional_bonded_images() {
        let cell = Cell::from_parameters(2.0, 2.0, 2.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "boundary-source bond test".to_string(),
            atoms: vec![
                Atom {
                    label: "A".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.99, 0.5, 0.5]),
                    fractional: Some([0.99, 0.5, 0.5]),
                },
                Atom {
                    label: "B".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.95, 0.5, 0.5]),
                    fractional: Some([0.95, 0.5, 0.5]),
                },
            ],
            cell: Some(cell),
            space_group: None,
        };

        let without_boundary = build_scene(&structure, false, true, 2.2);
        let has_second_shell_without = without_boundary
            .atoms
            .iter()
            .any(|a| a.is_image && a.base_index == 1 && (a.position[0] + 2.1).abs() < 1e-4);
        assert!(!has_second_shell_without);

        let with_boundary = build_scene(&structure, true, true, 2.2);
        let has_second_shell_with = with_boundary
            .atoms
            .iter()
            .any(|a| a.is_image && a.base_index == 1 && (a.position[0] + 2.1).abs() < 1e-4);
        assert!(has_second_shell_with);
    }

    #[test]
    fn toggles_boundary_and_bonded_image_sets_independently() {
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "toggle test".to_string(),
            atoms: vec![
                Atom {
                    label: "A".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.99, 0.5, 0.5]),
                    fractional: Some([0.99, 0.5, 0.5]),
                },
                Atom {
                    label: "B".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.50, 0.5, 0.5]),
                    fractional: Some([0.50, 0.5, 0.5]),
                },
                Atom {
                    label: "C".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.95, 0.2, 0.2]),
                    fractional: Some([0.95, 0.2, 0.2]),
                },
                Atom {
                    label: "D".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.05, 0.2, 0.2]),
                    fractional: Some([0.05, 0.2, 0.2]),
                },
            ],
            cell: Some(cell),
            space_group: None,
        };
        let mut app = App::new(structure);

        assert!(app.scene.boundary_image_count > 0);
        assert!(app.scene.bonded_image_count > 0);

        app.handle_key_press(KeyCode::Char('r'));
        assert!(!app.show_boundary_images);
        assert_eq!(app.scene.boundary_image_count, 0);
        assert!(app.scene.bonded_image_count > 0);

        app.handle_key_press(KeyCode::Char('t'));
        assert!(!app.show_bonded_images);
        assert_eq!(app.scene.boundary_image_count, 0);
        assert_eq!(app.scene.bonded_image_count, 0);
        assert_eq!(
            app.scene.boundary_image_count + app.scene.bonded_image_count,
            0
        );

        app.handle_key_press(KeyCode::Char('r'));
        assert!(app.show_boundary_images);
        assert!(app.scene.boundary_image_count > 0);
        assert_eq!(app.scene.bonded_image_count, 0);
    }

    #[test]
    fn toggles_cell_overlay_mode() {
        let structure = Structure {
            title: "overlay test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        assert!(!app.cell_on_top);

        app.handle_key_press(KeyCode::Char('x'));
        assert!(app.cell_on_top);

        app.handle_key_press(KeyCode::Char('x'));
        assert!(!app.cell_on_top);
    }

    #[test]
    fn toggles_render_theme_mode() {
        let structure = Structure {
            title: "theme test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        assert_eq!(app.render_theme, RenderTheme::Orbital);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Neon);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Dense);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Classic);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Orbital);
    }

    #[test]
    fn toggles_orientation_gizmo_mode() {
        let structure = Structure {
            title: "gizmo toggle test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        assert!(app.show_orientation_gizmo);

        app.handle_key_press(KeyCode::Char('v'));
        assert!(!app.show_orientation_gizmo);

        app.handle_key_press(KeyCode::Char('v'));
        assert!(app.show_orientation_gizmo);
    }

    #[test]
    fn labels_are_enabled_by_default_and_toggle() {
        let structure = Structure {
            title: "labels toggle test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        assert!(app.show_labels);

        app.handle_key_press(KeyCode::Char('L'));
        assert!(!app.show_labels);

        app.handle_key_press(KeyCode::Char('L'));
        assert!(app.show_labels);
    }

    #[test]
    fn orientation_gizmo_renders_xyz_labels_and_can_be_hidden() {
        let structure = Structure {
            title: "gizmo render test".to_string(),
            atoms: vec![Atom {
                label: "A".to_string(),
                element: "C".to_string(),
                position: [0.0, 0.0, 0.0],
                fractional: None,
            }],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        app.show_bonds = false;
        app.show_cell = false;
        app.show_labels = false;

        let with_gizmo = render_viewport(&app, 60, 20);
        assert!(with_gizmo.contains('x'));
        assert!(with_gizmo.contains('y'));
        assert!(with_gizmo.contains('z'));

        app.handle_key_press(KeyCode::Char('v'));
        let without_gizmo = render_viewport(&app, 60, 20);
        assert!(!without_gizmo.contains('x'));
        assert!(!without_gizmo.contains('y'));
        assert!(!without_gizmo.contains('z'));
    }

    #[test]
    fn viewport_transform_uses_equal_physical_scale_for_x_and_y() {
        for (width, height) in [(120, 24), (40, 60), (70, 30)] {
            let viewport = ViewportTransform::for_size(width, height)
                .expect("non-zero viewport should produce a transform");
            let (cx, cy) = viewport.screen_position(0.0, 0.0);
            let (x, _) = viewport.screen_position(0.6, 0.0);
            let (_, y) = viewport.screen_position(0.0, 0.6);
            let dx = (x - cx).abs();
            let dy_phys = (y - cy).abs() * CHAR_ASPECT;
            assert!(
                (dx - dy_phys).abs() < 1e-4,
                "physical x/y scale mismatch for {}x{} viewport: dx={}, dy_phys={}",
                width,
                height,
                dx,
                dy_phys
            );
        }
    }

    #[test]
    fn axis_snap_keys_align_lattice_axes_with_view_direction() {
        let cell = Cell::from_parameters(6.1, 7.3, 8.7, 82.0, 97.0, 74.0).expect("valid cell");
        let structure = Structure {
            title: "axis test".to_string(),
            atoms: vec![Atom {
                label: "X".to_string(),
                element: "C".to_string(),
                position: cell.frac_to_cart([0.5, 0.5, 0.5]),
                fractional: Some([0.5, 0.5, 0.5]),
            }],
            cell: Some(cell),
            space_group: None,
        };
        let mut app = App::new(structure);

        for (key, axis) in [
            (KeyCode::Char('A'), cell.lattice[0]),
            (KeyCode::Char('B'), cell.lattice[1]),
            (KeyCode::Char('C'), cell.lattice[2]),
        ] {
            app.handle_key_press(key);
            // Apply rot_mat directly: p_cam = R · (tip - center)
            let m = app.rot_mat;
            let aligned = [
                m[0][0] * axis[0] + m[0][1] * axis[1] + m[0][2] * axis[2],
                m[1][0] * axis[0] + m[1][1] * axis[1] + m[1][2] * axis[2],
            ];
            assert!(
                aligned[0].abs() < 1e-4,
                "axis x-component in camera space should be ~0, got {}",
                aligned[0]
            );
            assert!(
                aligned[1].abs() < 1e-4,
                "axis y-component in camera space should be ~0, got {}",
                aligned[1]
            );
            assert!(app.pan[0].abs() < 1e-6);
            assert!(app.pan[1].abs() < 1e-6);
        }
    }

    #[test]
    fn isometric_key_sets_expected_angles_and_recenters() {
        let structure = Structure {
            title: "iso test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
        };
        let mut app = App::new(structure);
        app.rot_mat = mat_from_euler(0.0, 0.0, 0.2); // some non-iso rotation with roll
        app.pan = [0.3, -0.4];

        app.handle_key_press(KeyCode::Char('i'));

        let expected = mat_from_euler(ISO_PITCH, ISO_YAW, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (app.rot_mat[i][j] - expected[i][j]).abs() < 1e-5,
                    "rot_mat[{i}][{j}]: got {} expected {}",
                    app.rot_mat[i][j],
                    expected[i][j]
                );
            }
        }
        assert!(app.pan[0].abs() < 1e-6);
        assert!(app.pan[1].abs() < 1e-6);
    }

    #[test]
    fn toggling_cell_keeps_atom_pixels_and_glyphs_stable() {
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "test".to_string(),
            atoms: vec![
                Atom {
                    label: "A".to_string(),
                    element: "C".to_string(),
                    position: cell.frac_to_cart([0.10, 0.10, 0.10]),
                    fractional: Some([0.10, 0.10, 0.10]),
                },
                Atom {
                    label: "B".to_string(),
                    element: "O".to_string(),
                    position: cell.frac_to_cart([0.85, 0.15, 0.20]),
                    fractional: Some([0.85, 0.15, 0.20]),
                },
                Atom {
                    label: "C".to_string(),
                    element: "N".to_string(),
                    position: cell.frac_to_cart([0.20, 0.80, 0.70]),
                    fractional: Some([0.20, 0.80, 0.70]),
                },
            ],
            cell: Some(cell),
            space_group: None,
        };

        let mut with_cell = App::new(structure.clone());
        with_cell.show_bonds = false;
        with_cell.show_cell = true;
        let with_cell_buf = render_viewport(&with_cell, 60, 20);

        let mut without_cell = App::new(structure);
        without_cell.show_bonds = false;
        without_cell.show_cell = false;
        let without_cell_buf = render_viewport(&without_cell, 60, 20);

        let with_cell_chars = flatten_grid(&with_cell_buf);
        let without_cell_chars = flatten_grid(&without_cell_buf);
        assert_eq!(with_cell_chars.len(), without_cell_chars.len());

        for (idx, ch) in without_cell_chars.iter().enumerate() {
            // In the no-cell, no-bonds render, every non-space pixel comes from atoms.
            if *ch != ' ' {
                assert_eq!(
                    *ch, with_cell_chars[idx],
                    "atom glyph changed at pixel index {idx}: off='{}' on='{}'",
                    *ch, with_cell_chars[idx]
                );
            }
        }
    }

    #[test]
    fn toggling_cell_for_fe_fixture_only_replaces_pixels_with_cell_glyph() {
        let cif = include_str!("../Fe.cif");
        let structure = parse_cif_str(cif, "fe_fixture").expect("Fe.cif should parse");

        let mut with_cell = App::new(structure.clone());
        with_cell.show_bonds = false;
        with_cell.show_cell = true;
        let with_cell_buf = render_viewport(&with_cell, 60, 20);

        let mut without_cell = App::new(structure);
        without_cell.show_bonds = false;
        without_cell.show_cell = false;
        let without_cell_buf = render_viewport(&without_cell, 60, 20);

        let with_cell_chars = flatten_grid(&with_cell_buf);
        let without_cell_chars = flatten_grid(&without_cell_buf);
        assert_eq!(with_cell_chars.len(), without_cell_chars.len());
        let cell_glyph = with_cell.render_theme.cell_line_glyph();

        for (idx, ch) in without_cell_chars.iter().enumerate() {
            if *ch != ' ' {
                let with_ch = with_cell_chars[idx];
                if with_ch != *ch {
                    assert_eq!(
                        with_ch, cell_glyph,
                        "Fe.cif unexpected glyph replacement at pixel index {idx}: off='{}' on='{}'",
                        *ch, with_ch
                    );
                }
            }
        }
    }

    #[test]
    fn toggling_cell_with_bonds_for_fe_fixture_only_replaces_pixels_with_cell_glyph() {
        let cif = include_str!("../Fe.cif");
        let structure = parse_cif_str(cif, "fe_fixture").expect("Fe.cif should parse");

        let mut with_cell = App::new(structure.clone());
        with_cell.show_bonds = true;
        with_cell.show_cell = true;
        let with_cell_buf = render_viewport(&with_cell, 60, 20);

        let mut without_cell = App::new(structure);
        without_cell.show_bonds = true;
        without_cell.show_cell = false;
        let without_cell_buf = render_viewport(&without_cell, 60, 20);

        let with_cell_chars = flatten_grid(&with_cell_buf);
        let without_cell_chars = flatten_grid(&without_cell_buf);
        assert_eq!(with_cell_chars.len(), without_cell_chars.len());
        let cell_glyph = with_cell.render_theme.cell_line_glyph();

        for (idx, ch) in without_cell_chars.iter().enumerate() {
            if *ch != ' ' {
                let with_ch = with_cell_chars[idx];
                if with_ch != *ch {
                    assert_eq!(
                        with_ch, cell_glyph,
                        "Fe.cif (+bonds) unexpected glyph replacement at pixel index {idx}: off='{}' on='{}'",
                        *ch, with_ch
                    );
                }
            }
        }
    }

    fn flatten_grid(s: &str) -> Vec<char> {
        s.lines().flat_map(|line| line.chars()).collect()
    }

    // ── Optimization regression tests ─────────────────────────────────────

    #[test]
    fn camera_params_project_matches_project_world() {
        // Verify that CameraParams::project (via from_mat) produces identical
        // results to the Euler-angle constructor used by project_world.
        use super::CameraParams;

        let test_cases: &[([f32; 3], [f32; 3], f32, [f32; 2], f32, f32, [f32; 2])] = &[
            // (position, center, scale, rotation, roll, fov_deg, pan)
            ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 1.0, [0.0, 0.0], 0.0, 45.0, [0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.5, 0.0, 0.0], 2.0, [0.3, 0.7], 0.1, 60.0, [0.1, -0.2]),
            ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], 0.5, [-0.5, 1.2], 0.8, 30.0, [0.0, 0.0]),
        ];

        for &(position, center, scale, rotation, roll, fov_deg, pan) in test_cases {
            let via_wrapper = project_world(position, center, scale, rotation, roll, fov_deg, pan);
            let via_camera =
                CameraParams::new(center, scale, rotation, roll, fov_deg, pan).project(position);

            match (via_wrapper, via_camera) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(
                        (a.x - b.x).abs() < 1e-5,
                        "x mismatch: project_world={} camera={}",
                        a.x,
                        b.x
                    );
                    assert!(
                        (a.y - b.y).abs() < 1e-5,
                        "y mismatch: project_world={} camera={}",
                        a.y,
                        b.y
                    );
                    assert!(
                        (a.z - b.z).abs() < 1e-5,
                        "z mismatch: project_world={} camera={}",
                        a.z,
                        b.z
                    );
                    assert!(
                        (a.screen_scale - b.screen_scale).abs() < 1e-5,
                        "screen_scale mismatch: {} vs {}",
                        a.screen_scale,
                        b.screen_scale
                    );
                }
                (a, b) => panic!("projection agreement failure: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn dist3_sq_equals_dist_squared() {
        use super::dist3_sq;
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];
        let sq = dist3_sq(a, b);
        let manual = (4.0 - 1.0_f32).powi(2) + (6.0 - 2.0_f32).powi(2) + 0.0;
        assert!((sq - manual).abs() < 1e-6);
        assert!((sq.sqrt() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn bounding_sphere_scale_consistent_for_symmetric_scene() {
        // A cell-less structure with atoms at ±1 on each axis should give a
        // consistent scale regardless of order and without inflating from sqrt.
        use super::{RenderAtom, SceneGeometry};

        let atoms = vec![
            RenderAtom { base_index: 0, position: [1.0, 0.0, 0.0], is_image: false },
            RenderAtom { base_index: 1, position: [-1.0, 0.0, 0.0], is_image: false },
            RenderAtom { base_index: 2, position: [0.0, 1.0, 0.0], is_image: false },
            RenderAtom { base_index: 3, position: [0.0, 0.0, 1.0], is_image: false },
        ];
        let scene = SceneGeometry {
            atoms,
            bonds: vec![],
            cell_edges: vec![],
            center: [0.0, 0.0, 0.0],
            boundary_image_count: 0,
            bonded_image_count: 0,
        };
        use super::bounding_sphere_scale;
        let scale = bounding_sphere_scale(&scene);
        // Max distance = 1.0, so scale = 0.92 / 1.0 = 0.92
        assert!((scale - 0.92).abs() < 1e-5, "expected scale ≈ 0.92, got {scale}");
    }

    #[test]
    fn cached_formula_and_counts_match_direct_computation() {
        // Ensure the cached values in App exactly equal what the free functions produce.
        use super::{element_counts, empirical_formula};
        let cif = include_str!("../U3Te4.cif");
        let structure = parse_cif_str(cif, "u3te4").expect("should parse");
        let app = App::new(structure.clone());

        let expected_counts = element_counts(&structure.atoms);
        let expected_formula = empirical_formula(&expected_counts);

        assert_eq!(app.cached_element_counts, expected_counts);
        assert_eq!(app.cached_formula, expected_formula);
    }
}
