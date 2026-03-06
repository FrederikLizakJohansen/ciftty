use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io::Stdout;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Padding, Paragraph, Wrap};
use ratatui::Terminal;

use crate::model::{Atom, Cell, Structure};
use crate::spacegroup;
use crate::xrd::{compute_pattern, XrdPattern};

const WAVELENGTH_PRESETS: &[(&str, f32)] = &[
    ("Cu Kα", 1.5406),
    ("Mo Kα", 0.7093),
    ("Co Kα", 1.7902),
    ("Ag Kα", 0.5594),
];
const XRD_TWO_THETA_MAX: f32 = 180.0;
const XRD_PANEL_HEIGHT: u16 = 14;

const ROTATION_STEP: f32 = 0.06;
const ROLL_STEP: f32 = 0.06;
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
const EDITOR_ATOM_ROWS: usize = 5;
const MOUSE_SENSITIVITY: f32 = 0.03; // radians per terminal column/row dragged
const MOUSE_WHEEL_ZOOM_FACTOR: f32 = 1.1;
const MOUSE_WHEEL_FOV_STEP_DEG: f32 = 2.0;
// Terminal character cells are roughly twice as tall as wide in pixels.
const CHAR_ASPECT: f32 = 2.0;
const DEFAULT_BOND_MAX_DISTANCE: f32 = 2.2;
const MIN_BOND_MAX_DISTANCE: f32 = 0.0;
const MAX_BOND_MAX_DISTANCE: f32 = 12.0;
const BOND_MAX_DISTANCE_STEP: f32 = 0.10;
// Classic ASCII shading ramp from dark to bright.
const SHADE_RAMP_CLASSIC: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
// Dense shading ramp from dark to bright; this fills terminal rows better.
const SHADE_RAMP_DENSE: &[char] = &[' ', '░', '▒', '▓', '█'];
// Orbital theme: circular shades for a softer, stylized look.
const SHADE_RAMP_ORBITAL: &[char] = &[' ', '·', '∘', '○', '◍', '●'];
// Neon theme: high-contrast punctuated ramp.
const SHADE_RAMP_NEON: &[char] = &[' ', '.', ':', '*', 'o', 'O', '@'];
// Wild theme: high-energy spark ramp.
const SHADE_RAMP_WILD: &[char] = &[' ', '·', ':', '*', '✶', '✸', '✹'];
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
const EMPTY_VIEW_HINT: &str = "Press Shift+O to open a CIF";
const DEFAULT_SPIN_SPEED: f32 = 1.0;
const MIN_SPIN_SPEED: f32 = 0.10;
const MAX_SPIN_SPEED: f32 = 5.00;
const SPIN_SPEED_STEP: f32 = 0.10;
const SPIN_BASE_RATE_RAD_PER_SEC: f32 = 1.2;
const ENSEMBLE_FP_BINS: usize = 720;
const ENSEMBLE_GRID_CAPACITY: usize = 9;
const ENSEMBLE_GRID_COLUMNS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderTheme {
    Dense,
    Classic,
    Orbital,
    Neon,
    Wild,
}

impl RenderTheme {
    fn next(self) -> Self {
        match self {
            Self::Dense => Self::Classic,
            Self::Classic => Self::Orbital,
            Self::Orbital => Self::Neon,
            Self::Neon => Self::Wild,
            Self::Wild => Self::Dense,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Classic => "classic",
            Self::Orbital => "orbital",
            Self::Neon => "neon",
            Self::Wild => "wild",
        }
    }

    fn shade_ramp(self) -> &'static [char] {
        match self {
            Self::Dense => SHADE_RAMP_DENSE,
            Self::Classic => SHADE_RAMP_CLASSIC,
            Self::Orbital => SHADE_RAMP_ORBITAL,
            Self::Neon => SHADE_RAMP_NEON,
            Self::Wild => SHADE_RAMP_WILD,
        }
    }

    fn cell_line_glyph(self) -> char {
        match self {
            Self::Dense => '▒',
            Self::Classic => ':',
            Self::Orbital => '·',
            Self::Neon => '=',
            Self::Wild => '✦',
        }
    }

    fn bond_line_glyph(self) -> char {
        match self {
            Self::Dense => '▓',
            Self::Classic => '-',
            Self::Orbital => '•',
            Self::Neon => '~',
            Self::Wild => '*',
        }
    }

    fn cell_line_color(self) -> Color {
        match self {
            Self::Orbital => Color::Cyan,
            Self::Neon => Color::LightCyan,
            Self::Wild => Color::LightMagenta,
            _ => CELL_LINE_COLOR,
        }
    }

    fn bond_line_color(self) -> Color {
        match self {
            Self::Orbital => Color::LightMagenta,
            Self::Neon => Color::LightYellow,
            Self::Wild => Color::LightRed,
            _ => BOND_LINE_COLOR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnsembleInput {
    pub source_dir: PathBuf,
    pub samples: Vec<(PathBuf, Structure)>,
    pub target: Option<(PathBuf, Structure)>,
    pub skipped_files: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EnsembleSortMode {
    TargetDistance,
    Novelty,
    FileName,
}

impl EnsembleSortMode {
    fn label(self) -> &'static str {
        match self {
            Self::TargetDistance => "target",
            Self::Novelty => "novelty",
            Self::FileName => "name",
        }
    }

    fn next(self, has_target: bool) -> Self {
        if has_target {
            match self {
                Self::TargetDistance => Self::Novelty,
                Self::Novelty => Self::FileName,
                Self::FileName => Self::TargetDistance,
            }
        } else {
            match self {
                Self::Novelty => Self::FileName,
                Self::FileName | Self::TargetDistance => Self::Novelty,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct EnsembleViewState {
    rot_mat: [[f32; 3]; 3],
    zoom: f32,
    fov_deg: f32,
    lock_fov_zoom: bool,
    pan: [f32; 2],
    sphere_scale: f32,
    show_bonds: bool,
    show_cell: bool,
    cell_on_top: bool,
    show_boundary_images: bool,
    show_bonded_images: bool,
    bond_max_distance: f32,
    show_labels: bool,
    show_orientation_gizmo: bool,
    render_theme: RenderTheme,
}

#[derive(Debug, Clone)]
struct EnsembleEntry {
    path: PathBuf,
    structure: Structure,
    fingerprint: Vec<f32>,
    distance_to_target: Option<f32>,
    novelty: f32,
    view_state: EnsembleViewState,
}

#[derive(Debug, Clone)]
struct EnsembleState {
    source_dir: PathBuf,
    target_path: Option<PathBuf>,
    target_pattern: Option<XrdPattern>,
    target_fingerprint: Option<Vec<f32>>,
    entries: Vec<EnsembleEntry>,
    order: Vec<usize>,
    selected_rank: usize,
    sort_mode: EnsembleSortMode,
    skipped_files: usize,
}

impl EnsembleState {
    fn from_input(input: EnsembleInput, default_view_state: EnsembleViewState) -> Option<Self> {
        if input.samples.is_empty() {
            return None;
        }

        let (_, wavelength) = WAVELENGTH_PRESETS[0];
        let target_path = input.target.as_ref().map(|(path, _)| path.clone());
        let target_pattern = input
            .target
            .as_ref()
            .map(|(_, structure)| compute_pattern(structure, wavelength, XRD_TWO_THETA_MAX));
        let target_fingerprint = target_pattern
            .as_ref()
            .map(|pattern| xrd_pattern_fingerprint(pattern, ENSEMBLE_FP_BINS));

        let mut entries: Vec<EnsembleEntry> = input
            .samples
            .into_iter()
            .map(|(path, structure)| {
                let pattern = compute_pattern(&structure, wavelength, XRD_TWO_THETA_MAX);
                let fingerprint = xrd_pattern_fingerprint(&pattern, ENSEMBLE_FP_BINS);
                let distance_to_target = target_fingerprint
                    .as_ref()
                    .map(|target_fp| cosine_distance(&fingerprint, target_fp));
                EnsembleEntry {
                    path,
                    structure,
                    fingerprint,
                    distance_to_target,
                    novelty: 0.0,
                    view_state: default_view_state.clone(),
                }
            })
            .collect();

        compute_ensemble_novelty(&mut entries);

        let sort_mode = if target_fingerprint.is_some() {
            EnsembleSortMode::TargetDistance
        } else {
            EnsembleSortMode::Novelty
        };
        let mut state = Self {
            source_dir: input.source_dir,
            target_path,
            target_pattern,
            target_fingerprint,
            entries,
            order: Vec::new(),
            selected_rank: 0,
            sort_mode,
            skipped_files: input.skipped_files,
        };
        state.refresh_order(None);
        Some(state)
    }

    fn has_target(&self) -> bool {
        self.target_fingerprint.is_some()
    }

    fn selected_entry_index(&self) -> Option<usize> {
        self.order.get(self.selected_rank).copied()
    }

    fn selected_entry(&self) -> Option<&EnsembleEntry> {
        self.selected_entry_index()
            .and_then(|idx| self.entries.get(idx))
    }

    fn cycle_sort_mode(&mut self) {
        let current_idx = self.selected_entry_index();
        self.sort_mode = self.sort_mode.next(self.has_target());
        self.refresh_order(current_idx);
    }

    fn select_offset(&mut self, delta: isize) -> Option<usize> {
        if self.order.is_empty() {
            return None;
        }
        let max_idx = self.order.len().saturating_sub(1) as isize;
        let next = (self.selected_rank as isize + delta).clamp(0, max_idx) as usize;
        self.selected_rank = next;
        self.order.get(self.selected_rank).copied()
    }

    fn refresh_order(&mut self, selected_entry_idx: Option<usize>) {
        self.order = (0..self.entries.len()).collect();
        self.order.sort_by(|&lhs, &rhs| {
            let left = &self.entries[lhs];
            let right = &self.entries[rhs];
            let by_metric = match self.sort_mode {
                EnsembleSortMode::TargetDistance => left
                    .distance_to_target
                    .partial_cmp(&right.distance_to_target)
                    .unwrap_or(Ordering::Equal),
                EnsembleSortMode::Novelty => right
                    .novelty
                    .partial_cmp(&left.novelty)
                    .unwrap_or(Ordering::Equal),
                EnsembleSortMode::FileName => {
                    file_name_key(&left.path).cmp(&file_name_key(&right.path))
                }
            };
            by_metric.then_with(|| file_name_key(&left.path).cmp(&file_name_key(&right.path)))
        });

        let current_idx = selected_entry_idx.or_else(|| self.selected_entry_index());
        self.selected_rank = current_idx
            .and_then(|idx| self.order.iter().position(|&candidate| candidate == idx))
            .unwrap_or(0);
    }

    fn store_view_state_for_selected(&mut self, view_state: &EnsembleViewState) {
        let Some(selected_idx) = self.selected_entry_index() else {
            return;
        };
        if let Some(entry) = self.entries.get_mut(selected_idx) {
            entry.view_state = view_state.clone();
        }
    }
}

#[derive(Debug, Clone)]
struct FilePickerEntry {
    path: PathBuf,
    name: String,
    is_dir: bool,
}

#[derive(Debug, Clone)]
struct FilePickerState {
    cwd: PathBuf,
    entries: Vec<FilePickerEntry>,
    selected: usize,
    error: Option<String>,
}

impl FilePickerState {
    fn new(cwd: PathBuf) -> Self {
        let mut state = Self {
            cwd,
            entries: Vec::new(),
            selected: 0,
            error: None,
        };
        state.refresh();
        state
    }

    fn refresh(&mut self) {
        let previously_selected = self
            .entries
            .get(self.selected)
            .map(|entry| entry.path.clone());

        let mut directories = Vec::new();
        let mut cif_files = Vec::new();

        match fs::read_dir(&self.cwd) {
            Ok(read_dir) => {
                self.error = None;
                for entry_result in read_dir {
                    let Ok(entry) = entry_result else {
                        continue;
                    };
                    let path = entry.path();
                    let Ok(file_type) = entry.file_type() else {
                        continue;
                    };
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if file_type.is_dir() {
                        directories.push(FilePickerEntry {
                            path,
                            name,
                            is_dir: true,
                        });
                    } else if file_type.is_file() && is_cif_path(&path) {
                        cif_files.push(FilePickerEntry {
                            path,
                            name,
                            is_dir: false,
                        });
                    }
                }
            }
            Err(err) => {
                self.entries.clear();
                self.selected = 0;
                self.error = Some(format!("Could not read directory: {err}"));
                return;
            }
        }

        let sort_key = |entry: &FilePickerEntry| entry.name.to_ascii_lowercase();
        directories.sort_by_key(sort_key);
        cif_files.sort_by_key(sort_key);

        let mut entries = Vec::new();
        if let Some(parent) = self.cwd.parent() {
            entries.push(FilePickerEntry {
                path: parent.to_path_buf(),
                name: "..".to_string(),
                is_dir: true,
            });
        }
        entries.extend(directories);
        entries.extend(cif_files);

        self.entries = entries;
        if self.entries.is_empty() {
            self.selected = 0;
            return;
        }
        if let Some(selected_path) = previously_selected {
            if let Some(index) = self
                .entries
                .iter()
                .position(|entry| entry.path == selected_path)
            {
                self.selected = index;
                return;
            }
        }
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
    }

    fn move_selection(&mut self, delta: isize) {
        if self.entries.is_empty() {
            self.selected = 0;
            return;
        }
        let max_index = self.entries.len().saturating_sub(1) as isize;
        let next = (self.selected as isize + delta).clamp(0, max_index);
        self.selected = next as usize;
    }

    fn open_selected(&mut self) -> Option<PathBuf> {
        let entry = self.entries.get(self.selected)?.clone();
        if entry.is_dir {
            self.cwd = entry.path;
            self.refresh();
            None
        } else {
            Some(entry.path)
        }
    }

    fn go_parent(&mut self) {
        if let Some(parent) = self.cwd.parent() {
            self.cwd = parent.to_path_buf();
            self.refresh();
        }
    }
}

// ── Editor types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DraftAtom {
    label: String,
    element: String,
    x: String,
    y: String,
    z: String,
}

impl DraftAtom {
    fn from_atom(atom: &Atom) -> Self {
        let frac = atom.fractional.unwrap_or(atom.position);
        Self {
            label: atom.label.clone(),
            element: atom.element.clone(),
            x: format!("{:.6}", frac[0]),
            y: format!("{:.6}", frac[1]),
            z: format!("{:.6}", frac[2]),
        }
    }

    fn to_atom(&self) -> Option<Atom> {
        let x = self.x.parse::<f32>().ok()?;
        let y = self.y.parse::<f32>().ok()?;
        let z = self.z.parse::<f32>().ok()?;
        let frac = [x, y, z];
        Some(Atom {
            label: if self.label.is_empty() {
                self.element.clone()
            } else {
                self.label.clone()
            },
            element: self.element.clone(),
            position: frac,
            fractional: Some(frac),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditorFocus {
    Title,
    CellA,
    CellB,
    CellC,
    CellAlpha,
    CellBeta,
    CellGamma,
    SpaceGroup,
    AtomLabel(usize),
    AtomElement(usize),
    AtomX(usize),
    AtomY(usize),
    AtomZ(usize),
}

impl EditorFocus {
    fn atom_row(&self) -> Option<usize> {
        match self {
            Self::AtomLabel(i)
            | Self::AtomElement(i)
            | Self::AtomX(i)
            | Self::AtomY(i)
            | Self::AtomZ(i) => Some(*i),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct EditorState {
    draft_title: String,
    draft_a: String,
    draft_b: String,
    draft_c: String,
    draft_alpha: String,
    draft_beta: String,
    draft_gamma: String,
    draft_space_group: String,
    resolved_sg_ops: Vec<String>,
    resolved_sg_name: Option<String>,
    draft_atoms: Vec<DraftAtom>,
    focus: EditorFocus,
    editing: bool,
    cursor_pos: usize,
    atom_scroll: usize,
    source_path: Option<PathBuf>,
    status: Option<String>,
}

impl EditorState {
    fn new_empty() -> Self {
        Self {
            draft_title: "new_structure".to_string(),
            draft_a: "5.000000".to_string(),
            draft_b: "5.000000".to_string(),
            draft_c: "5.000000".to_string(),
            draft_alpha: "90.000000".to_string(),
            draft_beta: "90.000000".to_string(),
            draft_gamma: "90.000000".to_string(),
            draft_space_group: "P1".to_string(),
            resolved_sg_ops: vec!["x,y,z".to_string()],
            resolved_sg_name: Some("P1".to_string()),
            draft_atoms: vec![DraftAtom {
                label: "A1".to_string(),
                element: "X".to_string(),
                x: "0.000000".to_string(),
                y: "0.000000".to_string(),
                z: "0.000000".to_string(),
            }],
            focus: EditorFocus::Title,
            editing: false,
            cursor_pos: 0,
            atom_scroll: 0,
            source_path: None,
            status: None,
        }
    }

    fn from_structure(structure: &Structure, path: Option<PathBuf>) -> Self {
        let cell = structure.cell;
        let (a, b, c, alpha, beta, gamma) = if let Some(c) = cell {
            (c.a, c.b, c.c, c.alpha_deg, c.beta_deg, c.gamma_deg)
        } else {
            (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        };
        let sg_str = structure
            .space_group
            .clone()
            .unwrap_or_else(|| "P1".to_string());

        // Resolve via spacegroup lookup.
        let (resolved_ops, resolved_name) = Self::resolve_sg(&sg_str, &structure.symmetry_ops);

        let draft_atoms = if !structure.asu_atoms.is_empty() {
            structure
                .asu_atoms
                .iter()
                .map(DraftAtom::from_atom)
                .collect()
        } else {
            structure.atoms.iter().map(DraftAtom::from_atom).collect()
        };

        Self {
            draft_title: structure.title.clone(),
            draft_a: format!("{a:.6}"),
            draft_b: format!("{b:.6}"),
            draft_c: format!("{c:.6}"),
            draft_alpha: format!("{alpha:.6}"),
            draft_beta: format!("{beta:.6}"),
            draft_gamma: format!("{gamma:.6}"),
            draft_space_group: sg_str,
            resolved_sg_ops: resolved_ops,
            resolved_sg_name: resolved_name,
            draft_atoms,
            focus: EditorFocus::Title,
            editing: false,
            cursor_pos: 0,
            atom_scroll: 0,
            source_path: path,
            status: None,
        }
    }

    fn resolve_sg(query: &str, fallback_ops: &[String]) -> (Vec<String>, Option<String>) {
        if let Some(sg) = spacegroup::lookup(query) {
            let ops = sg.ops.iter().map(|s| s.to_string()).collect();
            return (ops, Some(sg.canonical_name.to_string()));
        }
        if !fallback_ops.is_empty() {
            return (fallback_ops.to_vec(), None);
        }
        (vec!["x,y,z".to_string()], None)
    }

    fn get_field_text(&self, focus: EditorFocus) -> &str {
        match focus {
            EditorFocus::Title => &self.draft_title,
            EditorFocus::CellA => &self.draft_a,
            EditorFocus::CellB => &self.draft_b,
            EditorFocus::CellC => &self.draft_c,
            EditorFocus::CellAlpha => &self.draft_alpha,
            EditorFocus::CellBeta => &self.draft_beta,
            EditorFocus::CellGamma => &self.draft_gamma,
            EditorFocus::SpaceGroup => &self.draft_space_group,
            EditorFocus::AtomLabel(i) => self
                .draft_atoms
                .get(i)
                .map(|a| a.label.as_str())
                .unwrap_or(""),
            EditorFocus::AtomElement(i) => self
                .draft_atoms
                .get(i)
                .map(|a| a.element.as_str())
                .unwrap_or(""),
            EditorFocus::AtomX(i) => self.draft_atoms.get(i).map(|a| a.x.as_str()).unwrap_or(""),
            EditorFocus::AtomY(i) => self.draft_atoms.get(i).map(|a| a.y.as_str()).unwrap_or(""),
            EditorFocus::AtomZ(i) => self.draft_atoms.get(i).map(|a| a.z.as_str()).unwrap_or(""),
        }
    }

    fn set_field_text(&mut self, focus: EditorFocus, text: String) {
        match focus {
            EditorFocus::Title => self.draft_title = text,
            EditorFocus::CellA => self.draft_a = text,
            EditorFocus::CellB => self.draft_b = text,
            EditorFocus::CellC => self.draft_c = text,
            EditorFocus::CellAlpha => self.draft_alpha = text,
            EditorFocus::CellBeta => self.draft_beta = text,
            EditorFocus::CellGamma => self.draft_gamma = text,
            EditorFocus::SpaceGroup => {
                self.draft_space_group = text.clone();
                let (ops, name) = Self::resolve_sg(&text, &[]);
                self.resolved_sg_ops = ops;
                self.resolved_sg_name = name;
            }
            EditorFocus::AtomLabel(i) => {
                if let Some(a) = self.draft_atoms.get_mut(i) {
                    a.label = text;
                }
            }
            EditorFocus::AtomElement(i) => {
                if let Some(a) = self.draft_atoms.get_mut(i) {
                    a.element = text;
                }
            }
            EditorFocus::AtomX(i) => {
                if let Some(a) = self.draft_atoms.get_mut(i) {
                    a.x = text;
                }
            }
            EditorFocus::AtomY(i) => {
                if let Some(a) = self.draft_atoms.get_mut(i) {
                    a.y = text;
                }
            }
            EditorFocus::AtomZ(i) => {
                if let Some(a) = self.draft_atoms.get_mut(i) {
                    a.z = text;
                }
            }
        }
    }

    fn next_focus(&self) -> EditorFocus {
        match self.focus {
            EditorFocus::Title => EditorFocus::CellA,
            EditorFocus::CellA => EditorFocus::CellB,
            EditorFocus::CellB => EditorFocus::CellC,
            EditorFocus::CellC => EditorFocus::CellAlpha,
            EditorFocus::CellAlpha => EditorFocus::CellBeta,
            EditorFocus::CellBeta => EditorFocus::CellGamma,
            EditorFocus::CellGamma => EditorFocus::SpaceGroup,
            EditorFocus::SpaceGroup => {
                if self.draft_atoms.is_empty() {
                    EditorFocus::Title
                } else {
                    EditorFocus::AtomLabel(0)
                }
            }
            EditorFocus::AtomLabel(i) => EditorFocus::AtomElement(i),
            EditorFocus::AtomElement(i) => EditorFocus::AtomX(i),
            EditorFocus::AtomX(i) => EditorFocus::AtomY(i),
            EditorFocus::AtomY(i) => EditorFocus::AtomZ(i),
            EditorFocus::AtomZ(i) => {
                let next = i + 1;
                if next < self.draft_atoms.len() {
                    EditorFocus::AtomLabel(next)
                } else {
                    EditorFocus::Title
                }
            }
        }
    }

    fn prev_focus(&self) -> EditorFocus {
        match self.focus {
            EditorFocus::Title => {
                if self.draft_atoms.is_empty() {
                    EditorFocus::SpaceGroup
                } else {
                    EditorFocus::AtomZ(self.draft_atoms.len() - 1)
                }
            }
            EditorFocus::CellA => EditorFocus::Title,
            EditorFocus::CellB => EditorFocus::CellA,
            EditorFocus::CellC => EditorFocus::CellB,
            EditorFocus::CellAlpha => EditorFocus::CellC,
            EditorFocus::CellBeta => EditorFocus::CellAlpha,
            EditorFocus::CellGamma => EditorFocus::CellBeta,
            EditorFocus::SpaceGroup => EditorFocus::CellGamma,
            EditorFocus::AtomLabel(0) => EditorFocus::SpaceGroup,
            EditorFocus::AtomLabel(i) => EditorFocus::AtomZ(i - 1),
            EditorFocus::AtomElement(i) => EditorFocus::AtomLabel(i),
            EditorFocus::AtomX(i) => EditorFocus::AtomElement(i),
            EditorFocus::AtomY(i) => EditorFocus::AtomX(i),
            EditorFocus::AtomZ(i) => EditorFocus::AtomY(i),
        }
    }

    fn focus_row(&self) -> usize {
        match self.focus {
            EditorFocus::Title => 0,
            EditorFocus::CellA | EditorFocus::CellB | EditorFocus::CellC => 1,
            EditorFocus::CellAlpha | EditorFocus::CellBeta | EditorFocus::CellGamma => 2,
            EditorFocus::SpaceGroup => 3,
            EditorFocus::AtomLabel(i)
            | EditorFocus::AtomElement(i)
            | EditorFocus::AtomX(i)
            | EditorFocus::AtomY(i)
            | EditorFocus::AtomZ(i) => 4 + i,
        }
    }

    fn focus_col(&self) -> usize {
        match self.focus {
            EditorFocus::Title | EditorFocus::SpaceGroup => 0,
            EditorFocus::CellA | EditorFocus::CellAlpha | EditorFocus::AtomLabel(_) => 0,
            EditorFocus::CellB | EditorFocus::CellBeta | EditorFocus::AtomElement(_) => 1,
            EditorFocus::CellC | EditorFocus::CellGamma | EditorFocus::AtomX(_) => 2,
            EditorFocus::AtomY(_) => 3,
            EditorFocus::AtomZ(_) => 4,
        }
    }

    fn focus_at(&self, row: usize, col: usize) -> EditorFocus {
        match row {
            0 => EditorFocus::Title,
            1 => match col.min(2) {
                0 => EditorFocus::CellA,
                1 => EditorFocus::CellB,
                _ => EditorFocus::CellC,
            },
            2 => match col.min(2) {
                0 => EditorFocus::CellAlpha,
                1 => EditorFocus::CellBeta,
                _ => EditorFocus::CellGamma,
            },
            3 => EditorFocus::SpaceGroup,
            r => {
                let i = (r - 4).min(self.draft_atoms.len().saturating_sub(1));
                match col.min(4) {
                    0 => EditorFocus::AtomLabel(i),
                    1 => EditorFocus::AtomElement(i),
                    2 => EditorFocus::AtomX(i),
                    3 => EditorFocus::AtomY(i),
                    _ => EditorFocus::AtomZ(i),
                }
            }
        }
    }

    fn ensure_atom_visible(&mut self, atom_row: usize) {
        if atom_row < self.atom_scroll {
            self.atom_scroll = atom_row;
        } else if atom_row >= self.atom_scroll + EDITOR_ATOM_ROWS {
            self.atom_scroll = atom_row + 1 - EDITOR_ATOM_ROWS;
        }
    }

    fn move_up(&mut self) {
        let row = self.focus_row();
        let col = self.focus_col();
        if row == 0 {
            return;
        }
        let new_row = row - 1;
        // SpaceGroup (row 3) is a single field — clamp col to 0 when landing on it.
        let new_col = if new_row == 3 { 0 } else { col };
        let new_focus = self.focus_at(new_row, new_col);
        self.focus = new_focus;
        if let Some(atom_row) = self.focus.atom_row() {
            self.ensure_atom_visible(atom_row);
        }
    }

    fn move_down(&mut self) {
        let row = self.focus_row();
        let col = self.focus_col();
        // Atom index i lives at row 4+i, so last atom is at row 3+n.
        // With 0 atoms, last valid row is 3 (SpaceGroup). With n atoms, it is 3+n.
        if row >= 3 + self.draft_atoms.len() {
            return;
        }
        let new_row = row + 1;
        // Transitioning to SpaceGroup (row 3) — clamp col to 0.
        let new_col = if row == 2 { 0 } else { col };
        let new_focus = self.focus_at(new_row, new_col);
        self.focus = new_focus;
        if let Some(atom_row) = self.focus.atom_row() {
            self.ensure_atom_visible(atom_row);
        }
    }

    fn move_left(&mut self) {
        let row = self.focus_row();
        let col = self.focus_col();
        if col == 0 {
            return;
        }
        self.focus = self.focus_at(row, col - 1);
    }

    fn move_right(&mut self) {
        let row = self.focus_row();
        let col = self.focus_col();
        let max_col = match row {
            0 | 3 => 0,
            1 | 2 => 2,
            _ => 4,
        };
        if col >= max_col {
            return;
        }
        self.focus = self.focus_at(row, col + 1);
    }

    fn delete_focused_atom(&mut self) {
        let row = match self.focus.atom_row() {
            Some(r) => r,
            None => return,
        };
        if self.draft_atoms.is_empty() {
            return;
        }
        let col = self.focus_col();
        self.draft_atoms.remove(row);
        if self.draft_atoms.is_empty() {
            self.focus = EditorFocus::SpaceGroup;
            self.atom_scroll = 0;
        } else {
            let new_row = row.min(self.draft_atoms.len() - 1);
            self.focus = self.focus_at(4 + new_row, col);
            self.ensure_atom_visible(new_row);
        }
    }

    fn add_atom(&mut self) {
        self.draft_atoms.push(DraftAtom {
            label: format!("A{}", self.draft_atoms.len() + 1),
            element: "X".to_string(),
            x: "0.000000".to_string(),
            y: "0.000000".to_string(),
            z: "0.000000".to_string(),
        });
        let new_i = self.draft_atoms.len() - 1;
        self.focus = EditorFocus::AtomLabel(new_i);
        self.ensure_atom_visible(new_i);
    }

    fn build_structure(&self) -> Result<Structure, String> {
        let a = self
            .draft_a
            .parse::<f32>()
            .map_err(|_| "Invalid cell parameter a")?;
        let b = self
            .draft_b
            .parse::<f32>()
            .map_err(|_| "Invalid cell parameter b")?;
        let c = self
            .draft_c
            .parse::<f32>()
            .map_err(|_| "Invalid cell parameter c")?;
        let alpha = self
            .draft_alpha
            .parse::<f32>()
            .map_err(|_| "Invalid cell angle alpha")?;
        let beta = self
            .draft_beta
            .parse::<f32>()
            .map_err(|_| "Invalid cell angle beta")?;
        let gamma = self
            .draft_gamma
            .parse::<f32>()
            .map_err(|_| "Invalid cell angle gamma")?;

        let cell = Cell::from_parameters(a, b, c, alpha, beta, gamma);

        let asu_atoms: Vec<Atom> = self
            .draft_atoms
            .iter()
            .filter_map(|da| da.to_atom())
            .collect();

        if asu_atoms.is_empty() {
            return Err("No valid atoms defined".to_string());
        }

        let atoms = crate::cif::expand_asu_atoms(&asu_atoms, cell, &self.resolved_sg_ops);
        let atoms = if atoms.is_empty() {
            asu_atoms.clone()
        } else {
            atoms
        };

        let space_group = if self.draft_space_group.is_empty() {
            None
        } else {
            Some(self.draft_space_group.clone())
        };

        Ok(Structure {
            title: self.draft_title.clone(),
            atoms,
            cell,
            space_group,
            asu_atoms,
            symmetry_ops: self.resolved_sg_ops.clone(),
        })
    }
}

fn is_cif_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("cif"))
        .unwrap_or(false)
}

fn default_open_dialog_dir() -> PathBuf {
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn empty_structure() -> Structure {
    Structure {
        title: "No structure loaded".to_string(),
        atoms: Vec::new(),
        cell: None,
        space_group: None,
        asu_atoms: Vec::new(),
        symmetry_ops: Vec::new(),
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
    spin_lock: bool,
    spin_speed: f32,
    spin_velocity: [f32; 3], // [pitch, yaw, roll] direction; normalized when non-zero
    /// Column/row of the last mouse-drag position, for delta calculation.
    drag_last: Option<(u16, u16)>,
    /// Cached element → count map, derived from structure.atoms (immutable).
    cached_element_counts: BTreeMap<String, usize>,
    /// Cached Hill-order empirical formula derived from cached_element_counts.
    cached_formula: String,
    open_dialog_dir: PathBuf,
    file_picker: Option<FilePickerState>,
    show_xrd: bool,
    xrd_wavelength_idx: usize,
    cached_xrd_pattern: XrdPattern,
    editor: Option<EditorState>,
    current_cif_path: Option<PathBuf>,
    ensemble: Option<EnsembleState>,
    show_target_overlay: bool,
    show_ensemble_grid: bool,
}

impl App {
    fn new(structure: Structure) -> Self {
        Self::with_initial_structure(Some(structure), default_open_dialog_dir())
    }

    fn with_initial_structure(
        initial_structure: Option<Structure>,
        open_dialog_dir: PathBuf,
    ) -> Self {
        let structure = initial_structure.unwrap_or_else(empty_structure);
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
        let xrd_wavelength_idx = 0usize;
        let cached_xrd_pattern = compute_pattern(
            &structure,
            WAVELENGTH_PRESETS[xrd_wavelength_idx].1,
            XRD_TWO_THETA_MAX,
        );
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
            spin_lock: false,
            spin_speed: DEFAULT_SPIN_SPEED,
            spin_velocity: [0.0, 0.0, 0.0],
            drag_last: None,
            cached_element_counts,
            cached_formula,
            open_dialog_dir,
            file_picker: None,
            show_xrd: false,
            xrd_wavelength_idx,
            cached_xrd_pattern,
            editor: None,
            current_cif_path: None,
            ensemble: None,
            show_target_overlay: false,
            show_ensemble_grid: false,
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.editor.is_some() {
            if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                self.handle_editor_key(key);
            }
            return;
        }
        if self.file_picker.is_some() {
            if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                self.handle_file_picker_key(key.code);
            }
            return;
        }
        match key.kind {
            KeyEventKind::Press => self.handle_key_press(key.code),
            KeyEventKind::Repeat => self.handle_key_repeat(key.code),
            KeyEventKind::Release => {}
        }
        self.sync_current_view_into_selected_ensemble();
    }

    /// Actions that fire once per key-down (toggles, discrete steps).
    fn handle_key_press(&mut self, code: KeyCode) {
        if self.spin_lock && self.set_spin_direction_from_key(code) {
            return;
        }
        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('O') => self.open_file_picker(),
            KeyCode::Char('R') => self.toggle_spin_lock(),
            KeyCode::Char('<') => self.adjust_spin_speed(-SPIN_SPEED_STEP),
            KeyCode::Char('>') => self.adjust_spin_speed(SPIN_SPEED_STEP),
            KeyCode::Up => {
                self.select_ensemble_relative(-1);
            }
            KeyCode::Down => {
                self.select_ensemble_relative(1);
            }
            KeyCode::PageUp => {
                self.select_ensemble_relative(-5);
            }
            KeyCode::PageDown => {
                self.select_ensemble_relative(5);
            }
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
            KeyCode::Char('X') => self.show_xrd = !self.show_xrd,
            KeyCode::Char('W') => self.cycle_xrd_wavelength(),
            KeyCode::Char('P') => self.cycle_ensemble_sort_mode(),
            KeyCode::Char('Y') => self.toggle_target_overlay(),
            KeyCode::Char('G') => self.toggle_ensemble_grid(),
            KeyCode::Char('A') => self.snap_view_to_lattice_axis(0),
            KeyCode::Char('B') => self.snap_view_to_lattice_axis(1),
            KeyCode::Char('C') => self.snap_view_to_lattice_axis(2),
            KeyCode::Char('E') => self.open_editor_for_current(),
            KeyCode::Char('F') => self.open_editor_new(),
            KeyCode::Tab => {
                if !self.structure.atoms.is_empty() {
                    self.selected_atom = (self.selected_atom + 1) % self.structure.atoms.len();
                }
            }
            _ => self.apply_continuous_key(code),
        }
    }

    fn open_file_picker(&mut self) {
        self.file_picker = Some(FilePickerState::new(self.open_dialog_dir.clone()));
    }

    fn attach_ensemble(&mut self, input: EnsembleInput) {
        let ensemble_state = EnsembleState::from_input(input, self.current_ensemble_view_state());
        self.ensemble = ensemble_state;
        self.show_target_overlay = self
            .ensemble
            .as_ref()
            .and_then(|ensemble| ensemble.target_pattern.as_ref())
            .is_some();
        self.show_ensemble_grid = self.ensemble.is_some();
        self.select_ensemble_relative(0);
    }

    fn select_ensemble_relative(&mut self, delta: isize) {
        let current_view_state = self.current_ensemble_view_state();
        let selected_entry = if let Some(ensemble) = self.ensemble.as_mut() {
            ensemble.store_view_state_for_selected(&current_view_state);
            let idx = ensemble.select_offset(delta);
            idx.and_then(|selected_idx| {
                ensemble.entries.get(selected_idx).map(|entry| {
                    (
                        entry.path.clone(),
                        entry.structure.clone(),
                        entry.view_state.clone(),
                    )
                })
            })
        } else {
            None
        };
        if let Some((path, structure, view_state)) = selected_entry {
            self.current_cif_path = Some(path);
            self.apply_structure(structure);
            self.apply_ensemble_view_state(&view_state);
        }
    }

    fn cycle_ensemble_sort_mode(&mut self) {
        let current_view_state = self.current_ensemble_view_state();
        let selected_entry = if let Some(ensemble) = self.ensemble.as_mut() {
            ensemble.store_view_state_for_selected(&current_view_state);
            ensemble.cycle_sort_mode();
            ensemble.selected_entry().map(|entry| {
                (
                    entry.path.clone(),
                    entry.structure.clone(),
                    entry.view_state.clone(),
                )
            })
        } else {
            None
        };
        if let Some((path, structure, view_state)) = selected_entry {
            self.current_cif_path = Some(path);
            self.apply_structure(structure);
            self.apply_ensemble_view_state(&view_state);
        }
    }

    fn toggle_target_overlay(&mut self) {
        if let Some(ensemble) = &self.ensemble {
            if ensemble.target_pattern.is_some() {
                self.show_target_overlay = !self.show_target_overlay;
            }
        }
    }

    fn toggle_ensemble_grid(&mut self) {
        if self.ensemble.is_some() {
            self.show_ensemble_grid = !self.show_ensemble_grid;
        }
    }

    fn ensemble_grid_page_bounds(&self) -> Option<(usize, usize)> {
        if !self.show_ensemble_grid {
            return None;
        }
        let ensemble = self.ensemble.as_ref()?;
        if ensemble.order.is_empty() {
            return None;
        }
        let start = (ensemble.selected_rank / ENSEMBLE_GRID_CAPACITY) * ENSEMBLE_GRID_CAPACITY;
        let end = (start + ENSEMBLE_GRID_CAPACITY).min(ensemble.order.len());
        Some((start, end))
    }

    fn selected_ensemble_entry(&self) -> Option<&EnsembleEntry> {
        self.ensemble
            .as_ref()
            .and_then(|ensemble| ensemble.selected_entry())
    }

    fn current_ensemble_view_state(&self) -> EnsembleViewState {
        EnsembleViewState {
            rot_mat: self.rot_mat,
            zoom: self.zoom,
            fov_deg: self.fov_deg,
            lock_fov_zoom: self.lock_fov_zoom,
            pan: self.pan,
            sphere_scale: self.sphere_scale,
            show_bonds: self.show_bonds,
            show_cell: self.show_cell,
            cell_on_top: self.cell_on_top,
            show_boundary_images: self.show_boundary_images,
            show_bonded_images: self.show_bonded_images,
            bond_max_distance: self.bond_max_distance,
            show_labels: self.show_labels,
            show_orientation_gizmo: self.show_orientation_gizmo,
            render_theme: self.render_theme,
        }
    }

    fn apply_ensemble_view_state(&mut self, view_state: &EnsembleViewState) {
        self.rot_mat = view_state.rot_mat;
        self.zoom = view_state.zoom;
        self.fov_deg = view_state.fov_deg;
        self.lock_fov_zoom = view_state.lock_fov_zoom;
        self.pan = view_state.pan;
        self.sphere_scale = view_state.sphere_scale;
        self.show_bonds = view_state.show_bonds;
        self.show_cell = view_state.show_cell;
        self.cell_on_top = view_state.cell_on_top;
        self.show_boundary_images = view_state.show_boundary_images;
        self.show_bonded_images = view_state.show_bonded_images;
        self.bond_max_distance = view_state.bond_max_distance;
        self.show_labels = view_state.show_labels;
        self.show_orientation_gizmo = view_state.show_orientation_gizmo;
        self.render_theme = view_state.render_theme;
        self.rebuild_scene();
    }

    fn sync_current_view_into_selected_ensemble(&mut self) {
        let view_state = self.current_ensemble_view_state();
        if let Some(ensemble) = self.ensemble.as_mut() {
            ensemble.store_view_state_for_selected(&view_state);
        }
    }

    fn clear_ensemble_mode(&mut self) {
        self.ensemble = None;
        self.show_target_overlay = false;
        self.show_ensemble_grid = false;
    }

    fn handle_file_picker_key(&mut self, code: KeyCode) {
        let mut open_path: Option<PathBuf> = None;
        let mut close_picker = false;
        if let Some(picker) = self.file_picker.as_mut() {
            match code {
                KeyCode::Esc => close_picker = true,
                KeyCode::Up | KeyCode::Char('k') => picker.move_selection(-1),
                KeyCode::Down | KeyCode::Char('j') => picker.move_selection(1),
                KeyCode::PageUp => picker.move_selection(-10),
                KeyCode::PageDown => picker.move_selection(10),
                KeyCode::Backspace | KeyCode::Left => picker.go_parent(),
                KeyCode::Right | KeyCode::Enter => {
                    open_path = picker.open_selected();
                }
                _ => {}
            }
        }
        if close_picker {
            self.file_picker = None;
            return;
        }

        let Some(path) = open_path else {
            return;
        };

        match crate::cif::parse_cif_file(&path) {
            Ok(structure) => {
                self.clear_ensemble_mode();
                self.current_cif_path = Some(path.clone());
                self.apply_structure(structure);
                if let Some(parent) = path.parent() {
                    self.open_dialog_dir = parent.to_path_buf();
                }
                self.file_picker = None;
            }
            Err(err) => {
                if let Some(picker) = self.file_picker.as_mut() {
                    picker.error = Some(format!("Could not load {}: {err}", path.display()));
                }
            }
        }
    }

    fn handle_key_repeat(&mut self, code: KeyCode) {
        if self.spin_lock && self.set_spin_direction_from_key(code) {
            return;
        }
        match code {
            KeyCode::Up => {
                self.select_ensemble_relative(-1);
                return;
            }
            KeyCode::Down => {
                self.select_ensemble_relative(1);
                return;
            }
            _ => {}
        }
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
            KeyCode::Char('u') => self.rot_mat = mat_mul_3x3(mat_rot_z(-ROLL_STEP), self.rot_mat),
            KeyCode::Char('o') => self.rot_mat = mat_mul_3x3(mat_rot_z(ROLL_STEP), self.rot_mat),
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

    fn toggle_spin_lock(&mut self) {
        self.spin_lock = !self.spin_lock;
        if !self.spin_lock {
            self.spin_velocity = [0.0, 0.0, 0.0];
        }
    }

    fn adjust_spin_speed(&mut self, delta: f32) {
        self.spin_speed = (self.spin_speed + delta).clamp(MIN_SPIN_SPEED, MAX_SPIN_SPEED);
    }

    fn set_spin_direction_from_key(&mut self, code: KeyCode) -> bool {
        let direction = match code {
            KeyCode::Char('h') => Some([0.0, -1.0, 0.0]),
            KeyCode::Char('l') => Some([0.0, 1.0, 0.0]),
            KeyCode::Char('j') => Some([1.0, 0.0, 0.0]),
            KeyCode::Char('k') => Some([-1.0, 0.0, 0.0]),
            KeyCode::Char('u') => Some([0.0, 0.0, -1.0]),
            KeyCode::Char('o') => Some([0.0, 0.0, 1.0]),
            _ => None,
        };
        let Some(direction) = direction else {
            return false;
        };
        self.set_spin_direction(direction);
        true
    }

    fn set_spin_direction(&mut self, direction: [f32; 3]) {
        let norm = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if norm <= f32::EPSILON {
            return;
        }
        self.spin_velocity = [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ];
    }

    fn update_spin_motion(&mut self, delta_seconds: f32) {
        if !self.spin_lock {
            return;
        }
        let vx = self.spin_velocity[0];
        let vy = self.spin_velocity[1];
        let vz = self.spin_velocity[2];
        if vx.abs() <= f32::EPSILON && vy.abs() <= f32::EPSILON && vz.abs() <= f32::EPSILON {
            return;
        }
        let step = SPIN_BASE_RATE_RAD_PER_SEC * self.spin_speed * delta_seconds.max(0.0);
        if step <= 0.0 {
            return;
        }
        self.rot_mat = mat_mul_3x3(mat_rot_x(vx * step), self.rot_mat);
        self.rot_mat = mat_mul_3x3(mat_rot_y(vy * step), self.rot_mat);
        self.rot_mat = mat_mul_3x3(mat_rot_z(vz * step), self.rot_mat);
        self.sync_current_view_into_selected_ensemble();
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
        let camera =
            CameraParams::from_mat(self.scene.center, scale, self.rot_mat, fov_deg, [0.0, 0.0]);
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

        if saw_point {
            Some(max_extent)
        } else {
            None
        }
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
        if self.file_picker.is_some() {
            return;
        }
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
                    if self.spin_lock {
                        self.set_spin_direction([dpitch, dyaw, 0.0]);
                    } else {
                        self.rot_mat = mat_mul_3x3(mat_rot_x(dpitch), self.rot_mat);
                        self.rot_mat = mat_mul_3x3(mat_rot_y(dyaw), self.rot_mat);
                    }
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
        self.sync_current_view_into_selected_ensemble();
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

    fn apply_structure(&mut self, structure: Structure) {
        self.structure = structure;
        self.selected_atom = 0;
        self.rebuild_scene();
        self.cached_element_counts = element_counts(&self.structure.atoms);
        self.cached_formula = empirical_formula(&self.cached_element_counts);
        self.cached_xrd_pattern = compute_pattern(
            &self.structure,
            WAVELENGTH_PRESETS[self.xrd_wavelength_idx].1,
            XRD_TWO_THETA_MAX,
        );
    }

    fn cycle_xrd_wavelength(&mut self) {
        self.xrd_wavelength_idx = (self.xrd_wavelength_idx + 1) % WAVELENGTH_PRESETS.len();
        self.cached_xrd_pattern = compute_pattern(
            &self.structure,
            WAVELENGTH_PRESETS[self.xrd_wavelength_idx].1,
            XRD_TWO_THETA_MAX,
        );
    }

    fn open_editor_for_current(&mut self) {
        if self.editor.is_some() {
            self.editor = None;
        } else {
            self.editor = Some(EditorState::from_structure(
                &self.structure,
                self.current_cif_path.clone(),
            ));
        }
    }

    fn open_editor_new(&mut self) {
        self.editor = Some(EditorState::new_empty());
    }

    fn editor_apply(&mut self) {
        let result = if let Some(editor) = &self.editor {
            editor.build_structure()
        } else {
            return;
        };
        match result {
            Ok(structure) => {
                let atom_count = structure.atoms.len();
                self.clear_ensemble_mode();
                self.apply_structure(structure);
                if let Some(editor) = &mut self.editor {
                    editor.status = Some(format!("Applied — {atom_count} atoms after expansion."));
                }
            }
            Err(msg) => {
                if let Some(editor) = &mut self.editor {
                    editor.status = Some(format!("Error: {msg}"));
                }
            }
        }
    }

    fn editor_save(&mut self) {
        let (structure, path) = match &self.editor {
            Some(editor) => {
                let s = match editor.build_structure() {
                    Ok(s) => s,
                    Err(msg) => {
                        if let Some(editor) = &mut self.editor {
                            editor.status = Some(format!("Error: {msg}"));
                        }
                        return;
                    }
                };
                let p = editor
                    .source_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from(format!("{}.cif", editor.draft_title)));
                (s, p)
            }
            None => return,
        };
        match crate::cif::write_cif(&structure, &path) {
            Ok(()) => {
                self.current_cif_path = Some(path.clone());
                if let Some(editor) = &mut self.editor {
                    editor.source_path = Some(path.clone());
                    editor.status = Some(format!("Saved to {}", path.display()));
                }
            }
            Err(err) => {
                if let Some(editor) = &mut self.editor {
                    editor.status = Some(format!("Save failed: {err}"));
                }
            }
        }
    }

    fn handle_editor_key(&mut self, key: KeyEvent) {
        // Ctrl+A / Ctrl+S are handled regardless of editing mode.
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('a') => {
                    self.editor_apply();
                    return;
                }
                KeyCode::Char('s') => {
                    self.editor_save();
                    return;
                }
                _ => {}
            }
        }

        let editing = self.editor.as_ref().map(|e| e.editing).unwrap_or(false);

        if editing {
            self.handle_editor_edit_mode_key(key.code);
        } else {
            self.handle_editor_nav_mode_key(key.code);
        }
    }

    fn handle_editor_nav_mode_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('E') => {
                self.editor = None;
            }
            KeyCode::Enter => {
                if let Some(editor) = &mut self.editor {
                    editor.editing = true;
                    let len = editor.get_field_text(editor.focus).len();
                    editor.cursor_pos = len;
                }
            }
            KeyCode::Tab => {
                if let Some(editor) = &mut self.editor {
                    let next = editor.next_focus();
                    editor.focus = next;
                    if let Some(atom_row) = editor.focus.atom_row() {
                        editor.ensure_atom_visible(atom_row);
                    }
                }
            }
            KeyCode::BackTab => {
                if let Some(editor) = &mut self.editor {
                    let prev = editor.prev_focus();
                    editor.focus = prev;
                }
            }
            KeyCode::Up => {
                if let Some(editor) = &mut self.editor {
                    editor.move_up();
                }
            }
            KeyCode::Down => {
                if let Some(editor) = &mut self.editor {
                    editor.move_down();
                }
            }
            KeyCode::Left => {
                if let Some(editor) = &mut self.editor {
                    editor.move_left();
                }
            }
            KeyCode::Right => {
                if let Some(editor) = &mut self.editor {
                    editor.move_right();
                }
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                // Only 'A' (without Ctrl) in nav mode adds atom.
                if let Some(editor) = &mut self.editor {
                    editor.add_atom();
                }
            }
            KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete => {
                if let Some(editor) = &mut self.editor {
                    editor.delete_focused_atom();
                }
            }
            _ => {}
        }
    }

    fn handle_editor_edit_mode_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                if let Some(editor) = &mut self.editor {
                    editor.editing = false;
                }
            }
            KeyCode::Enter => {
                // Commit value, exit edit mode.
                if let Some(editor) = &mut self.editor {
                    editor.editing = false;
                }
            }
            KeyCode::Tab => {
                // Commit + advance to next field (stay in edit mode).
                if let Some(editor) = &mut self.editor {
                    let next = editor.next_focus();
                    editor.focus = next;
                    if let Some(atom_row) = editor.focus.atom_row() {
                        editor.ensure_atom_visible(atom_row);
                    }
                    let len = editor.get_field_text(next).len();
                    editor.cursor_pos = len;
                }
            }
            KeyCode::BackTab => {
                if let Some(editor) = &mut self.editor {
                    let prev = editor.prev_focus();
                    editor.focus = prev;
                    let len = editor.get_field_text(prev).len();
                    editor.cursor_pos = len;
                }
            }
            KeyCode::Backspace => {
                if let Some(editor) = &mut self.editor {
                    let focus = editor.focus;
                    let mut text = editor.get_field_text(focus).to_string();
                    if editor.cursor_pos > 0 && editor.cursor_pos <= text.len() {
                        text.remove(editor.cursor_pos - 1);
                        editor.cursor_pos -= 1;
                        editor.set_field_text(focus, text);
                    }
                }
            }
            KeyCode::Char(ch) => {
                if let Some(editor) = &mut self.editor {
                    let focus = editor.focus;
                    let mut text = editor.get_field_text(focus).to_string();
                    let pos = editor.cursor_pos.min(text.len());
                    text.insert(pos, ch);
                    editor.cursor_pos = pos + 1;
                    editor.set_field_text(focus, text);
                }
            }
            KeyCode::Left => {
                if let Some(editor) = &mut self.editor {
                    editor.cursor_pos = editor.cursor_pos.saturating_sub(1);
                }
            }
            KeyCode::Right => {
                if let Some(editor) = &mut self.editor {
                    let len = editor.get_field_text(editor.focus).len();
                    if editor.cursor_pos < len {
                        editor.cursor_pos += 1;
                    }
                }
            }
            KeyCode::Up => {
                if let Some(editor) = &mut self.editor {
                    editor.move_up();
                    let len = editor.get_field_text(editor.focus).len();
                    editor.cursor_pos = len;
                }
            }
            KeyCode::Down => {
                if let Some(editor) = &mut self.editor {
                    editor.move_down();
                    let len = editor.get_field_text(editor.focus).len();
                    editor.cursor_pos = len;
                }
            }
            _ => {}
        }
    }
}

pub fn run(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    structure: Option<Structure>,
    open_dialog_dir: PathBuf,
    ensemble: Option<EnsembleInput>,
) -> Result<()> {
    let mut app = if let Some(structure) = structure {
        let mut app = App::new(structure);
        app.open_dialog_dir = open_dialog_dir;
        app
    } else {
        App::with_initial_structure(None, open_dialog_dir)
    };
    if let Some(ensemble_input) = ensemble {
        app.attach_ensemble(ensemble_input);
    }
    let tick_rate = Duration::from_millis(16);
    let mut previous_frame_time = std::time::Instant::now();

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

        let now = std::time::Instant::now();
        let delta_seconds = now
            .saturating_duration_since(previous_frame_time)
            .as_secs_f32()
            .clamp(0.0, 0.1);
        previous_frame_time = now;
        app.update_spin_motion(delta_seconds);

        terminal.draw(|frame| {
            draw(frame, &app);
        })?;
    }

    Ok(())
}

fn draw(frame: &mut ratatui::Frame, app: &App) {
    let area = frame.area();

    // Optionally carve a horizontal XRD strip from the bottom.
    let (top_area, xrd_area) = if app.show_xrd {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(8), Constraint::Length(XRD_PANEL_HEIGHT)])
            .split(area);
        (rows[0], Some(rows[1]))
    } else {
        (area, None)
    };

    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(24), Constraint::Length(44)])
        .split(top_area);

    let structure_panel_height = if app.ensemble.is_some() { 24 } else { 17 };
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(structure_panel_height),
            Constraint::Min(8),
        ])
        .split(root[1]);

    let viewport_block = Block::default()
        .borders(Borders::ALL)
        .padding(Padding::new(1, 1, 0, 0))
        .title(viewport_title(app));
    let viewport_inner = viewport_block.inner(root[0]);
    let viewport = render_primary_viewport_text(
        app,
        viewport_inner.width as usize,
        viewport_inner.height as usize,
    );
    let viewport_paragraph = Paragraph::new(viewport).block(viewport_block);
    frame.render_widget(viewport_paragraph, root[0]);

    let structure_block = Block::default()
        .borders(Borders::ALL)
        .padding(Padding::new(1, 1, 0, 0))
        .title("Structure");
    let side = Paragraph::new(structure_panel_lines(app))
        .block(structure_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(side, right[0]);

    let controls_block = Block::default()
        .borders(Borders::ALL)
        .padding(Padding::new(1, 1, 0, 0))
        .title("Controls");
    let view_panel = Paragraph::new(controls_keys_panel_lines(app))
        .block(controls_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(view_panel, right[1]);

    if let Some(xrd_rect) = xrd_area {
        draw_xrd_panel(frame, app, xrd_rect);
    }

    if let Some(file_picker) = &app.file_picker {
        draw_file_picker(frame, file_picker);
    }

    if let Some(editor) = &app.editor {
        draw_editor(frame, editor, root[0]);
    }
}

fn draw_xrd_panel(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let pattern = &app.cached_xrd_pattern;
    let (wl_name, _) = WAVELENGTH_PRESETS[app.xrd_wavelength_idx];
    let target_distance = app
        .selected_ensemble_entry()
        .and_then(|entry| entry.distance_to_target);
    let title = if let Some(distance) = target_distance {
        format!(
            " XRD  {wl_name} {:.4}Å  dXRD(Cu)={distance:.4} ",
            pattern.wavelength
        )
    } else {
        format!(" XRD  {wl_name} {:.4}Å ", pattern.wavelength)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .padding(Padding::new(1, 1, 0, 0))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if pattern.peaks.is_empty() {
        let msg = if app.structure.cell.is_none() {
            "No unit cell — XRD requires fractional coordinates"
        } else {
            "No diffraction peaks in range"
        };
        frame.render_widget(
            Paragraph::new(Span::styled(msg, Style::default().fg(Color::DarkGray))),
            inner,
        );
        return;
    }

    let inner_w = inner.width as usize;
    let inner_h = inner.height as usize;
    if inner_w < 20 || inner_h < 4 {
        return;
    }

    // Layout (rows, top to bottom):
    //   CHART_ROWS   stick bars with y-axis labels
    //   1            baseline "     └──┴──────┴─── 2θ"
    //   1            2θ tick labels
    //   remaining    peak table
    const CHART_ROWS: usize = 5;
    const AXIS_ROWS: usize = 2;
    let table_rows = inner_h.saturating_sub(CHART_ROWS + AXIS_ROWS);

    // Left margin: "100% " (5 chars) + "│" (1 char) = 6 cols
    const Y_LABEL_W: usize = 5;
    const Y_SEP: usize = 1;
    let plot_w = inner_w.saturating_sub(Y_LABEL_W + Y_SEP);
    if plot_w == 0 {
        return;
    }

    let theta_min = 5.0_f32;
    let theta_max = XRD_TWO_THETA_MAX;
    let theta_to_col = |t: f32| -> usize {
        ((t - theta_min) / (theta_max - theta_min) * (plot_w as f32 - 1.0))
            .round()
            .clamp(0.0, (plot_w - 1) as f32) as usize
    };

    // Assign labels 1–5 to the 5 tallest peaks (by 2θ order among the top-5).
    // All other peaks appear in the chart with ▲ (label = 0).
    let mut top_indices: Vec<usize> = (0..pattern.peaks.len()).collect();
    top_indices.sort_by(|&a, &b| {
        pattern.peaks[b]
            .intensity
            .partial_cmp(&pattern.peaks[a].intensity)
            .unwrap()
    });
    top_indices.truncate(5);
    top_indices.sort_by(|&a, &b| {
        pattern.peaks[a]
            .two_theta
            .partial_cmp(&pattern.peaks[b].two_theta)
            .unwrap()
    });
    let mut peak_labels = vec![0usize; pattern.peaks.len()]; // 0 = ▲
    for (lbl, &i) in top_indices.iter().enumerate() {
        peak_labels[i] = lbl + 1;
    }
    let top_peaks: Vec<(usize, &crate::xrd::BraggPeak)> = top_indices
        .iter()
        .map(|&i| (peak_labels[i], &pattern.peaks[i]))
        .collect();

    // For each plot column, keep the highest-intensity peak that lands there.
    let mut col_data: Vec<Option<(usize, f32)>> = vec![None; plot_w]; // (label 0=▲ or 1–5, intensity)
    for (idx, peak) in pattern.peaks.iter().enumerate() {
        let label = peak_labels[idx];
        let col = theta_to_col(peak.two_theta);
        match col_data[col] {
            None => col_data[col] = Some((label, peak.intensity)),
            Some((_, prev_i)) if peak.intensity > prev_i => {
                col_data[col] = Some((label, peak.intensity));
            }
            _ => {}
        }
    }

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(inner_h);

    // ── Stick chart rows ─────────────────────────────────────────────────────
    for row in 0..CHART_ROWS {
        let y_label = match row {
            0 => "100% ",
            r if r == CHART_ROWS / 2 => " 50% ",
            r if r == CHART_ROWS - 1 => "  0% ",
            _ => "     ",
        };
        let mut spans: Vec<Span<'static>> = vec![
            Span::styled(y_label, Style::default().fg(Color::DarkGray)),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
        ];

        for col in 0..plot_w {
            let (ch, color) = match col_data[col] {
                None => (' ', Color::Reset),
                Some((label, intensity)) => {
                    // top_row: first row (from top) where the bar is drawn.
                    // Clamped so even a 1% peak shows one row.
                    let top_row = ((1.0 - intensity / 100.0) * CHART_ROWS as f32).floor() as usize;
                    let top_row = top_row.min(CHART_ROWS - 1);
                    let color = if intensity >= 50.0 {
                        Color::LightCyan
                    } else if intensity >= 15.0 {
                        Color::Cyan
                    } else {
                        Color::DarkGray
                    };
                    let ch = if row < top_row {
                        ' '
                    } else if row == top_row {
                        // Peak tip: numbered for top-5, ▲ for all others.
                        if label > 0 {
                            char::from_digit(label as u32, 10).unwrap_or('▲')
                        } else {
                            '▲'
                        }
                    } else {
                        '│'
                    };
                    (ch, color)
                }
            };

            if ch == ' ' {
                spans.push(Span::raw(" "));
            } else {
                // Numbered tips are bold white; ▲ and bars use the peak's color.
                let style = if ch.is_ascii_digit() {
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(color)
                };
                spans.push(Span::styled(ch.to_string(), style));
            }
        }
        lines.push(Line::from(spans));
    }

    // ── Baseline "     └──┴──────────────────────── 2θ" ────────────────────
    {
        let prefix = " ".repeat(Y_LABEL_W) + "└";
        let ticks: String = (0..plot_w)
            .map(|col| {
                if col_data[col].is_some() {
                    '┴'
                } else {
                    '─'
                }
            })
            .collect();
        let suffix = " 2θ";
        let full = format!("{prefix}{ticks}{suffix}");
        lines.push(Line::from(Span::styled(
            full,
            Style::default().fg(Color::DarkGray),
        )));
    }

    // ── 2θ tick labels "      10°   20°   30°   40°   50°   60°   70°   80°" ─
    {
        let total = Y_LABEL_W + Y_SEP + plot_w;
        let mut label_buf: Vec<u8> = vec![b' '; total];
        for tick in (20u32..=160).step_by(20) {
            let col = theta_to_col(tick as f32);
            let label = format!("{tick}°");
            let start = (Y_LABEL_W + Y_SEP + col).saturating_sub(label.len() / 2);
            for (i, byte) in label.bytes().enumerate() {
                let pos = start + i;
                if pos < label_buf.len() {
                    label_buf[pos] = byte;
                }
            }
        }
        let s = String::from_utf8_lossy(&label_buf).into_owned();
        lines.push(Line::from(Span::styled(
            s,
            Style::default().fg(Color::DarkGray),
        )));
    }

    // ── Peak table ───────────────────────────────────────────────────────────
    let mut used_table_rows = 0usize;
    if table_rows > 0 {
        if let Some(ensemble) = &app.ensemble {
            if ensemble.has_target() {
                let target_name = ensemble
                    .target_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
                    .unwrap_or("target.cif");
                let state = if app.show_target_overlay { "on" } else { "off" };
                let distance_text = target_distance
                    .map(|distance| format!("  dXRD(Cu)={distance:.4}"))
                    .unwrap_or_default();
                lines.push(Line::from(vec![
                    Span::styled("target ", Style::default().fg(Color::LightMagenta)),
                    Span::styled(
                        elide_right(target_name, 20),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled("  [Y] ", Style::default().fg(Color::DarkGray)),
                    Span::styled(state, Style::default().fg(Color::LightYellow)),
                    Span::styled(distance_text, Style::default().fg(Color::LightCyan)),
                ]));
                used_table_rows += 1;

                if app.show_target_overlay && table_rows > used_table_rows {
                    if let Some(target_pattern) = &ensemble.target_pattern {
                        let mut strongest = target_pattern.peaks.clone();
                        strongest.sort_by(|a, b| {
                            b.intensity
                                .partial_cmp(&a.intensity)
                                .unwrap_or(Ordering::Equal)
                        });
                        strongest.truncate(3);
                        strongest.sort_by(|a, b| {
                            a.two_theta
                                .partial_cmp(&b.two_theta)
                                .unwrap_or(Ordering::Equal)
                        });
                        let summary = strongest
                            .iter()
                            .map(|peak| format!("{:.1}°", peak.two_theta))
                            .collect::<Vec<_>>()
                            .join("  ");
                        lines.push(Line::from(vec![
                            Span::styled("peaks ", Style::default().fg(Color::LightMagenta)),
                            Span::styled(
                                if summary.is_empty() {
                                    "-".to_string()
                                } else {
                                    summary
                                },
                                Style::default().fg(Color::White),
                            ),
                        ]));
                        used_table_rows += 1;
                    }
                }
            }
        }
    }

    // The 5 tallest peaks, labeled 1–5 left-to-right, matching chart tips.
    // Each row: "  N  44.7°  d=2.027Å  [ 1  1  0]  ██████████  100%"
    let bar_cap = (inner_w.saturating_sub(32)).clamp(4, 24);
    for &(label, peak) in top_peaks
        .iter()
        .take(table_rows.saturating_sub(used_table_rows))
    {
        let [h, k, l] = peak.hkl;
        let bar_len = ((peak.intensity / 100.0) * bar_cap as f32).round() as usize;
        let bar = "█".repeat(bar_len);
        let bar_color = if peak.intensity >= 50.0 {
            Color::LightCyan
        } else {
            Color::Cyan
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{label:2} "),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:5.1}°", peak.two_theta),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled(
                format!("  d={:.3}Å", peak.d_spacing),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(
                format!("  [{:2} {:2} {:2}]", h, k, l),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("  ", Style::default()),
            Span::styled(bar, Style::default().fg(bar_color)),
            Span::styled(
                format!("  {:.0}%", peak.intensity),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_file_picker(frame: &mut ratatui::Frame, picker: &FilePickerState) {
    let popup = centered_rect_sized(68, 22, frame.area());
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .padding(Padding::new(1, 1, 0, 0))
        .title(" CIF Browser ");
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let max_inner_width = inner.width.saturating_sub(2) as usize;
    let cwd_display = elide_from_left(&picker.cwd.display().to_string(), max_inner_width);
    let mut lines: Vec<Line<'static>> = vec![
        Line::from(Span::styled(
            format!("cwd {}", cwd_display),
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "j/k move  Enter open  <- parent  Esc close",
            Style::default().fg(Color::Gray),
        )),
        Line::default(),
    ];

    let has_error = picker.error.is_some();
    let top_reserved = 3usize;
    let bottom_reserved = if has_error { 3usize } else { 1usize };
    let max_entries = inner
        .height
        .saturating_sub((top_reserved + bottom_reserved) as u16) as usize;

    if picker.entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no folders or .cif files)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let visible_count = max_entries.max(1).min(picker.entries.len());
        let mut start = picker.selected.saturating_sub(visible_count / 2);
        if start + visible_count > picker.entries.len() {
            start = picker.entries.len().saturating_sub(visible_count);
        }
        let end = (start + visible_count).min(picker.entries.len());

        if start > 0 {
            lines.push(Line::from(Span::styled(
                "  ...",
                Style::default().fg(Color::DarkGray),
            )));
        }

        for (idx, entry) in picker.entries[start..end].iter().enumerate() {
            let absolute_idx = start + idx;
            let is_last_visible =
                absolute_idx + 1 == picker.entries.len() || idx + 1 == end - start;
            let selected = absolute_idx == picker.selected;
            let marker = if selected { "❯" } else { " " };
            let branch = if is_last_visible { "└─" } else { "├─" };
            let item = if entry.is_dir {
                format!("{branch} ▸ {}/", entry.name)
            } else {
                format!("{branch} ◦ {}", entry.name)
            };
            let item = elide_right(&item, max_inner_width.saturating_sub(4));
            let color = if entry.is_dir {
                Color::LightCyan
            } else {
                Color::LightGreen
            };
            let style = if selected {
                Style::default()
                    .fg(color)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(color)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{marker} "),
                    Style::default().fg(Color::LightYellow),
                ),
                Span::styled(item, style),
            ]));
        }

        if end < picker.entries.len() {
            lines.push(Line::from(Span::styled(
                "  ...",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    if let Some(error) = &picker.error {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            elide_right(&format!("error: {error}"), max_inner_width),
            Style::default().fg(Color::LightRed),
        )));
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn draw_editor(frame: &mut ratatui::Frame, editor: &EditorState, viewport: Rect) {
    let popup = bottom_right_rect(71, 20, viewport);
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .padding(Padding::new(1, 1, 0, 0))
        .title(" Structure Editor ");
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let max_w = inner.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Helper: render a labeled field with [ ] brackets, showing cursor if active.
    let field_line = |label: &str,
                      value: &str,
                      is_focused: bool,
                      is_editing: bool,
                      cursor_pos: usize,
                      field_w: usize|
     -> Vec<Span<'static>> {
        let label_style = Style::default().fg(Color::Cyan);
        let mut spans: Vec<Span<'static>> = if label.is_empty() {
            vec![]
        } else {
            vec![Span::styled(format!("{label}: "), label_style)]
        };
        let bracket_style = if is_focused {
            Style::default().fg(Color::LightYellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let edit_bracket_style = if is_focused && is_editing {
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightYellow)
                .add_modifier(Modifier::BOLD)
        } else {
            bracket_style
        };
        spans.push(Span::styled("[", edit_bracket_style));
        let inner_w = field_w.saturating_sub(2);
        if is_focused && is_editing {
            let mut display = value.to_string();
            let cursor = cursor_pos.min(display.len());
            display.insert(cursor, '_');
            // Elide from left so cursor remains visible at end.
            let display = elide_from_left(&display, inner_w);
            let padded = format!("{:<width$}", display, width = inner_w);
            spans.push(Span::styled(
                padded,
                Style::default().fg(Color::Black).bg(Color::LightYellow),
            ));
        } else {
            let display = elide_right(value, inner_w);
            let padded = format!("{:<width$}", display, width = inner_w);
            let style = if is_focused {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            spans.push(Span::styled(padded, style));
        }
        spans.push(Span::styled("]", edit_bracket_style));
        spans
    };

    // Title line.
    {
        let is_f = editor.focus == EditorFocus::Title;
        let title_w = (max_w / 2).max(16);
        let mut spans = field_line(
            "Title",
            &editor.draft_title,
            is_f,
            editor.editing,
            editor.cursor_pos,
            title_w,
        );
        lines.push(Line::from(spans.drain(..).collect::<Vec<_>>()));
    }

    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "  CELL PARAMETERS",
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
    )));

    // Cell a/b/c on one line.
    {
        let fw = 12usize;
        let is_a = editor.focus == EditorFocus::CellA;
        let is_b = editor.focus == EditorFocus::CellB;
        let is_c = editor.focus == EditorFocus::CellC;
        let mut spans = vec![Span::styled(" a ", Style::default().fg(Color::Red))];
        spans.extend(field_line(
            "",
            &editor.draft_a,
            is_a,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled("b ", Style::default().fg(Color::Green)));
        spans.extend(field_line(
            "",
            &editor.draft_b,
            is_b,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled("c ", Style::default().fg(Color::Blue)));
        spans.extend(field_line(
            "",
            &editor.draft_c,
            is_c,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::styled(" Å", Style::default().fg(Color::DarkGray)));
        lines.push(Line::from(spans));
    }

    // Cell alpha/beta/gamma on one line.
    {
        let fw = 12usize;
        let is_al = editor.focus == EditorFocus::CellAlpha;
        let is_be = editor.focus == EditorFocus::CellBeta;
        let is_ga = editor.focus == EditorFocus::CellGamma;
        let mut spans = vec![Span::styled(" α ", Style::default().fg(Color::LightRed))];
        spans.extend(field_line(
            "",
            &editor.draft_alpha,
            is_al,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled("β ", Style::default().fg(Color::LightGreen)));
        spans.extend(field_line(
            "",
            &editor.draft_beta,
            is_be,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::raw("  "));
        spans.push(Span::styled("γ ", Style::default().fg(Color::LightBlue)));
        spans.extend(field_line(
            "",
            &editor.draft_gamma,
            is_ga,
            editor.editing,
            editor.cursor_pos,
            fw,
        ));
        spans.push(Span::styled(" °", Style::default().fg(Color::DarkGray)));
        lines.push(Line::from(spans));
    }

    // Space group line.
    {
        let is_f = editor.focus == EditorFocus::SpaceGroup;
        let mut spans = vec![Span::styled(
            " Space group: ",
            Style::default().fg(Color::Cyan),
        )];
        spans.extend(field_line(
            "",
            &editor.draft_space_group,
            is_f,
            editor.editing,
            editor.cursor_pos,
            16,
        ));
        let sg_info = match &editor.resolved_sg_name {
            Some(name) => format!("  → {} ({} ops)", name, editor.resolved_sg_ops.len()),
            None => format!("  → {} ops (custom)", editor.resolved_sg_ops.len()),
        };
        spans.push(Span::styled(sg_info, Style::default().fg(Color::DarkGray)));
        lines.push(Line::from(spans));
    }

    lines.push(Line::default());
    {
        let header = Line::from(vec![
            Span::styled(
                "  ASYMMETRIC UNIT",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("                               ", Style::default()),
            Span::styled("[A] Add  [D] Del", Style::default().fg(Color::DarkGray)),
        ]);
        lines.push(header);
    }
    lines.push(Line::from(Span::styled(
        "   #   Label   Elem        x              y              z",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(Span::styled(
        "  ─────────────────────────────────────────────────────────────",
        Style::default().fg(Color::DarkGray),
    )));

    // Atom rows — show up to EDITOR_ATOM_ROWS rows with scroll indicators.
    let scroll = editor.atom_scroll;
    let end = (scroll + EDITOR_ATOM_ROWS).min(editor.draft_atoms.len());
    let has_above = scroll > 0;
    let has_below = end < editor.draft_atoms.len();
    for i in scroll..end {
        let atom = &editor.draft_atoms[i];
        let is_selected = editor.focus.atom_row() == Some(i);
        let marker = if is_selected { "▶" } else { " " };
        let row_style = if is_selected {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(Color::Gray)
        };

        let fw_label = 8usize;
        let fw_elem = 6usize;
        let fw_coord = 14usize;

        let scroll_hint = if i == scroll && has_above {
            "↑"
        } else if i == end - 1 && has_below {
            "↓"
        } else {
            " "
        };
        let num_style = if is_selected {
            Style::default().fg(Color::LightYellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let mut spans = vec![
            Span::styled(format!("{marker}{:>3}", i + 1), num_style),
            Span::styled(scroll_hint, Style::default().fg(Color::Cyan)),
            Span::raw(" "),
        ];

        let is_label = editor.focus == EditorFocus::AtomLabel(i);
        let is_elem = editor.focus == EditorFocus::AtomElement(i);
        let is_x = editor.focus == EditorFocus::AtomX(i);
        let is_y = editor.focus == EditorFocus::AtomY(i);
        let is_z = editor.focus == EditorFocus::AtomZ(i);

        spans.extend(field_line(
            "",
            &atom.label,
            is_label,
            editor.editing,
            editor.cursor_pos,
            fw_label,
        ));
        spans.push(Span::raw(" "));
        spans.extend(field_line(
            "",
            &atom.element,
            is_elem,
            editor.editing,
            editor.cursor_pos,
            fw_elem,
        ));
        spans.push(Span::raw(" "));
        spans.extend(field_line(
            "",
            &atom.x,
            is_x,
            editor.editing,
            editor.cursor_pos,
            fw_coord,
        ));
        spans.push(Span::raw(" "));
        spans.extend(field_line(
            "",
            &atom.y,
            is_y,
            editor.editing,
            editor.cursor_pos,
            fw_coord,
        ));
        spans.push(Span::raw(" "));
        spans.extend(field_line(
            "",
            &atom.z,
            is_z,
            editor.editing,
            editor.cursor_pos,
            fw_coord,
        ));

        lines.push(Line::from(spans));
        let _ = row_style; // suppress warning
    }

    if editor.draft_atoms.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no atoms — press A to add)",
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines.push(Line::default());

    // Expanded atom count.
    {
        let expanded_count = if let Ok(s) = editor.build_structure() {
            s.atoms.len()
        } else {
            0
        };
        let sg_note = editor.resolved_sg_name.as_deref().unwrap_or("custom ops");
        lines.push(Line::from(Span::styled(
            format!("  Expanded: {expanded_count} atoms ({sg_note} symmetry)"),
            Style::default().fg(Color::DarkGray),
        )));
    }

    lines.push(Line::from(vec![
        Span::styled(
            "[Enter]",
            Style::default().fg(Color::LightYellow).bg(Color::DarkGray),
        ),
        Span::raw(" Edit  "),
        Span::styled(
            "[Tab]",
            Style::default().fg(Color::LightYellow).bg(Color::DarkGray),
        ),
        Span::raw(" Next  "),
        Span::styled(
            "[^A]",
            Style::default().fg(Color::LightYellow).bg(Color::DarkGray),
        ),
        Span::raw(" Apply  "),
        Span::styled(
            "[^S]",
            Style::default().fg(Color::LightYellow).bg(Color::DarkGray),
        ),
        Span::raw(" Save  "),
        Span::styled(
            "[Esc]",
            Style::default().fg(Color::LightYellow).bg(Color::DarkGray),
        ),
        Span::raw(" Close"),
    ]));

    if let Some(status) = &editor.status {
        lines.push(Line::from(Span::styled(
            format!(
                "  Status: {}",
                elide_right(status, max_w.saturating_sub(10))
            ),
            Style::default().fg(Color::LightGreen),
        )));
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn bottom_right_rect(width: u16, height: u16, area: Rect) -> Rect {
    let margin = 1u16;
    let usable_w = area.width.saturating_sub(margin * 2).max(10);
    let usable_h = area.height.saturating_sub(margin * 2).max(8);
    let w = width.min(usable_w);
    let h = height.min(usable_h);
    Rect {
        x: area.x + area.width.saturating_sub(w + margin),
        y: area.y + area.height.saturating_sub(h + margin),
        width: w,
        height: h,
    }
}

fn centered_rect_sized(width: u16, height: u16, area: Rect) -> Rect {
    let usable_width = area.width.saturating_sub(2).max(10);
    let usable_height = area.height.saturating_sub(2).max(8);
    let width = width.min(usable_width).max(10);
    let height = height.min(usable_height).max(8);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn elide_right(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let keep = max_chars - 3;
    let mut out = String::new();
    for ch in text.chars().take(keep) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn elide_from_left(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let keep = max_chars - 3;
    let tail: String = text
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("...{tail}")
}

fn xrd_pattern_fingerprint(pattern: &XrdPattern, bins: usize) -> Vec<f32> {
    if bins == 0 {
        return Vec::new();
    }
    let theta_min = 0.0_f32;
    let theta_span = (XRD_TWO_THETA_MAX - theta_min).max(f32::EPSILON);
    let mut fingerprint = vec![0.0_f32; bins];

    for peak in &pattern.peaks {
        if peak.two_theta < theta_min || peak.two_theta > XRD_TWO_THETA_MAX {
            continue;
        }
        let ratio = (peak.two_theta - theta_min) / theta_span;
        let col = (ratio * (bins.saturating_sub(1) as f32))
            .round()
            .clamp(0.0, bins.saturating_sub(1) as f32) as usize;
        fingerprint[col] += peak.intensity.max(0.0);
    }

    let norm_sq: f32 = fingerprint.iter().map(|value| value * value).sum();
    if norm_sq <= f32::EPSILON {
        return fingerprint;
    }
    let norm = norm_sq.sqrt();
    for value in &mut fingerprint {
        *value /= norm;
    }
    fingerprint
}

fn cosine_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let lhs_norm_sq: f32 = lhs.iter().map(|value| value * value).sum();
    let rhs_norm_sq: f32 = rhs.iter().map(|value| value * value).sum();
    let lhs_nonzero = lhs_norm_sq > f32::EPSILON;
    let rhs_nonzero = rhs_norm_sq > f32::EPSILON;
    match (lhs_nonzero, rhs_nonzero) {
        (false, false) => return 0.0,
        (false, true) | (true, false) => return 1.0,
        (true, true) => {}
    }

    let dot = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let sim = dot / (lhs_norm_sq.sqrt() * rhs_norm_sq.sqrt());
    (1.0 - sim).clamp(0.0, 2.0)
}

fn compute_ensemble_novelty(entries: &mut [EnsembleEntry]) {
    if entries.is_empty() {
        return;
    }
    if entries.len() == 1 {
        entries[0].novelty = 0.0;
        return;
    }

    let fingerprints: Vec<Vec<f32>> = entries
        .iter()
        .map(|entry| entry.fingerprint.clone())
        .collect();
    let count = entries.len();

    for idx in 0..count {
        let mut sum = 0.0_f32;
        for other_idx in 0..count {
            if idx == other_idx {
                continue;
            }
            sum += cosine_distance(&fingerprints[idx], &fingerprints[other_idx]);
        }
        entries[idx].novelty = sum / (count as f32 - 1.0);
    }
}

fn file_name_key(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_else(|| path.to_string_lossy().to_ascii_lowercase())
}

fn structure_panel_lines(app: &App) -> Vec<Line<'static>> {
    let selected = app.structure.atoms.get(app.selected_atom);
    let total_atoms = app.structure.atoms.len();
    let element_counts = &app.cached_element_counts;
    let formula = app.cached_formula.clone();

    let mut lines = vec![
        Line::from(Span::styled(
            app.structure.title.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        kv_line("Formula", formula, Color::Cyan, Color::LightYellow),
        kv_line(
            "Group",
            app.structure
                .space_group
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            Color::Cyan,
            Color::LightBlue,
        ),
        elements_line(element_counts),
    ];

    if let Some(ensemble) = &app.ensemble {
        lines.push(Line::default());
        lines.push(section_header("ENSEMBLE"));
        lines.push(kv_line(
            "Source",
            elide_from_left(&ensemble.source_dir.to_string_lossy(), 34),
            Color::Cyan,
            Color::Gray,
        ));
        lines.push(kv_line(
            "Sample",
            format!(
                "{}/{}",
                ensemble.selected_rank.saturating_add(1),
                ensemble.order.len()
            ),
            Color::Cyan,
            Color::LightGreen,
        ));
        lines.push(kv_line(
            "Sort",
            ensemble.sort_mode.label().to_string(),
            Color::Cyan,
            Color::LightMagenta,
        ));
        if let Some(entry) = ensemble.selected_entry() {
            let sample_name = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            lines.push(kv_line(
                "File",
                elide_right(sample_name, 30),
                Color::Cyan,
                Color::White,
            ));
            lines.push(kv_line(
                "Novelty",
                format!("{:.4}", entry.novelty),
                Color::Cyan,
                Color::LightBlue,
            ));
            if let Some(distance) = entry.distance_to_target {
                lines.push(kv_line(
                    "dXRD(Cu)",
                    format!("{distance:.4}"),
                    Color::Cyan,
                    Color::LightYellow,
                ));
            }
        }
        if ensemble.skipped_files > 0 {
            lines.push(kv_line(
                "Skipped",
                ensemble.skipped_files.to_string(),
                Color::Cyan,
                Color::LightRed,
            ));
        }
    }

    lines.push(Line::default());
    lines.push(section_header("LATTICE"));

    if let Some(cell) = app.structure.cell {
        lines.push(Line::from(vec![
            Span::styled("a ", Style::default().fg(Color::Red)),
            Span::styled(format!("{:.3}", cell.a), Style::default().fg(Color::White)),
            Span::styled("  b ", Style::default().fg(Color::Green)),
            Span::styled(format!("{:.3}", cell.b), Style::default().fg(Color::White)),
            Span::styled("  c ", Style::default().fg(Color::Blue)),
            Span::styled(format!("{:.3}", cell.c), Style::default().fg(Color::White)),
            Span::styled(" Å", Style::default().fg(Color::Gray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("α ", Style::default().fg(Color::LightRed)),
            Span::styled(
                format!("{:.1}", cell.alpha_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled("  β ", Style::default().fg(Color::LightGreen)),
            Span::styled(
                format!("{:.1}", cell.beta_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled("  γ ", Style::default().fg(Color::LightBlue)),
            Span::styled(
                format!("{:.1}", cell.gamma_deg),
                Style::default().fg(Color::LightYellow),
            ),
            Span::styled("°", Style::default().fg(Color::Gray)),
        ]));
        lines.push(kv_line(
            "Vol",
            format!("{:.1} Å³", cell_volume(cell)),
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
    lines.push(section_header("ATOM"));

    if let Some(atom) = selected {
        let atom_col = atom_color(&atom.element, false);
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} / {}  ", app.selected_atom + 1, total_atoms),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(
                atom.label.clone(),
                Style::default().fg(atom_col).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default()),
            Span::styled(atom.element.clone(), Style::default().fg(atom_col)),
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
            Span::styled(" Å", Style::default().fg(Color::Gray)),
        ]));
        if let Some(frac) = atom.fractional {
            lines.push(Line::from(vec![
                Span::styled("frac ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.3}", frac[0]), Style::default().fg(Color::White)),
                Span::styled("  ", Style::default()),
                Span::styled(format!("{:.3}", frac[1]), Style::default().fg(Color::White)),
                Span::styled("  ", Style::default()),
                Span::styled(format!("{:.3}", frac[2]), Style::default().fg(Color::White)),
            ]));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "none",
            Style::default().fg(Color::DarkGray),
        )));
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
    let has_target = app
        .ensemble
        .as_ref()
        .and_then(|ensemble| ensemble.target_pattern.as_ref())
        .is_some();

    vec![
        section_header("VIEW"),
        control_short_value_line("Shift+O", "Open", "CIF dialog".to_string(), Color::Gray),
        control_short_value_line(
            "g",
            "Theme",
            app.render_theme.label().to_string(),
            theme_color(app.render_theme),
        ),
        control_short_bool_line("R", "Spin lock", app.spin_lock),
        control_short_value_line(
            "<>",
            "Spin speed",
            format!("{:.2}x", app.spin_speed),
            Color::LightYellow,
        ),
        control_short_bool_line("b", "Bonds", app.show_bonds),
        control_short_bool_line("c", "Cell", app.show_cell),
        control_short_bool_line("x", "Cell overlay", app.cell_on_top),
        control_short_bool_line("v", "Gizmo", app.show_orientation_gizmo),
        control_short_bool_line("L", "Labels", app.show_labels),
        section_header("IMAGES"),
        control_short_value_line(
            "r",
            "Periodic",
            if app.show_boundary_images {
                "on".to_string()
            } else {
                "off".to_string()
            },
            boundary_color,
        ),
        control_short_value_line(
            "t",
            "Bonded",
            if app.show_bonded_images {
                "on".to_string()
            } else {
                "off".to_string()
            },
            bonded_color,
        ),
        section_header("CAMERA"),
        control_short_value_line(
            "+/-",
            "Zoom",
            format!("{:.2}", app.zoom),
            Color::LightYellow,
        ),
        control_short_value_line(
            ",/.",
            "FOV",
            format!("{:.1}°", app.fov_deg),
            Color::LightYellow,
        ),
        control_short_bool_line("z", "Size lock", app.lock_fov_zoom),
        control_short_value_line(
            "wasd",
            "Pan",
            format!("{:.2}  {:.2}", app.pan[0], app.pan[1]),
            Color::White,
        ),
        control_short_value_line(
            "hjkl/uo",
            "Look",
            {
                let d = app.rot_mat[2];
                format!("{:.2}  {:.2}  {:.2}", d[0], d[1], d[2])
            },
            Color::White,
        ),
        section_header("RENDER"),
        control_short_value_line(
            "[/]",
            "Sphere",
            format!("{:.2}", app.sphere_scale),
            Color::LightYellow,
        ),
        control_short_value_line(
            "nm",
            "Bond Å",
            format!("{:.2}", app.bond_max_distance),
            Color::LightYellow,
        ),
        control_short_value_line(
            "Tab",
            "Atom",
            format!("{}/{}", app.selected_atom + 1, app.structure.atoms.len()),
            Color::LightGreen,
        ),
        section_header("DIFFRACTION"),
        control_short_bool_line("X", "XRD panel", app.show_xrd),
        control_short_value_line(
            "W",
            "Source",
            WAVELENGTH_PRESETS[app.xrd_wavelength_idx].0.to_string(),
            Color::LightCyan,
        ),
        control_short_bool_line("Y", "Target ref", app.show_target_overlay && has_target),
        section_header("ENSEMBLE"),
        control_short_value_line("↑↓", "Sample", String::new(), Color::Gray),
        control_short_value_line("Pg↑↓", "Sample ±5", String::new(), Color::Gray),
        control_short_value_line("P", "Cycle sort", String::new(), Color::Gray),
        control_short_bool_line(
            "G",
            "Grid view",
            app.show_ensemble_grid && app.ensemble.is_some(),
        ),
        section_header("EDITOR"),
        control_short_value_line("E", "Edit struct", String::new(), Color::Gray),
        control_short_value_line("F", "New struct", String::new(), Color::Gray),
    ]
}

fn section_header(label: &str) -> Line<'static> {
    Line::from(Span::styled(
        format!("  {label}"),
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
    ))
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
            format!("{key}"),
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
        RenderTheme::Wild => Color::LightRed,
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
    if out.is_empty() {
        "-".to_string()
    } else {
        out
    }
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

fn viewport_title(app: &App) -> String {
    if let Some((start, end)) = app.ensemble_grid_page_bounds() {
        if let Some(ensemble) = &app.ensemble {
            return format!(
                "3D View · Ensemble Grid {}-{} / {}",
                start + 1,
                end,
                ensemble.order.len()
            );
        }
    }
    "3D View".to_string()
}

fn render_primary_viewport_text(app: &App, width: usize, height: usize) -> Text<'static> {
    if let Some(grid_text) = render_ensemble_grid_text(app, width, height) {
        return grid_text;
    }
    render_viewport_text(app, width, height)
}

fn render_ensemble_grid_text(app: &App, width: usize, height: usize) -> Option<Text<'static>> {
    if width == 0 || height == 0 {
        return Some(Text::default());
    }
    let ensemble = app.ensemble.as_ref()?;
    let (start, end) = app.ensemble_grid_page_bounds()?;
    let visible = &ensemble.order[start..end];
    let count = visible.len();
    if count == 0 {
        return None;
    }

    let columns = ENSEMBLE_GRID_COLUMNS.min(count.max(1));
    let rows = (count + columns - 1) / columns;
    let separator_cols = columns.saturating_sub(1);
    let separator_rows = rows.saturating_sub(1);
    if width <= separator_cols || height <= separator_rows {
        return None;
    }
    let tile_w = (width - separator_cols) / columns;
    let tile_h = (height - separator_rows) / rows;
    if tile_w < 12 || tile_h < 6 {
        return None;
    }

    let mut chars = vec![' '; width * height];
    let mut colors = vec![Color::Reset; width * height];

    for col_idx in 1..columns {
        let col = col_idx * tile_w + (col_idx - 1);
        if col >= width {
            continue;
        }
        for row in 0..height {
            let idx = row * width + col;
            chars[idx] = '│';
            colors[idx] = Color::DarkGray;
        }
    }
    for row_idx in 1..rows {
        let row = row_idx * tile_h + (row_idx - 1);
        if row >= height {
            continue;
        }
        for col in 0..width {
            let idx = row * width + col;
            chars[idx] = '─';
            colors[idx] = Color::DarkGray;
        }
    }

    let selected_entry_idx = ensemble.selected_entry_index();
    for (slot_idx, &entry_idx) in visible.iter().enumerate() {
        let row_idx = slot_idx / columns;
        let col_idx = slot_idx % columns;
        let x0 = col_idx * (tile_w + 1);
        let y0 = row_idx * (tile_h + 1);
        if x0 + tile_w > width || y0 + tile_h > height {
            continue;
        }

        let entry = &ensemble.entries[entry_idx];
        let view = &entry.view_state;
        let scene = build_scene(
            &entry.structure,
            view.show_boundary_images,
            view.show_bonded_images,
            view.bond_max_distance,
        );
        let base_scale = bounding_sphere_scale(&scene);
        let tile_buffer = render_viewport_buffer_with_config(
            &ViewportRenderConfig {
                structure: &entry.structure,
                scene: &scene,
                base_scale,
                rot_mat: view.rot_mat,
                zoom: view.zoom,
                fov_deg: view.fov_deg,
                pan: view.pan,
                sphere_scale: view.sphere_scale,
                show_bonds: view.show_bonds,
                show_cell: view.show_cell,
                cell_on_top: view.cell_on_top,
                show_labels: false,
                show_orientation_gizmo: false,
                render_theme: view.render_theme,
                selected_atom: None,
                empty_hint: "",
            },
            tile_w,
            tile_h,
        );
        blit_viewport_buffer(
            &tile_buffer,
            &mut chars,
            &mut colors,
            width,
            x0,
            y0,
            tile_w,
            tile_h,
        );

        let is_selected_tile = selected_entry_idx == Some(entry_idx);
        if is_selected_tile {
            draw_buffer_rect_border(
                &mut chars,
                &mut colors,
                width,
                height,
                x0,
                y0,
                tile_w,
                tile_h,
                Color::LightYellow,
            );
        }

        let sample_name = entry
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown.cif");
        let rank = start + slot_idx + 1;
        let mut tile_label = format!("{rank}: {}", sample_name);
        if let Some(distance) = entry.distance_to_target {
            tile_label.push_str(&format!("  d={distance:.3}"));
        }
        tile_label = elide_right(&tile_label, tile_w.saturating_sub(2));
        let label_color = if is_selected_tile {
            Color::LightYellow
        } else {
            Color::Gray
        };
        let label_start_col = x0 + usize::from(is_selected_tile);
        let max_label_width = tile_w.saturating_sub(label_start_col.saturating_sub(x0));
        for (offset, ch) in tile_label.chars().enumerate() {
            if offset >= max_label_width {
                break;
            }
            put_buffer_char(
                &mut chars,
                &mut colors,
                width,
                height,
                label_start_col + offset,
                y0,
                ch,
                label_color,
            );
        }
    }

    Some(to_colored_text(&chars, &colors, width, height))
}

fn blit_viewport_buffer(
    tile: &ViewportBuffer,
    chars: &mut [char],
    colors: &mut [Color],
    target_width: usize,
    x0: usize,
    y0: usize,
    tile_w: usize,
    tile_h: usize,
) {
    for row in 0..tile_h {
        for col in 0..tile_w {
            let src_idx = row * tile_w + col;
            let dst_idx = (y0 + row) * target_width + (x0 + col);
            chars[dst_idx] = tile.chars[src_idx];
            colors[dst_idx] = tile.colors[src_idx];
        }
    }
}

fn draw_buffer_rect_border(
    chars: &mut [char],
    colors: &mut [Color],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    rect_w: usize,
    rect_h: usize,
    color: Color,
) {
    if rect_w < 2 || rect_h < 2 {
        return;
    }
    let x1 = x0 + rect_w - 1;
    let y1 = y0 + rect_h - 1;
    for col in x0..=x1 {
        put_buffer_char(chars, colors, width, height, col, y0, '─', color);
        put_buffer_char(chars, colors, width, height, col, y1, '─', color);
    }
    for row in y0..=y1 {
        put_buffer_char(chars, colors, width, height, x0, row, '│', color);
        put_buffer_char(chars, colors, width, height, x1, row, '│', color);
    }
    put_buffer_char(chars, colors, width, height, x0, y0, '┌', color);
    put_buffer_char(chars, colors, width, height, x1, y0, '┐', color);
    put_buffer_char(chars, colors, width, height, x0, y1, '└', color);
    put_buffer_char(chars, colors, width, height, x1, y1, '┘', color);
}

fn put_buffer_char(
    chars: &mut [char],
    colors: &mut [Color],
    width: usize,
    height: usize,
    col: usize,
    row: usize,
    ch: char,
    color: Color,
) {
    if col >= width || row >= height {
        return;
    }
    let idx = row * width + col;
    chars[idx] = ch;
    colors[idx] = color;
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

struct ViewportRenderConfig<'a> {
    structure: &'a Structure,
    scene: &'a SceneGeometry,
    base_scale: f32,
    rot_mat: [[f32; 3]; 3],
    zoom: f32,
    fov_deg: f32,
    pan: [f32; 2],
    sphere_scale: f32,
    show_bonds: bool,
    show_cell: bool,
    cell_on_top: bool,
    show_labels: bool,
    show_orientation_gizmo: bool,
    render_theme: RenderTheme,
    selected_atom: Option<usize>,
    empty_hint: &'a str,
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
/// for every projected point.
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
    render_viewport_buffer_with_config(
        &ViewportRenderConfig {
            structure: &app.structure,
            scene: &app.scene,
            base_scale: app.base_scale,
            rot_mat: app.rot_mat,
            zoom: app.zoom,
            fov_deg: app.fov_deg,
            pan: app.pan,
            sphere_scale: app.sphere_scale,
            show_bonds: app.show_bonds,
            show_cell: app.show_cell,
            cell_on_top: app.cell_on_top,
            show_labels: app.show_labels,
            show_orientation_gizmo: app.show_orientation_gizmo,
            render_theme: app.render_theme,
            selected_atom: Some(app.selected_atom),
            empty_hint: EMPTY_VIEW_HINT,
        },
        width,
        height,
    )
}

fn render_viewport_buffer_with_config(
    config: &ViewportRenderConfig<'_>,
    width: usize,
    height: usize,
) -> ViewportBuffer {
    if width == 0 || height == 0 {
        return ViewportBuffer {
            chars: Vec::new(),
            colors: Vec::new(),
        };
    }
    if config.scene.atoms.is_empty() {
        let message = config.empty_hint;
        let mut chars = vec![' '; width * height];
        let mut colors = vec![Color::Reset; width * height];
        let row = height / 2;
        let message_len = message.chars().count();
        let start_col = width.saturating_sub(message_len) / 2;
        for (offset, ch) in message.chars().enumerate() {
            let col = start_col + offset;
            if col >= width {
                break;
            }
            let idx = row * width + col;
            chars[idx] = ch;
            colors[idx] = LABEL_COLOR;
        }
        return ViewportBuffer { chars, colors };
    }

    let mut chars = vec![' '; width * height];
    let mut colors = vec![Color::Reset; width * height];
    let mut z_buffer = vec![f32::INFINITY; width * height];
    let scale = config.base_scale * config.zoom;
    let Some(viewport) = ViewportTransform::for_size(width, height) else {
        return ViewportBuffer { chars, colors };
    };
    // Build camera params once — avoids recomputing trig values per point.
    let camera = CameraParams::from_mat(
        config.scene.center,
        scale,
        config.rot_mat,
        config.fov_deg,
        config.pan,
    );

    if config.show_cell && !config.cell_on_top {
        for (start, end) in &config.scene.cell_edges {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                *start,
                *end,
                config.render_theme.cell_line_glyph(),
                config.render_theme.cell_line_color(),
                &camera,
                CELL_LINE_DEPTH_BIAS,
                false,
            );
        }
    }

    if config.show_bonds {
        for bond in &config.scene.bonds {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                bond.start,
                bond.end,
                config.render_theme.bond_line_glyph(),
                config.render_theme.bond_line_color(),
                &camera,
                BOND_LINE_DEPTH_BIAS,
                false,
            );
        }
    }

    // Rasterize atoms as sphere disks, back-to-front so the z-buffer correctly
    // resolves overlaps even at equal depth (painter's order as tie-breaker).
    let mut atoms_projected: Vec<(&RenderAtom, ProjectedPoint)> = config
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
        let element = config
            .structure
            .atoms
            .get(atom.base_index)
            .map(|a| a.element.as_str())
            .unwrap_or("X");
        let r_world = display_radius(element) * config.sphere_scale;
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

        let selected = config
            .selected_atom
            .map(|selected_atom| atom.base_index == selected_atom && !atom.is_image)
            .unwrap_or(false);
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
                chars[idx] = sphere_glyph(nx, ny, nz, selected, config.render_theme);
                colors[idx] = color;
            }
        }
    }

    if config.show_labels {
        if let Some(selected_atom) = config.selected_atom {
            if let Some(selected) = config.structure.atoms.get(selected_atom) {
                let label = format!("Selected: {} ({})", selected.label, selected.element);
                for (i, ch) in label.chars().enumerate().take(width) {
                    chars[i] = ch;
                    colors[i] = LABEL_COLOR;
                }
            }
        }
    }

    if config.show_cell && config.cell_on_top {
        for (start, end) in &config.scene.cell_edges {
            rasterize_segment(
                &mut chars,
                &mut colors,
                &mut z_buffer,
                width,
                height,
                viewport,
                *start,
                *end,
                config.render_theme.cell_line_glyph(),
                config.render_theme.cell_line_color(),
                &camera,
                CELL_LINE_DEPTH_BIAS,
                true,
            );
        }
    }

    if config.show_orientation_gizmo {
        draw_orientation_gizmo(&mut chars, &mut colors, width, height, config.rot_mat);
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

                        let target_in_cell = target_shift == [0, 0, 0];
                        let target_is_boundary = boundary_image_keys.contains(&target_key);
                        let target_rendered =
                            target_in_cell || target_is_boundary || include_bonded_images;
                        if target_rendered {
                            bonds.push(BondSegment { start, end });
                        }

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
    mat_mul_3x3(
        mat_rot_z(roll),
        mat_mul_3x3(mat_rot_y(yaw), mat_rot_x(pitch)),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        best_image_shift_and_distance, boundary_axis_shifts, build_scene, mat_from_euler,
        project_world, render_viewport, App, RenderTheme, ViewportTransform, CHAR_ASPECT,
        DEFAULT_BOND_MAX_DISTANCE, ISO_PITCH, ISO_YAW, MAX_SPIN_SPEED, MIN_FOV_DEG, MIN_SPIN_SPEED,
        MOUSE_SENSITIVITY,
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
        let cif = include_str!("../data/Fe.cif");
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
            ..Default::default()
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
            ..Default::default()
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
        let cif = include_str!("../data/Fe.cif");
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
            ..Default::default()
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
            ..Default::default()
        };

        let strict = build_scene(&structure, false, true, 2.0);
        assert_eq!(strict.bonds.len(), 0);
        assert_eq!(strict.bonded_image_count, 0);

        let relaxed = build_scene(&structure, false, true, 2.2);
        assert!(!relaxed.bonds.is_empty());
        assert!(relaxed.bonded_image_count > 0);
    }

    #[test]
    fn bonds_to_hidden_image_atoms_are_excluded() {
        // Two atoms whose only bond crosses the periodic boundary (distance ~1 Å
        // in a 10 Å cell, well within DEFAULT_BOND_MAX_DISTANCE).
        let cell = Cell::from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).expect("valid cell");
        let structure = Structure {
            title: "dangling bond test".to_string(),
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
            ..Default::default()
        };

        // With both image types off, the cross-boundary bond target is a
        // non-rendered atom — no bond should appear.
        let no_images = build_scene(&structure, false, false, DEFAULT_BOND_MAX_DISTANCE);
        assert_eq!(
            no_images.bonds.len(),
            0,
            "dangling bond to hidden image atom must not be generated"
        );

        // With bonded images on, the bond to the image atom is rendered → bond appears.
        let with_bonded = build_scene(&structure, false, true, DEFAULT_BOND_MAX_DISTANCE);
        assert!(
            !with_bonded.bonds.is_empty(),
            "cross-boundary bond should appear when bonded images are enabled"
        );
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };
        let mut app = App::new(structure);
        assert_eq!(app.render_theme, RenderTheme::Orbital);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Neon);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Wild);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Dense);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Classic);

        app.handle_key_press(KeyCode::Char('g'));
        assert_eq!(app.render_theme, RenderTheme::Orbital);
    }

    #[test]
    fn spin_lock_directional_input_starts_continuous_rotation() {
        let structure = Structure {
            title: "spin lock test".to_string(),
            atoms: vec![Atom {
                label: "A".to_string(),
                element: "C".to_string(),
                position: [0.0, 0.0, 0.0],
                fractional: None,
            }],
            cell: None,
            space_group: None,
            ..Default::default()
        };
        let mut app = App::new(structure);
        let initial = app.rot_mat;

        app.handle_key_press(KeyCode::Char('R'));
        assert!(app.spin_lock);
        app.handle_key_press(KeyCode::Char('l'));
        assert!(app.spin_velocity[1] > 0.99);
        app.update_spin_motion(0.05);

        let mut changed = false;
        for i in 0..3 {
            for j in 0..3 {
                if (app.rot_mat[i][j] - initial[i][j]).abs() > 1e-6 {
                    changed = true;
                }
            }
        }
        assert!(
            changed,
            "rotation matrix should change while spin lock is active"
        );

        app.handle_key_press(KeyCode::Char('R'));
        assert!(!app.spin_lock);
        assert_eq!(app.spin_velocity, [0.0, 0.0, 0.0]);
        let after_disable = app.rot_mat;
        app.update_spin_motion(0.05);
        for i in 0..3 {
            for j in 0..3 {
                assert!((app.rot_mat[i][j] - after_disable[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn spin_speed_keys_adjust_and_clamp() {
        let structure = Structure {
            title: "spin speed test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
            ..Default::default()
        };
        let mut app = App::new(structure);
        let initial = app.spin_speed;

        app.handle_key_press(KeyCode::Char('>'));
        assert!(app.spin_speed > initial);

        for _ in 0..200 {
            app.handle_key_press(KeyCode::Char('>'));
        }
        assert!((app.spin_speed - MAX_SPIN_SPEED).abs() < 1e-6);

        for _ in 0..400 {
            app.handle_key_press(KeyCode::Char('<'));
        }
        assert!((app.spin_speed - MIN_SPIN_SPEED).abs() < 1e-6);
    }

    #[test]
    fn toggles_orientation_gizmo_mode() {
        let structure = Structure {
            title: "gizmo toggle test".to_string(),
            atoms: vec![],
            cell: None,
            space_group: None,
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
        let cif = include_str!("../data/Fe.cif");
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
        let cif = include_str!("../data/Fe.cif");
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
            (
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                1.0,
                [0.0, 0.0],
                0.0,
                45.0,
                [0.0, 0.0],
            ),
            (
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                2.0,
                [0.3, 0.7],
                0.1,
                60.0,
                [0.1, -0.2],
            ),
            (
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                0.5,
                [-0.5, 1.2],
                0.8,
                30.0,
                [0.0, 0.0],
            ),
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
            RenderAtom {
                base_index: 0,
                position: [1.0, 0.0, 0.0],
                is_image: false,
            },
            RenderAtom {
                base_index: 1,
                position: [-1.0, 0.0, 0.0],
                is_image: false,
            },
            RenderAtom {
                base_index: 2,
                position: [0.0, 1.0, 0.0],
                is_image: false,
            },
            RenderAtom {
                base_index: 3,
                position: [0.0, 0.0, 1.0],
                is_image: false,
            },
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
        assert!(
            (scale - 0.92).abs() < 1e-5,
            "expected scale ≈ 0.92, got {scale}"
        );
    }

    #[test]
    fn cosine_distance_handles_zero_and_orthogonal_vectors() {
        use super::cosine_distance;

        assert_eq!(cosine_distance(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine_distance(&[1.0, 0.0], &[0.0, 0.0]), 1.0);
        assert!((cosine_distance(&[1.0, 0.0], &[1.0, 0.0]) - 0.0).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn xrd_pattern_fingerprint_is_l2_normalized() {
        use super::{xrd_pattern_fingerprint, XrdPattern};
        use crate::xrd::BraggPeak;

        let pattern = XrdPattern {
            peaks: vec![
                BraggPeak {
                    hkl: [1, 0, 0],
                    two_theta: 20.0,
                    d_spacing: 2.0,
                    intensity: 30.0,
                },
                BraggPeak {
                    hkl: [1, 1, 0],
                    two_theta: 40.0,
                    d_spacing: 1.4,
                    intensity: 40.0,
                },
            ],
            wavelength: 1.5406,
        };
        let fingerprint = xrd_pattern_fingerprint(&pattern, 360);
        let norm = fingerprint
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cached_formula_and_counts_match_direct_computation() {
        // Ensure the cached values in App exactly equal what the free functions produce.
        use super::{element_counts, empirical_formula};
        let cif = "data_U3Te4
_symmetry_space_group_name_H-M   I-43d
_cell_length_a   9.48509980
_cell_length_b   9.48509980
_cell_length_c   9.48509980
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   220
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  'y+1/4, -x+3/4, -z+1/4'
  3  '-x, -y+1/2, z'
  4  '-y+3/4, x+3/4, -z+1/4'
  5  'x, -y, -z+1/2'
  6  'y+1/4, x+1/4, z+1/4'
  7  '-x, y+1/2, -z+1/2'
  8  '-y+3/4, -x+1/4, z+1/4'
  9  'z, x, y'
  10  '-z+1/4, y+1/4, -x+3/4'
  11  'z, -x, -y+1/2'
  12  '-z+1/4, -y+3/4, x+3/4'
  13  '-z+1/2, x, -y'
  14  'z+1/4, y+1/4, x+1/4'
  15  '-z+1/2, -x, y+1/2'
  16  'z+1/4, -y+3/4, -x+1/4'
  17  'y, z, x'
  18  '-x+3/4, -z+1/4, y+1/4'
  19  '-y+1/2, z, -x'
  20  'x+3/4, -z+1/4, -y+3/4'
  21  '-y, -z+1/2, x'
  22  'x+1/4, z+1/4, y+1/4'
  23  'y+1/2, -z+1/2, -x'
  24  '-x+1/4, z+1/4, -y+3/4'
  25  'x+1/2, y+1/2, z+1/2'
  26  'y+3/4, -x+1/4, -z+3/4'
  27  '-x+1/2, -y, z+1/2'
  28  '-y+1/4, x+1/4, -z+3/4'
  29  'x+1/2, -y+1/2, -z'
  30  'y+3/4, x+3/4, z+3/4'
  31  '-x+1/2, y, -z'
  32  '-y+1/4, -x+3/4, z+3/4'
  33  'z+1/2, x+1/2, y+1/2'
  34  '-z+3/4, y+3/4, -x+1/4'
  35  'z+1/2, -x+1/2, -y'
  36  '-z+3/4, -y+1/4, x+1/4'
  37  '-z, x+1/2, -y+1/2'
  38  'z+3/4, y+3/4, x+3/4'
  39  '-z, -x+1/2, y'
  40  'z+3/4, -y+1/4, -x+3/4'
  41  'y+1/2, z+1/2, x+1/2'
  42  '-x+1/4, -z+3/4, y+3/4'
  43  '-y, z+1/2, -x+1/2'
  44  'x+1/4, -z+3/4, -y+1/4'
  45  '-y+1/2, -z, x+1/2'
  46  'x+3/4, z+3/4, y+3/4'
  47  'y, -z, -x+1/2'
  48  '-x+3/4, z+3/4, -y+1/4'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  U  U0  12  0.00000000  0.25000000  0.37500000  1
  Te  Te1  16  0.07670302  0.92329698  0.42329698  1
";
        let structure = parse_cif_str(cif, "u3te4").expect("should parse");
        let app = App::new(structure.clone());

        let expected_counts = element_counts(&structure.atoms);
        let expected_formula = empirical_formula(&expected_counts);

        assert_eq!(app.cached_element_counts, expected_counts);
        assert_eq!(app.cached_formula, expected_formula);
    }
}
