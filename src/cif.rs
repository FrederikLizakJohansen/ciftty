use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::model::{Atom, Cell, Structure};

#[derive(Debug, Clone)]
enum ParsedPosition {
    Cartesian([f32; 3]),
    Fractional([f32; 3]),
}

#[derive(Debug, Clone)]
struct ParsedAtom {
    label: String,
    element: String,
    position: ParsedPosition,
}

#[derive(Debug, Default, Clone, Copy)]
struct CellParams {
    a: Option<f32>,
    b: Option<f32>,
    c: Option<f32>,
    alpha: Option<f32>,
    beta: Option<f32>,
    gamma: Option<f32>,
}

impl CellParams {
    fn update(&mut self, tag: &str, value: f32) {
        match tag {
            "_cell_length_a" => self.a = Some(value),
            "_cell_length_b" => self.b = Some(value),
            "_cell_length_c" => self.c = Some(value),
            "_cell_angle_alpha" => self.alpha = Some(value),
            "_cell_angle_beta" => self.beta = Some(value),
            "_cell_angle_gamma" => self.gamma = Some(value),
            _ => {}
        }
    }

    fn build_cell(&self) -> Option<Cell> {
        Cell::from_parameters(
            self.a?,
            self.b?,
            self.c?,
            self.alpha?,
            self.beta?,
            self.gamma?,
        )
    }
}

pub fn parse_cif_file(path: &Path) -> Result<Structure> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Could not read CIF file at {}", path.display()))?;
    let fallback_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("structure");
    parse_cif_str(&content, fallback_title)
}

pub fn parse_cif_str(input: &str, fallback_title: &str) -> Result<Structure> {
    let lines: Vec<&str> = input.lines().collect();
    let mut title = fallback_title.to_string();
    let mut cell = CellParams::default();
    let mut space_group: Option<String> = None;
    let mut parsed_atoms: Vec<ParsedAtom> = Vec::new();
    let mut symmetry_ops: Vec<String> = Vec::new();
    let mut i = 0usize;

    while i < lines.len() {
        let line = strip_inline_comment(lines[i]).trim();
        if line.is_empty() {
            i += 1;
            continue;
        }

        if let Some(data_name) = line.strip_prefix("data_") {
            let data_name = data_name.trim();
            if !data_name.is_empty() {
                title = data_name.to_string();
            }
            i += 1;
            continue;
        }

        if line.starts_with('_') {
            let tokens = tokenize_row(line);
            if tokens.len() >= 2 {
                let tag = tokens[0].to_ascii_lowercase();
                if tag.starts_with("_cell_") {
                    if let Some(value) = parse_cif_number(&tokens[1]) {
                        cell.update(&tag, value);
                    }
                    i += 1;
                    continue;
                }
                if is_space_group_name_tag(&tag) {
                    if let Some(value) = parse_tag_value(&tokens[1..]) {
                        space_group = Some(value);
                    }
                    i += 1;
                    continue;
                }
                if is_space_group_number_tag(&tag) {
                    if let Some(value) = parse_tag_value(&tokens[1..]) {
                        if space_group.is_none() {
                            space_group = Some(format!("#{}", value));
                        }
                    }
                    i += 1;
                    continue;
                }
            }
        }

        if line.eq_ignore_ascii_case("loop_") {
            i += 1;
            let mut headers: Vec<String> = Vec::new();
            while i < lines.len() {
                let header_line = strip_inline_comment(lines[i]).trim();
                if header_line.starts_with('_') {
                    headers.push(header_line.to_ascii_lowercase());
                    i += 1;
                    continue;
                }
                break;
            }

            if headers.is_empty() {
                continue;
            }

            let atom_loop = headers.iter().any(|h| h.starts_with("_atom_site_"));
            let symop_column = headers.iter().position(|h| is_symmetry_operation_header(h));

            while i < lines.len() {
                let value_line = strip_inline_comment(lines[i]).trim();
                if value_line.is_empty() {
                    i += 1;
                    continue;
                }
                if value_line.starts_with('_')
                    || value_line.eq_ignore_ascii_case("loop_")
                    || value_line.starts_with("data_")
                {
                    break;
                }

                let values = tokenize_row(value_line);
                if atom_loop {
                    if let Some(atom) = parse_atom_row(&headers, &values) {
                        parsed_atoms.push(atom);
                    }
                } else if let Some(index) = symop_column {
                    if let Some(op) = values.get(index) {
                        let op = op.trim();
                        if !op.is_empty() {
                            symmetry_ops.push(op.to_string());
                        }
                    }
                }
                i += 1;
            }

            continue;
        }

        i += 1;
    }

    let cell_model = cell.build_cell();
    let atoms = expand_atoms(parsed_atoms, cell_model, &symmetry_ops);

    if atoms.is_empty() {
        bail!("No atoms were parsed from `_atom_site_*` CIF loops");
    }

    Ok(Structure {
        title,
        atoms,
        cell: cell_model,
        space_group,
    })
}

fn is_space_group_name_tag(tag: &str) -> bool {
    matches!(
        tag,
        "_symmetry_space_group_name_h-m"
            | "_space_group_name_h-m_alt"
            | "_space_group_name_h-m_full"
            | "_space_group_name_h-m_ref"
            | "_space_group_name_hall"
    )
}

fn is_space_group_number_tag(tag: &str) -> bool {
    matches!(
        tag,
        "_space_group_it_number" | "_symmetry_int_tables_number"
    )
}

fn parse_tag_value(tokens: &[String]) -> Option<String> {
    if tokens.is_empty() {
        return None;
    }
    let joined = tokens.join(" ");
    let trimmed = joined.trim().trim_matches('\'').trim_matches('"').trim();
    if trimmed.is_empty() || trimmed == "?" || trimmed == "." {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn is_symmetry_operation_header(header: &str) -> bool {
    matches!(
        header,
        "_symmetry_equiv_pos_as_xyz"
            | "_space_group_symop_operation_xyz"
            | "_space_group_symop.operation_xyz"
            | "_space_group_symop_operation_xyz_"
    )
}

fn expand_atoms(
    parsed_atoms: Vec<ParsedAtom>,
    cell: Option<Cell>,
    symmetry_ops: &[String],
) -> Vec<Atom> {
    let mut atoms = Vec::new();
    let mut seen_fractional: HashSet<(String, i32, i32, i32)> = HashSet::new();
    let has_symmetry_ops = !symmetry_ops.is_empty();

    for parsed in parsed_atoms {
        match parsed.position {
            ParsedPosition::Cartesian(position) => {
                atoms.push(Atom {
                    label: parsed.label,
                    element: parsed.element,
                    position,
                    fractional: None,
                });
            }
            ParsedPosition::Fractional(frac) => {
                let mut generated = Vec::new();
                if has_symmetry_ops {
                    for op in symmetry_ops {
                        if let Some(new_frac) = apply_symmetry_operation(op, frac) {
                            generated.push(new_frac);
                        }
                    }
                }
                if generated.is_empty() {
                    generated.push(normalize_fractional(frac));
                }

                for frac_pos in generated {
                    let key = (
                        parsed.element.clone(),
                        quantize_fractional(frac_pos[0]),
                        quantize_fractional(frac_pos[1]),
                        quantize_fractional(frac_pos[2]),
                    );
                    if !seen_fractional.insert(key) {
                        continue;
                    }
                    let position = if let Some(cell) = cell {
                        cell.frac_to_cart(frac_pos)
                    } else {
                        frac_pos
                    };
                    atoms.push(Atom {
                        label: parsed.label.clone(),
                        element: parsed.element.clone(),
                        position,
                        fractional: Some(frac_pos),
                    });
                }
            }
        }
    }

    atoms
}

fn apply_symmetry_operation(operation: &str, frac: [f32; 3]) -> Option<[f32; 3]> {
    let parts: Vec<&str> = operation.split(',').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return None;
    }
    Some([
        wrap_fractional(eval_symmetry_component(parts[0], frac)?),
        wrap_fractional(eval_symmetry_component(parts[1], frac)?),
        wrap_fractional(eval_symmetry_component(parts[2], frac)?),
    ])
}

fn eval_symmetry_component(component: &str, frac: [f32; 3]) -> Option<f32> {
    let compact: String = component.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() {
        return None;
    }

    let mut total = 0.0f32;
    let bytes = compact.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let mut sign = 1.0f32;
        match bytes[i] as char {
            '+' => i += 1,
            '-' => {
                sign = -1.0;
                i += 1;
            }
            _ => {}
        }
        if i >= bytes.len() {
            return None;
        }

        let start = i;
        while i < bytes.len() {
            let ch = bytes[i] as char;
            if ch == '+' || ch == '-' {
                break;
            }
            i += 1;
        }

        let term = &compact[start..i];
        if let Some((axis, coeff)) = parse_axis_term(term) {
            total += sign * coeff * frac[axis];
        } else if let Some(value) = parse_cif_number(term) {
            total += sign * value;
        } else {
            return None;
        }
    }

    Some(total)
}

fn parse_axis_term(term: &str) -> Option<(usize, f32)> {
    let lower = term.to_ascii_lowercase();
    if lower == "x" {
        return Some((0, 1.0));
    }
    if lower == "y" {
        return Some((1, 1.0));
    }
    if lower == "z" {
        return Some((2, 1.0));
    }

    if let Some(coeff) = lower.strip_suffix('x').and_then(parse_cif_number) {
        return Some((0, coeff));
    }
    if let Some(coeff) = lower.strip_suffix('y').and_then(parse_cif_number) {
        return Some((1, coeff));
    }
    if let Some(coeff) = lower.strip_suffix('z').and_then(parse_cif_number) {
        return Some((2, coeff));
    }
    None
}

fn normalize_fractional(value: [f32; 3]) -> [f32; 3] {
    [
        wrap_fractional(value[0]),
        wrap_fractional(value[1]),
        wrap_fractional(value[2]),
    ]
}

fn wrap_fractional(value: f32) -> f32 {
    let mut out = value - value.floor();
    if out < 0.0 {
        out += 1.0;
    }
    if out >= 1.0 {
        out -= 1.0;
    }
    out
}

fn quantize_fractional(value: f32) -> i32 {
    (wrap_fractional(value) * 1_000_000.0).round() as i32
}

fn parse_atom_row(headers: &[String], values: &[String]) -> Option<ParsedAtom> {
    let label = get_field(
        headers,
        values,
        &[
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_auth_atom_id",
        ],
    )
    .unwrap_or_else(|| format!("atom_{}", values.len()));

    let element = get_field(
        headers,
        values,
        &[
            "_atom_site_type_symbol",
            "_atom_site_symbol",
            "_atom_site_label",
        ],
    )
    .and_then(|v| normalize_element_symbol(&v))
    .or_else(|| infer_element_from_label(&label))
    .unwrap_or_else(|| "X".to_string());

    if let Some(cart) = parse_xyz(
        headers,
        values,
        &[
            "_atom_site_cartn_x",
            "_atom_site_cartn_y",
            "_atom_site_cartn_z",
        ],
    ) {
        return Some(ParsedAtom {
            label,
            element,
            position: ParsedPosition::Cartesian(cart),
        });
    }

    parse_xyz(
        headers,
        values,
        &[
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ],
    )
    .map(|frac| ParsedAtom {
        label,
        element,
        position: ParsedPosition::Fractional(frac),
    })
}

fn parse_xyz(headers: &[String], values: &[String], fields: &[&str; 3]) -> Option<[f32; 3]> {
    let x = get_field(headers, values, &[fields[0]]).and_then(|v| parse_cif_number(&v));
    let y = get_field(headers, values, &[fields[1]]).and_then(|v| parse_cif_number(&v));
    let z = get_field(headers, values, &[fields[2]]).and_then(|v| parse_cif_number(&v));
    Some([x?, y?, z?])
}

fn get_field(headers: &[String], values: &[String], aliases: &[&str]) -> Option<String> {
    for alias in aliases {
        if let Some((index, _)) = headers.iter().enumerate().find(|(_, h)| h == alias) {
            if let Some(value) = values.get(index) {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn parse_cif_number(raw: &str) -> Option<f32> {
    let mut value = raw.trim().trim_matches('\'').trim_matches('"');
    if value.is_empty() || value == "." || value == "?" {
        return None;
    }

    if let Some(paren) = value.find('(') {
        value = &value[..paren];
    }

    if let Some((num, den)) = value.split_once('/') {
        let num = num.trim().parse::<f32>().ok()?;
        let den = den.trim().parse::<f32>().ok()?;
        if den.abs() < f32::EPSILON {
            return None;
        }
        return Some(num / den);
    }

    value.parse::<f32>().ok()
}

fn infer_element_from_label(label: &str) -> Option<String> {
    normalize_element_symbol(label)
}

fn normalize_element_symbol(raw: &str) -> Option<String> {
    let mut chars = raw
        .trim_matches('\'')
        .trim_matches('"')
        .chars()
        .filter(|c| c.is_ascii_alphabetic());
    let first = chars.next()?;
    let mut out = String::new();
    out.push(first.to_ascii_uppercase());
    if let Some(second) = chars.next() {
        if second.is_ascii_lowercase() {
            out.push(second);
        } else {
            out.push(second.to_ascii_lowercase());
        }
    }
    Some(out)
}

fn strip_inline_comment(line: &str) -> &str {
    match line.find('#') {
        Some(index) => &line[..index],
        None => line,
    }
}

fn tokenize_row(line: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in line.chars() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else {
                current.push(ch);
            }
            continue;
        }

        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }

        if ch.is_whitespace() {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            continue;
        }

        current.push(ch);
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::parse_cif_str;

    #[test]
    fn parses_fractional_atoms_with_cell() {
        let cif = r#"
data_example
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  C1 C 0.0 0.0 0.0
  O1 O 0.5 0.5 0.5
"#;
        let structure = parse_cif_str(cif, "fallback").expect("should parse");
        assert_eq!(structure.atoms.len(), 2);
        assert_eq!(structure.atoms[0].element, "C");
        assert!((structure.atoms[1].position[0] - 2.5).abs() < 1e-4);
        assert_eq!(structure.title, "example");
    }

    #[test]
    fn expands_symmetry_to_match_fe_multiplicity() {
        let cif = include_str!("../Fe.cif");
        let structure = parse_cif_str(cif, "fe_fixture").expect("Fe.cif should parse");
        assert_eq!(structure.atoms.len(), 2);
    }

    #[test]
    fn expands_symmetry_to_match_u3te4_multiplicity() {
        let cif = include_str!("../U3Te4.cif");
        let structure = parse_cif_str(cif, "u3te4_fixture").expect("U3Te4.cif should parse");
        assert_eq!(structure.atoms.len(), 28);
    }

    #[test]
    fn expands_symmetry_to_match_cega2_multiplicity() {
        let cif = include_str!("../CeGa2.cif");
        let structure = parse_cif_str(cif, "cega2_fixture").expect("CeGa2.cif should parse");
        assert_eq!(structure.atoms.len(), 3);
    }
}
