use crate::model::{Atom, Cell, Structure};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BraggPeak {
    pub hkl: [i32; 3],
    pub two_theta: f32,
    pub d_spacing: f32,
    /// Normalised intensity (strongest peak = 100.0).
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub struct XrdPattern {
    pub peaks: Vec<BraggPeak>,
    pub wavelength: f32,  // Å; stored for reference
}

impl XrdPattern {
    pub fn empty(wavelength: f32) -> Self {
        Self { peaks: Vec::new(), wavelength }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute a simulated powder XRD pattern for `structure`.
///
/// `wavelength` is the X-ray wavelength in Å (e.g. 1.5406 for Cu Kα).
/// `two_theta_max` caps the angular range (degrees).
/// Returns peaks sorted by 2θ, normalised to strongest = 100.
pub fn compute_pattern(
    structure: &Structure,
    wavelength: f32,
    two_theta_max: f32,
) -> XrdPattern {
    let Some(cell) = structure.cell else {
        return XrdPattern::empty(wavelength);
    };
    if structure.atoms.is_empty() {
        return XrdPattern::empty(wavelength);
    }
    // Only atoms with fractional coordinates participate in structure factors.
    let frac_atoms: Vec<&Atom> =
        structure.atoms.iter().filter(|a| a.fractional.is_some()).collect();
    if frac_atoms.is_empty() {
        return XrdPattern::empty(wavelength);
    }

    let recip = reciprocal_lattice(&cell);
    // d_min from Bragg's law: d = λ/(2·sin(θ_max)) with θ_max = two_theta_max/2.
    let theta_max = (two_theta_max / 2.0).to_radians();
    let d_min = if theta_max.sin() > 0.0 { wavelength / (2.0 * theta_max.sin()) } else { 0.3 };
    // Upper bound on |h|, |k|, |l|: longest cell axis / d_min.
    let max_hkl = ((cell.a.max(cell.b).max(cell.c) / d_min).ceil() as i32).clamp(1, 20);

    let mut raw: Vec<RawPeak> = Vec::new();

    for h in -max_hkl..=max_hkl {
        for k in -max_hkl..=max_hkl {
            for l in -max_hkl..=max_hkl {
                if h == 0 && k == 0 && l == 0 {
                    continue;
                }
                let d = d_spacing(h, k, l, &recip);
                if d < d_min || !d.is_finite() || d <= 0.0 {
                    continue;
                }
                let sin_theta = wavelength / (2.0 * d);
                if sin_theta > 1.0 {
                    continue;
                }
                let theta = sin_theta.asin();
                let two_theta = 2.0 * theta.to_degrees();
                if two_theta > two_theta_max || two_theta < 1.0 {
                    continue;
                }
                let s = sin_theta / wavelength; // = 1/(2d)
                let f_sq = structure_factor_sq(h, k, l, &frac_atoms, s);
                if f_sq < 1e-4 {
                    continue;
                }
                let lp = lorentz_polarization(theta);
                raw.push(RawPeak { hkl: [h, k, l], two_theta, d, f_sq, lp });
            }
        }
    }

    // Merge peaks with the same d-spacing (within tolerance) — these are
    // symmetry-equivalent reflections in a powder pattern.
    let peaks = merge_and_normalise(raw, wavelength);
    XrdPattern { peaks, wavelength }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct RawPeak {
    hkl: [i32; 3],
    two_theta: f32,
    d: f32,
    f_sq: f32,
    lp: f32,
}

/// Merge symmetry-equivalent peaks (same d within 0.001 Å), accumulate
/// intensity (multiplicity), normalise, and return sorted by 2θ.
fn merge_and_normalise(mut raw: Vec<RawPeak>, wavelength: f32) -> Vec<BraggPeak> {
    if raw.is_empty() {
        return Vec::new();
    }
    raw.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<BraggPeak> = Vec::new();
    let d_tol = 0.001_f32;

    for raw_peak in raw {
        let intensity = raw_peak.f_sq * raw_peak.lp;
        if let Some(last) = merged.last_mut() {
            if (last.d_spacing - raw_peak.d).abs() < d_tol {
                last.intensity += intensity;
                continue;
            }
        }
        merged.push(BraggPeak {
            hkl: raw_peak.hkl,
            two_theta: raw_peak.two_theta,
            d_spacing: raw_peak.d,
            intensity,
        });
    }

    // Normalise to strongest = 100.
    let max_i = merged.iter().map(|p| p.intensity).fold(0.0f32, f32::max);
    if max_i > 0.0 {
        for p in &mut merged {
            p.intensity = p.intensity / max_i * 100.0;
        }
    }

    // Filter weak peaks (< 0.5 relative) and sort by 2θ.
    merged.retain(|p| p.intensity >= 0.5);
    merged.sort_by(|a, b| a.two_theta.partial_cmp(&b.two_theta).unwrap_or(std::cmp::Ordering::Equal));

    // Re-assign two_theta from d using the stored wavelength (it came from d anyway,
    // but keep consistent after merging).
    for p in &mut merged {
        let sin_t = wavelength / (2.0 * p.d_spacing);
        if sin_t <= 1.0 {
            p.two_theta = 2.0 * sin_t.asin().to_degrees();
        }
    }

    merged
}

/// Reciprocal lattice matrix: rows are a*, b*, c* such that
/// G* = (A^{-1})^T where A has columns = real-space lattice vectors.
/// d_{hkl} = 1 / |G* · [h,k,l]^T|
fn reciprocal_lattice(cell: &Cell) -> [[f32; 3]; 3] {
    // A = cell.lattice: lattice[i] is the i-th basis vector (a=0, b=1, c=2).
    // Invert 3×3 using cofactor / determinant.
    let a = cell.lattice;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return [[0.0; 3]; 3];
    }
    let inv_det = 1.0 / det;
    // Cofactor matrix (transposed = adjugate / det).
    let inv = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ];
    // inv is already (A^{-1}), and its transpose gives the reciprocal lattice
    // row vectors.  For crystallographic convention (no 2π), d = 1/|G*·hkl|.
    // Transpose inv so rows are a*, b*, c*.
    [
        [inv[0][0], inv[1][0], inv[2][0]],
        [inv[0][1], inv[1][1], inv[2][1]],
        [inv[0][2], inv[1][2], inv[2][2]],
    ]
}

fn d_spacing(h: i32, k: i32, l: i32, recip: &[[f32; 3]; 3]) -> f32 {
    let hf = h as f32;
    let kf = k as f32;
    let lf = l as f32;
    // G*·[h,k,l]: each recip[i] is a reciprocal lattice vector (a*, b*, c*).
    let gx = recip[0][0] * hf + recip[1][0] * kf + recip[2][0] * lf;
    let gy = recip[0][1] * hf + recip[1][1] * kf + recip[2][1] * lf;
    let gz = recip[0][2] * hf + recip[1][2] * kf + recip[2][2] * lf;
    let len = (gx * gx + gy * gy + gz * gz).sqrt();
    if len < 1e-9 { f32::INFINITY } else { 1.0 / len }
}

/// |F(hkl)|² summed over all fractional atoms.
fn structure_factor_sq(h: i32, k: i32, l: i32, atoms: &[&Atom], s: f32) -> f32 {
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    for atom in atoms {
        let frac = atom.fractional.unwrap();
        let phase = two_pi * (h as f32 * frac[0] + k as f32 * frac[1] + l as f32 * frac[2]);
        let f = atomic_scattering_factor(&atom.element, s);
        re += f * phase.cos();
        im += f * phase.sin();
    }
    re * re + im * im
}

/// Lorentz-polarization factor for a powder diffractometer.
fn lorentz_polarization(theta: f32) -> f32 {
    let s = theta.sin();
    let s2 = (2.0 * theta).sin();
    if s.abs() < 1e-6 || s2.abs() < 1e-6 {
        return 1.0;
    }
    (1.0 + (2.0 * theta).cos().powi(2)) / (s * s * s2)
}

/// Cromer–Mann atomic scattering factor:
///   f(s) = Σᵢ aᵢ·exp(−bᵢ·s²) + c    where s = sinθ/λ
///
/// Coefficients from International Tables for Crystallography, Vol. C.
/// Elements not listed fall back to Z (a rough approximation).
fn atomic_scattering_factor(element: &str, s: f32) -> f32 {
    // Each entry: [a1,b1, a2,b2, a3,b3, a4,b4, c]
    let params: Option<[f32; 9]> = match element {
        "H"  => Some([0.489918,20.6593, 0.262003,7.74039, 0.196767,49.5519, 0.049879,2.20159, 0.001305]),
        "He" => Some([0.873400,9.10370, 0.630900,3.35680, 0.311200,22.9276, 0.178000,0.98200, 0.006400]),
        "Li" => Some([1.128200,3.95460, 0.750800,1.05240, 0.617500,85.3905, 0.465300,168.261, 0.037700]),
        "Be" => Some([1.591900,43.6427, 1.127800,1.86230, 0.539100,103.483, 0.702900,0.54200, 0.038500]),
        "B"  => Some([2.054500,23.2185, 1.332600,1.02100, 1.097900,60.3498, 0.706800,0.14030, -0.193200]),
        "C"  => Some([2.310000,20.8439, 1.020000,10.2075, 1.588600,0.56870, 0.865000,51.6512, 0.215600]),
        "N"  => Some([12.2126,0.00570, 3.132200,9.89330, 2.012500,28.9975, 1.166300,0.58260, -11.529]),
        "O"  => Some([3.048500,13.2771, 2.286800,5.70110, 1.546300,0.32390, 0.867000,32.9089, 0.250800]),
        "F"  => Some([3.539200,10.2825, 2.641200,4.29440, 1.517000,0.26150, 1.024300,26.1476, 0.277600]),
        "Ne" => Some([3.955300,8.40420, 3.112500,3.42620, 1.454600,0.23060, 1.125100,21.7184, 0.351500]),
        "Na" => Some([4.762600,3.28500, 3.173600,8.84220, 1.267400,0.31360, 1.112800,129.424, 0.676000]),
        "Mg" => Some([5.420400,2.82750, 2.173500,79.2611, 1.226900,0.38080, 2.307300,7.19370, 0.858400]),
        "Al" => Some([6.420200,3.03870, 1.900200,0.74260, 1.593600,31.5472, 1.964600,85.0886, 1.115100]),
        "Si" => Some([6.291500,2.43860, 3.035300,32.3337, 1.989100,0.67850, 1.541000,81.6937, 1.140700]),
        "P"  => Some([6.434500,1.90670, 4.179100,27.1570, 1.780000,0.52600, 1.490800,68.1645, 1.114900]),
        "S"  => Some([6.905300,1.46790, 5.203400,22.2151, 1.437900,0.25360, 1.586300,56.1720, 0.866900]),
        "Cl" => Some([11.4604,0.01040, 7.196400,1.16620, 6.255600,18.5194, 1.645500,47.7784, -9.5574]),
        "Ar" => Some([7.484500,0.90720, 6.772300,14.8407, 0.653900,43.8983, 1.644200,33.3929, 1.444500]),
        "K"  => Some([8.218600,12.7949, 7.439800,0.77480, 1.051900,213.187, 0.865900,41.6841, 1.422800]),
        "Ca" => Some([8.626600,10.4422, 7.387300,0.65990, 1.589900,85.7484, 1.021100,178.437, 1.375100]),
        "Ti" => Some([9.759500,7.85080, 7.355800,0.50000, 1.699100,35.6338, 1.902100,116.105, 1.280700]),
        "V"  => Some([10.2971,6.86570, 7.351100,0.43850, 2.070300,26.8938, 2.057100,102.478, 1.219900]),
        "Cr" => Some([10.6406,6.10380, 7.353700,0.39200, 3.324000,20.2626, 1.492200,98.7399, 1.183200]),
        "Mn" => Some([11.2819,5.34090, 7.357300,0.34320, 3.019300,17.8674, 2.244100,83.7543, 1.089600]),
        "Fe" => Some([11.7695,4.76110, 7.357300,0.30720, 3.522200,15.3535, 2.304500,76.8805, 1.036900]),
        "Co" => Some([12.2841,4.27910, 7.340900,0.27840, 4.003400,13.5359, 2.348800,71.1692, 1.011800]),
        "Ni" => Some([12.8376,3.87850, 7.292000,0.25650, 4.443800,12.1763, 2.380000,66.3421, 1.034100]),
        "Cu" => Some([13.3380,3.58280, 7.167600,0.24700, 5.615800,11.3966, 1.673500,64.8126, 1.191000]),
        "Zn" => Some([14.0743,3.26550, 7.031800,0.23330, 5.165200,10.3163, 2.410000,58.7097, 1.304100]),
        "Ga" => Some([15.2354,3.06690, 6.700600,0.24120, 4.359100,10.7805, 2.962300,61.4135, 1.718900]),
        "Ge" => Some([16.0816,2.85090, 6.374700,0.25160, 3.706800,11.4468, 3.683000,54.7625, 2.131300]),
        "As" => Some([16.6723,2.63450, 6.070100,0.26470, 3.431300,12.9479, 4.277900,47.7972, 2.531000]),
        "Se" => Some([17.0006,2.40980, 5.819600,0.27260, 3.973100,15.2372, 4.354300,43.8163, 2.840900]),
        "Br" => Some([17.1789,2.17230, 5.235800,16.5796, 5.637700,0.26090, 3.985100,41.4328, 2.955700]),
        "Zr" => Some([17.8765,1.27618, 10.9480,11.9160, 5.417320,0.117622, 3.657210,87.6627, 2.06929]),
        "Mo" => Some([3.702500,0.27720, 17.2360,1.09560, 12.8870,11.0040, 3.742900,61.6584, 4.387500]),
        "Ag" => Some([19.2808,0.64460, 16.6885,7.47260, 4.804500,24.6605, 1.046300,99.8156, 5.179000]),
        "Ba" => Some([20.3361,3.21600, 19.2970,0.27560, 10.8880,20.2073, 2.695900,167.202, 2.773100]),
        "Ce" => Some([21.1671,2.81219, 19.7695,0.226836, 11.8513,17.6083, 3.330490,127.113, 1.86264]),
        "U"  => Some([36.0228,0.102340, 23.4128,3.21436, 14.9491,23.9533, 4.188000,118.202, 13.3966]),
        "Te" => Some([19.9644,4.81742, 19.0138,0.420885, 6.14487,28.5284, 2.52390,70.8403, 4.35200]),
        _    => None,
    };

    match params {
        Some(p) => {
            let s2 = s * s;
            p[0] * (-p[1] * s2).exp()
                + p[2] * (-p[3] * s2).exp()
                + p[4] * (-p[5] * s2).exp()
                + p[6] * (-p[7] * s2).exp()
                + p[8]
        }
        None => {
            // Fall back to atomic number (crude but finite)
            atomic_number_fallback(element) as f32
        }
    }
}

fn atomic_number_fallback(element: &str) -> u8 {
    match element {
        "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5,
        "C" => 6, "N" => 7, "O" => 8, "F" => 9, "Ne" => 10,
        "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P" => 15,
        "S" => 16, "Cl" => 17, "Ar" => 18, "K" => 19, "Ca" => 20,
        "Sc" => 21, "Ti" => 22, "V" => 23, "Cr" => 24, "Mn" => 25,
        "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30,
        "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35,
        "Kr" => 36, "Rb" => 37, "Sr" => 38, "Y" => 39, "Zr" => 40,
        "Nb" => 41, "Mo" => 42, "Tc" => 43, "Ru" => 44, "Rh" => 45,
        "Pd" => 46, "Ag" => 47, "Cd" => 48, "In" => 49, "Sn" => 50,
        "Sb" => 51, "Te" => 52, "I" => 53, "Xe" => 54, "Cs" => 55,
        "Ba" => 56, "La" => 57, "Ce" => 58, "Pr" => 59, "Nd" => 60,
        "Pm" => 61, "Sm" => 62, "Eu" => 63, "Gd" => 64, "Tb" => 65,
        "Dy" => 66, "Ho" => 67, "Er" => 68, "Tm" => 69, "Yb" => 70,
        "Lu" => 71, "Hf" => 72, "Ta" => 73, "W" => 74, "Re" => 75,
        "Os" => 76, "Ir" => 77, "Pt" => 78, "Au" => 79, "Hg" => 80,
        "Tl" => 81, "Pb" => 82, "Bi" => 83, "Th" => 90, "U" => 92,
        _ => 6,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Atom, Cell, Structure};

    #[test]
    fn reciprocal_lattice_cubic_gives_correct_d_spacings() {
        let a = 4.0_f32;
        let cell = Cell::from_parameters(a, a, a, 90.0, 90.0, 90.0).unwrap();
        let recip = reciprocal_lattice(&cell);
        // For cubic: d_100 = a, d_110 = a/√2, d_111 = a/√3
        let d100 = d_spacing(1, 0, 0, &recip);
        let d110 = d_spacing(1, 1, 0, &recip);
        let d111 = d_spacing(1, 1, 1, &recip);
        assert!((d100 - a).abs() < 1e-4, "d_100={d100} expected {a}");
        assert!((d110 - a / 2.0_f32.sqrt()).abs() < 1e-4, "d_110={d110}");
        assert!((d111 - a / 3.0_f32.sqrt()).abs() < 1e-4, "d_111={d111}");
    }

    #[test]
    fn compute_pattern_returns_peaks_for_simple_cubic_fe() {
        let cif = include_str!("../data/Fe.cif");
        let structure = crate::cif::parse_cif_str(cif, "fe").expect("parse");
        let pattern = compute_pattern(&structure, 1.5406, 90.0);
        assert!(!pattern.peaks.is_empty(), "Fe should produce Bragg peaks");
        // Strongest peak should be at 100.
        let max = pattern.peaks.iter().map(|p| p.intensity).fold(0.0f32, f32::max);
        assert!((max - 100.0).abs() < 0.5, "max intensity should be 100, got {max}");
    }

    #[test]
    fn d_spacing_consistent_with_bragg_law() {
        let a = 3.5_f32;
        let cell = Cell::from_parameters(a, a, a, 90.0, 90.0, 90.0).unwrap();
        let recip = reciprocal_lattice(&cell);
        let lambda = 1.5406_f32;
        let d = d_spacing(1, 1, 0, &recip);
        let two_theta = 2.0 * (lambda / (2.0 * d)).asin().to_degrees();
        // For bcc Fe a≈2.87; a=3.5 gives a similar range — just check it's physical.
        assert!(two_theta > 0.0 && two_theta < 180.0);
    }

    #[test]
    fn no_cell_returns_empty_pattern() {
        let structure = Structure {
            title: "no cell".to_string(),
            atoms: vec![Atom {
                label: "C".to_string(),
                element: "C".to_string(),
                position: [0.0; 3],
                fractional: None,
            }],
            cell: None,
            space_group: None,
        };
        let pattern = compute_pattern(&structure, 1.5406, 90.0);
        assert!(pattern.peaks.is_empty());
    }
}
