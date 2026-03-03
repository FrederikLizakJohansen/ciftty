use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct Atom {
    pub label: String,
    pub element: String,
    pub position: [f32; 3],
    pub fractional: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub alpha_deg: f32,
    pub beta_deg: f32,
    pub gamma_deg: f32,
    pub lattice: [[f32; 3]; 3],
}

impl Cell {
    pub fn from_parameters(
        a: f32,
        b: f32,
        c: f32,
        alpha_deg: f32,
        beta_deg: f32,
        gamma_deg: f32,
    ) -> Option<Self> {
        let alpha = alpha_deg * PI / 180.0;
        let beta = beta_deg * PI / 180.0;
        let gamma = gamma_deg * PI / 180.0;
        let sin_gamma = gamma.sin();
        if sin_gamma.abs() < 1e-6 {
            return None;
        }

        let ax = a;
        let ay = 0.0;
        let az = 0.0;

        let bx = b * gamma.cos();
        let by = b * sin_gamma;
        let bz = 0.0;

        let cx = c * beta.cos();
        let cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma;
        let cz_sq = (c * c) - (cx * cx) - (cy * cy);
        let cz = cz_sq.max(0.0).sqrt();

        Some(Self {
            a,
            b,
            c,
            alpha_deg,
            beta_deg,
            gamma_deg,
            lattice: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]],
        })
    }

    pub fn frac_to_cart(&self, frac: [f32; 3]) -> [f32; 3] {
        [
            (self.lattice[0][0] * frac[0])
                + (self.lattice[1][0] * frac[1])
                + (self.lattice[2][0] * frac[2]),
            (self.lattice[0][1] * frac[0])
                + (self.lattice[1][1] * frac[1])
                + (self.lattice[2][1] * frac[2]),
            (self.lattice[0][2] * frac[0])
                + (self.lattice[1][2] * frac[1])
                + (self.lattice[2][2] * frac[2]),
        ]
    }

    pub fn shift_to_cart(&self, shift: [i32; 3]) -> [f32; 3] {
        [
            (self.lattice[0][0] * shift[0] as f32)
                + (self.lattice[1][0] * shift[1] as f32)
                + (self.lattice[2][0] * shift[2] as f32),
            (self.lattice[0][1] * shift[0] as f32)
                + (self.lattice[1][1] * shift[1] as f32)
                + (self.lattice[2][1] * shift[2] as f32),
            (self.lattice[0][2] * shift[0] as f32)
                + (self.lattice[1][2] * shift[1] as f32)
                + (self.lattice[2][2] * shift[2] as f32),
        ]
    }

    pub fn corners(&self) -> [[f32; 3]; 8] {
        let o = [0.0, 0.0, 0.0];
        let a = self.lattice[0];
        let b = self.lattice[1];
        let c = self.lattice[2];
        let ab = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let ac = [a[0] + c[0], a[1] + c[1], a[2] + c[2]];
        let bc = [b[0] + c[0], b[1] + c[1], b[2] + c[2]];
        let abc = [ab[0] + c[0], ab[1] + c[1], ab[2] + c[2]];
        [o, a, b, c, ab, ac, bc, abc]
    }

    pub fn edge_segments(&self) -> Vec<([f32; 3], [f32; 3])> {
        let c = self.corners();
        let edges: [(usize, usize); 12] = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ];
        edges.into_iter().map(|(i, j)| (c[i], c[j])).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Structure {
    pub title: String,
    pub atoms: Vec<Atom>,
    pub cell: Option<Cell>,
}

impl Structure {
    pub fn center(&self) -> [f32; 3] {
        if self.atoms.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let mut sum = [0.0; 3];
        for atom in &self.atoms {
            sum[0] += atom.position[0];
            sum[1] += atom.position[1];
            sum[2] += atom.position[2];
        }

        let n = self.atoms.len() as f32;
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}
