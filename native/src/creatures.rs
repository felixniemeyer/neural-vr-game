//! Sea creatures, from David's steam-frame-vr Aquarium (experiments/steam-frame-vr,
//! native/src/aquarium.rs, catalog: creatures.json): procedural animated meshes, built on
//! the CPU each frame. Changes here: School::pose (one animal at fixed detail, for
//! swarm.rs) instead of the culled, level-of-detail School::animate, and a GPU vertex
//! layout (std430) with a velocity.

use anyhow::{ensure, Context, Result};
use glam::{Quat, Vec3};
use serde::Deserialize;
use std::f32::consts::{PI, TAU};

pub const RADIUS: f32 = 38.0;
pub const COUNT: usize = 100;
pub const MAX_VERTICES: usize = 260_000;

// All local pose frequencies are multiples of 0.1 rad/s. Use a prime sample
// count so none of those waves repeatedly land on the same few phases.
const MOTION_CYCLE: f32 = TAU * 10.0;
const ENVELOPE_SAMPLES: usize = 157;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
    /// m/s (filled by swarm.rs).
    pub velocity: [f32; 3],
    pub _pad2: f32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Family {
    Ray,
    Jelly,
    Arthropod,
    Ribbon,
}

#[derive(Deserialize)]
struct Specimen {
    id: usize,
    name: String,
    family: Family,
    anatomy: String,
    size: f32,
    proportion: f32,
    detail: f32,
    appendages: usize,
}

#[derive(Clone, Copy)]
struct Form {
    seed: f32,
    family: Family,
    size: f32,
    spread: f32,
    density: f32,
    limbs: usize,
    horns: bool,
    plated: bool,
    hooks: bool,
    fins: bool,
    open_frame: bool,
    offset_fins: bool,
    crown: bool,
    shelled: bool,
    nested: bool,
    upright: bool,
    split_tail: bool,
    morph: f32,
    body: [f32; 4],
    membrane: [f32; 4],
    light: [f32; 4],
}

impl Form {
    fn new(specimen: &Specimen) -> Self {
        let seed = specimen.id as f32;
        let anatomy = specimen.anatomy.to_lowercase();
        let base = match specimen.family {
            Family::Ray => [0.50, 0.79, 0.78, 0.04],
            Family::Jelly => [0.42, 0.64, 0.86, 0.18],
            Family::Arthropod => [0.71, 0.76, 0.60, 0.03],
            Family::Ribbon => [0.73, 0.62, 0.81, 0.07],
        };
        let shift = (hash(seed * 9.7) - 0.5) * 0.24;
        Self {
            seed,
            family: specimen.family,
            size: specimen.size,
            spread: specimen.proportion,
            density: specimen.detail,
            limbs: specimen.appendages,
            horns: has_any(&anatomy, &["antler", "crown", "antenna", "horn", "feather"]),
            plated: has_any(
                &anatomy,
                &["armor", "plate", "rib", "shell", "carapace", "bone"],
            ),
            hooks: has_any(
                &anatomy,
                &["hook", "mantis", "scorpion", "claw", "pincer", "saw"],
            ),
            fins: has_any(&anatomy, &["wing", "sail", "paddle", "bone oar"]),
            open_frame: has_any(
                &anatomy,
                &[
                    "open ",
                    "fenestrat",
                    "window",
                    "perforat",
                    "keyhole",
                    "hollow",
                ],
            ),
            offset_fins: has_any(&anatomy, &["asymmetr", "one side", "one-sided", "one high"]),
            crown: has_any(
                &anatomy,
                &["inverted fan", "crown", "pinnacle", "crest spine"],
            ),
            shelled: has_any(&anatomy, &["bell", "umbrella", "hood", "dome", "carapace"]),
            nested: has_any(
                &anatomy,
                &[
                    "nested",
                    "tiers",
                    "chain of",
                    "linked",
                    "two lobes",
                    "hourglass",
                ],
            ),
            upright: has_any(
                &anatomy,
                &[
                    "tall",
                    "vertical",
                    "upright",
                    "spire",
                    "chrysalis",
                    "pupa",
                    "bottle",
                ],
            ),
            split_tail: has_any(
                &anatomy,
                &["fork", "split tail", "y-shaped", "twin-pronged"],
            ),
            morph: hash(seed * 5.83),
            body: [base[0] + shift, base[1], base[2] - shift, base[3]],
            membrane: [base[0] * 0.56, base[1] * 0.66, base[2] * 0.75, 0.32],
            light: [base[0] * 0.65 + 0.20, base[1] * 0.88 + 0.10, 0.96, 0.82],
        }
    }

    fn geometry(self, mesh: &mut Mesh<'_>, time: f32, angular_size: f32) {
        let budget = ((self.size - 0.5) / 9.5).sqrt() * 0.72 + self.density * 0.28;
        let detail = (2.0 + budget * 5.0).round() as usize;
        let quality = if angular_size >= 0.27 {
            detail
        } else if angular_size >= 0.11 {
            detail.min(4)
        } else if angular_size >= 0.035 {
            detail.min(2)
        } else {
            1
        };
        mesh.facets = if quality >= 3 { 4 } else { 3 };
        match self.family {
            Family::Ray => self.ray(mesh, time, quality),
            Family::Jelly => self.jelly(mesh, time, quality),
            Family::Arthropod => self.arthropod(mesh, time, quality),
            Family::Ribbon => self.ribbon(mesh, time, quality),
        };
    }

    fn envelope(self, scale: f32, phase_offset: f32, vertices: &mut Vec<Vertex>) -> (f32, f32) {
        let mut max_span: f32 = 0.0;
        let mut max_radius_squared: f32 = 0.0;
        for sample in 0..ENVELOPE_SAMPLES {
            vertices.clear();
            let mut mesh = Mesh {
                vertices,
                origin: Vec3::ZERO,
                basis: Quat::IDENTITY,
                size: scale,
                facets: 3,
                truncated: false,
            };
            let phase = (sample as f32 + phase_offset) * MOTION_CYCLE / ENVELOPE_SAMPLES as f32;
            self.geometry(&mut mesh, phase, f32::INFINITY);
            let mut minimum = Vec3::splat(f32::INFINITY);
            let mut maximum = Vec3::splat(f32::NEG_INFINITY);
            for vertex in vertices.iter() {
                let position = Vec3::from_array(vertex.position);
                minimum = minimum.min(position);
                maximum = maximum.max(position);
                max_radius_squared = max_radius_squared.max(position.length_squared());
            }
            max_span = max_span.max((maximum - minimum).max_element());
        }
        (max_span, max_radius_squared.sqrt())
    }

    fn ray(self, mesh: &mut Mesh<'_>, time: f32, detail: usize) {
        let span = 0.38 + 0.14 * self.spread;
        let stroke = (time * 3.1).sin() * 0.16;
        let sweep = if self.hooks {
            -0.25
        } else {
            0.12 + self.morph * 0.23
        };
        let head = Vec3::new(0.0, 0.0, -0.30);
        mesh.ellipsoid(
            Vec3::new(0.0, 0.0, -0.08),
            Vec3::new(0.11, 0.075, 0.27),
            detail,
            self.body,
        );
        for side in [-1.0, 1.0] {
            let hinges = |u: f32| {
                let side_span = if self.offset_fins && side < 0.0 {
                    span * 0.57
                } else {
                    span
                };
                let x = side * side_span * u;
                let fan = if self.crown {
                    (u * PI).sin() * 0.25
                } else {
                    0.
                };
                let y = stroke * u.powi(2) + fan;
                let z = -0.28 + sweep * u.powi(2);
                let trailing = z + (1.0 - u).sqrt() * (0.36 + self.morph * 0.20);
                (Vec3::new(x, y, z), Vec3::new(x, y - 0.032 * u, trailing))
            };
            for i in 0..detail * 2 {
                let u = i as f32 / (detail * 2) as f32;
                let v = (i + 1) as f32 / (detail * 2) as f32;
                let (front, rear) = hinges(u);
                let (next_front, next_rear) = hinges(v);
                // Alternating insets leave windows between the ribbed fin lobes.
                let window = if self.open_frame {
                    0.23
                } else if self.plated {
                    0.40
                } else {
                    0.66
                };
                let edge = front.lerp(rear, window);
                let next_edge = next_front.lerp(next_rear, window);
                mesh.quad(front, next_front, next_edge, edge, self.membrane);
                if !self.crown && (!self.open_frame || i % 2 == 0) {
                    mesh.triangle(edge, next_edge, rear, self.membrane);
                }
                if self.crown {
                    let crown_tip = front
                        + Vec3::new(side * (u - 0.15) * 0.12, 0.10 + (1.0 - u) * 0.14, -0.035);
                    mesh.bone(front, crown_tip, 0.012, 0.001, self.body);
                }
                let rib = if i % 2 == 0 { self.body } else { self.membrane };
                mesh.bone(front, rear, 0.008 + (1.0 - u) * 0.010, 0.003, rib);
                if i % 2 == 0 {
                    mesh.bone(front, next_front, 0.009, 0.005, self.light);
                }
            }
            mesh.ellipsoid(
                head + Vec3::new(side * 0.068, 0.034, 0.014),
                Vec3::splat(0.022),
                1,
                self.light,
            );
            let jaw = head + Vec3::X * side * 0.05;
            if self.hooks {
                self.claw(mesh, jaw, side, time);
            } else if self.horns {
                self.horn(mesh, jaw, side, detail);
            }
        }
        self.keel_tail(mesh, time, detail);
        self.tentacles(
            mesh,
            time,
            detail,
            self.limbs,
            Vec3::new(0.0, -0.025, 0.20),
            0.1,
            Vec3::Z,
            0.31,
        );
    }

    fn jelly(self, mesh: &mut Mesh<'_>, time: f32, detail: usize) {
        let pulse = (time * 2.7).sin();
        let width = (0.19 + self.spread * 0.16) * (1.0 + pulse * 0.17);
        let height = if self.upright {
            0.41
        } else {
            0.17 + (1.0 - self.spread) * 0.20
        };
        let height = height * (1.0 - pulse * 0.10);
        let ribs = self.limbs.max(6);
        let arcs = ribs * 2;
        let bell = |angle: f32, t: f32| {
            let flare = (1.0 - (1.0 - t).powi(2)).sqrt();
            let r = width * flare * (1.0 + 0.045 * (angle * ribs as f32).cos() * t);
            Vec3::new(r * angle.cos(), height * (1.0 - t) - 0.06, r * angle.sin())
        };
        for edge in 0..arcs {
            let a = edge as f32 * TAU / arcs as f32;
            let b = (edge + 1) as f32 * TAU / arcs as f32;
            for j in 0..detail {
                let s = j as f32 / detail as f32;
                let t = (j + 1) as f32 / detail as f32;
                mesh.quad(
                    bell(a, s),
                    bell(a, t),
                    bell(b, t),
                    bell(b, s),
                    self.membrane,
                );
                if edge % 2 == 0 {
                    mesh.bone(bell(a, s), bell(a, t), 0.010, 0.012, self.body);
                }
            }
            mesh.bone(bell(a, 1.0), bell(b, 1.0), 0.012, 0.012, self.light);
            if (self.horns || self.crown) && edge % 2 == 0 {
                mesh.bone(
                    bell(a, 1.0),
                    bell(a, 1.0) + Vec3::new(a.cos(), 1.5, a.sin()) * 0.13,
                    0.016,
                    0.001,
                    self.body,
                );
            }
        }
        mesh.ellipsoid(
            Vec3::new(0.0, 0.04, 0.0),
            Vec3::new(width * 0.33, height * 0.54, width * 0.33),
            detail,
            self.body,
        );
        if self.fins {
            self.wings(mesh, Vec3::new(0.0, 0.07, 0.0), time, detail);
        }
        if self.nested {
            for i in 0..3 {
                let center = Vec3::new(0.0, -0.09 - i as f32 * 0.08, 0.0);
                let spread = width * (0.73 - i as f32 * 0.18);
                self.shell(mesh, center, spread, 0.065, detail);
            }
        }
        self.tentacles(
            mesh,
            time,
            detail,
            self.limbs,
            Vec3::new(0.0, -0.06, 0.0),
            width * 0.92,
            Vec3::NEG_Y,
            0.49,
        );
    }

    fn arthropod(self, mesh: &mut Mesh<'_>, time: f32, detail: usize) {
        let width = 0.075 + 0.15 * self.spread;
        let segments = (detail + 2).min(8);
        for i in 0..segments {
            let t = i as f32 / (segments - 1) as f32;
            let shape = 0.64 + 0.36 * (t * PI).sin();
            let hump = if self.upright {
                0.14
            } else {
                0.014 + self.morph * 0.060
            };
            let center = Vec3::new(0.0, hump * (t * PI).sin(), -0.25 + 0.48 * t);
            mesh.ellipsoid(
                center,
                Vec3::new(width * shape, 0.065 * shape, 0.31 / segments as f32),
                detail.min(4),
                self.body,
            );
            if self.plated || detail > 3 {
                mesh.bone(
                    center + Vec3::Y * 0.032,
                    center + Vec3::new(0.0, 0.14 * shape, 0.075),
                    0.019,
                    0.001,
                    self.light,
                );
            }
        }
        if self.fins {
            self.wings(mesh, Vec3::new(0.0, 0.035, -0.06), time, detail);
        }
        if self.shelled && !self.fins {
            self.shell(
                mesh,
                Vec3::new(0.0, 0.13, 0.0),
                width * 1.17,
                0.085 + self.morph * 0.12,
                detail,
            );
        }
        let pairs = self.limbs.div_ceil(2);
        for side in [-1.0, 1.0] {
            mesh.ellipsoid(
                Vec3::new(side * 0.065, 0.023, -0.28),
                Vec3::splat(0.023),
                1,
                self.light,
            );
            let jaw = Vec3::new(side * 0.065, 0.0, -0.28);
            if self.hooks {
                self.claw(mesh, jaw, side, time);
            } else if self.horns {
                self.horn(mesh, jaw, side, detail);
            }
            for limb in 0..pairs {
                if self.limbs % 2 == 1 && limb + 1 == pairs && side < 0.0 {
                    continue;
                }
                let t = limb as f32 / pairs as f32;
                let z = -0.24 + 0.47 * t;
                let hip = Vec3::new(side * width * 0.7, -0.03, z);
                let stride = (time * 4.4 + t * TAU + side).sin();
                let lift = 0.040 + (time * 4.4 + t * TAU + side).cos() * 0.056;
                let thigh = 0.082 + 0.122 * self.spread + 0.028 * (-3.0 * t).exp() * self.morph;
                let thigh = if self.offset_fins && side < 0.0 {
                    thigh * 0.65
                } else {
                    thigh
                };
                let knee =
                    hip + Vec3::new(side * 0.67, 0.32 + lift, stride * 0.72).normalize() * thigh;
                let ankle = knee
                    + Vec3::new(side * 0.61, -0.70, stride * 0.72 + 0.3).normalize()
                        * (thigh * 1.3);
                let foot =
                    ankle + Vec3::new(side * 0.16, -0.7, stride * 0.56 + 0.25).normalize() * 0.13;
                mesh.bone(hip, knee, 0.022, 0.018, self.body);
                mesh.bone(knee, ankle, 0.019, 0.011, self.body);
                mesh.bone(ankle, foot, 0.013, 0.001, self.light);
                mesh.ellipsoid(knee, Vec3::splat(0.023), 1, self.body);
            }
        }
        self.tentacles(
            mesh,
            time,
            detail,
            pairs.max(2),
            Vec3::new(0.0, -0.025, 0.23),
            width * 0.5,
            Vec3::Z,
            0.37,
        );
    }

    fn ribbon(self, mesh: &mut Mesh<'_>, time: f32, detail: usize) {
        let segments = 5 + detail * 2;
        let width = 0.062 + self.spread * 0.17;
        let spine = |t: f32| {
            Vec3::new(
                (time * 3.3 - t * (4.5 + self.morph * 6.0)).sin() * t * (0.12 + self.spread * 0.16),
                (time * 2.4 - t * 8.).cos() * t * (0.06 + self.morph * 0.70 * 0.13),
                t - 0.48,
            )
        };
        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let next = (i + 1) as f32 / segments as f32;
            let p = spine(t);
            let q = spine(next);
            if self.nested && i % 3 == 0 {
                self.shell(
                    mesh,
                    p + Vec3::Y * 0.04,
                    width * (1.0 - t) * 0.9,
                    0.07 * (1.0 - t),
                    detail.min(3),
                );
            }
            let r = width * (1.0 - t).powf(0.85);
            let n = width * (1.0 - next).powf(0.85);
            mesh.ellipsoid(
                p,
                Vec3::new(r * 0.56, r * 0.33, 0.57 / segments as f32),
                detail.min(3),
                self.body,
            );
            mesh.bone(p, q, r * 0.35, n * 0.35, self.body);
            for side in [-1.0, 1.0] {
                let wing = p + Vec3::new(side * r * 1.55, -r * 0.42, 0.10);
                let next_wing = q + Vec3::new(side * n * 1.56, -n * 0.42, 0.10);
                mesh.triangle(p, q, wing, self.membrane);
                if self.plated {
                    mesh.triangle(q, wing, next_wing, self.membrane);
                }
                mesh.bone(p, wing, r * 0.13, 0.001, self.light);
            }
        }
        for side in [-1.0, 1.0] {
            let anchor = spine(0.) + Vec3::new(side * width * 0.42, 0.008, -0.015);
            mesh.ellipsoid(anchor, Vec3::splat(0.020), 1, self.light);
            if self.horns || self.hooks {
                self.horn(mesh, anchor, side, detail);
            }
        }
        if self.fins {
            self.wings(mesh, spine(0.15), time, detail);
        }
        if self.split_tail {
            for side in [-1.0, 1.0] {
                let center = spine(0.83);
                let tip = spine(1.0) + Vec3::new(side * 0.09, 0.0, 0.07);
                mesh.bone(center, tip, 0.018, 0.002, self.body);
            }
        }
        self.tentacles(
            mesh,
            time,
            detail,
            self.limbs,
            spine(0.82),
            width * 0.18,
            Vec3::Z,
            0.29,
        );
    }

    fn keel_tail(self, mesh: &mut Mesh<'_>, time: f32, detail: usize) {
        let count = if self.split_tail { 2 } else { 1 };
        for branch in 0..count {
            let side = if count == 1 {
                0.
            } else {
                branch as f32 * 2. - 1.
            };
            let path = |t: f32| {
                let sway = (time * 2.6 - t * 6.).sin() * t * 0.040;
                let curve = if self.hooks {
                    (t * PI * 0.7).sin() * t * 0.18
                } else {
                    0.
                };
                Vec3::new(side * 0.11 * t + sway, curve, 0.14 + t * 0.36)
            };
            for i in 0..detail {
                let t = i as f32 / detail as f32;
                let point = path(t);
                let next = path((i + 1) as f32 / detail as f32);
                mesh.bone(
                    point,
                    next,
                    0.023 * (1.05 - t),
                    0.012 * (1.01 - t),
                    self.body,
                );
                if self.plated || self.hooks {
                    for side in [-1.0, 1.0] {
                        mesh.bone(
                            point,
                            point + Vec3::new(side * 0.04, 0.018, 0.055) * (1. - t),
                            0.011,
                            0.001,
                            self.light,
                        );
                    }
                }
            }
        }
    }

    fn shell(self, mesh: &mut Mesh<'_>, center: Vec3, radius: f32, height: f32, detail: usize) {
        let spokes = 5 + detail;
        for i in 0..spokes {
            let a = i as f32 / spokes as f32 * TAU;
            let b = (i + 1) as f32 / spokes as f32 * TAU;
            let rim =
                |angle: f32| center + Vec3::new(radius * angle.cos(), 0., radius * angle.sin());
            let ridge = |angle: f32| {
                center
                    + Vec3::new(
                        radius * 0.72 * angle.cos(),
                        height * 0.65,
                        radius * 0.72 * angle.sin(),
                    )
            };
            let cap = center + Vec3::Y * height;
            mesh.triangle(cap, ridge(a), ridge(b), self.membrane);
            if !self.open_frame {
                mesh.quad(ridge(a), rim(a), rim(b), ridge(b), self.membrane);
            }
            mesh.bone(cap, ridge(a), 0.009, 0.008, self.body);
            mesh.bone(ridge(a), rim(a), 0.008, 0.006, self.body);
            mesh.bone(rim(a), rim(b), 0.005, 0.005, self.light);
        }
    }

    fn wings(self, mesh: &mut Mesh<'_>, anchor: Vec3, time: f32, detail: usize) {
        for side in [-1.0, 1.0] {
            let hinge = (time * 3.8 + side * 0.11).sin() * 0.40;
            let swivel = Quat::from_rotation_z(side * hinge);
            let span = 0.24 + self.spread * 0.22;
            for wing in 0..2 {
                let sail = |t: f32| {
                    let lead = -0.12 + wing as f32 * 0.12;
                    let z = lead + t * t * (0.20 + wing as f32 * 0.16);
                    let chord = (t * PI).sin() * (0.09 + self.morph * 0.13);
                    (
                        anchor + swivel * Vec3::new(side * span * t, t * 0.025, z),
                        anchor + swivel * Vec3::new(side * span * t, -t * 0.025, z + chord),
                    )
                };
                for i in 0..detail * 2 {
                    let (a, b) = sail(i as f32 / (detail * 2) as f32);
                    let (c, d) = sail((i + 1) as f32 / (detail * 2) as f32);
                    if self.open_frame && i % 2 != 0 {
                        mesh.triangle(a, c, d, self.membrane);
                    } else {
                        mesh.quad(a, c, d, b, self.membrane);
                    }
                    mesh.bone(a, c, 0.007, 0.004, self.body);
                    mesh.bone(a, d, 0.006, 0.003, self.light);
                }
            }
        }
    }

    fn claw(self, mesh: &mut Mesh<'_>, anchor: Vec3, side: f32, time: f32) {
        let stroke = (time * 2.8 + side * 0.77).sin() * 0.31;
        let hinge = Quat::from_rotation_y(side * stroke);
        let elbow = anchor + hinge * Vec3::new(side * 0.10, 0.071, -0.060);
        let wrist = elbow + hinge * Vec3::new(side * 0.037, 0.027, -0.105);
        mesh.bone(anchor, elbow, 0.017, 0.024, self.body);
        mesh.bone(elbow, wrist, 0.024, 0.017, self.body);
        let gape = 0.018 + (time * 4.0).cos().powi(2) * 0.031;
        for digit in [-1.0, 1.0] {
            let bend = wrist + hinge * Vec3::new(gape * digit, 0.022, -0.065);
            let tip = bend + hinge * Vec3::new(-gape * digit * 0.75, 0.010, -0.042);
            mesh.bone(wrist, bend, 0.015, 0.010, self.body);
            mesh.bone(bend, tip, 0.010, 0.001, self.light);
        }
    }

    fn horn(self, mesh: &mut Mesh<'_>, anchor: Vec3, side: f32, detail: usize) {
        let points = if self.horns { 3 } else { 2 };
        let mut start = anchor;
        for i in 0..points {
            let end = start
                + Vec3::new(
                    side * (0.08 - 0.018 * i as f32),
                    0.05,
                    -0.035 - 0.018 * i as f32,
                );
            mesh.bone(
                start,
                end,
                0.027 / (i + 1) as f32,
                0.011 / (i + 1) as f32,
                self.body,
            );
            if self.horns && detail > 2 {
                let branch = start.lerp(end, 0.64);
                mesh.bone(
                    branch,
                    branch + Vec3::new(side * 0.027, 0.071, -0.014),
                    0.01,
                    0.001,
                    self.light,
                );
            }
            start = end;
        }
        mesh.bone(
            start,
            start + Vec3::new(side * 0.01, 0.025, -0.05),
            0.009,
            0.001,
            self.body,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn tentacles(
        self,
        mesh: &mut Mesh<'_>,
        time: f32,
        detail: usize,
        count: usize,
        origin: Vec3,
        width: f32,
        direction: Vec3,
        length: f32,
    ) {
        let steps = 3 + detail * 2;
        let lateral = if direction == Vec3::Z {
            Vec3::Y
        } else {
            Vec3::Z
        };
        for index in 0..count {
            let phi = index as f32 / count as f32 * TAU;
            let sideways = Vec3::X * phi.cos() + lateral * phi.sin();
            let extent = length * (0.70 + 0.55 * hash(self.seed + index as f32 * 11.23));
            let path = |t: f32| {
                origin
                    + direction * (extent * t)
                    + sideways * (width + t * 0.08 + (time * 3.4 - t * 9. + phi).sin() * t * 0.085)
                    + lateral * (t * (time * 2.2 - t * 7. + phi).cos() * 0.055)
            };
            let mut prev = path(0.0);
            for i in 0..steps {
                let t = (i + 1) as f32 / steps as f32;
                let next = path(t);
                mesh.bone(
                    prev,
                    next,
                    0.012 * (1.1 - t),
                    0.012 * (1.0 - t) + 0.0007,
                    self.light,
                );
                prev = next;
            }
        }
    }
}

fn has_any(text: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| text.contains(term))
}

struct Mesh<'a> {
    vertices: &'a mut Vec<Vertex>,
    origin: Vec3,
    basis: Quat,
    size: f32,
    facets: usize,
    truncated: bool,
}

impl Mesh<'_> {
    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        if self.vertices.len() + 3 > MAX_VERTICES {
            self.truncated = true;
            return;
        }
        for point in [a, b, c] {
            self.vertices.push(Vertex {
                position: (self.origin + self.basis * (point * self.size)).to_array(),
                color,
                ..Vertex::default()
            });
        }
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
        self.triangle(a, b, c, color);
        self.triangle(a, c, d, color);
    }

    fn ellipsoid(&mut self, center: Vec3, scale: Vec3, detail: usize, color: [f32; 4]) {
        let rings = 2 + detail;
        let slices = 4 + detail * 2;
        let point = |ring: usize, slice: usize| {
            let theta = ring as f32 * PI / rings as f32;
            let phi = slice as f32 * TAU / slices as f32;
            center
                + scale
                    * Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    )
        };
        for ring in 0..rings {
            for slice in 0..slices {
                let a = point(ring, slice);
                let b = point(ring + 1, slice);
                let c = point(ring + 1, slice + 1);
                let d = point(ring, slice + 1);
                if ring != 0 {
                    self.triangle(a, c, d, color);
                }
                if ring + 1 != rings {
                    self.triangle(a, b, c, color);
                }
            }
        }
    }

    fn bone(&mut self, start: Vec3, end: Vec3, r0: f32, r1: f32, color: [f32; 4]) {
        let tangent = (end - start).normalize_or_zero();
        if tangent == Vec3::ZERO {
            return;
        }
        let (a, b) = tangent.any_orthonormal_pair();
        let quad = [a, b, -a, -b];
        let tri = [a, -0.5 * a + 0.866_025_4 * b, -0.5 * a - 0.866_025_4 * b];
        let cross_section: &[Vec3] = if self.facets == 4 { &quad } else { &tri };
        for i in 0..self.facets {
            let v = cross_section[i];
            let w = cross_section[(i + 1) % self.facets];
            self.quad(
                start + v * r0,
                end + v * r1,
                end + w * r1,
                start + w * r0,
                color,
            );
        }
    }
}

struct Animal {
    form: Form,
    model_scale: f32,
    #[cfg_attr(not(test), allow(dead_code))]
    clearance_radius: f32,
    anchor: Vec3,
    drift: f32,
    pace: f32,
    phase: f32,
}

impl Animal {
    fn new(specimen: &Specimen, scratch: &mut Vec<Vertex>) -> Self {
        let form = Form::new(specimen);
        let (span, mesh_radius) = form.envelope(1.0, 0.0, scratch);
        let model_scale = specimen.size / span;
        // Allow for extrema between the sampled phases and minor LOD changes.
        let clearance_radius = mesh_radius * model_scale * 1.08;
        let index = (specimen.id - 1) as f32;
        let spherical_y = 1.0 - 2.0 * (index + 0.5) / COUNT as f32;
        let polar = index * 2.399_963_1;
        let radial = (1.0 - spherical_y * spherical_y).sqrt();
        let direction = Vec3::new(radial * polar.cos(), spherical_y, radial * polar.sin());
        let near = specimen.id % 4 == 0;
        let radius = if near {
            4. + hash(index * 2.71) * 10.
        } else {
            15. + hash(index * 2.71) * 14.
        };
        let drift = 1.4 + hash(index * 5.8) * 3.8;
        let max_anchor = RADIUS - clearance_radius - drift * 1.12 - 0.5;
        Self {
            form,
            model_scale,
            clearance_radius,
            anchor: direction * radius.min(max_anchor),
            drift,
            pace: 0.11 + 0.41 * (1.0 - specimen.size / 10.0),
            phase: hash(index * 7.9) * TAU,
        }
    }

    fn posture(&self, time: f32) -> (Vec3, Quat) {
        let phase = time * self.pace + self.phase;
        let tilt = self.phase.sin();
        let offset = Vec3::new(phase.cos(), phase.sin() * tilt * 0.48, phase.sin()) * self.drift;
        let heading = Vec3::new(-phase.sin(), phase.cos() * tilt * 0.48, phase.cos()).normalize();
        let basis = match self.form.family {
            Family::Jelly => Quat::from_rotation_y(phase) * Quat::from_rotation_z(0.24 * tilt),
            _ => {
                Quat::from_rotation_arc(Vec3::NEG_Z, heading)
                    * Quat::from_rotation_z((phase * 1.5).sin() * 0.23)
            }
        };
        (self.anchor + offset, basis)
    }
}

fn hash(x: f32) -> f32 {
    (x.sin() * 12_345.679_f32).rem_euclid(1.0)
}

pub struct School {
    animals: Vec<Animal>,
}

impl School {
    pub fn load() -> Result<Self> {
        let mut specimens: Vec<Specimen> =
            serde_json::from_str(include_str!("creatures.json"))
                .context("Reading the Aquarium specimen catalog")?;
        specimens.sort_unstable_by_key(|specimen| specimen.id);
        ensure!(specimens.len() == COUNT, "Aquarium needs {COUNT} specimens");
        for (index, spec) in specimens.iter().enumerate() {
            ensure!(
                spec.id == index + 1
                    && !spec.name.trim().is_empty()
                    && !spec.anatomy.trim().is_empty()
                    && (0.5..=10.0).contains(&spec.size)
                    && (0.0..=1.0).contains(&spec.proportion)
                    && (0.0..=1.0).contains(&spec.detail)
                    && (2..=12).contains(&spec.appendages),
                "Invalid Aquarium specimen entry for {} ({})",
                spec.id,
                spec.name
            );
        }
        let smallest = specimens
            .iter()
            .map(|s| s.size)
            .fold(f32::INFINITY, f32::min);
        let largest = specimens.iter().map(|s| s.size).fold(0., f32::max);
        ensure!(
            (largest / smallest - 20.).abs() < 0.01,
            "Aquarium size range must span 20×"
        );
        let mut scratch = Vec::new();
        Ok(Self {
            animals: specimens
                .iter()
                .map(|specimen| Animal::new(specimen, &mut scratch))
                .collect(),
        })
    }
}

impl School {
    /// Animal `index`'s posed triangles at time `time` (fixed detail, so the vertex count
    /// stays the same from frame to frame), appended to `out`. Returns its centre.
    pub fn pose(&self, index: usize, time: f32, out: &mut Vec<Vertex>) -> Vec3 {
        let animal = &self.animals[index % self.animals.len()];
        let (origin, basis) = animal.posture(time);
        let form = animal.form;
        let mut mesh = Mesh { vertices: out, origin, basis, size: animal.model_scale, facets: 3, truncated: false };
        let cycle = animal.phase + time * (1.12 - (form.size / 10.0).sqrt() * 0.65);
        form.geometry(&mut mesh, cycle, 0.);
        origin
    }

    pub fn len(&self) -> usize {
        self.animals.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_catalog_animals_make_distinct_finite_animated_geometry() {
        let school = School::load().unwrap();
        let mut signatures = std::collections::HashSet::new();
        for animal in school.animals {
            let mut frames = [Vec::new(), Vec::new()];
            for (i, vertices) in frames.iter_mut().enumerate() {
                let mut mesh = Mesh {
                    vertices,
                    origin: Vec3::ZERO,
                    basis: Quat::IDENTITY,
                    size: animal.model_scale,
                    facets: 3,
                    truncated: false,
                };
                animal.form.geometry(&mut mesh, i as f32 * 0.53, 0.25);
                assert!(vertices.len() >= 90 && vertices.len() % 3 == 0);
                assert!(vertices
                    .iter()
                    .all(|v| v.position.iter().all(|p| p.is_finite())));
            }
            assert!(frames[0]
                .iter()
                .zip(&frames[1])
                .any(|(a, b)| a.position != b.position));
            let key = frames[0]
                .iter()
                .map(|v| v.position.map(f32::to_bits))
                .collect::<Vec<_>>();
            assert!(signatures.insert(key));
            for frame in 0..24 {
                let (position, _) = animal.posture(frame as f32);
                assert!(position.length() + animal.clearance_radius < RADIUS);
            }
        }
    }

    #[test]
    fn animated_meshes_span_twentyfold_in_meters() {
        let school = School::load().unwrap();
        let mut scratch = Vec::new();
        let (mut smallest, mut largest) = (f32::INFINITY, 0.0_f32);
        for animal in school.animals {
            let (span, radius) = animal.form.envelope(animal.model_scale, 0.5, &mut scratch);
            assert!(
                (span / animal.form.size - 1.0).abs() < 0.02,
                "specimen {} span {} vs catalog {}",
                animal.form.seed,
                span,
                animal.form.size
            );
            assert!(radius <= animal.clearance_radius);
            assert!(animal.anchor.length() + animal.drift * 1.12 + radius < RADIUS);
            smallest = smallest.min(span);
            largest = largest.max(span);
        }
        assert!((smallest - 0.5).abs() < 0.01);
        assert!((largest - 10.0).abs() < 0.10);
        assert!((largest / smallest - 20.0).abs() < 0.25);
    }
}
