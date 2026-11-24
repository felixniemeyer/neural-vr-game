//! Particle bodies: a few of the sea creatures (creatures.rs) as mass-spring soft bodies.
//! Every mesh point is a particle (the triangle soup's coincident vertices welded); every edge a spring whose rest length follows
//! the animated pose (the swimming acts as muscles); a weak pull toward the pose keeps the
//! shape; a divergence-free current pushes the particles around. So the bodies lag, wobble
//! and smear instead of replaying the animation rigidly.
//!
//! The school swims in a sphere (creatures::RADIUS, scaled) whose centre follows the
//! player slowly. Particles are drawn as points in the scene pass (creatures.vert).

use crate::creatures::{School, Vertex};
use anyhow::Result;
use glam::Vec3;

#[derive(Clone, Copy)]
pub struct Params {
    /// How many creatures (the first n of the catalogue, spread over it).
    pub count: usize,
    /// World metres per school metre.
    pub scale: f32,
    /// Pull toward the pose (1/s²), edge springs (1/s²), drag (1/s).
    pub hold: f32,
    pub link: f32,
    pub drag: f32,
    /// The current's acceleration (m/s²) and its wavelength (m).
    pub flow: f32,
    pub swirl: f32,
    /// Seconds the school's centre needs to follow the player.
    pub follow: f32,
}

pub struct Swarm {
    school: School,
    count: usize,
    target: Vec<Vertex>,
    /// The meshes are triangle soup: coincident vertices are welded into one particle
    /// (weld[vertex] = particle), so neighbouring triangles share particles and springs.
    weld: Vec<usize>,
    /// Springs: particles a, b and two vertices whose posed distance is the rest length.
    edges: Vec<[usize; 4]>,
    /// Per particle: the pose (mean of its welded vertices), position, velocity, colour.
    pose: Vec<Vec3>,
    x: Vec<Vec3>,
    v: Vec<Vec3>,
    color: Vec<[f32; 4]>,
    centre: Option<Vec3>,
    pub out: Vec<Vertex>,
}

impl Swarm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            school: School::load()?, count: 0, target: Vec::new(), weld: Vec::new(), edges: Vec::new(),
            pose: Vec::new(), x: Vec::new(), v: Vec::new(), color: Vec::new(), centre: None, out: Vec::new(),
        })
    }

    /// Weld the current targets into particles and springs; particles start on the pose.
    fn rebuild(&mut self, scale: f32) {
        let grid = 1e-3 * scale.max(1e-3);
        let mut index = std::collections::HashMap::new();
        self.weld = self.target.iter().map(|t| {
            let key = Vec3::from(t.position) / grid;
            let key = (key.x.round() as i64, key.y.round() as i64, key.z.round() as i64);
            let next = index.len();
            *index.entry(key).or_insert(next)
        }).collect();
        let particles = index.len();
        let mut seen = std::collections::HashSet::new();
        self.edges.clear();
        for tri in (0..self.target.len() / 3 * 3).step_by(3) {
            for (a, b) in [(tri, tri + 1), (tri + 1, tri + 2), (tri + 2, tri)] {
                let (pa, pb) = (self.weld[a], self.weld[b]);
                if pa != pb && seen.insert((pa.min(pb), pa.max(pb))) {
                    self.edges.push([pa, pb, a, b]);
                }
            }
        }
        self.pose = vec![Vec3::ZERO; particles];
        self.color = vec![[0.; 4]; particles];
        for (vertex, &particle) in self.weld.iter().enumerate() {
            self.color[particle] = self.target[vertex].color;
        }
        self.x = vec![Vec3::ZERO; particles];
        self.v = vec![Vec3::ZERO; particles];
    }

    /// Advance by dt seconds at animation time `time`; `player`: the player's world position.
    pub fn update(&mut self, p: Params, time: f32, dt: f32, player: Vec3) {
        // The school floats above the ground around the player.
        let goal = player + Vec3::Y * (crate::creatures::RADIUS * 0.5 * p.scale);
        let centre = self.centre.get_or_insert(goal);
        let before = *centre;
        *centre += (goal - *centre) * (1. - (-dt / p.follow.max(0.1)).exp());
        let (centre, shift) = (*centre, *centre - before);

        // Targets: the chosen animals' posed triangles in the world.
        self.target.clear();
        let total = self.school.len();
        for k in 0..p.count.min(total) {
            self.school.pose(k * total / p.count.max(1), time, &mut self.target);
        }
        for t in &mut self.target {
            t.position = (centre + Vec3::from(t.position) * p.scale).to_array();
        }
        let fresh = p.count != self.count || self.weld.len() != self.target.len();
        if fresh {
            self.count = p.count;
            self.rebuild(p.scale);
        }
        let n = self.x.len();
        let mut welded = vec![0u32; n];
        self.pose.fill(Vec3::ZERO);
        for (vertex, &particle) in self.weld.iter().enumerate() {
            self.pose[particle] += Vec3::from(self.target[vertex].position);
            welded[particle] += 1;
        }
        for (pose, &k) in self.pose.iter_mut().zip(&welded) {
            *pose /= k.max(1) as f32;
        }
        if fresh {
            self.x.copy_from_slice(&self.pose);
        }
        // The followed centre moves everything along (no rubber band when the player moves).
        for x in &mut self.x {
            *x += shift;
        }

        let steps = 2;
        let h = dt.min(0.05) / steps as f32;
        let mut force = vec![Vec3::ZERO; n];
        let s = time * 0.3;
        for _ in 0..steps {
            for i in 0..n {
                let x = self.x[i];
                let q = x / p.swirl.max(0.1);
                // Each component ignores its own axis: divergence-free.
                let current = Vec3::new((q.y + s).sin() + (q.z * 1.3).cos(), (q.z + s).sin() + (q.x * 1.3).cos(), (q.x + s).sin() + (q.y * 1.3).cos());
                force[i] = p.hold * (self.pose[i] - x) + p.flow * current - p.drag * self.v[i];
            }
            for &[a, b, va, vb] in &self.edges {
                let rest = (Vec3::from(self.target[va].position) - Vec3::from(self.target[vb].position)).length();
                let d = self.x[b] - self.x[a];
                let length = d.length();
                if length > 1e-6 {
                    let f = p.link * (length - rest) * d / length;
                    force[a] += f;
                    force[b] -= f;
                }
            }
            for i in 0..n {
                self.v[i] += force[i] * h;
                self.x[i] += self.v[i] * h;
            }
        }

        let follow = shift / dt.max(1e-4);
        self.out.clear();
        self.out.extend((0..n).map(|i| Vertex {
            position: self.x[i].to_array(),
            color: self.color[i],
            velocity: (self.v[i] + follow).to_array(),
            ..Vertex::default()
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_stay_stable_and_move() {
        let mut swarm = Swarm::new().unwrap();
        let p = Params { count: 8, scale: 1., hold: 4., link: 120., drag: 1.5, flow: 2., swirl: 6., follow: 10. };
        let mut start = Vec::new();
        let clock = std::time::Instant::now();
        for frame in 0..900 {
            swarm.update(p, frame as f32 / 90., 1. / 90., Vec3::new(frame as f32 * 0.02, 1.6, 0.));
            if frame == 0 {
                start = swarm.out.iter().map(|v| Vec3::from(v.position)).collect();
            }
        }
        println!("{:.2} ms per update", clock.elapsed().as_secs_f64() * 1000. / 900.);
        let n = swarm.out.len();
        assert!(n > 1000, "{n} particles");
        let spread = swarm.out.iter().zip(&swarm.pose).map(|(o, t)| (Vec3::from(o.position) - *t).length()).fold(0f32, f32::max);
        let moved = swarm.out.iter().zip(&start).map(|(o, s)| (Vec3::from(o.position) - *s).length()).sum::<f32>() / n as f32;
        println!("{n} particles ({} vertices, {} springs), max distance from pose {spread:.2} m, mean travel {moved:.2} m", swarm.target.len(), swarm.edges.len());
        assert!(swarm.out.iter().all(|v| v.position.iter().all(|x| x.is_finite())));
        assert!(spread < 5., "bodies fly apart: {spread}");
        assert!(moved > 0.1);
    }
}
