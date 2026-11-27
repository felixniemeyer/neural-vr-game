//! CPU copy of the block layout in shaders/scene.vert (and hash from common.glsl), for
//! raycasts from the controllers. Keep both in sync.

use glam::{IVec2, Vec2, Vec3};

fn fract(v: Vec3) -> Vec3 {
    v - v.floor()
}

/// common.glsl hash(vec3).
fn hash(p: Vec3) -> f32 {
    let mut p = fract(p * Vec3::new(0.1031, 0.1030, 0.0973));
    p += p.dot(Vec3::new(p.y, p.x, p.z) + 33.33);
    let h = (p.x + p.y) * p.z;
    h - h.floor()
}

/// The rise's gaussian width, as a fraction of the field radius (common.glsl SPREAD).
const SPREAD: f32 = 0.45;

pub struct City {
    /// Cell size (m), maximum height (m), blocks per side.
    pub cell: f32,
    pub height: f32,
    pub blocks: u32,
    /// The player's ground position (x, z): blocks rise around it.
    pub player: Vec2,
}

impl City {
    /// The block of a grid cell as (low, high) corners, or None where it has sunk.
    fn block(&self, cell: IVec2) -> Option<(Vec3, Vec3)> {
        let c = cell.as_vec2();
        let d = (c + 0.5) * self.cell - self.player;
        // common.glsl fieldRadius(), scene.vert rise(): 0 at the field radius.
        let field = (self.blocks as f32 * 0.5 - 1.).max(1.) * self.cell;
        let spread = 2. * (SPREAD * field).powi(2);
        let edge = (-field * field / spread).exp();
        let rise = (((-d.dot(d) / spread).exp() - edge) / (1. - edge)).max(0.);
        let height = self.height * (0.25 + 0.75 * hash(c.extend(3.7))) * rise;
        if height < 1e-3 {
            return None;
        }
        let inset = self.cell * (0.12 + 0.12 * hash(c.extend(9.1)));
        let low = c * self.cell + inset;
        let high = low + self.cell - 2. * inset;
        Some((Vec3::new(low.x, 0., low.y), Vec3::new(high.x, height, high.y)))
    }

    /// The cells around the player, as scene.vert instances them.
    fn cells(&self) -> impl Iterator<Item = IVec2> {
        let n = self.blocks as i32;
        let first = (self.player / self.cell).floor().as_ivec2() - n / 2;
        (0..n * n).map(move |i| first + IVec2::new(i % n, i / n))
    }

    /// Nearest point where the ray (origin, unit direction) hits a block, within `range`.
    pub fn hit(&self, origin: Vec3, dir: Vec3, range: f32) -> Option<Vec3> {
        let inv = dir.recip();
        let nearest = self
            .cells()
            .filter_map(|cell| self.block(cell))
            .filter_map(|(lo, hi)| {
                let (a, b) = ((lo - origin) * inv, (hi - origin) * inv);
                let enter = a.min(b).max_element().max(0.);
                (enter <= a.max(b).min_element()).then_some(enter)
            })
            .fold(range, f32::min);
        (nearest < range).then(|| origin + dir * nearest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_hit_blocks_and_miss_the_sky() {
        let city = City { cell: 4., height: 7., blocks: 24, player: Vec2::ZERO };
        let up = Vec3::new(0., 1., 0.);
        assert!(city.hit(Vec3::new(0., 30., 0.), up, 150.).is_none());
        // Looking down from above: the hit is a roof (or the street level is missed).
        let mut roofs = 0;
        for i in 0..20 {
            let at = Vec3::new(i as f32 * 1.3 - 13., 30., 2.1);
            if let Some(hit) = city.hit(at, -up, 150.) {
                assert!(hit.y > 0.5 && hit.y <= 7.);
                roofs += 1;
            }
        }
        assert!(roofs > 5, "{roofs} roofs");
        // Horizontal through a row of blocks ends at a wall.
        assert!(city.hit(Vec3::new(0., 0.3, 2.), Vec3::X, 150.).is_some());
    }
}
