//! Collisions against the image's own geometry as a sensor field (World::sensors): per
//! direction from a centre, the distance to the geometry. For a body at `position`, a
//! sensor's clearance is its distance minus the body's offset from the centre along it
//! (first order; exact for a wall facing the sensor). The clearance goes negative past the
//! surface, so the field knows inside from outside: a body that got through is still
//! pushed back out the way it came, never further in.
//! Moving toward a sensor within `reach` is braked (hard to go closer, free to depart), and
//! a gentle push (scaled by the caller) eases the body back out to the reach.

use glam::Vec3;

/// The sensor field: its centre (world) and per sensor (unit direction, distance in m).
#[derive(Default)]
pub struct Sensors {
    pub centre: Vec3,
    pub field: Vec<(Vec3, f32)>,
}

impl Sensors {
    pub fn len(&self) -> usize {
        self.field.len()
    }

    /// The sensors with less than `reach` clearance for a body at `position`: (direction,
    /// clearance; negative: past the surface).
    fn within(&self, position: Vec3, reach: f32) -> impl Iterator<Item = (Vec3, f32)> + '_ {
        let offset = position - self.centre;
        self.field.iter().map(move |&(dir, distance)| (dir, distance - offset.dot(dir))).filter(move |&(_, gap)| gap < reach)
    }

    /// `motion` (a step or a velocity) of a body at `position` without its part toward the
    /// sensors with less than `reach` clearance: none of it within `firm` (or past the
    /// surface), fading out from there to the reach. First along their mean direction (a
    /// wall's normal: no sideways drift from its oblique sensors), then sensor by sensor
    /// down to each one's allowance (corners), so a wall of many sensors brakes like one.
    /// Moving away or along is free.
    pub fn brake(&self, position: Vec3, mut motion: Vec3, reach: f32, firm: f32) -> Vec3 {
        let strength = |gap: f32| ((reach - gap) / (reach - firm).max(1e-3)).min(1.);
        let near: Vec<(Vec3, f32)> = self.within(position, reach).collect();
        // Each sensor allows (1 - strength) of the approach toward it the motion had.
        let allowed: Vec<f32> = near.iter().map(|&(dir, gap)| motion.dot(dir).max(0.) * (1. - strength(gap))).collect();
        let (sum, weight) = near.iter().fold((Vec3::ZERO, 0.), |(sum, weight), &(dir, gap)| (sum + dir * strength(gap), weight + strength(gap)));
        if let Some(normal) = sum.try_normalize() {
            motion -= normal * motion.dot(normal).max(0.) * (weight / near.len() as f32);
        }
        for (&(dir, _), allowed) in near.iter().zip(allowed) {
            motion -= dir * (motion.dot(dir) - allowed).max(0.);
        }
        motion
    }

    /// The push that would bring a body at `position` back to `reach` clearance: the mean
    /// over the sensors within reach of their shortfall, away along each (so the sensors'
    /// density doesn't matter); zero when none is within reach.
    pub fn push(&self, position: Vec3, reach: f32) -> Vec3 {
        let (mut push, mut hits) = (Vec3::ZERO, 0);
        for (dir, gap) in self.within(position, reach) {
            push -= dir * (reach - gap);
            hits += 1;
        }
        if hits == 0 {
            return Vec3::ZERO;
        }
        let push = push / hits as f32;
        if push.is_finite() { push } else { Vec3::ZERO }
    }

    /// The smallest clearance at `position` (m; infinite without sensors), for logging.
    pub fn nearest(&self, position: Vec3) -> f32 {
        self.within(position, f32::INFINITY).map(|(_, gap)| gap).fold(f32::INFINITY, f32::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wall 2 m along +x from the centre (1, 2, 3), seen by a fan of sensors.
    fn wall() -> Sensors {
        let field = (-5..=5)
            .map(|i| {
                let dir = Vec3::new(1., i as f32 * 0.05, 0.).normalize();
                (dir, 2. / dir.x)
            })
            .collect();
        Sensors { centre: Vec3::new(1., 2., 3.), field }
    }

    #[test]
    fn braking_stops_approach_but_not_departure() {
        let wall = wall();
        // 0.4 m from the wall, reach 1, firm 0.5: no closer, sliding along stays.
        let at = wall.centre + Vec3::X * 1.6;
        let toward = wall.brake(at, Vec3::new(0.1, 0., 0.1), 1., 0.5);
        assert!(toward.x.abs() < 1e-3 && toward.y.abs() < 1e-3 && (toward.z - 0.1).abs() < 1e-5, "{toward}");
        let away = wall.brake(at, Vec3::new(-0.1, 0., 0.), 1., 0.5);
        assert!((away.x + 0.1).abs() < 1e-5, "{away}");
        // Past the wall: still no further in.
        let inside = wall.brake(wall.centre + Vec3::X * 2.5, Vec3::X * 0.1, 1., 0.5);
        assert!(inside.x.abs() < 1e-3, "{inside}");
        // Between firm and the reach: part of the approach is left.
        let facing = Sensors { centre: Vec3::ZERO, field: vec![(Vec3::X, 2.)] };
        let half = facing.brake(Vec3::X * 1.25, Vec3::X * 0.1, 1., 0.5);
        assert!((half.x - 0.05).abs() < 1e-5, "{half}");
    }

    #[test]
    fn the_push_points_out_even_from_inside() {
        let wall = wall();
        assert_eq!(wall.push(wall.centre, 1.), Vec3::ZERO);
        // 0.2 m inside the reach of the facing sensor.
        let facing = Sensors { centre: Vec3::ZERO, field: vec![(Vec3::X, 2.)] };
        let near = facing.push(Vec3::X * 1.2, 1.);
        assert!((near - Vec3::NEG_X * 0.2).length() < 1e-5, "{near}");
        for x in [1.5, 2., 2.5, 4.] {
            let p = wall.push(wall.centre + Vec3::X * x, 1.);
            assert!(p.x < -0.4 && p.y.abs() < 1e-4, "at {x}: {p}");
        }
        assert!((wall.nearest(wall.centre + Vec3::X * 2.5) + 0.5).abs() < 1e-5);
        assert_eq!(Sensors::default().push(Vec3::ONE, 1.), Vec3::ZERO);
    }
}
