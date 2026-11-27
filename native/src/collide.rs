//! The desktop camera's collisions against the sphere's sensor field (World::sensors):
//! per direction from the sphere's centre, the distance to the geometry. For the camera at
//! `offset` from that centre, a sensor's clearance is its distance minus the offset along
//! it (first order; exact for a wall facing the sensor). The clearance goes negative past
//! the surface, so the field knows inside from outside: a camera that got through is still
//! pushed back out the way it came, never further in.
//! Moving toward a sensor within `reach` is braked (hard to go closer, free to depart), and
//! a gentle push (scaled by the caller) eases the camera back out to the reach.

use glam::Vec3;

/// The camera's clearance along a sensor (m; negative: past the surface).
fn clearance((dir, distance): (Vec3, f32), offset: Vec3) -> f32 {
    distance - offset.dot(dir)
}

/// `step` (the camera's movement this frame) without its part toward the sensors with less
/// than `reach` clearance: none of it within half the reach (or past the surface), fading
/// out toward the reach. First along their mean direction (a wall's normal: no sideways
/// drift from its oblique sensors), then sensor by sensor (corners), so a wall of many
/// sensors brakes like one. Moving away or along is untouched.
pub fn brake(sensors: &[(Vec3, f32)], offset: Vec3, mut step: Vec3, reach: f32) -> Vec3 {
    let strength = |gap: f32| ((reach - gap) / (0.5 * reach)).min(1.);
    let near: Vec<(Vec3, f32)> = sensors
        .iter()
        .map(|&sensor| (sensor.0, clearance(sensor, offset)))
        .filter(|&(_, gap)| gap < reach)
        .collect();
    let (sum, weight) = near.iter().fold((Vec3::ZERO, 0.), |(sum, weight), &(dir, gap)| (sum + dir * strength(gap), weight + strength(gap)));
    if let Some(normal) = sum.try_normalize() {
        step -= normal * step.dot(normal).max(0.) * (weight / near.len() as f32);
    }
    for (dir, gap) in near {
        step -= dir * step.dot(dir).max(0.) * strength(gap);
    }
    step
}

/// The push that would bring the camera back to `reach` clearance: the mean over the
/// sensors within reach of their shortfall, away along each (so the sensors' density
/// doesn't matter); zero when none is within reach.
pub fn push(sensors: &[(Vec3, f32)], offset: Vec3, reach: f32) -> Vec3 {
    let (mut push, mut hits) = (Vec3::ZERO, 0);
    for &sensor in sensors {
        let gap = clearance(sensor, offset);
        if gap < reach {
            push -= sensor.0 * (reach - gap);
            hits += 1;
        }
    }
    if hits == 0 {
        return Vec3::ZERO;
    }
    let push = push / hits as f32;
    if push.is_finite() { push } else { Vec3::ZERO }
}

/// The smallest clearance (m; infinite without sensors), for the status line.
pub fn nearest(sensors: &[(Vec3, f32)], offset: Vec3) -> f32 {
    sensors.iter().map(|&s| clearance(s, offset)).fold(f32::INFINITY, f32::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wall 2 m along +x from the centre, seen by a fan of sensors.
    fn wall() -> Vec<(Vec3, f32)> {
        (-5..=5)
            .map(|i| {
                let dir = Vec3::new(1., i as f32 * 0.05, 0.).normalize();
                (dir, 2. / dir.x)
            })
            .collect()
    }

    #[test]
    fn braking_stops_approach_but_not_departure() {
        let wall = wall();
        // 0.4 m from the wall, reach 1 (within half of it): no closer, sliding along stays.
        let at = Vec3::X * 1.6;
        let toward = brake(&wall, at, Vec3::new(0.1, 0., 0.1), 1.);
        assert!(toward.x.abs() < 1e-3 && toward.y.abs() < 1e-3 && (toward.z - 0.1).abs() < 1e-5, "{toward}");
        let away = brake(&wall, at, Vec3::new(-0.1, 0., 0.), 1.);
        assert!((away.x + 0.1).abs() < 1e-5, "{away}");
        // Past the wall: still no further in.
        let inside = brake(&wall, Vec3::X * 2.5, Vec3::X * 0.1, 1.);
        assert!(inside.x.abs() < 1e-3, "{inside}");
    }

    #[test]
    fn the_push_points_out_even_from_inside() {
        let wall = wall();
        assert_eq!(push(&wall, Vec3::ZERO, 1.), Vec3::ZERO);
        // 0.2 m inside the reach (the facing sensor's shortfall); always toward -x.
        let near = push(&[(Vec3::X, 2.)], Vec3::X * 1.2, 1.);
        assert!((near - Vec3::NEG_X * 0.2).length() < 1e-5, "{near}");
        for x in [1.5, 2., 2.5, 4.] {
            let p = push(&wall, Vec3::X * x, 1.);
            assert!(p.x < -0.4 && p.y.abs() < 1e-4, "at {x}: {p}");
        }
        assert!((nearest(&wall, Vec3::X * 2.5) + 0.5).abs() < 1e-5);
    }
}
