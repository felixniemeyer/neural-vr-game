//! The desktop camera's collisions (fabric-artwork's sensor push), from the sphere's own
//! depth: every point within `reach` of the camera pushes it straight away by its
//! penetration (at most half the reach); the push is their mean, so the probe's density
//! doesn't matter.

use glam::Vec3;

/// The push away from `points` for a camera at `origin`; zero when none is within reach.
pub fn push(points: &[Vec3], origin: Vec3, reach: f32) -> Vec3 {
    let (mut push, mut hits) = (Vec3::ZERO, 0);
    for &point in points {
        let to = point - origin;
        let distance = to.length();
        if distance < reach && distance > 1e-3 {
            push -= to / distance * (reach - distance).min(0.5 * reach);
            hits += 1;
        }
    }
    if hits == 0 {
        return Vec3::ZERO;
    }
    let push = push / hits as f32;
    if push.is_finite() { push } else { Vec3::ZERO }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_points_push_away_far_ones_do_not() {
        let origin = Vec3::new(1., 2., 3.);
        // 0.2 m inside a 1 m reach: pushed back 0.2 m (0.8 m away; at most half the reach).
        let near = push(&[origin + Vec3::X * 0.8], origin, 1.);
        assert!((near - Vec3::NEG_X * 0.2).length() < 1e-5, "{near}");
        let deep = push(&[origin + Vec3::X * 0.1], origin, 1.);
        assert!((deep - Vec3::NEG_X * 0.5).length() < 1e-5, "{deep}");
        assert_eq!(push(&[origin + Vec3::Y * 2.], origin, 1.), Vec3::ZERO);
        // Points on both sides cancel; the mean doesn't grow with the count.
        let both = push(&[origin + Vec3::X * 0.8, origin - Vec3::X * 0.8], origin, 1.);
        assert!(both.length() < 1e-5);
        let many = push(&[origin + Vec3::X * 0.8; 10], origin, 1.);
        assert!((many - near).length() < 1e-5);
    }
}
