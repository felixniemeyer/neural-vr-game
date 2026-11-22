//! Swinging: an anchor held by a controller is a rope of fixed (slowly reeling) length.
//! It only pulls: inside its length the player flies freely (gravity, momentum), at the
//! end it holds them on the sphere around the anchor and takes the outward velocity, so
//! falling past the anchor turns into a pendulum swing.

use glam::Vec3;

/// Gravity while swinging (m/s²).
pub const GRAVITY: f32 = 9.81;
/// Shortest rope (m): reeling in stops here.
pub const SHORTEST: f32 = 1.5;

/// Hold `player` (moving at `velocity`) within `length` of `anchor`: the corrected
/// position and velocity.
pub fn rope(player: Vec3, velocity: Vec3, anchor: Vec3, length: f32) -> (Vec3, Vec3) {
    let out = player - anchor;
    let distance = out.length();
    if distance <= length || distance < 1e-5 {
        return (player, velocity);
    }
    let radial = out / distance;
    let outward = velocity.dot(radial).max(0.);
    (anchor + radial * length, velocity - radial * outward)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rope_swings_instead_of_falling() {
        // Hanging 10 m out sideways from an anchor at 20 m, released from rest.
        let anchor = Vec3::new(0., 20., 0.);
        let (mut p, mut v) = (Vec3::new(10., 20., 0.), Vec3::ZERO);
        let dt = 1. / 90.;
        let mut lowest = p.y;
        let mut max_speed: f32 = 0.;
        for _ in 0..270 {
            v.y -= GRAVITY * dt;
            p += v * dt;
            (p, v) = rope(p, v, anchor, 10.);
            assert!(p.distance(anchor) <= 10. + 1e-3);
            lowest = lowest.min(p.y);
            max_speed = max_speed.max(v.length());
        }
        // Within 3 s it swings through the bottom (10 m below the anchor, about 1.9 s in)
        // at about sqrt(2 g 10 m) = 14 m/s, and never leaves the rope's sphere.
        assert!(lowest < 10.6, "lowest {lowest}");
        assert!(max_speed > 11. && max_speed < 14.5, "speed {max_speed}");
    }

    #[test]
    fn a_slack_rope_does_nothing() {
        let (p, v) = rope(Vec3::ONE, Vec3::X, Vec3::ZERO, 5.);
        assert_eq!((p, v), (Vec3::ONE, Vec3::X));
    }
}
