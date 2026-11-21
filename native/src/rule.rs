//! Random neural-CA weights that slowly morph over time.
//!
//! The CA update is `h <- mix(h, tanh(W * perception), rate)` with
//! perception = [self, ring1, ring2, ring4, ring8], each group being the hidden
//! channels plus one extra slot (the bias for `self`, the mean inverse depth relative
//! to the pixel for a ring). Two hidden sizes are supported:
//!   1 hidden image: 3 channels, a group is one vec4 (h0, h1, h2, extra)
//!   2 hidden images: 7 channels, a group is two vec4s (h0..h3), (h4, h5, h6, extra)
//! Only the first `rings` rings are used. A readout matrix maps (hidden + bias) to
//! (r, g, b, radial speed). Several random bases are blended with sines, normalised so
//! the overall weight scale stays constant, and a gain sine moves the rule between
//! order and chaos. Training would later replace the bases.

/// Maximum hidden channels (2 hidden images).
pub const HIDDEN: usize = 7;
pub const GROUPS: usize = 5;
/// Rule: HIDDEN rows x GROUPS x 2 vec4s (the full 2-image layout; the 1-image case
/// uses the first vec4 of each group and the first 3 rows).
pub const RULE_VEC4S: usize = HIDDEN * GROUPS * 2;
/// Readout: 4 rows (r, g, b, radial speed) x 2 vec4s.
pub const READOUT_VEC4S: usize = 4 * 2;

const BASES: usize = 4;
const PERIODS: [f32; BASES] = [41., 59., 73., 97.];

pub struct Rule {
    rule: Vec<[f32; RULE_VEC4S * 4]>,
    readout: Vec<[f32; READOUT_VEC4S * 4]>,
}

pub struct RuleFrame {
    /// Two rules; the shader mixes them per pixel with a world-space region field.
    pub rule: [[f32; 4]; RULE_VEC4S],
    pub rule_b: [[f32; 4]; RULE_VEC4S],
    pub readout: [[f32; 4]; READOUT_VEC4S],
    /// update rate, noise, fire probability, seed probability
    pub params: [f32; 4],
}

fn uniform_unit() -> f32 {
    // Uniform with unit variance: [-sqrt(3), sqrt(3)].
    (rand::random::<f32>() * 2. - 1.) * 3f32.sqrt()
}

fn wave(time: f32, period: f32, phase: f32) -> f32 {
    (std::f32::consts::TAU * time / period + phase).sin()
}

/// Normalised blend weights of the bases at `time`.
fn blend(time: f32, offset: f32) -> [f32; BASES] {
    let mut weights: [f32; BASES] =
        std::array::from_fn(|k| wave(time, PERIODS[k], k as f32 * 1.7 + offset));
    let norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-3);
    weights.iter_mut().for_each(|w| *w /= norm);
    weights
}

/// Whether vec4 `v` (0 or 1) of a perception group / readout row is used.
fn used(hidden_images: usize, v: usize) -> bool {
    hidden_images == 2 || v == 0
}

impl Rule {
    pub fn random() -> Self {
        Self {
            rule: (0..BASES).map(|_| std::array::from_fn(|_| uniform_unit())).collect(),
            readout: (0..BASES).map(|_| std::array::from_fn(|_| uniform_unit())).collect(),
        }
    }

    /// Blend of the bases at `time`, with a phase offset (different offsets give
    /// different rules from the same bases). Unused weights are zero.
    fn rule_at(&self, time: f32, offset: f32, hidden_images: usize, rings: usize) -> [[f32; 4]; RULE_VEC4S] {
        let weights = blend(time, offset);
        let outputs = if hidden_images == 2 { 7 } else { 3 };
        let inputs = (1 + rings) * 4 * hidden_images;
        // Gain: ~0.7 (ordered, smooth) .. ~2.2 (chaotic), per input sqrt-normalised.
        let gain = 1.45 + 0.75 * wave(time, 31., 0.5 + offset);
        let scale = gain / (inputs as f32).sqrt();
        let mut rule = [[0f32; 4]; RULE_VEC4S];
        for o in 0..outputs {
            for group in 0..=rings {
                for v in 0..2 {
                    let slot = o * GROUPS * 2 + group * 2 + v;
                    for c in 0..4 {
                        if used(hidden_images, v) {
                            let i = slot * 4 + c;
                            rule[slot][c] =
                                (0..BASES).map(|k| self.rule[k][i] * weights[k]).sum::<f32>() * scale;
                        }
                    }
                }
            }
        }
        rule
    }

    pub fn at(&self, time: f32, hidden_images: usize, rings: usize) -> RuleFrame {
        let weights = blend(time, 0.);
        let mut readout = [[0f32; 4]; READOUT_VEC4S];
        let scale = 0.8 * (8. / (4 * hidden_images) as f32).sqrt();
        for (slot, value) in readout.iter_mut().enumerate() {
            for c in 0..4 {
                if used(hidden_images, slot % 2) {
                    let i = slot * 4 + c;
                    value[c] = (0..BASES).map(|k| self.readout[k][i] * weights[k]).sum::<f32>() * scale;
                }
            }
        }
        RuleFrame {
            rule: self.rule_at(time, 0., hidden_images, rings),
            rule_b: self.rule_at(time, 2.4, hidden_images, rings),
            readout,
            params: [
                0.25 + 0.15 * wave(time, 23., 2.),
                0.02,
                0.5 + 0.3 * wave(time, 19., 4.),
                0.0002 * (1. + wave(time, 47., 1.)),
            ],
        }
    }
}
