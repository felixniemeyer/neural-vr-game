//! Random neural-CA weights that slowly morph over time.
//!
//! The CA update is `h <- mix(h, tanh(W * perception), rate)` with
//! perception = [self, ring1, ring2, ring4, ring8] x (7 hidden + 1 extra) = 40 inputs
//! (the extra slot of `self` is a constant 1, i.e. the bias; the extra slot of each
//! ring is its mean inverse depth relative to the pixel). A readout matrix maps
//! (7 hidden + bias) to (r, g, b, radial speed). Several random bases are blended with
//! sines, normalised so the overall weight scale stays constant, and a gain sine
//! moves the rule between order and chaos. Training would later replace the bases.

pub const HIDDEN: usize = 7;
pub const GROUPS: usize = 5;
/// Rule: HIDDEN rows x GROUPS * 8 inputs, packed as vec4s.
pub const RULE_VEC4S: usize = HIDDEN * GROUPS * 2;
/// Readout: 4 rows (r, g, b, radial speed) x 8 inputs, packed as vec4s.
pub const READOUT_VEC4S: usize = 4 * 2;

const BASES: usize = 4;
const RULE_INPUTS: usize = GROUPS * 8;

pub struct Rule {
    rule: Vec<[f32; HIDDEN * RULE_INPUTS]>,
    readout: Vec<[f32; 4 * 8]>,
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

impl Rule {
    pub fn random() -> Self {
        Self {
            rule: (0..BASES)
                .map(|_| std::array::from_fn(|_| uniform_unit()))
                .collect(),
            readout: (0..BASES)
                .map(|_| std::array::from_fn(|_| uniform_unit()))
                .collect(),
        }
    }

    /// Blend of the bases at `time`, with a phase offset (different offsets give
    /// different rules from the same bases).
    fn rule_at(&self, time: f32, offset: f32) -> [[f32; 4]; RULE_VEC4S] {
        let periods = [41., 59., 73., 97.];
        let blend: Vec<f32> = (0..BASES)
            .map(|k| wave(time, periods[k], k as f32 * 1.7 + offset))
            .collect();
        let norm = blend.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-3);
        // Gain: ~0.7 (ordered, smooth) .. ~2.2 (chaotic), per input sqrt-normalised.
        let gain = 1.45 + 0.75 * wave(time, 31., 0.5 + offset);
        let scale = gain / (RULE_INPUTS as f32).sqrt();
        let mut rule = [[0f32; 4]; RULE_VEC4S];
        for (i, value) in rule.iter_mut().flatten().enumerate() {
            *value = (0..BASES).map(|k| self.rule[k][i] * blend[k]).sum::<f32>() / norm * scale;
        }
        rule
    }

    pub fn at(&self, time: f32) -> RuleFrame {
        let rule = self.rule_at(time, 0.);
        let rule_b = self.rule_at(time, 2.4);
        let periods = [41., 59., 73., 97.];
        let blend: Vec<f32> = (0..BASES)
            .map(|k| wave(time, periods[k], k as f32 * 1.7))
            .collect();
        let norm = blend.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-3);
        let mut readout = [[0f32; 4]; READOUT_VEC4S];
        for (i, value) in readout.iter_mut().flatten().enumerate() {
            *value = (0..BASES).map(|k| self.readout[k][i] * blend[k]).sum::<f32>() / norm * 0.8;
        }
        RuleFrame {
            rule,
            rule_b,
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
