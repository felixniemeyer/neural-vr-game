//! Weights of the tiny per-pixel net (see shaders/net.frag), random for now and
//! slowly morphing; a trained network would be exported to the same layout.
//!
//!   layer 1: 14 input vec4s (12 spiral taps + 2 self) -> 16 hidden, as 56 mat4
//!            (w1[k * 4 + q]: input k to hidden quad q), relu
//!   layer 2: 16 hidden -> 2 output vec4s, as 8 mat4 (w2[j * 4 + q])
//!
//! Several random bases are blended with sines, normalised so the weight scale stays
//! constant, and a gain sine moves the net between calm and chaotic. The spiral tap
//! offsets are computed here too (golden angle, exponential radius, slowly turning).

pub const TAPS: usize = 12;
pub const W1: usize = (TAPS + 2) * 4;
pub const W2: usize = 2 * 4;

const BASES: usize = 4;
const PERIODS: [f32; BASES] = [41., 59., 73., 97.];
const GOLDEN_ANGLE: f32 = 2.399_963;

type Mat4 = [f32; 16];

pub struct Rule {
    w1: Vec<[Mat4; W1]>,
    w2: Vec<[Mat4; W2]>,
}

pub struct RuleFrame {
    pub taps: [[f32; 4]; TAPS / 2],
    pub w1: [Mat4; W1],
    pub w2: [Mat4; W2],
    /// update rate, noise, alive, seed probability
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
fn blend(time: f32) -> [f32; BASES] {
    let mut weights: [f32; BASES] = std::array::from_fn(|k| wave(time, PERIODS[k], k as f32 * 1.7));
    let norm = weights.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-3);
    weights.iter_mut().for_each(|w| *w /= norm);
    weights
}

fn random<const N: usize>() -> [Mat4; N] {
    std::array::from_fn(|_| std::array::from_fn(|_| uniform_unit()))
}

fn mix<const N: usize>(bases: &[[Mat4; N]], weights: &[f32; BASES], scale: f32) -> [Mat4; N] {
    std::array::from_fn(|m| {
        std::array::from_fn(|i| (0..BASES).map(|k| bases[k][m][i] * weights[k]).sum::<f32>() * scale)
    })
}

impl Rule {
    pub fn random() -> Self {
        Self {
            w1: (0..BASES).map(|_| random()).collect(),
            w2: (0..BASES).map(|_| random()).collect(),
        }
    }

    /// Weights and taps at `time`; `reach` is the outermost tap radius in pixels,
    /// `alive` the rate floor on certain pixels, `seeds` scales the seed probability.
    pub fn at(&self, time: f32, reach: f32, alive: f32, seeds: f32) -> RuleFrame {
        let weights = blend(time);
        // Gain ~0.7 (calm) .. ~2.2 (chaotic), He-style per-layer fan-in scaling.
        let gain = 1.45 + 0.75 * wave(time, 31., 0.5);
        let w1 = mix(&self.w1, &weights, gain * (2. / (W1 * 4) as f32).sqrt());
        let w2 = mix(&self.w2, &weights, 1.5 * (2. / 16f32).sqrt());
        let turn = 0.05 * time;
        let reach = reach.max(1.);
        let taps = std::array::from_fn(|i| {
            let offset = |k: usize| {
                let radius = reach.powf(k as f32 / (TAPS - 1) as f32);
                let angle = k as f32 * GOLDEN_ANGLE + turn;
                [(radius * angle.cos()).round(), (radius * angle.sin()).round()]
            };
            let [a, b] = [offset(2 * i), offset(2 * i + 1)];
            [a[0], a[1], b[0], b[1]]
        });
        RuleFrame {
            taps,
            w1,
            w2,
            params: [
                0.25 + 0.15 * wave(time, 23., 2.),
                0.02,
                alive,
                0.0032 * (1. + wave(time, 47., 1.)) * seeds,
            ],
        }
    }
}
