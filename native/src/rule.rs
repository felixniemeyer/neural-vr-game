//! Weights of the per-pixel net (see shaders/net.frag), in the layout train.py exports:
//! net.json (inputs, layers, taps, near, max_change) + weights.bin (float32, per layer
//! W [outputs][inputs] row-major, then the bias).
//!
//! Time: net.json "rate" (ticks per second) and "max_tau" (the largest step it trained
//! on); nets from before (no rate) trained on 10 fps steps without seeing tau.
//!
//! `WEIGHTS=dir` loads a trained net; without it the default layout runs with random
//! weights only. The `wild` fader blends the trained weights toward random ones that
//! slowly morph (several random bases mixed with sines, normalised): 0 = as trained,
//! 1 = fully random, the artistic distortion knob.

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;

pub const MAX_TAPS: usize = 32;
pub const MAX_LAYERS: usize = 4;
const MAX_UNITS: usize = 64;
const BASES: usize = 4;
const PERIODS: [f32; BASES] = [41., 59., 73., 97.];

pub struct Rule {
    /// (mip level, dx, dy) per tap, pixels of level 0, y down.
    taps: Vec<[f32; 3]>,
    /// (inputs, outputs) per layer.
    pub layers: Vec<(usize, usize)>,
    near: f32,
    max_change: f32,
    /// Ticks per second, the largest trained step (ticks), and whether the net sees tau.
    pub rate: f32,
    pub max_tau: f32,
    tau_input: bool,
    trained: Option<Vec<f32>>,
    bases: Vec<Vec<f32>>,
}

/// Per-frame uniforms of the net.
pub struct RuleFrame {
    /// taps, layers, near, max change
    pub net: [f32; 4],
    /// per layer: inputs, outputs, offset (vec4s), 0
    pub layers: [[f32; 4]; MAX_LAYERS],
    /// per tap: level, dx, dy, 0
    pub taps: [[f32; 4]; MAX_TAPS],
    /// tau input (1 or 0), 0, 0, 0
    pub run: [f32; 4],
}

/// The layout train.py uses by default (net.levels 5, points 3, hidden [16]).
fn default_taps() -> Vec<[f32; 3]> {
    let mut taps = Vec::new();
    for level in 0..5 {
        let radius = 2f32.powi(level);
        for k in 0..3 {
            let angle = (90. + 60. * level as f32 + 120. * k as f32).to_radians();
            taps.push([level as f32, radius * angle.cos(), -radius * angle.sin()]);
        }
    }
    taps
}

fn wave(time: f32, period: f32, phase: f32) -> f32 {
    (std::f32::consts::TAU * time / period + phase).sin()
}

impl Rule {
    pub fn load(dir: Option<&Path>) -> Result<Self> {
        let (taps, layers, near, max_change, trained, time) = match dir {
            None => {
                let taps = default_taps();
                let inputs = 4 * (2 * (1 + taps.len()) + 1);
                (taps, vec![(inputs, 16), (16, 8)], 10., 0.5, None, (24., 1., true))
            }
            Some(dir) => {
                let json: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("net.json"))?)?;
                let number = |v: &Value| v.as_f64().context("number expected in net.json").map(|x| x as f32);
                let taps = json["taps"]
                    .as_array()
                    .context("taps")?
                    .iter()
                    .map(|t| Ok([number(&t[0])?, number(&t[1])?, number(&t[2])?]))
                    .collect::<Result<Vec<_>>>()?;
                let layers = json["layers"]
                    .as_array()
                    .context("layers")?
                    .iter()
                    .map(|l| Ok((number(&l[0])? as usize, number(&l[1])? as usize)))
                    .collect::<Result<Vec<_>>>()?;
                ensure!(json["hidden_state"].as_u64() == Some(4), "the runtime state holds 4 hidden channels");
                let bytes = std::fs::read(dir.join("weights.bin"))?;
                let weights: Vec<f32> =
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let expected: usize = layers.iter().map(|&(i, o)| i * o + o).sum();
                ensure!(weights.len() == expected, "weights.bin has {} floats, net.json needs {expected}", weights.len());
                let time = match json.get("rate") {
                    Some(rate) => (number(rate)?, json.get("max_tau").map_or(Ok(1.), number)?, true),
                    None => (10., 1., false),
                };
                (taps, layers, number(&json["near"])?, number(&json["max_change"])?, Some(weights), time)
            }
        };
        ensure!(taps.len() <= MAX_TAPS && layers.len() <= MAX_LAYERS, "net too large for the shader");
        ensure!(layers[0].0 == 4 * (2 * (1 + taps.len()) + 1), "layer 1 inputs do not match the taps");
        ensure!(layers.last().unwrap().1 == 8, "the net must output 8 values");
        for &(i, o) in &layers {
            ensure!(i % 4 == 0 && o % 4 == 0 && o <= MAX_UNITS, "layer sizes must be multiples of 4, at most {MAX_UNITS}");
        }
        let count: usize = layers.iter().map(|&(i, o)| i * o + o).sum();
        let bases = (0..BASES)
            .map(|_| (0..count).map(|_| (rand::random::<f32>() * 2. - 1.) * 3f32.sqrt()).collect())
            .collect();
        let (rate, max_tau, tau_input) = time;
        Ok(Self { taps, layers, near, max_change, rate, max_tau, tau_input, trained, bases })
    }

    pub fn trained(&self) -> bool {
        self.trained.is_some()
    }

    /// Mip levels the net reads (the pyramid depth).
    pub fn levels(&self) -> u32 {
        self.taps.iter().map(|t| t[0] as u32 + 1).max().unwrap_or(1)
    }

    /// Specialization constants of net.frag: taps, levels, layers, units per layer (4).
    pub fn specialization(&self) -> [i32; 7] {
        let mut out = [self.taps.len() as i32, self.levels() as i32, self.layers.len() as i32, 8, 8, 8, 8];
        for (slot, &(_, o)) in out[3..].iter_mut().zip(&self.layers) {
            *slot = o as i32;
        }
        out
    }

    /// Number of floats in the weights buffer.
    pub fn len(&self) -> usize {
        self.bases[0].len()
    }

    pub fn frame(&self) -> RuleFrame {
        let mut layers = [[0.; 4]; MAX_LAYERS];
        let mut offset = 0;
        for (slot, &(i, o)) in layers.iter_mut().zip(&self.layers) {
            *slot = [i as f32, o as f32, (offset / 4) as f32, 0.];
            offset += i * o + o;
        }
        let mut taps = [[0.; 4]; MAX_TAPS];
        for (slot, t) in taps.iter_mut().zip(&self.taps) {
            *slot = [t[0], t[1], t[2], 0.];
        }
        RuleFrame {
            net: [self.taps.len() as f32, self.layers.len() as f32, self.near, self.max_change],
            layers,
            taps,
            run: [if self.tau_input { 1. } else { 0. }, 0., 0., 0.],
        }
    }

    /// The weights at `time`: trained blended toward random by `wild` (always random
    /// without a trained net). Random weights use He scaling per layer (gain on the
    /// hidden layers, 0.1 on the output layer), the bases blended with normalised sines.
    pub fn weights(&self, time: f32, wild: f32, out: &mut [f32]) {
        let wild = if self.trained.is_some() { wild.clamp(0., 1.) } else { 1. };
        let mut blend: [f32; BASES] = std::array::from_fn(|k| wave(time, PERIODS[k], k as f32 * 1.7));
        let norm = blend.iter().map(|w| w * w).sum::<f32>().sqrt().max(1e-3);
        blend.iter_mut().for_each(|w| *w /= norm);
        let gain = 1.45 + 0.75 * wave(time, 31., 0.5);
        let mut start = 0;
        for (l, &(i, o)) in self.layers.iter().enumerate() {
            // The output layer small, as train.py initialises it: random weights then
            // perturb by small steps instead of throwing depth and colour around.
            let scale = if l + 1 < self.layers.len() { gain } else { 0.1 } * (2. / i as f32).sqrt();
            for index in start..start + i * o + o {
                let random = (0..BASES).map(|k| self.bases[k][index] * blend[k]).sum::<f32>() * scale;
                let trained = self.trained.as_ref().map_or(0., |w| w[index]);
                out[index] = trained + (random - trained) * wild;
            }
            start += i * o + o;
        }
    }
}

/// Weights as the GPU reads them (state.glsl Weights, net.frag): float16, per layer in
/// 4x4 blocks (4 outputs x 4 inputs), block (q, k) column by column, i.e. for each input
/// the 4 outputs' weights side by side (one packed fma per input), then the bias. Same
/// offsets (in groups of 4) as the float32 layout.
pub fn pack(layers: &[(usize, usize)], weights: &[f32], out: &mut [u16]) {
    let half = |w: f32| half::f16::from_f32(w).to_bits();
    let mut start = 0;
    for &(i, o) in layers {
        let mut at = start;
        for q in 0..o / 4 {
            for k in 0..i / 4 {
                for c in 0..4 {
                    for r in 0..4 {
                        out[at] = half(weights[start + (4 * q + r) * i + 4 * k + c]);
                        at += 1;
                    }
                }
            }
        }
        for b in 0..o {
            out[at + b] = half(weights[start + i * o + b]);
        }
        start += i * o + o;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pack_puts_each_inputs_four_outputs_together() {
        // One layer, 8 inputs -> 4 outputs: W[r][j] = 10 r + j, bias 100 + r.
        let mut w: Vec<f32> = (0..4).flat_map(|r| (0..8).map(move |j| (10 * r + j) as f32)).collect();
        w.extend((0..4).map(|r| (100 + r) as f32));
        let mut out = vec![0u16; w.len()];
        super::pack(&[(8, 4)], &w, &mut out);
        let f: Vec<f32> = out.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
        // Block k = 1, column c = 2 (input 6): outputs 0..3.
        assert_eq!(&f[16 + 8..16 + 12], &[6., 16., 26., 36.]);
        assert_eq!(&f[32..], &[100., 101., 102., 103.]);
    }
}
