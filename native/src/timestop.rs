//! Personal time-stop: holding the grip slows the net's world to a halt (the image keeps
//! reprojecting with the head, it only stops evolving), limited by stamina. Empty stamina
//! lets time run again until the grip is released and stamina has recovered a little.

/// Seconds of full stop that full stamina buys, and seconds to refill it.
pub const LASTS: f32 = 6.;
pub const REFILL: f32 = 12.;
/// How fast time slows down and speeds up again (1/s).
const RESPONSE: f32 = 4.;
/// Stamina needed to start a stop.
const START: f32 = 0.2;

#[derive(Debug, Clone, Copy)]
pub struct TimeStop {
    /// 0..1.
    pub stamina: f32,
    /// The world's time factor now, 0 (stopped) .. 1.
    pub factor: f32,
    stopping: bool,
    was_held: bool,
}

impl Default for TimeStop {
    fn default() -> Self {
        Self { stamina: 1., factor: 1., stopping: false, was_held: false }
    }
}

impl TimeStop {
    /// Advance by `dt` seconds with the grip `held`; returns the time factor.
    pub fn update(&mut self, held: bool, dt: f32) -> f32 {
        // A stop starts on a fresh press with enough stamina, and lasts while held.
        let press = held && !self.was_held;
        self.was_held = held;
        self.stopping = held && (self.stopping || (press && self.stamina >= START)) && self.stamina > 0.;
        if self.stopping {
            self.stamina = (self.stamina - dt / LASTS).max(0.);
        } else {
            self.stamina = (self.stamina + dt / REFILL).min(1.);
        }
        let goal = if self.stopping { 0. } else { 1. };
        self.factor += (goal - self.factor) * (1. - (-RESPONSE * dt).exp());
        self.factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(t: &mut TimeStop, held: bool, seconds: f32) -> f32 {
        let mut f = 1.;
        for _ in 0..(seconds * 90.) as usize {
            f = t.update(held, 1. / 90.);
        }
        f
    }

    #[test]
    fn holding_stops_time_until_stamina_runs_out() {
        let mut t = TimeStop::default();
        assert!(run(&mut t, true, 2.) < 0.01, "stopped after 2 s");
        assert!(t.stamina > 0.6 && t.stamina < 0.7);
        // Still holding: at 6 s stamina is gone and time runs again (refilling from then).
        assert!(run(&mut t, true, 5.) > 0.95);
        assert!(t.stamina < 0.1);
        // Holding on does not restart it; a new press does, with stamina >= 0.2.
        assert!(run(&mut t, true, 2.) > 0.95, "held on: no restart");
        run(&mut t, false, 0.1);
        assert!(t.stamina > 0.2);
        assert!(run(&mut t, true, 1.) < 0.05, "new press stops again");
    }
}
