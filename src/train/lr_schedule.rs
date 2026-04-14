//! Warmup-cosine learning rate schedule.

/// Cosine LR decay with linear warm-up.
///
/// - Steps `[0, warmup_steps)`: linear ramp from `0` to `base_lr`
/// - Steps `[warmup_steps, total_steps]`: cosine decay to `base_lr * min_lr_scale`
#[derive(Debug, Clone)]
pub struct WarmupCosineSchedule {
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub base_lr: f64,
    pub min_lr_scale: f64,
}

impl WarmupCosineSchedule {
    pub fn lr_at_step(&self, step: usize) -> f64 {
        let min_lr = self.base_lr * self.min_lr_scale;
        if step < self.warmup_steps {
            // Linear warm-up (avoid div-by-zero when warmup_steps == 0)
            let warmup = self.warmup_steps.max(1) as f64;
            self.base_lr * (step as f64 + 1.0) / warmup
        } else if self.total_steps <= self.warmup_steps {
            min_lr
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps) as f64;
            let decay_pos = (step - self.warmup_steps) as f64;
            let progress = (decay_pos / decay_steps).min(1.0);
            min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_schedule() -> WarmupCosineSchedule {
        WarmupCosineSchedule {
            warmup_steps: 100,
            total_steps: 1000,
            base_lr: 1e-4,
            min_lr_scale: 0.1,
        }
    }

    #[test]
    fn warmup_ramp_is_linear() {
        let s = default_schedule();
        let lr0 = s.lr_at_step(0);
        let lr49 = s.lr_at_step(49);
        let lr99 = s.lr_at_step(99);
        // Step 0 → base_lr * 1/100 = 1e-6
        assert!((lr0 - 1e-6).abs() < 1e-12);
        // Step 49 → base_lr * 50/100 = 5e-5
        assert!((lr49 - 5e-5).abs() < 1e-12);
        // Step 99 → base_lr * 100/100 = 1e-4
        assert!((lr99 - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn at_warmup_boundary_equals_base_lr() {
        let s = default_schedule();
        // Step 100 is first decay step; progress=0 → cos(0)=1 → full base_lr
        let lr = s.lr_at_step(100);
        assert!((lr - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn at_total_steps_equals_min_lr() {
        let s = default_schedule();
        // Step 1000 → progress=1.0 → cos(PI)=-1 → min_lr
        let lr = s.lr_at_step(1000);
        let min_lr = 1e-4 * 0.1;
        assert!((lr - min_lr).abs() < 1e-12);
    }

    #[test]
    fn mid_decay_is_between_base_and_min() {
        let s = default_schedule();
        let lr = s.lr_at_step(550); // midpoint of decay
        let min_lr = 1e-4 * 0.1;
        assert!(lr > min_lr);
        assert!(lr < 1e-4);
    }

    #[test]
    fn decay_is_monotonically_decreasing() {
        let s = default_schedule();
        let lrs: Vec<f64> = (100..=1000).step_by(10).map(|i| s.lr_at_step(i)).collect();
        for w in lrs.windows(2) {
            assert!(w[0] >= w[1], "LR should decrease: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn zero_warmup_starts_at_base_lr() {
        let s = WarmupCosineSchedule {
            warmup_steps: 0,
            total_steps: 100,
            base_lr: 1e-3,
            min_lr_scale: 0.0,
        };
        // Step 0 → warmup_steps=0, so we're in decay, progress=0 → base_lr
        let lr = s.lr_at_step(0);
        assert!((lr - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn beyond_total_steps_clamps_to_min() {
        let s = default_schedule();
        let lr = s.lr_at_step(2000);
        let min_lr = 1e-4 * 0.1;
        // progress clamped to 1.0 → min_lr
        assert!((lr - min_lr).abs() < 1e-12);
    }

    #[test]
    fn warmup_equals_total_returns_min_lr() {
        let s = WarmupCosineSchedule {
            warmup_steps: 100,
            total_steps: 100,
            base_lr: 1e-3,
            min_lr_scale: 0.1,
        };
        // total_steps <= warmup_steps → min_lr for post-warmup
        let lr = s.lr_at_step(100);
        let min_lr = 1e-3 * 0.1;
        assert!((lr - min_lr).abs() < 1e-12);
    }
}
