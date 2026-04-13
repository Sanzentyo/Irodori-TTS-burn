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
