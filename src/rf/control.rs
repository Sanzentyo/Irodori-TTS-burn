use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Request-scoped cooperative cancellation, checked between model forwards.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    pub fn check(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            Err(crate::IrodoriError::Cancelled)
        } else {
            Ok(())
        }
    }
}
