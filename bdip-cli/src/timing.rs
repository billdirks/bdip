use std::time::Instant;

/// Records wall-clock durations for named pipeline stages and optionally
/// prints a summary to stderr. When `enabled` is false every method is a
/// no-op, so call sites require no conditional logic.
pub struct PipelineTimer {
    enabled: bool,
    start: Instant,
    last: Instant,
    laps: Vec<(String, std::time::Duration)>,
}

impl PipelineTimer {
    pub fn new(enabled: bool) -> Self {
        let now = Instant::now();
        Self {
            enabled,
            start: now,
            last: now,
            laps: Vec::new(),
        }
    }

    /// Records the elapsed time since the previous `lap` (or construction)
    /// under `label`. Does nothing when the timer is disabled.
    pub fn lap(&mut self, label: &str) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        self.laps
            .push((label.to_string(), now.duration_since(self.last)));
        self.last = now;
    }

    /// Prints the recorded laps plus a total to stderr. Does nothing when
    /// the timer is disabled.
    pub fn report(&self) {
        if !self.enabled {
            return;
        }
        eprintln!("\n--- Pipeline Timings ---");
        for (label, dur) in &self.laps {
            eprintln!("  {:<25} {:>8.2} ms", label, dur.as_secs_f64() * 1000.0);
        }
        let total = self.start.elapsed();
        eprintln!("  {:<25} {:>8.2} ms", "total", total.as_secs_f64() * 1000.0);
        eprintln!("------------------------");
    }
}
