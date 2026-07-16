use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
pub struct EnvMetrics {
    pub calls: u64,
    pub states: u64,
    pub model_queries: u64,
    pub rust_resolved_decisions: u64,
    pub failures: u64,
    pub validation: Duration,
    pub env_step: Duration,
    pub materialization: Duration,
    pub exchange: Duration,
}

impl EnvMetrics {
    pub fn as_pairs(self) -> [(&'static str, u64); 9] {
        [
            ("calls", self.calls),
            ("states", self.states),
            ("model_queries", self.model_queries),
            ("rust_resolved_decisions", self.rust_resolved_decisions),
            ("failures", self.failures),
            ("validation_ns", self.validation.as_nanos() as u64),
            ("env_step_ns", self.env_step.as_nanos() as u64),
            ("materialization_ns", self.materialization.as_nanos() as u64),
            ("exchange_ns", self.exchange.as_nanos() as u64),
        ]
    }
}
