use std::time::Duration;

/// Delay before reconnect attempt `attempt` (0-based): 0.5 s, 1 s, 2 s, then 5 s.
pub fn backoff_delay(attempt: u32) -> Duration {
    match attempt {
        0 => Duration::from_millis(500),
        1 => Duration::from_secs(1),
        2 => Duration::from_secs(2),
        _ => Duration::from_secs(5),
    }
}
