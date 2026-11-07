//! ms-audio: device-independent audio helpers (downmix, sample ring, backoff).

mod backoff;
mod downmix;
mod ring;

pub use backoff::backoff_delay;
pub use downmix::{downmix, layout_for, Channel};
pub use ring::{ring, RingConsumer, RingProducer};
