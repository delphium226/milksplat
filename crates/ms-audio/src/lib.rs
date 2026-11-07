//! ms-audio: system-audio capture (WASAPI loopback, Windows only) plus the
//! device-independent helpers it uses (downmix, sample ring, backoff).

mod backoff;
mod downmix;
// Only the Windows capture service uses it; still built (and tested) everywhere.
#[cfg_attr(not(windows), allow(dead_code))]
mod restart;
mod ring;
#[cfg(windows)]
mod win;

pub use backoff::backoff_delay;
pub use downmix::{downmix, layout_for, Channel};
pub use ring::{ring, RingConsumer, RingProducer};
#[cfg(windows)]
pub use win::{AudioEvent, AudioService, DeviceChoice};
