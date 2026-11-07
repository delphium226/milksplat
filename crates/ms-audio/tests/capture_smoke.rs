//! Real-device smoke tests (need a Windows output device). Run manually:
//! cargo test -p ms-audio -- --ignored
#![cfg(windows)]

use ms_audio::{AudioEvent, AudioService, DeviceChoice};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
#[ignore = "needs a Windows output device"]
fn capture_starts() {
    let t0 = Instant::now();
    let (service, _consumer, events) = AudioService::start(
        DeviceChoice::Default,
        Arc::new(move || t0.elapsed().as_secs_f64()),
    );
    let deadline = t0 + Duration::from_secs(3);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match events.recv_timeout(left) {
            Ok(AudioEvent::Started {
                name,
                sample_rate,
                channels,
                bluetooth,
            }) => {
                eprintln!(
                    "Started after {:?}: {name:?} {sample_rate} Hz {channels} ch bluetooth={bluetooth}",
                    t0.elapsed()
                );
                assert!(sample_rate > 0);
                assert!(channels > 0);
                break;
            }
            Ok(other) => eprintln!("event: {other:?}"),
            Err(e) => panic!("no Started within 3 s: {e}"),
        }
    }
    let t = Instant::now();
    drop(service);
    assert!(
        t.elapsed() < Duration::from_millis(2500),
        "shutdown took {:?}",
        t.elapsed()
    );
}

#[test]
#[ignore = "needs a Windows output device"]
fn lists_output_devices() {
    let names = AudioService::list_output_devices();
    eprintln!("{names:?}");
    assert!(!names.is_empty());
}
