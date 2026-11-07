//! Prints the rolling RMS of system audio (loopback of the default output device)
//! and capture events once a second.
//!
//! cargo run -p ms-audio --example meter [-- --seconds N]
//!
//! Without `--seconds` it runs until Ctrl+C.

#[cfg(not(windows))]
fn main() {
    eprintln!("meter: system-audio capture is Windows-only");
}

#[cfg(windows)]
fn main() {
    use ms_analysis::AudioHistory;
    use ms_audio::{AudioEvent, AudioService, DeviceChoice};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();

    let seconds = match parse_seconds() {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("meter: {msg}\nusage: meter [--seconds N]");
            std::process::exit(2);
        }
    };

    println!("output devices: {:?}", AudioService::list_output_devices());

    let t0 = Instant::now();
    let clock = Arc::new(move || t0.elapsed().as_secs_f64());
    let (service, mut consumer, events) = AudioService::start(DeviceChoice::Default, clock.clone());

    let mut rate = 48_000u32;
    let mut hist = AudioHistory::new(rate, 2.0);
    let mut window = vec![[0.0f32; 2]; rate as usize];
    let mut next_tick = t0 + Duration::from_secs(1);
    let mut tick = 0u64;

    loop {
        let wait = next_tick.saturating_duration_since(Instant::now());
        match events.recv_timeout(wait) {
            Ok(event) => {
                println!("[{:6.2}s] event: {event:?}", clock());
                if let AudioEvent::Started { sample_rate, .. } = event {
                    if sample_rate != rate {
                        rate = sample_rate;
                        hist.reset(rate);
                        window.resize(rate as usize, [0.0; 2]);
                    }
                }
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                println!("event channel closed");
                break;
            }
        }
        next_tick += Duration::from_secs(1);
        tick += 1;

        let now = clock();
        consumer.drain_into(&mut hist);
        hist.fill_silence_until(now);
        hist.window(now, &mut window);
        let sum: f64 = window
            .iter()
            .map(|[l, r]| (*l as f64).powi(2) + (*r as f64).powi(2))
            .sum();
        let rms = (sum / (2 * window.len()) as f64).sqrt();
        println!("[{now:6.2}s] rms {rms:.5}");

        if seconds.is_some_and(|s| tick >= s) {
            break;
        }
    }

    let t = Instant::now();
    drop(service);
    println!("stopped (shutdown took {:?})", t.elapsed());
}

/// Parses `--seconds N` (None: run until Ctrl+C).
#[cfg(windows)]
fn parse_seconds() -> Result<Option<u64>, String> {
    let mut args = std::env::args().skip(1);
    let mut seconds = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seconds" => {
                let v = args.next().ok_or("--seconds needs a value")?;
                let n = v
                    .parse::<u64>()
                    .map_err(|e| format!("bad --seconds value {v:?}: {e}"))?;
                seconds = Some(n);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(seconds)
}
