//! Logging (spec §9.3): every line goes to stderr, to a daily log file in
//! the logs directory (7 files kept) and to an in-memory ring of the last
//! 200 lines, which a crash report quotes.
//!
//! The file is written by `tracing-appender`'s worker thread, so logging
//! from the render thread never waits on the disk (when that thread falls
//! far behind, lines are dropped rather than blocking).

use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::fmt::MakeWriter;

/// Lines kept for crash reports.
pub const RING_LINES: usize = 200;
/// Daily log files kept.
pub const LOG_FILES_KEPT: usize = 7;

/// The last `cap` log lines.
#[derive(Debug)]
pub struct LogRing {
    cap: usize,
    lines: Mutex<VecDeque<String>>,
}

impl LogRing {
    pub fn new(cap: usize) -> LogRing {
        LogRing {
            cap: cap.max(1),
            lines: Mutex::new(VecDeque::with_capacity(cap.max(1))),
        }
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<String>> {
        self.lines.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds a line, dropping the oldest when full.
    pub fn push(&self, line: String) {
        let mut lines = self.lock();
        while lines.len() >= self.cap {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// The lines, oldest first.
    pub fn lines(&self) -> Vec<String> {
        self.lock().iter().cloned().collect()
    }
}

/// The app's ring, filled by the logging set up in [`init_logging`].
pub fn ring() -> &'static LogRing {
    static RING: OnceLock<LogRing> = OnceLock::new();
    RING.get_or_init(|| LogRing::new(RING_LINES))
}

/// A `MakeWriter` that puts each formatted event into a [`LogRing`].
#[derive(Debug, Clone, Copy)]
pub struct RingWriter(pub &'static LogRing);

/// One event being formatted; pushed into the ring when dropped.
pub struct RingLine {
    ring: &'static LogRing,
    buf: Vec<u8>,
}

impl Write for RingLine {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for RingLine {
    fn drop(&mut self) {
        let text = String::from_utf8_lossy(&self.buf);
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            self.ring.push(line.to_string());
        }
    }
}

impl<'a> MakeWriter<'a> for RingWriter {
    type Writer = RingLine;

    fn make_writer(&'a self) -> RingLine {
        RingLine {
            ring: self.0,
            buf: Vec::new(),
        }
    }
}

/// Sets up logging: info and up (only warnings from wgpu, naga and winit)
/// to stderr, to `milksplat.<date>.log` in `dir` (daily, 7 files kept) and
/// to [`ring`]. Keep the guard alive until exit: dropping it flushes the
/// file. If the log file cannot be opened, logging goes on without it and
/// says so. Does nothing if a logger is already installed (tests).
pub fn init_logging(dir: &Path) -> WorkerGuard {
    use std::io::IsTerminal;
    use tracing::Level;
    use tracing_subscriber::filter::Targets;
    use tracing_subscriber::prelude::*;

    let filter = Targets::new()
        .with_default(Level::INFO)
        .with_target("wgpu", Level::WARN)
        .with_target("wgpu_core", Level::WARN)
        .with_target("wgpu_hal", Level::WARN)
        .with_target("naga", Level::WARN)
        .with_target("winit", Level::WARN);
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("milksplat")
        .filename_suffix("log")
        .max_log_files(LOG_FILES_KEPT)
        .build(dir);
    let (file_error, (file_writer, guard)) = match appender {
        Ok(a) => (None, tracing_appender::non_blocking(a)),
        Err(e) => (Some(e), tracing_appender::non_blocking(std::io::sink())),
    };
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_thread_names(true);
    let ring = tracing_subscriber::fmt::layer()
        .with_writer(RingWriter(ring()))
        .with_ansi(false)
        .with_thread_names(true);
    let _ = tracing_subscriber::registry()
        .with(stderr)
        .with(file)
        .with(ring)
        .with(filter)
        .try_init();
    if let Some(e) = file_error {
        tracing::warn!("cannot open a log file in {}: {e}", dir.display());
    }
    guard
}
