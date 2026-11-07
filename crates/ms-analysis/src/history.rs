//! Ring buffer of stereo audio frames with timing and gap filling.

/// One stereo frame (left, right).
pub type Frame = [f32; 2];

/// Gaps longer than this (seconds) are padded with silence.
const GAP_SECONDS: f64 = 0.050;
/// Amplitude above which a frame counts as non-silent.
const LOUD: f32 = 1e-4;

/// Time-stamped history of captured stereo audio.
pub struct AudioHistory {
    rate: u32,
    seconds: f64,
    buf: Vec<Frame>,
    /// Index where the next frame will be written.
    head: usize,
    /// Number of valid frames (<= buf.len()).
    len: usize,
    /// Time of the newest frame.
    newest: Option<f64>,
    /// Time of the last frame with |l| or |r| above `LOUD`.
    last_loud: Option<f64>,
    /// Time of the first push/fill after creation/reset.
    origin: Option<f64>,
}

fn capacity_for(rate: u32, seconds: f64) -> usize {
    let secs = if seconds.is_finite() {
        seconds.max(0.0)
    } else {
        0.0
    };
    let want = (secs * rate as f64).ceil();
    let want = if want < 1.0e9 {
        want as usize
    } else {
        1_000_000_000
    };
    want.max(rate as usize).max(1)
}

impl AudioHistory {
    /// Creates a history holding at least `seconds` (min 1 s) at `sample_rate`.
    pub fn new(sample_rate: u32, seconds: f64) -> Self {
        let rate = sample_rate.max(1);
        let cap = capacity_for(rate, seconds);
        Self {
            rate,
            seconds,
            buf: vec![[0.0; 2]; cap],
            head: 0,
            len: 0,
            newest: None,
            last_loud: None,
            origin: None,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.rate
    }

    /// Clears all data and switches to `sample_rate`.
    pub fn reset(&mut self, sample_rate: u32) {
        self.rate = sample_rate.max(1);
        let cap = capacity_for(self.rate, self.seconds);
        self.buf.clear();
        self.buf.resize(cap, [0.0; 2]);
        self.head = 0;
        self.len = 0;
        self.newest = None;
        self.last_loud = None;
        self.origin = None;
    }

    fn append(&mut self, frames: &[Frame]) {
        let cap = self.buf.len();
        // Only the last `cap` frames can survive.
        let frames = &frames[frames.len().saturating_sub(cap)..];
        for f in frames {
            self.buf[self.head] = *f;
            self.head += 1;
            if self.head == cap {
                self.head = 0;
            }
        }
        self.len = (self.len + frames.len()).min(cap);
    }

    fn append_zeros(&mut self, count: usize) {
        let cap = self.buf.len();
        let count = count.min(cap);
        for _ in 0..count {
            self.buf[self.head] = [0.0; 2];
            self.head += 1;
            if self.head == cap {
                self.head = 0;
            }
        }
        self.len = (self.len + count).min(cap);
    }

    fn frames_for(&self, secs: f64) -> usize {
        let n = (secs * self.rate as f64).round();
        if n <= 0.0 {
            0
        } else if n >= self.buf.len() as f64 {
            self.buf.len()
        } else {
            n as usize
        }
    }

    /// Appends `frames`; `capture_time` is the time of the last frame.
    pub fn push(&mut self, frames: &[Frame], capture_time: f64) {
        if frames.is_empty() {
            return;
        }
        if !capture_time.is_finite() {
            self.append(frames);
            return;
        }
        let rate = self.rate as f64;
        let first_time = capture_time - (frames.len() - 1) as f64 / rate;
        if let Some(newest) = self.newest {
            let gap = first_time - (newest + 1.0 / rate);
            if gap > GAP_SECONDS {
                let n = self.frames_for(gap);
                self.append_zeros(n);
            }
        }
        self.append(frames);
        self.newest = Some(capture_time);
        if self.origin.is_none() {
            self.origin = Some(capture_time);
        }
        if let Some(k) = frames
            .iter()
            .rposition(|f| f[0].abs() > LOUD || f[1].abs() > LOUD)
        {
            self.last_loud = Some(capture_time - (frames.len() - 1 - k) as f64 / rate);
        }
    }

    /// Pads silence up to `now` if the newest frame is more than 50 ms old.
    pub fn fill_silence_until(&mut self, now: f64) {
        if !now.is_finite() {
            return;
        }
        match self.newest {
            None => {
                self.newest = Some(now);
                self.origin = Some(now);
            }
            Some(newest) => {
                if now - newest > GAP_SECONDS {
                    let n = self.frames_for(now - newest);
                    self.append_zeros(n);
                    self.newest = Some(now);
                }
            }
        }
    }

    /// Fills `out` with frames ending at `end_time`; zeros where no data.
    pub fn window(&self, end_time: f64, out: &mut [Frame]) {
        let zero = [0.0; 2];
        let newest = match self.newest {
            Some(t) if end_time.is_finite() => t,
            _ => {
                out.fill(zero);
                return;
            }
        };
        // Frames back from the newest frame to the window's last element.
        let d = ((newest - end_time) * self.rate as f64).round();
        let d = d.clamp(-1.0e12, 1.0e12) as i64;
        let n = out.len() as i64;
        let cap = self.buf.len() as i64;
        let len = self.len as i64;
        for (i, o) in out.iter_mut().enumerate() {
            let back = d + (n - 1 - i as i64);
            *o = if back < 0 || back >= len {
                zero
            } else {
                // head - 1 is the newest frame.
                let idx = (self.head as i64 - 1 - back).rem_euclid(cap);
                self.buf[idx as usize]
            };
        }
    }

    /// Seconds since the last non-silent frame (or since the history began).
    pub fn silent_for(&self, now: f64) -> f64 {
        if !now.is_finite() {
            return 0.0;
        }
        match self.last_loud.or(self.origin) {
            Some(t) => (now - t).max(0.0),
            None => 0.0,
        }
    }
}
