//! Per-tick audio analysis: waveform, spectrum, band levels.

use crate::beat::BeatDetector;
use crate::history::{AudioHistory, Frame};
use crate::spectrum::{Spectrum, FFT_SIZE, RFFT_BINS, SPECTRUM_BINS};

const WAVE_LEN: usize = 576;
const BAND_EDGES: [(f64, f64); 3] = [(20.0, 250.0), (250.0, 4000.0), (4000.0, 16000.0)];
const LONG_TAU: f64 = 5.0;
const INSTANT_TAU: f64 = 0.05;
const ATT_TAU: f64 = 0.4;
const AVG_FLOOR: f64 = 1e-6;
const SILENT_SUM: f64 = 1e-9;
/// A tick is silent when the history has been quiet for longer than this.
const SILENT_SECONDS: f64 = 0.1;

/// Everything presets read from the audio, once per tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioFrame {
    /// `[left, right]` channel samples ending at `now - av_offset`.
    pub wave: [[f32; WAVE_LEN]; 2],
    pub spectrum: [f32; SPECTRUM_BINS],
    pub bass: f32,
    pub mid: f32,
    pub treb: f32,
    pub bass_att: f32,
    pub mid_att: f32,
    pub treb_att: f32,
    pub beat: f32,
    pub beat_strength: f32,
    pub silent_for: f32,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            wave: [[0.0; WAVE_LEN]; 2],
            spectrum: [0.0; SPECTRUM_BINS],
            bass: 0.0,
            mid: 0.0,
            treb: 0.0,
            bass_att: 0.0,
            mid_att: 0.0,
            treb_att: 0.0,
            beat: 0.0,
            beat_strength: 0.0,
            silent_for: 0.0,
        }
    }
}

fn finite(v: f64) -> f32 {
    if v.is_finite() {
        v as f32
    } else {
        0.0
    }
}

fn smooth(y: &mut f64, x: f64, dt: f64, tau: f64) {
    *y += (x - *y) * (1.0 - (-dt / tau).exp());
}

/// Half-open rfft bin ranges per band at `rate`.
fn band_ranges(rate: u32) -> [(usize, usize); 3] {
    let nyquist = rate as f64 / 2.0;
    let bin_hz = rate as f64 / FFT_SIZE as f64;
    let mut r = [(0, 0); 3];
    for (slot, &(lo, hi)) in r.iter_mut().zip(&BAND_EDGES) {
        let (lo, hi) = (lo.min(nyquist), hi.min(nyquist));
        let first = (0..RFFT_BINS).find(|&k| k as f64 * bin_hz >= lo);
        let end = (0..RFFT_BINS).find(|&k| k as f64 * bin_hz >= hi);
        *slot = match (first, end) {
            (Some(a), Some(b)) if b > a => (a, b),
            (Some(a), None) if a < RFFT_BINS => (a, RFFT_BINS),
            _ => (0, 0),
        };
    }
    r
}

/// Computes an [`AudioFrame`] per simulation tick.
pub struct Analyzer {
    rate: u32,
    fft: Spectrum,
    frames: Vec<Frame>,
    wave_frames: [Frame; WAVE_LEN],
    ranges: [(usize, usize); 3],
    started: bool,
    /// Non-silent ticks since the last reset.
    warm_ticks: u32,
    long_avg: [f64; 3],
    instant: [f64; 3],
    att: [f64; 3],
    raw: [f32; 3],
    beat: BeatDetector,
    out: AudioFrame,
}

impl Analyzer {
    pub fn new(sample_rate: u32) -> Self {
        let rate = sample_rate.max(1);
        Self {
            rate,
            fft: Spectrum::new(),
            frames: vec![[0.0; 2]; FFT_SIZE],
            wave_frames: [[0.0; 2]; WAVE_LEN],
            ranges: band_ranges(rate),
            started: false,
            warm_ticks: 0,
            long_avg: [0.0; 3],
            instant: [0.0; 3],
            att: [0.0; 3],
            raw: [0.0; 3],
            beat: BeatDetector::new(),
            out: AudioFrame::default(),
        }
    }

    /// Raw (bass, mid, treb) band energies from the last tick.
    #[doc(hidden)]
    pub fn raw_energy(&self) -> [f32; 3] {
        self.raw
    }

    fn reset(&mut self, rate: u32) {
        self.rate = rate;
        self.fft = Spectrum::new();
        self.ranges = band_ranges(rate);
        self.started = false;
        self.warm_ticks = 0;
        self.long_avg = [0.0; 3];
        self.instant = [0.0; 3];
        self.att = [0.0; 3];
        self.raw = [0.0; 3];
        self.beat.reset();
        self.out = AudioFrame::default();
    }

    /// Analyses the audio ending at `now - av_offset`; call once per tick.
    pub fn tick(&mut self, hist: &AudioHistory, now: f64, av_offset: f64, dt: f64) -> &AudioFrame {
        if hist.sample_rate() != self.rate {
            self.reset(hist.sample_rate());
        }
        let end = now - av_offset;
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };

        hist.window(end, &mut self.wave_frames);
        for (i, f) in self.wave_frames.iter().enumerate() {
            self.out.wave[0][i] = finite(f[0] as f64);
            self.out.wave[1][i] = finite(f[1] as f64);
        }

        hist.window(end, &mut self.frames);
        let bins = self.fft.process(&self.frames);
        for (o, c) in self.out.spectrum.iter_mut().zip(bins) {
            let mag = ((c.re * c.re + c.im * c.im) as f64).sqrt();
            *o = finite((1.0 + mag * 0.5).ln());
        }

        let mut energy = [0.0f64; 3];
        for (e, &(a, b)) in energy.iter_mut().zip(&self.ranges) {
            let sum: f64 = bins[a..b]
                .iter()
                .map(|c| (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64))
                .sum();
            *e = if sum.is_finite() { sum } else { 0.0 };
        }
        self.raw = [finite(energy[0]), finite(energy[1]), finite(energy[2])];

        let silent =
            energy.iter().sum::<f64>() < SILENT_SUM || hist.silent_for(now) > SILENT_SECONDS;
        let (beat, strength) = self.beat.update(self.raw[0], now, dt, silent);
        if !silent {
            self.warm_ticks = self.warm_ticks.saturating_add(1);
            if !self.started {
                self.started = true;
                self.long_avg = energy;
            } else {
                // Running mean while warming up, then the 5 s exponential.
                let w = (1.0 - (-dt / LONG_TAU).exp()).max(1.0 / self.warm_ticks as f64);
                for (avg, &e) in self.long_avg.iter_mut().zip(&energy) {
                    *avg += (e - *avg) * w;
                }
            }
        }

        for (i, &(a, b)) in self.ranges.iter().enumerate() {
            if b <= a {
                self.instant[i] = 1.0;
                self.att[i] = 1.0;
            } else if self.started {
                let ratio = energy[i] / self.long_avg[i].max(AVG_FLOOR);
                let ratio = if ratio.is_finite() { ratio } else { 0.0 };
                smooth(&mut self.instant[i], ratio, dt, INSTANT_TAU);
                smooth(&mut self.att[i], ratio, dt, ATT_TAU);
            } else {
                self.instant[i] = 0.0;
                self.att[i] = 0.0;
            }
        }
        self.out.bass = finite(self.instant[0]);
        self.out.mid = finite(self.instant[1]);
        self.out.treb = finite(self.instant[2]);
        self.out.bass_att = finite(self.att[0]);
        self.out.mid_att = finite(self.att[1]);
        self.out.treb_att = finite(self.att[2]);
        self.out.beat = beat;
        self.out.beat_strength = strength;
        self.out.silent_for = finite(hist.silent_for(now));
        &self.out
    }
}
