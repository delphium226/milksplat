//! Windowed real FFT of the mono mix.

use crate::history::Frame;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// FFT length in frames.
pub const FFT_SIZE: usize = 2048;
/// Number of rfft bins (0..=FFT_SIZE/2).
pub const RFFT_BINS: usize = FFT_SIZE / 2 + 1;
/// Number of bins exposed to presets.
pub const SPECTRUM_BINS: usize = 512;

/// Planned FFT with preallocated buffers; `process` never allocates.
pub struct Spectrum {
    plan: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    input: Vec<f32>,
    output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl Spectrum {
    pub fn new() -> Self {
        let plan = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos())
            .collect();
        Self {
            input: plan.make_input_vec(),
            output: plan.make_output_vec(),
            scratch: plan.make_scratch_vec(),
            plan,
            window,
        }
    }

    /// Transforms the mono mix of `frames` (exactly `FFT_SIZE`); returns the
    /// complex rfft output.
    pub fn process(&mut self, frames: &[Frame]) -> &[Complex<f32>] {
        for ((x, f), w) in self.input.iter_mut().zip(frames).zip(&self.window) {
            *x = 0.5 * (f[0] + f[1]) * w;
        }
        // Lengths are fixed by construction, so this cannot fail; on the
        // impossible error the output is zeroed rather than panicking.
        if self
            .plan
            .process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch)
            .is_err()
        {
            self.output.fill(Complex::new(0.0, 0.0));
        }
        &self.output
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::new()
    }
}
