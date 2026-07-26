//! Rate conversion to 16 kHz.
//!
//! Runs once, offline, on the complete clip. A device rate that is not an
//! integer multiple of 16 kHz is the normal case, not a special one: sinc
//! interpolation reconstructs the continuous signal and evaluates it at
//! arbitrary fractional positions, so 44100 (ratio 160/441) takes exactly the
//! same code path as 48000 (ratio 1/3).

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use super::{AudioError, TARGET_RATE};

/// Filter length. Longer means a steeper transition band for more CPU; 256 is
/// rubato's suggested starting point and lands far inside our time budget.
const SINC_LEN: usize = 256;

/// Sub-sample positions precomputed per filter. With 256 of them, the linear
/// interpolation below picks between points 1/256 of a sample apart, which
/// puts its error well under the 16-bit noise floor.
const OVERSAMPLING: usize = 256;

/// Frames fed to the resampler per iteration. Only affects internal chunking.
const CHUNK_SIZE: usize = 1024;

/// Converts a mono clip to 16 kHz. Returns the input untouched when the device
/// already runs at 16 kHz, so no filtering is applied for nothing.
pub fn to_target_rate(input: &[f32], from_rate: u32) -> Result<Vec<f32>, AudioError> {
    if from_rate == TARGET_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }

    let ratio = f64::from(TARGET_RATE) / f64::from(from_rate);

    // `f_cutoff` is left at its default: rubato picks the highest cutoff that
    // keeps aliasing under the window's sidelobe level, then multiplies it by
    // the ratio when downsampling, which is the anti-alias filter this needs.
    let parameters = SincInterpolationParameters::new(SINC_LEN, WindowFunction::BlackmanHarris2)
        .oversampling_factor(OVERSAMPLING)
        .interpolation(SincInterpolationType::Linear);

    let mut resampler =
        Async::<f32>::new_sinc(ratio, 1.0, &parameters, CHUNK_SIZE, 1, FixedAsync::Input)?;

    // One channel, one frame per sample, so the slice maps straight through.
    let adapter = InterleavedSlice::new(input, 1, input.len())
        .expect("a mono adapter over the whole slice always has consistent dimensions");

    // `process_all` feeds the trailing partial chunk, trims the filter's
    // startup delay, flushes the tail, and returns exactly the expected number
    // of frames.
    let output = resampler.process_all(&adapter, input.len(), None)?;

    Ok(output.take_data())
}

/// Frames `to_target_rate` produces for an input of `input_len` frames.
#[cfg(test)]
fn expected_len(input_len: usize, from_rate: u32) -> usize {
    let ratio = f64::from(TARGET_RATE) / f64::from(from_rate);
    (input_len as f64 * ratio).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{num_complex::Complex, FftPlanner};

    fn sine(freq: f64, rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| {
                let t = n as f64 / f64::from(rate);
                (std::f64::consts::TAU * freq * t).sin() as f32
            })
            .collect()
    }

    /// Peak amplitude found in `[low, high]` Hz, scaled so that a pure sine of
    /// amplitude 1.0 reads back as 1.0 regardless of the transform length.
    fn peak_amplitude(signal: &[f32], rate: u32, low: f64, high: f64) -> f32 {
        // Analyse the middle of the clip: the edges carry the filter's ramp.
        const N: usize = 4096;
        assert!(signal.len() >= N, "clip too short to analyse");
        let start = (signal.len() - N) / 2;

        // Hann window, so a tone that is not periodic in the window does not
        // smear across the spectrum and hide the alias we are looking for.
        let window: Vec<f32> = (0..N)
            .map(|n| {
                let x = std::f32::consts::TAU * n as f32 / N as f32;
                0.5 - 0.5 * x.cos()
            })
            .collect();
        let coherent_gain: f32 = window.iter().sum();

        let mut buffer: Vec<Complex<f32>> = signal[start..start + N]
            .iter()
            .zip(&window)
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        FftPlanner::new().plan_fft_forward(N).process(&mut buffer);

        let bin_hz = f64::from(rate) / N as f64;
        let first = (low / bin_hz).floor().max(0.0) as usize;
        let last = ((high / bin_hz).ceil() as usize).min(N / 2);

        buffer[first..=last]
            .iter()
            // A real signal splits its energy between the positive and
            // negative frequency bins, hence the factor 2.
            .map(|c| 2.0 * c.norm() / coherent_gain)
            .fold(0.0, f32::max)
    }

    /// A 12 kHz tone at 44100 Hz is above the 8 kHz Nyquist of the target rate.
    /// Without an anti-alias filter it folds back to |16000 - 12000| = 4 kHz at
    /// nearly full amplitude. It must be at least 60 dB down instead.
    #[test]
    fn rejects_aliasing_when_downsampling() {
        let input = sine(12_000.0, 44_100, 44_100 / 2);
        let output = to_target_rate(&input, 44_100).expect("resample");

        let alias = peak_amplitude(&output, TARGET_RATE, 3_500.0, 4_500.0);
        let attenuation_db = 20.0 * alias.max(f32::MIN_POSITIVE).log10();

        assert!(
            attenuation_db <= -60.0,
            "12 kHz folded back to {alias:.6} at ~4 kHz ({attenuation_db:.1} dB), \
             expected at least 60 dB of rejection"
        );
    }

    /// The other half of the same story: a tone inside the passband has to
    /// survive, at the right frequency and at roughly the right level.
    #[test]
    fn preserves_a_tone_inside_the_passband() {
        let input = sine(1_000.0, 44_100, 44_100 / 2);
        let output = to_target_rate(&input, 44_100).expect("resample");

        let in_band = peak_amplitude(&output, TARGET_RATE, 950.0, 1_050.0);
        assert!(
            (in_band - 1.0).abs() < 0.02,
            "1 kHz came back at {in_band:.4}, expected ~1.0"
        );

        let elsewhere = peak_amplitude(&output, TARGET_RATE, 2_000.0, 8_000.0);
        assert!(
            elsewhere < 0.001,
            "unexpected energy at {elsewhere:.6} outside the tone"
        );
    }

    #[test]
    fn output_length_matches_the_ratio() {
        for (rate, samples) in [(44_100, 44_100), (48_000, 48_000), (32_000, 5_000)] {
            let output = to_target_rate(&sine(440.0, rate, samples), rate).expect("resample");
            assert_eq!(
                output.len(),
                expected_len(samples, rate),
                "wrong length for {rate} Hz"
            );
        }
    }

    #[test]
    fn passes_16k_through_untouched() {
        let input = sine(440.0, TARGET_RATE, 8_000);
        let output = to_target_rate(&input, TARGET_RATE).expect("resample");
        assert_eq!(input, output);
    }

    #[test]
    fn handles_an_empty_clip() {
        assert!(to_target_rate(&[], 44_100).expect("resample").is_empty());
    }

    /// Timing for a worst-case clip, to check the filter length stays within
    /// budget: this runs between releasing the key and the WAV appearing.
    /// Ignored by default because the result is a property of the machine, not
    /// of the code. Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn reports_time_for_a_full_length_clip() {
        for rate in [44_100, 48_000] {
            let input = sine(440.0, rate, rate as usize * 120);
            let start = std::time::Instant::now();
            let output = to_target_rate(&input, rate).expect("resample");
            let elapsed = start.elapsed();

            println!(
                "120 s at {rate} Hz -> {} frames in {:.0} ms",
                output.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}
