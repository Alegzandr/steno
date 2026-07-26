//! Generator for the committed test clip.
//!
//! Not part of the app: the whole file is test-only. It exists so there is one
//! WAV in the tree that Steno's own pipeline produced — a signal that actually
//! carries audio — separate from anything captured on a real microphone. It
//! runs the synthetic clip through the real `resample` and `wav` code, so a
//! green run also proves that path writes a listenable 16 kHz file.
//!
//! Regenerate with:
//!   cargo test --package steno -- --ignored generate_pipeline_check_wav --nocapture

use std::path::PathBuf;

use super::{resample, wav};

/// A native capture rate WASAPI would hand us; 48 kHz down to 16 kHz is the
/// common 1/3 path, so the resampler does real work here.
const NATIVE_RATE: u32 = 48_000;

/// Committed clip location: `src-tauri/fixtures/pipeline-check.wav`.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("pipeline-check.wav")
}

/// A 2 s linear chirp from 220 Hz to 3 kHz at 0.6 amplitude, with 10 ms fades
/// so the ends do not click. The whole sweep stays under the 8 kHz target
/// Nyquist, so it survives resampling intact and reads as a clean rising tone.
fn chirp(rate: u32) -> Vec<f32> {
    const SECONDS: f64 = 2.0;
    const F0: f64 = 220.0;
    const F1: f64 = 3_000.0;
    const AMPLITUDE: f64 = 0.6;

    let total = (SECONDS * f64::from(rate)) as usize;
    let fade = (0.010 * f64::from(rate)) as usize;
    let sweep = (F1 - F0) / SECONDS;

    (0..total)
        .map(|n| {
            let t = n as f64 / f64::from(rate);
            // Instantaneous frequency F0 + sweep*t, so phase integrates to this.
            let phase = std::f64::consts::TAU * (F0 * t + 0.5 * sweep * t * t);
            let ramp_in = (n as f64 / fade as f64).min(1.0);
            let ramp_out = ((total - n) as f64 / fade as f64).min(1.0);
            (AMPLITUDE * ramp_in.min(ramp_out) * phase.sin()) as f32
        })
        .collect()
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0, |m, &s| f32::max(m, s.abs()))
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|&s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Ignored by default: it writes into the source tree, so it only runs when
/// asked. Exercises the same `resample` + `wav` path a real recording uses,
/// then copies the temp WAV to the committed fixture location.
#[test]
#[ignore]
fn generate_pipeline_check_wav() {
    let raw = chirp(NATIVE_RATE);
    let samples = resample::to_target_rate(&raw, NATIVE_RATE).expect("resample the chirp");

    let temp = wav::write_temp(&samples).expect("write the temp WAV");
    let dest = fixture_path();
    std::fs::create_dir_all(dest.parent().expect("fixtures has a parent"))
        .expect("create fixtures dir");
    std::fs::copy(&temp, &dest).expect("copy into fixtures");
    std::fs::remove_file(&temp).ok();

    println!(
        "wrote {} ({} Hz -> {} Hz, {} frames, peak {:.3}, rms {:.3})",
        dest.display(),
        NATIVE_RATE,
        super::TARGET_RATE,
        samples.len(),
        peak(&samples),
        rms(&samples),
    );
}
