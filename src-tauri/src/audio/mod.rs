//! Push-to-talk capture: microphone in, 16 kHz mono WAV out.
//!
//! The device is opened at whatever configuration it advertises, downmixed to
//! mono in the audio callback, then resampled and written to disk once, when
//! the recording stops.

pub mod capture;
pub mod error;
pub mod events;
pub mod recorder;
pub mod resample;
pub mod wav;

// Test-only: generates the committed `fixtures/pipeline-check.wav`.
#[cfg(test)]
mod fixture;

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

pub use error::AudioError;
pub use recorder::{Recorder, RecordingState};

/// Takes a lock, recovering from poisoning instead of propagating a panic.
/// A thread that died holding the sample buffer must not leave the shortcut
/// dead for the rest of the session.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The rate Whisper wants. Nothing reaches disk at any other rate.
pub const TARGET_RATE: u32 = 16_000;

/// Anything shorter is an accidental tap on the shortcut, not a dictation.
pub const MIN_DURATION: Duration = Duration::from_millis(300);

/// Hard cap. Capture auto-stops here and still produces a WAV.
pub const MAX_DURATION: Duration = Duration::from_secs(120);

/// How often the session thread publishes an input level.
pub const LEVEL_INTERVAL: Duration = Duration::from_millis(50);

/// A sample at or above this counts as clipped. Not 1.0: converters that
/// saturate rarely land exactly on full scale.
pub const CLIP_THRESHOLD: f32 = 0.99;
