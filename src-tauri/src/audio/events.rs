//! Everything the frontend hears about a recording. Capture is driven by the
//! global shortcut, so events are the only way the UI learns what happened.

use serde::Serialize;

pub const RECORDING_STARTED: &str = "recording-started";
pub const AUDIO_LEVEL: &str = "audio-level";
pub const RECORDING_COMPLETE: &str = "recording-complete";
pub const RECORDING_DISCARDED: &str = "recording-discarded";
pub const RECORDING_ERROR: &str = "recording-error";

/// Emitted on the first audio callback, not when the shortcut fires, so the
/// red dot means "the microphone is live" rather than "we asked for it".
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingStarted {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    /// Milliseconds between the start request and that first callback.
    pub onset_ms: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioLevel {
    /// Peak amplitude over the last interval, 0.0 to 1.0.
    pub peak: f32,
    pub elapsed_ms: u64,
    /// Latched for the whole clip once any sample reaches full scale.
    pub clipped: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingComplete {
    pub path: String,
    pub duration_ms: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub clipped: bool,
    /// `released` or `max-duration`.
    pub reason: &'static str,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingDiscarded {
    pub duration_ms: u64,
    /// `too-short` or `cancelled`. No WAV was written either way.
    pub reason: &'static str,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingError {
    pub message: String,
}
