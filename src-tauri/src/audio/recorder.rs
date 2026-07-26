//! Session lifecycle: the state the shortcut handler touches, and the thread
//! that owns a recording from the first callback to the finished WAV.
//!
//! The shortcut handler runs on the event loop, so nothing here blocks it:
//! `start` spawns, `stop` sends. Opening the device and writing the file both
//! take tens to hundreds of milliseconds and happen on the session thread.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::capture::{self, Capture, InputConfig};
use super::{
    events, lock, resample, wav, LEVEL_INTERVAL, MAX_DURATION, MIN_DURATION, TARGET_RATE,
};
use crate::config::Config;

/// What the UI needs to know when it resyncs after a reload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecordingState {
    Idle,
    Recording,
    /// Capture stopped; the clip is still being resampled and written.
    Finalizing,
}

/// Why a session ended.
enum Stop {
    Released,
    MaxDuration,
    Cancelled,
    DeviceError(String),
}

/// The internal counterpart of `RecordingState`, carrying the channel that
/// reaches the running session.
enum Slot {
    Idle,
    Recording(Sender<Stop>),
    Finalizing,
}

struct Inner {
    slot: Mutex<Slot>,
}

impl Inner {
    fn set(&self, slot: Slot) {
        *lock(&self.slot) = slot;
    }
}

/// Tauri managed state. The cpal stream itself lives on the session thread,
/// which is where it is built and dropped.
pub struct Recorder {
    inner: Arc<Inner>,
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                slot: Mutex::new(Slot::Idle),
            }),
        }
    }

    /// Starts a recording. Ignored when one is already running or when the
    /// previous clip is still being written.
    pub fn start<R: Runtime>(&self, app: &AppHandle<R>) {
        let mut slot = lock(&self.inner.slot);

        // The guard. On Windows the hotkey is registered with MOD_NOREPEAT so
        // Pressed does not repeat, but other platforms do repeat it while the
        // key is held, and a second Pressed must never open a second device.
        if !matches!(*slot, Slot::Idle) {
            return;
        }

        let (stop, stops) = mpsc::channel();
        *slot = Slot::Recording(stop.clone());
        drop(slot);

        let inner = self.inner.clone();
        let handle = app.clone();

        let spawned = thread::Builder::new()
            .name("steno-audio".to_owned())
            .spawn(move || run(handle, inner, stop, stops));

        if let Err(error) = spawned {
            self.inner.set(Slot::Idle);
            emit_error(app, &format!("could not start the audio thread: {error}"));
        }
    }

    /// Ends the recording and writes the clip.
    pub fn stop(&self) {
        self.signal(Stop::Released);
    }

    /// Ends the recording and throws the clip away without writing a WAV.
    pub fn cancel(&self) {
        self.signal(Stop::Cancelled);
    }

    pub fn state(&self) -> RecordingState {
        match &*lock(&self.inner.slot) {
            Slot::Idle => RecordingState::Idle,
            Slot::Recording(_) => RecordingState::Recording,
            Slot::Finalizing => RecordingState::Finalizing,
        }
    }

    fn signal(&self, stop: Stop) {
        // Ignored unless a capture is running: a Released can arrive with no
        // matching Pressed if the shortcut was registered mid-hold.
        if let Slot::Recording(sender) = &*lock(&self.inner.slot) {
            let _ = sender.send(stop);
        }
    }
}

/// Returns the recorder to `Idle` however the session thread ends, including
/// on a panic. A stuck `Recording` would leave the shortcut dead for the rest
/// of the session.
struct ResetOnDrop(Arc<Inner>);

impl Drop for ResetOnDrop {
    fn drop(&mut self) {
        self.0.set(Slot::Idle);
    }
}

fn run<R: Runtime>(
    app: AppHandle<R>,
    inner: Arc<Inner>,
    stop: Sender<Stop>,
    stops: Receiver<Stop>,
) {
    let _reset = ResetOnDrop(inner.clone());

    // Read at capture time, not at start(): changing the device in the UI takes
    // effect on the next recording without restarting the app.
    let preferred = app.state::<Config>().input_device();

    let device_errors = stop.clone();
    let capture = match capture::start(preferred.as_deref(), move |error| {
        // Unplugged device, driver gone, format changed underneath us. Folding
        // it into the stop channel means one exit path, so the guard is
        // released exactly once.
        let _ = device_errors.send(Stop::DeviceError(error.to_string()));
    }) {
        Ok(capture) => capture,
        Err(error) => return emit_error(&app, &error.to_string()),
    };

    let requested_at = Instant::now();
    let live_at = wait_until_live(&capture, requested_at);

    let _ = app.emit(
        events::RECORDING_STARTED,
        events::RecordingStarted {
            device_name: capture.config.device_name.clone(),
            sample_rate: capture.config.sample_rate,
            channels: capture.config.channels,
            onset_ms: live_at.duration_since(requested_at).as_millis() as u64,
        },
    );

    let reason = meter(&app, &capture, &stops, live_at);

    // Release the device before spending time on the file.
    drop(capture.stream);
    inner.set(Slot::Finalizing);

    let clipped = capture.clipped.load(Ordering::Relaxed);
    let raw = std::mem::take(&mut *lock(&capture.samples));

    finalize(&app, raw, &capture.config, clipped, reason);
}

/// Waits for the device's first callback, so `recording-started` means the
/// microphone is live rather than "we asked for it". Measured at a 13 ms
/// median on this machine, 25 ms worst case; the ceiling is a safety net for a
/// device that never delivers, not an expected wait.
fn wait_until_live(capture: &Capture, requested_at: Instant) -> Instant {
    const POLL: Duration = Duration::from_millis(2);
    const CEILING: Duration = Duration::from_millis(500);

    while !capture.live.load(Ordering::Relaxed) && requested_at.elapsed() < CEILING {
        thread::sleep(POLL);
    }

    Instant::now()
}

/// Publishes input levels until something stops the recording.
fn meter<R: Runtime>(
    app: &AppHandle<R>,
    capture: &Capture,
    stops: &Receiver<Stop>,
    live_at: Instant,
) -> Stop {
    let mut cursor = 0;

    loop {
        match stops.recv_timeout(LEVEL_INTERVAL) {
            Ok(stop) => return stop,
            // Cannot happen while this thread holds a sender, but treat a lost
            // channel as a normal stop rather than spinning forever.
            Err(RecvTimeoutError::Disconnected) => return Stop::Released,
            Err(RecvTimeoutError::Timeout) => {}
        }

        let _ = app.emit(
            events::AUDIO_LEVEL,
            events::AudioLevel {
                peak: peak_since(capture, &mut cursor),
                elapsed_ms: live_at.elapsed().as_millis() as u64,
                clipped: capture.clipped.load(Ordering::Relaxed),
            },
        );

        if live_at.elapsed() >= MAX_DURATION {
            return Stop::MaxDuration;
        }
    }
}

/// Peak over the samples that arrived since the previous call. Scanning
/// happens here rather than in the audio callback, and holds the buffer lock
/// only for one interval's worth of samples.
fn peak_since(capture: &Capture, cursor: &mut usize) -> f32 {
    let buffer = lock(&capture.samples);
    let peak = buffer[*cursor..]
        .iter()
        .fold(0.0, |peak, sample| f32::max(peak, sample.abs()));

    *cursor = buffer.len();
    peak
}

fn finalize<R: Runtime>(
    app: &AppHandle<R>,
    raw: Vec<f32>,
    config: &InputConfig,
    clipped: bool,
    reason: Stop,
) {
    let captured_ms = duration_ms(raw.len(), config.sample_rate);

    // The device died mid-recording, but the samples we captured before it
    // dropped are real dictation. Save them and still report the error.
    if let Stop::DeviceError(message) = reason {
        return salvage(app, &raw, config, clipped, captured_ms, &message);
    }

    let reason = match reason {
        // Handled above; the compiler cannot see that.
        Stop::DeviceError(_) => unreachable!(),
        Stop::Cancelled => return emit_discarded(app, captured_ms, "cancelled"),
        Stop::Released => "released",
        Stop::MaxDuration => "max-duration",
    };

    // An accidental tap on the shortcut. Silently dropped, but the frontend
    // still hears about it so the UI can leave the recording state.
    if captured_ms < MIN_DURATION.as_millis() as u64 {
        return emit_discarded(app, captured_ms, "too-short");
    }

    match write_clip(&raw, config) {
        Ok((path, samples)) => emit_complete(app, &path, samples, clipped, reason),
        Err(error) => emit_error(app, &error.to_string()),
    }
}

/// Losing 100 seconds of dictation to a USB glitch is avoidable: whatever
/// reached memory before the device dropped is written out as a normal clip,
/// tagged `device-lost`. The error is emitted alongside it either way, so the
/// UI leaves the recording state and the user knows the mic went. A clip too
/// short to be a dictation is dropped — only the error goes out.
fn salvage<R: Runtime>(
    app: &AppHandle<R>,
    raw: &[f32],
    config: &InputConfig,
    clipped: bool,
    captured_ms: u64,
    message: &str,
) {
    if captured_ms >= MIN_DURATION.as_millis() as u64 {
        // Best effort: if the salvage write itself fails, the error below is
        // still the user's signal.
        if let Ok((path, samples)) = write_clip(raw, config) {
            emit_complete(app, &path, samples, clipped, "device-lost");
        }
    }

    emit_error(app, message);
}

/// Resamples the raw capture to 16 kHz and writes the temp WAV. Returns its
/// path and the final sample count (the file's own length).
fn write_clip(raw: &[f32], config: &InputConfig) -> Result<(PathBuf, usize), super::AudioError> {
    let samples = resample::to_target_rate(raw, config.sample_rate)?;
    let len = samples.len();
    let path = wav::write_temp(&samples)?;
    Ok((path, len))
}

fn emit_complete<R: Runtime>(
    app: &AppHandle<R>,
    path: &Path,
    sample_len: usize,
    clipped: bool,
    reason: &'static str,
) {
    let _ = app.emit(
        events::RECORDING_COMPLETE,
        events::RecordingComplete {
            path: path.to_string_lossy().into_owned(),
            duration_ms: duration_ms(sample_len, TARGET_RATE),
            sample_rate: TARGET_RATE,
            channels: 1,
            clipped,
            reason,
        },
    );
}

fn duration_ms(samples: usize, rate: u32) -> u64 {
    samples as u64 * 1_000 / u64::from(rate)
}

fn emit_error<R: Runtime>(app: &AppHandle<R>, message: &str) {
    let _ = app.emit(
        events::RECORDING_ERROR,
        events::RecordingError {
            message: message.to_owned(),
        },
    );
}

fn emit_discarded<R: Runtime>(app: &AppHandle<R>, duration_ms: u64, reason: &'static str) {
    let _ = app.emit(
        events::RECORDING_DISCARDED,
        events::RecordingDiscarded {
            duration_ms,
            reason,
        },
    );
}
