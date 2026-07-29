//! Clip in, text out.
//!
//! Transcription is chained in Rust off the end of a recording rather than
//! invoked from the webview. The temp WAV's whole life — written, read,
//! deleted or deliberately kept — then stays in one place, and a webview reload
//! in the middle of a dictation cannot strand a file in the temp directory or
//! drop a transcript on the floor.

pub mod engine;
pub mod filter;
pub mod prompt;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::config::Config;
use crate::model;
use crate::resident::Resident;
use engine::Engine;

pub const TRANSCRIPTION_STARTED: &str = "transcription-started";
pub const TRANSCRIPTION_COMPLETE: &str = "transcription-complete";
pub const TRANSCRIPTION_EMPTY: &str = "transcription-empty";
pub const TRANSCRIPTION_ERROR: &str = "transcription-error";

/// The managed Whisper engine. `Arc` because warming needs to hand ownership
/// to a background thread.
pub type Whisper = Arc<Resident<Engine>>;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Started {
    pub clip_duration_ms: u64,
    /// The model still has to load. Lets the spinner say so rather than
    /// looking hung for the first few seconds after a cold start.
    pub model_cold: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Complete {
    pub text: String,
    /// Wall clock spent inside Whisper, excluding any model load.
    pub duration_ms: u64,
    pub clip_duration_ms: u64,
    /// `duration_ms / clip_duration_ms`. Below 1.0 is faster than real time.
    pub realtime_factor: f32,
    pub segment_count: usize,
    pub dropped_count: usize,
    pub model_id: String,
    pub backend: &'static str,
}

/// Whisper ran, or was skipped, and there is nothing to insert. Distinct from
/// `Complete` with an empty string so the editor never appends a blank line
/// for a clip that said nothing.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Empty {
    /// `rms-floor`, `no-speech`, `denylist`, or `empty`.
    pub reason: &'static str,
    pub rms_dbfs: f32,
    pub clip_duration_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failed {
    pub message: String,
    /// The clip is deliberately left on disk when transcription fails, so the
    /// dictation is not lost and the failure can be reproduced.
    pub wav_path: String,
}

/// How many clips are inside Whisper right now.
///
/// Not derivable from anything else. The recorder is already idle by the time a
/// clip is transcribed, and the residency says whether the model is loaded, not
/// whether it is busy — it is warmed on window show with nothing to do. The tray
/// is the only reader, and what it needs is a count rather than a flag: the retry
/// command can queue a second clip while the first is still running.
#[derive(Default)]
pub struct InFlight(AtomicUsize);

impl InFlight {
    pub fn any(&self) -> bool {
        self.0.load(Ordering::Acquire) > 0
    }
}

/// Counts one clip as in flight for as long as it exists.
///
/// A guard rather than a pair of calls because `run` returns from a dozen places
/// — every guard clause, every failure — and a decrement that is missed once
/// leaves the tray claiming Steno is working forever.
struct Busy<R: Runtime>(AppHandle<R>);

impl<R: Runtime> Busy<R> {
    fn enter(app: &AppHandle<R>) -> Self {
        app.state::<Arc<InFlight>>().0.fetch_add(1, Ordering::AcqRel);
        crate::tray::refresh(app);
        Self(app.clone())
    }
}

impl<R: Runtime> Drop for Busy<R> {
    fn drop(&mut self) {
        self.0
            .state::<Arc<InFlight>>()
            .0
            .fetch_sub(1, Ordering::AcqRel);
        crate::tray::refresh(&self.0);
    }
}

/// Runs a clip through Whisper on its own thread and reports the result.
///
/// Called from the audio session thread, which must return promptly to release
/// the recorder, so this never blocks its caller.
pub fn spawn<R: Runtime>(app: &AppHandle<R>, wav: PathBuf) {
    // Taken here rather than inside the thread, on purpose: the recorder returns
    // to Idle as soon as this call comes back, and a count raised a few
    // milliseconds later would let the tray blink through Idle between the clip
    // being written and Whisper starting on it.
    let busy = Busy::enter(app);
    let app = app.clone();

    let spawned = thread::Builder::new()
        .name("steno-transcribe".to_owned())
        .spawn(move || {
            let _busy = busy;
            run(app, wav)
        });

    if let Err(error) = spawned {
        // `busy` moved into the closure, which `spawn` dropped on failure, so
        // the count is already back down.
        eprintln!("transcribe: could not spawn the transcription thread ({error})");
    }
}

fn run<R: Runtime>(app: AppHandle<R>, wav: PathBuf) {
    let settings = app.state::<Config>().get().whisper;

    let samples = match read_clip(&wav) {
        Ok(samples) => samples,
        Err(error) => return fail(&app, &wav, &error),
    };

    let clip_duration_ms = (samples.len() as u64) * 1_000 / u64::from(crate::audio::TARGET_RATE);
    let whisper = app.state::<Whisper>().inner().clone();

    let _ = app.emit(
        TRANSCRIPTION_STARTED,
        Started {
            clip_duration_ms,
            model_cold: !whisper.is_warm(),
        },
    );

    // Guard one, before anything expensive. A gated microphone delivers a
    // buffer of near-zeros when you speak too quietly or miss the key, and
    // Whisper answers near-silence with confident subtitle boilerplate.
    let rms_dbfs = filter::rms_dbfs(&samples);
    if rms_dbfs < settings.rms_floor_dbfs {
        eprintln!(
            "transcribe: clip is {rms_dbfs:.1} dBFS, below the {:.1} dBFS floor — not transcribed",
            settings.rms_floor_dbfs
        );
        discard(&wav);
        return empty(&app, filter::Guard::RmsFloor.as_str(), rms_dbfs, clip_duration_ms);
    }

    let spec = model::resolve(&app);
    let path = model::path(&app, spec);

    let lease = match whisper.acquire(|| Engine::load(&path, spec.id)) {
        Ok(lease) => lease,
        Err(error) => return fail(&app, &wav, &error),
    };

    let started = Instant::now();
    let transcript = match lease.run(&samples, &settings) {
        Ok(transcript) => transcript,
        Err(error) => return fail(&app, &wav, &error),
    };
    let duration_ms = started.elapsed().as_millis() as u64;
    crate::lifecycle::touch(&app);

    eprintln!(
        "transcribe: {} ms of audio in {duration_ms} ms on {} ({} segment(s), {} dropped)",
        clip_duration_ms,
        lease.backend,
        transcript.segment_count,
        transcript.dropped()
    );

    if transcript.text.is_empty() {
        let reason = transcript
            .emptied_by()
            .map(|guard| guard.as_str())
            .unwrap_or("empty");
        discard(&wav);
        return empty(&app, reason, rms_dbfs, clip_duration_ms);
    }

    let realtime_factor = if clip_duration_ms > 0 {
        duration_ms as f32 / clip_duration_ms as f32
    } else {
        0.0
    };

    let dropped_count = transcript.dropped();
    let _ = app.emit(
        TRANSCRIPTION_COMPLETE,
        Complete {
            text: transcript.text,
            duration_ms,
            clip_duration_ms,
            realtime_factor,
            segment_count: transcript.segment_count,
            dropped_count,
            model_id: lease.model_id.clone(),
            backend: lease.backend,
        },
    );

    discard(&wav);
}

/// Reads the 16 kHz mono clip the recorder wrote.
///
/// Re-reading rather than passing the samples through from the session thread:
/// the WAV has to exist anyway — it is what survives a failure — and this keeps
/// the retry command on exactly the same path as the automatic run.
pub fn read_clip(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("could not open {} ({error})", path.display()))?;

    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != crate::audio::TARGET_RATE {
        return Err(format!(
            "{} is {} channel(s) at {} Hz, expected mono at {} Hz",
            path.display(),
            spec.channels,
            spec.sample_rate,
            crate::audio::TARGET_RATE
        ));
    }

    const FULL_SCALE: f32 = 32_768.0;
    reader
        .samples::<i16>()
        .map(|sample| {
            sample
                .map(|value| f32::from(value) / FULL_SCALE)
                .map_err(|error| format!("could not read {} ({error})", path.display()))
        })
        .collect()
}

/// Removes the temp clip. Only ever called once the audio has been accounted
/// for: transcribed, or positively identified as silence.
fn discard(wav: &Path) {
    if let Err(error) = std::fs::remove_file(wav) {
        eprintln!("transcribe: could not delete {} ({error})", wav.display());
    }
}

/// Keeps the clip and says where it is. Losing a dictation to a transient
/// failure is worse than leaving a file in the temp directory.
fn fail<R: Runtime>(app: &AppHandle<R>, wav: &Path, message: &str) {
    eprintln!(
        "transcribe: failed ({message}); the clip is kept at {}",
        wav.display()
    );

    let _ = app.emit(
        TRANSCRIPTION_ERROR,
        Failed {
            message: message.to_owned(),
            wav_path: wav.to_string_lossy().into_owned(),
        },
    );
}

fn empty<R: Runtime>(
    app: &AppHandle<R>,
    reason: &'static str,
    rms_dbfs: f32,
    clip_duration_ms: u64,
) {
    let _ = app.emit(
        TRANSCRIPTION_EMPTY,
        Empty {
            reason,
            rms_dbfs,
            clip_duration_ms,
        },
    );
}
