//! Fetching the model file, resumably, with the digest checked before it is
//! ever used.
//!
//! This is the first thing a new user sees. A silent multi-gigabyte transfer
//! behind a frozen window is indistinguishable from a broken app, so progress
//! is emitted continuously, an interrupted transfer picks up where it left off,
//! and the file is only put in place once its SHA-256 matches.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Runtime};

use super::ModelSpec;
use crate::audio::lock;

pub const DOWNLOAD_PROGRESS: &str = "model-download-progress";
pub const DOWNLOAD_COMPLETE: &str = "model-download-complete";
pub const DOWNLOAD_ERROR: &str = "model-download-error";

/// How often progress reaches the UI. Fast enough to look alive, slow enough
/// that a 3 GB transfer does not emit a hundred thousand events.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(200);

/// A stalled connection has to surface as an error rather than as a progress
/// bar that stopped moving for no stated reason.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub model_id: &'static str,
    pub received_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: u64,
    pub eta_ms: u64,
    /// The transfer restarted from a partial file rather than from zero.
    pub resumed: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Complete {
    pub model_id: &'static str,
    pub path: String,
    pub size_bytes: u64,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failed {
    pub model_id: &'static str,
    pub message: String,
    /// Whether pressing the button again continues rather than starts over.
    pub resumable: bool,
}

/// Tauri managed state: at most one download at a time, and a way to stop it.
#[derive(Default)]
pub struct Downloads {
    cancel: Mutex<Option<Arc<AtomicBool>>>,
}

impl Downloads {
    /// Claims the single download slot.
    fn begin(&self) -> Result<Arc<AtomicBool>, String> {
        let mut slot = lock(&self.cancel);
        if slot.is_some() {
            return Err("a download is already running".to_owned());
        }
        let flag = Arc::new(AtomicBool::new(false));
        *slot = Some(flag.clone());
        Ok(flag)
    }

    fn end(&self) {
        *lock(&self.cancel) = None;
    }

    /// Asks the running download to stop. The partial file is kept.
    pub fn cancel(&self) {
        if let Some(flag) = lock(&self.cancel).as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    pub fn is_running(&self) -> bool {
        lock(&self.cancel).is_some()
    }
}

/// Frees the slot however the download ends, including on an early return.
struct EndOnDrop<'a>(&'a Downloads);

impl Drop for EndOnDrop<'_> {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// The in-progress file sitting next to the finished one. A distinct name is
/// what stops a half-downloaded model from ever being loaded.
pub fn partial_path(destination: &Path) -> PathBuf {
    let mut name = destination
        .file_name()
        .unwrap_or_default()
        .to_os_string();
    name.push(".part");
    destination.with_file_name(name)
}

/// Downloads `spec` to `destination`, resuming an interrupted transfer, and
/// emits progress as it goes.
///
/// Errors are returned *and* emitted: the caller may be a command with nobody
/// awaiting it, and the UI needs to hear either way.
pub async fn run<R: Runtime>(
    app: AppHandle<R>,
    downloads: &Downloads,
    spec: &'static ModelSpec,
    destination: PathBuf,
) -> Result<(), String> {
    let cancel = downloads.begin()?;
    let _end = EndOnDrop(downloads);

    let emitter = app.clone();
    let report = move |progress: Progress| {
        let _ = emitter.emit(DOWNLOAD_PROGRESS, progress);
    };

    match transfer(spec, &destination, &cancel, &report).await {
        Ok(duration) => {
            let _ = app.emit(
                DOWNLOAD_COMPLETE,
                Complete {
                    model_id: spec.id,
                    path: destination.to_string_lossy().into_owned(),
                    size_bytes: spec.bytes,
                    duration_ms: duration.as_millis() as u64,
                },
            );
            Ok(())
        }
        Err(error) => {
            let resumable = partial_path(&destination).exists();
            let _ = app.emit(
                DOWNLOAD_ERROR,
                Failed {
                    model_id: spec.id,
                    message: error.clone(),
                    resumable,
                },
            );
            Err(error)
        }
    }
}

/// The download itself, with no dependency on Tauri.
///
/// Progress goes to a sink rather than straight to `AppHandle::emit`, so the
/// exact code the app runs — resume, range handling, checksum — can also be
/// driven from a headless harness. A download path that is only ever exercised
/// on a user's first launch is a download path nobody has tested.
pub async fn transfer(
    spec: &'static ModelSpec,
    destination: &Path,
    cancel: &AtomicBool,
    report: &(dyn Fn(Progress) + Send + Sync),
) -> Result<Duration, String> {
    let started = Instant::now();

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!("could not create {} ({error})", parent.display())
        })?;
    }

    let partial = partial_path(destination);
    let mut have = resume_point(&partial, spec)?;

    // A previous run finished the bytes but died before verifying. Skip
    // straight to the digest rather than re-fetching gigabytes.
    if have < spec.bytes {
        have = fetch(spec, &partial, have, cancel, report).await?;
    }

    if have != spec.bytes {
        return Err(format!(
            "{} is {have} bytes, expected {}",
            spec.id, spec.bytes
        ));
    }

    verify(spec, partial.clone(), report).await?;

    fs::rename(&partial, destination)
        .map_err(|error| format!("could not move the model into place ({error})"))?;

    Ok(started.elapsed())
}

/// How many bytes of `partial` can be kept. A file longer than the model
/// itself is not a partial download of it, so it is thrown away.
fn resume_point(partial: &Path, spec: &ModelSpec) -> Result<u64, String> {
    let Ok(meta) = fs::metadata(partial) else {
        return Ok(0);
    };

    if meta.len() > spec.bytes {
        fs::remove_file(partial)
            .map_err(|error| format!("could not discard an oversized partial file ({error})"))?;
        return Ok(0);
    }

    Ok(meta.len())
}

async fn fetch(
    spec: &'static ModelSpec,
    partial: &Path,
    mut have: u64,
    cancel: &AtomicBool,
    report: &(dyn Fn(Progress) + Send + Sync),
) -> Result<u64, String> {
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|error| format!("could not build an HTTP client ({error})"))?;

    let mut request = client.get(spec.url);
    if have > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }

    let response = request
        .send()
        .await
        .map_err(|error| format!("could not reach huggingface.co ({error})"))?;

    if !response.status().is_success() {
        return Err(format!(
            "huggingface.co answered {} for {}",
            response.status(),
            spec.id
        ));
    }

    // A server that ignores `Range` answers 200 with the whole file. Honouring
    // that means starting over, otherwise the bytes would be appended to what
    // is already there and the digest would never match.
    let resumed = have > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !resumed {
        eprintln!("model: the server ignored our range request, starting over");
        have = 0;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(have == 0)
        .append(have > 0)
        .open(partial)
        .map_err(|error| format!("could not open {} ({error})", partial.display()))?;

    let mut stream = response.bytes_stream();
    let began_at = have;
    let began = Instant::now();
    let mut last_emit = Instant::now() - PROGRESS_INTERVAL;

    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            // Flush what we have: the point of cancelling is to resume later.
            let _ = file.flush();
            return Err("download cancelled".to_owned());
        }

        let chunk = chunk.map_err(|error| format!("the transfer failed ({error})"))?;
        file.write_all(&chunk)
            .map_err(|error| format!("could not write to {} ({error})", partial.display()))?;
        have += chunk.len() as u64;

        if last_emit.elapsed() >= PROGRESS_INTERVAL {
            last_emit = Instant::now();
            report(measure(spec, have, have - began_at, began.elapsed(), resumed));
        }
    }

    file.flush()
        .map_err(|error| format!("could not flush {} ({error})", partial.display()))?;

    report(measure(spec, have, have - began_at, began.elapsed(), resumed));
    Ok(have)
}

/// Turns raw counters into the shape the UI draws. Rate is measured over this
/// session only: a resumed transfer must not average in the bytes some earlier
/// attempt fetched at a different speed.
fn measure(
    spec: &'static ModelSpec,
    received: u64,
    this_session: u64,
    elapsed: Duration,
    resumed: bool,
) -> Progress {
    let seconds = elapsed.as_secs_f64();
    let rate = if seconds > 0.1 {
        (this_session as f64 / seconds) as u64
    } else {
        0
    };

    let remaining = spec.bytes.saturating_sub(received);
    let eta_ms = if rate > 0 {
        remaining * 1_000 / rate
    } else {
        0
    };

    Progress {
        model_id: spec.id,
        received_bytes: received,
        total_bytes: spec.bytes,
        bytes_per_second: rate,
        eta_ms,
        resumed,
    }
}

/// Hashes the finished file and compares it against the published digest.
///
/// Runs on a blocking thread: this reads gigabytes, and doing it on an async
/// worker would stall every other task on the runtime. A mismatch deletes the
/// file — a corrupt partial never converges by resuming, it only wastes the
/// next attempt too.
async fn verify(
    spec: &'static ModelSpec,
    partial: PathBuf,
    report: &(dyn Fn(Progress) + Send + Sync),
) -> Result<(), String> {
    // The bar has nothing left to advance, so say what is happening instead of
    // going quiet for the seconds this takes.
    report(Progress {
        model_id: spec.id,
        received_bytes: spec.bytes,
        total_bytes: spec.bytes,
        bytes_per_second: 0,
        eta_ms: 0,
        resumed: false,
    });

    let expected = spec.sha256;
    let target = partial.clone();
    let digest = tauri::async_runtime::spawn_blocking(move || hash(&target))
        .await
        .map_err(|error| format!("the checksum task failed ({error})"))??;

    if digest != expected {
        // Deleted, not kept: the bytes are wrong somewhere unknown, so resuming
        // would append to corruption and fail the same way, for ever.
        let _ = fs::remove_file(&partial);
        return Err(format!(
            "checksum mismatch for {}: expected {expected}, got {digest}. \
             The partial file was discarded; the download will start over.",
            spec.id
        ));
    }

    Ok(())
}

fn hash(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("could not read the model ({error})"))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("could not read the model ({error})"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}
