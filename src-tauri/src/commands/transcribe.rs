use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Manager, Runtime};

use crate::format::Formatter;
use crate::resident::ResidentState;
use crate::transcribe::{self, Whisper};

/// Re-runs a clip that failed. Transcription keeps the WAV on failure and
/// reports its path precisely so this is possible without dictating again.
///
/// Returns as soon as the work is queued; the result arrives as the same events
/// an automatic run emits.
#[tauri::command]
pub fn transcribe_file<R: Runtime>(app: AppHandle<R>, path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !path.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }

    transcribe::spawn(&app, path);
    Ok(())
}

/// What is currently resident. Lets the UI resync after a webview reload
/// instead of waiting for the next state change.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Residency {
    pub whisper: ResidentState,
    pub llm: ResidentState,
}

#[tauri::command]
pub fn residency<R: Runtime>(app: AppHandle<R>) -> Residency {
    Residency {
        whisper: app.state::<Whisper>().state(),
        llm: app.state::<Formatter>().state(),
    }
}
