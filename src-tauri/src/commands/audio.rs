use tauri::{AppHandle, Runtime, State};

use crate::audio::{Recorder, RecordingState};

/// The global shortcut is the real trigger; this exists so a recording can be
/// driven from the webview console while debugging.
#[tauri::command]
pub fn start_recording<R: Runtime>(app: AppHandle<R>, recorder: State<'_, Recorder>) {
    recorder.start(&app);
}

/// Counterpart of `start_recording`. Writes the clip, like releasing the key.
#[tauri::command]
pub fn stop_recording(recorder: State<'_, Recorder>) {
    recorder.stop();
}

/// Backs the cancel control shown while recording: ends the capture and drops
/// the clip without writing a WAV.
#[tauri::command]
pub fn cancel_recording(recorder: State<'_, Recorder>) {
    recorder.cancel();
}

/// Lets the UI resync after a webview reload, which in dev happens on every
/// hot reload and can land in the middle of a recording.
#[tauri::command]
pub fn recording_state(recorder: State<'_, Recorder>) -> RecordingState {
    recorder.state()
}
