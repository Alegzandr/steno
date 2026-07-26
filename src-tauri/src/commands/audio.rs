use tauri::{AppHandle, Runtime, State};

use crate::audio::capture::{self, InputDevice};
use crate::audio::{Recorder, RecordingState};
use crate::config::Config;

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

/// Backs the microphone dropdown: the input devices present right now, with the
/// system default flagged.
#[tauri::command]
pub fn enumerate_input_devices() -> Result<Vec<InputDevice>, String> {
    capture::enumerate().map_err(|error| error.to_string())
}

/// The saved device override, or `None` when Steno follows the system default.
#[tauri::command]
pub fn input_device(config: State<'_, Config>) -> Option<String> {
    config.input_device()
}

/// Persists the chosen device. `None` clears the override back to the system
/// default. Takes effect on the next recording, not the current one.
#[tauri::command]
pub fn set_input_device(config: State<'_, Config>, name: Option<String>) -> Result<(), String> {
    config.set_input_device(name)
}
