use std::sync::Arc;

use tauri::{AppHandle, Manager, Runtime};

use crate::config::Config;
use crate::format::cleanup::{self, Cleanup};
use crate::format::model::{self, Availability};

/// Whether a cleanup could run right now, and the exact command to type if it
/// could not.
///
/// Checked before phase 4's Clean up button does anything, so "Ollama is not
/// running" arrives as an instruction rather than as a timeout.
#[tauri::command]
pub fn ollama_availability<R: Runtime>(app: AppHandle<R>) -> Availability {
    let settings = app.state::<Config>().get().ollama;
    model::availability(&settings.endpoint, &settings.model)
}

/// Restructures the whole buffer, streaming the result back as events.
///
/// Takes the text rather than reading it from anywhere: the editor is the
/// single source of truth for what is in the buffer, and a cleanup must operate
/// on exactly what the user is looking at, including whatever they typed
/// between dictations.
#[tauri::command]
pub fn clean_up<R: Runtime>(app: AppHandle<R>, text: String) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("there is nothing to clean up yet".to_owned());
    }

    cleanup::spawn(app, text)
}

/// Stops a cleanup in flight. Returns whether there was one to stop, so Esc can
/// fall through to cancelling a recording when there was not.
#[tauri::command]
pub fn cancel_cleanup<R: Runtime>(app: AppHandle<R>) -> bool {
    app.state::<Arc<Cleanup>>().cancel()
}

#[tauri::command]
pub fn cleanup_running<R: Runtime>(app: AppHandle<R>) -> bool {
    app.state::<Arc<Cleanup>>().is_running()
}
