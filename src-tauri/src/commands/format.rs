use std::sync::Arc;

use tauri::{AppHandle, Manager, Runtime};

use crate::format::cleanup::{self, Cleanup};
use crate::format::model::{self, Availability};

/// Whether a cleanup could run right now, and what to do if it could not.
///
/// Checked before the Clean up button does anything, so a missing model file
/// arrives as a sentence rather than as a failure thirty seconds in.
#[tauri::command]
pub fn llm_availability<R: Runtime>(app: AppHandle<R>) -> Availability {
    model::availability(&crate::format::model_path(&app))
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
