use tauri::{AppHandle, Runtime};

use crate::window;

/// Hides the mini editor.
///
/// The frontend must go through this rather than calling `window.hide()` on
/// the webview: hiding is also the trigger that releases the Whisper context
/// and the formatting model, and a hide that bypasses it leaves twelve
/// gigabytes of video memory held by a window nobody can see.
#[tauri::command]
pub fn hide_window<R: Runtime>(app: AppHandle<R>) {
    window::hide(&app);
}

/// Ends Steno.
///
/// Steno needs an explicit one. The close button hides rather than quits, and
/// nothing else closes the last window, so without this the process can only be
/// ended from the Task Manager — which skips `RunEvent::Exit`, and with it the
/// unload of the formatting model and the shutdown of a server Steno started.
/// The job object still catches that case, but relying on the backstop as the
/// normal path is not a design.
///
/// `exit` runs the Tauri exit sequence, so `lifecycle::on_exit` gets to release
/// both models before the process goes away.
#[tauri::command]
pub fn quit_app<R: Runtime>(app: AppHandle<R>) {
    app.exit(0);
}
