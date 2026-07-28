use tauri::{AppHandle, Runtime, State};

use crate::gpu::{self, runtime, Blocker};
use crate::model::download::Downloads;

/// What stops this build from working at all, if anything does.
///
/// `None` is the normal case and the one the UI is built around. A `Some` is
/// not a warning to put in a corner: nothing Steno does — dictating, cleaning
/// up — can happen, so the window shows this instead of the editor.
///
/// Stable for as long as the frontend can observe it, so it is asked once on
/// mount rather than subscribed to. The one thing that can change the answer is
/// the completion of the cuBLAS download, and `install_cublas` returns the new
/// state to whoever asked for it.
#[tauri::command]
pub fn gpu_blocker() -> Option<Blocker> {
    gpu::blocker()
}

/// Everything the panel needs to explain the download before starting it: what
/// is missing, how big it is, where it goes, what the driver supports, and
/// whether there is room.
#[tauri::command]
pub fn cublas_status<R: Runtime>(app: AppHandle<R>) -> runtime::Status {
    runtime::status(&app)
}

/// Downloads and installs cuBLAS, and answers with what Steno can do afterwards.
///
/// Awaited by the caller so a failure surfaces as a rejected promise. Progress
/// arrives as `model-download-*` events, the same ones the model screen draws,
/// keyed on the archive's id; the inflation that follows the download has no
/// byte count to report and emits `cublas-install-stage` instead.
#[tauri::command]
pub async fn install_cublas<R: Runtime>(
    app: AppHandle<R>,
    downloads: State<'_, Downloads>,
) -> Result<runtime::Status, String> {
    runtime::install(app.clone(), downloads.inner()).await
}
