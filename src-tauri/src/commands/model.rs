use tauri::{AppHandle, Runtime, State};

use crate::model::{self, download::Downloads, ModelStatus};

/// Everything the first-launch screen needs: which model was chosen, whether it
/// is already on disk, and how much of it a previous attempt got through.
#[tauri::command]
pub fn model_status<R: Runtime>(app: AppHandle<R>) -> ModelStatus {
    model::status(&app)
}

/// Downloads the chosen model, resuming an interrupted attempt.
///
/// Awaited by the caller so a failure surfaces as a rejected promise, but the
/// progress the UI actually draws from arrives as events: the command does not
/// return until gigabytes later.
#[tauri::command]
pub async fn download_model<R: Runtime>(
    app: AppHandle<R>,
    downloads: State<'_, Downloads>,
) -> Result<(), String> {
    let spec = model::resolve(&app);
    let destination = model::path(&app, spec);

    model::download::run(app.clone(), downloads.inner(), spec, destination).await
}

/// Stops a running download. The partial file is kept, so pressing the button
/// again continues from where it stopped.
#[tauri::command]
pub fn cancel_model_download(downloads: State<'_, Downloads>) {
    downloads.cancel();
}

/// Lets the UI resync after a webview reload that landed mid-download.
#[tauri::command]
pub fn model_download_running(downloads: State<'_, Downloads>) -> bool {
    downloads.is_running()
}
