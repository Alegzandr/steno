use tauri::{AppHandle, Manager, Runtime, State};

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

/// Whether the formatting model is on disk, and what it would take to get it
/// there.
///
/// Separate from `model_status` because the two models are independent: Steno
/// dictates without the formatter and formats without a new Whisper model, and
/// a single combined status would make the first-launch screen block on both.
#[tauri::command]
pub fn formatter_status<R: Runtime>(app: AppHandle<R>) -> Option<model::ModelStatus> {
    let file = app
        .state::<crate::config::Config>()
        .get()
        .llm
        .model_file;

    model::formatter_spec(&file).map(|spec| model::status_of(&app, spec))
}

/// Downloads the formatting model, resuming an interrupted attempt.
#[tauri::command]
pub async fn download_formatter_model<R: Runtime>(
    app: AppHandle<R>,
    downloads: State<'_, Downloads>,
) -> Result<(), String> {
    let file = app
        .state::<crate::config::Config>()
        .get()
        .llm
        .model_file;

    let Some(spec) = model::formatter_spec(&file) else {
        return Err(format!(
            "settings.json names {file}, which Steno does not know how to download. \
             Put the file in the models directory yourself, or set llm.modelFile back \
             to {}.",
            model::QWEN3_14B_Q4_K_M.id
        ));
    };

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
