//! The local formatting model: loading it into this process, streaming a
//! cleanup through it, and keeping it off the GPU whenever Steno is not using
//! it.

pub mod backends;
pub mod cleanup;
pub mod model;

use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Manager, Runtime};

use crate::config::Config;
use crate::resident::Resident;

/// The GGUF Steno formats with by default.
///
/// Qwen3-14B at Q4_K_M, the file the phase 4 measurements were taken against —
/// 8.63 GiB, 41 layers, all of them offloaded on a 16 GB card. Named as a bare
/// file so it resolves inside `model::directory`; see `LlmSettings::model_file`.
pub const DEFAULT_MODEL_FILE: &str = crate::model::QWEN3_14B_Q4_K_M.id;

/// The managed handle on the resident formatting model.
///
/// Before 5.1 the value here was a marker and the weights lived in the Ollama
/// process; `Drop` sent an HTTP request and then polled to find out whether it
/// had worked. Now the value *is* the model, and `Drop` is what frees the video
/// memory. `Resident` is unchanged: it already guaranteed that an eviction
/// waits for an in-flight lease, which is exactly what stops a cleanup having
/// its model pulled out from under it mid-stream.
pub type Formatter = Arc<Resident<model::Loaded>>;

/// Where the formatting model file is expected to be.
///
/// The same directory as the Whisper models, deliberately: it is one place for
/// a user to find, one directory to check the drive of, and one thing to move
/// when it is on the wrong disk.
pub fn model_path<R: Runtime>(app: &AppHandle<R>) -> PathBuf {
    let file = app.state::<Config>().get().llm.model_file;
    crate::model::directory(app).join(file)
}
