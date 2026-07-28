//! The Whisper model file: which one, where it lives, and getting it there.
//!
//! The model is not bundled. It is 0.5 to 3 GB depending on the variant, it
//! never changes, and shipping it inside the installer would quadruple the
//! download for every update. It lives in the app data directory instead and is
//! fetched once, on first launch.

pub mod detect;
pub mod download;

use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Manager, Runtime};

use crate::config::Config;

/// One downloadable model. Sizes and digests are the real values published by
/// the `ggerganov/whisper.cpp` repository; they are checked after every
/// download, so a wrong one here shows up immediately rather than as a corrupt
/// model months later.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    /// Filename on disk, and the identifier stored in `settings.json`.
    pub id: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// Shown in the UI while downloading.
    pub label: &'static str,
}

/// Full-precision large-v3. Best quality, and the default when there is a GPU
/// backend compiled in with enough memory to hold it.
pub const LARGE_V3: ModelSpec = ModelSpec {
    id: "ggml-large-v3.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    bytes: 3_095_033_483,
    sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    label: "Whisper large-v3",
};

/// Quantised turbo. A fifth of the size, several times faster, and what a CPU
/// build gets: large-v3 on CPU is slower than real time and unusable for
/// push-to-talk.
pub const LARGE_V3_TURBO_Q5_0: ModelSpec = ModelSpec {
    id: "ggml-large-v3-turbo-q5_0.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
    bytes: 574_041_195,
    sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    label: "Whisper large-v3 turbo (q5_0)",
};

/// Unquantised turbo, for anyone who wants the middle ground. Never chosen
/// automatically; reachable by editing `model.id` in `settings.json`.
pub const LARGE_V3_TURBO: ModelSpec = ModelSpec {
    id: "ggml-large-v3-turbo.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    bytes: 1_624_555_275,
    sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    label: "Whisper large-v3 turbo",
};

pub const CATALOGUE: &[ModelSpec] = &[LARGE_V3, LARGE_V3_TURBO_Q5_0, LARGE_V3_TURBO];

/// The formatting model, from Qwen's own GGUF repository rather than a
/// third-party requantisation.
///
/// Deliberately not in `CATALOGUE`: that list is what `default_spec` chooses
/// between and what `settings.json`'s `model.id` names, and both are about
/// transcription. This is a separate axis with its own setting
/// (`llm.modelFile`) and its own download.
///
/// Size and digest are the published values for `Qwen/Qwen3-14B-GGUF`. Note
/// that Ollama's `qwen3:14b` blob is a *different* Q4_K_M of the same weights —
/// 9,276,184,896 bytes against 9,001,752,960 here — so a file copied out of an
/// Ollama store will not match this digest. It is close enough in size that the
/// load-time measurements carry over, and not close enough to pretend they are
/// the same file.
pub const QWEN3_14B_Q4_K_M: ModelSpec = ModelSpec {
    id: "Qwen3-14B-Q4_K_M.gguf",
    url: "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
    bytes: 9_001_752_960,
    sha256: "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
    label: "Qwen3 14B (Q4_K_M)",
};

pub const FORMATTER_CATALOGUE: &[ModelSpec] = &[QWEN3_14B_Q4_K_M];

/// The formatting model named in `settings.json`, when Steno knows how to fetch
/// it.
///
/// `None` is not an error: `llm.modelFile` may name any GGUF the user has put
/// in the models directory themselves, and Steno will load it happily. It only
/// means there is nothing to offer to download.
pub fn formatter_spec(file: &str) -> Option<&'static ModelSpec> {
    FORMATTER_CATALOGUE.iter().find(|spec| spec.id == file)
}

/// Video memory below which large-v3 is not worth attempting.
///
/// The nominal figure is 12 GB, but adapters report a little under their
/// marketing capacity — this machine's 16 GB card reports 16376 MiB — so the
/// bar is set slightly low rather than excluding a genuine 12 GB card by a few
/// megabytes.
const LARGE_MODEL_VRAM_FLOOR: u64 = 11_500 * 1024 * 1024;

/// Whether this binary has a GPU backend compiled in at all.
///
/// This gates the model choice as much as the amount of memory does. A CPU-only
/// build that downloads 2.9 GB to then transcribe at several times slower than
/// real time has made the user wait twice for nothing.
pub const fn gpu_backend_compiled() -> bool {
    cfg!(any(
        feature = "cuda",
        feature = "vulkan",
        all(target_os = "macos", target_arch = "aarch64"),
    ))
}

/// The model to use when the user has not chosen one.
pub fn default_spec() -> &'static ModelSpec {
    if !gpu_backend_compiled() {
        return &LARGE_V3_TURBO_Q5_0;
    }

    match detect::dedicated_video_memory() {
        Some(bytes) if bytes >= LARGE_MODEL_VRAM_FLOOR => &LARGE_V3,
        _ => &LARGE_V3_TURBO_Q5_0,
    }
}

/// Looks a model up by filename.
pub fn find(id: &str) -> Option<&'static ModelSpec> {
    CATALOGUE.iter().find(|spec| spec.id == id)
}

/// The model named in `settings.json`, probing the hardware and recording the
/// answer the first time.
///
/// An unrecognised `model.id` is not an error the user can be left guessing
/// about: it is reported and the automatic choice is used, rather than silently
/// substituting something they did not ask for.
pub fn resolve<R: Runtime>(app: &AppHandle<R>) -> &'static ModelSpec {
    let config = app.state::<Config>();

    if let Some(id) = config.get().model.id {
        if let Some(spec) = find(&id) {
            return spec;
        }
        eprintln!("model: settings.json names an unknown model {id:?}, falling back");
    }

    let spec = default_spec();
    if let Err(error) = config.set_model_id(spec.id) {
        eprintln!("model: could not record the chosen model ({error})");
    }
    spec
}

/// Where model files live. Created on demand by the downloader.
pub fn directory<R: Runtime>(app: &AppHandle<R>) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("models")
}

pub fn path<R: Runtime>(app: &AppHandle<R>, spec: &ModelSpec) -> PathBuf {
    directory(app).join(spec.id)
}

/// What the frontend needs to decide between "let me dictate" and "download
/// this first".
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub spec: ModelSpec,
    pub path: String,
    /// Present, and the right size. The digest is only checked at download
    /// time; re-hashing 3 GB on every launch would cost seconds for nothing.
    pub installed: bool,
    /// A resumable partial download is sitting on disk.
    pub partial_bytes: u64,
    /// Video memory found on the best adapter, if it could be read at all.
    pub video_memory_bytes: Option<u64>,
    pub gpu_backend: &'static str,
}

pub fn status<R: Runtime>(app: &AppHandle<R>) -> ModelStatus {
    status_of(app, resolve(app))
}

/// The same answer for a named model rather than the chosen transcription one.
///
/// `installed` is a size check, not a digest check, for both: re-hashing nine
/// gigabytes on every launch would cost seconds to re-prove something the
/// download already proved once.
pub fn status_of<R: Runtime>(app: &AppHandle<R>, spec: &ModelSpec) -> ModelStatus {
    let spec = *spec;
    let path = path(app, &spec);

    let installed = std::fs::metadata(&path)
        .map(|meta| meta.len() == spec.bytes)
        .unwrap_or(false);

    let partial_bytes = std::fs::metadata(download::partial_path(&path))
        .map(|meta| meta.len())
        .unwrap_or(0);

    ModelStatus {
        spec,
        path: path.to_string_lossy().into_owned(),
        installed,
        partial_bytes,
        video_memory_bytes: detect::dedicated_video_memory(),
        gpu_backend: backend_name(),
    }
}

/// Which whisper.cpp backend this binary was compiled with. Reported rather
/// than inferred, so a build that silently fell back to CPU is visible.
pub const fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "metal"
    } else {
        "cpu"
    }
}
