//! Loading the formatting model into this process, and giving the video memory
//! back by dropping it.
//!
//! This replaced an Ollama client, and nearly everything that file did has no
//! counterpart here. There is no server to find or start, no `keep_alive` to
//! set finite in case Steno dies, no polling `/api/ps` to prove an unload
//! happened, and no question of whether a resident model belongs to somebody
//! else. The weights are in this address space; `Drop` frees them; the process
//! exiting frees them too. What used to need three hundred lines and a job
//! object is now the ordinary Rust lifetime of a struct field.
//!
//! Two things about the load are measured rather than assumed, and both are
//! recorded here so they cannot quietly regress:
//!
//! - **`mmap` is off.** It is llama.cpp's default and it is the wrong default
//!   for Steno. Every byte of this file is destined for video memory, so
//!   mapping it buys nothing but a page fault per tensor, taken in tensor order
//!   rather than file order. Measured on the dev machine with the file fully
//!   cached, mmap on cost 2784 ms against 1098 ms for the explicit read; on a
//!   mechanical drive the fault pattern is what turned a 57-second read into a
//!   57-second stall with all the work at the end. Ollama passes `mmap = false`
//!   for the same model, which is what put us onto it. Steno never re-reads the
//!   file after the upload, so the one thing mmap is good for does not apply.
//! - **The backend is initialised exactly once per process.** `LlamaBackend::init`
//!   is a global one-shot — a second call returns `BackendAlreadyInitialized` —
//!   so it lives in a `OnceLock` and outlives every load/unload cycle. That is
//!   not a leak being tolerated: what Steno unloads is the model, and the model
//!   is what holds the nine gigabytes.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use serde::Serialize;

use super::backends;

/// The process-wide llama.cpp backend.
///
/// `Result` inside the `OnceLock` rather than around it: a failed init is
/// permanent for the life of the process, and retrying it would call `init` a
/// second time, which fails for a different reason and would report the wrong
/// cause.
static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

pub fn backend() -> Result<&'static LlamaBackend, String> {
    BACKEND
        .get_or_init(|| {
            // Reported, not loaded. Calling `load_backends_from_path` here
            // registers the modules in whisper.cpp's statically linked ggml,
            // not in the one `llama.dll` consults — see `format::backends` for
            // how that was found and why it looked like it worked. llama.cpp
            // loads them itself, from its own fixed search path, and this line
            // exists so a support question can be answered by reading a log
            // rather than by guessing which copy was used.
            match backends::locate() {
                Some(location) => eprintln!(
                    "llm: ggml backends in {}{}",
                    location.path.display(),
                    if location.from_build_tree {
                        " (build tree)"
                    } else {
                        ""
                    }
                ),
                None => eprintln!("llm: no backend directory found; see the startup check"),
            }

            LlamaBackend::init().map_err(|error| format!("llama.cpp would not start ({error})"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The formatting model, resident in this process.
///
/// Held by `Resident`, which is what guarantees no cleanup is streaming when
/// this is dropped.
pub struct Loaded {
    pub model: LlamaModel,
    pub path: PathBuf,
    pub load_ms: u64,
    /// Bytes read and how long it took, kept so the storage advisory can fall
    /// back to an observation when the drive declines to describe itself. See
    /// `crate::storage`.
    pub read: (u64, Duration),
}

/// Everything the load needs that a user can change.
pub struct Params {
    pub path: PathBuf,
    /// 999 means "all of them"; llama.cpp clamps to what the model has.
    pub n_gpu_layers: u32,
}

impl Loaded {
    /// Brings the weights into video memory. Blocking, and the whole point of
    /// warming on window show.
    pub fn load(params: Params) -> Result<Self, String> {
        let backend = backend()?;

        if !params.path.exists() {
            return Err(format!(
                "the formatting model is not downloaded yet ({})",
                params.path.display()
            ));
        }

        let bytes = std::fs::metadata(&params.path)
            .map(|meta| meta.len())
            .unwrap_or(0);

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(params.n_gpu_layers)
            // See the module comment. Do not "fix" this back to the default.
            .with_use_mmap(false);

        let started = Instant::now();
        let model = LlamaModel::load_from_file(backend, &params.path, &model_params)
            .map_err(|error| load_failure(&params.path, &error.to_string()))?;
        let elapsed = started.elapsed();
        let load_ms = elapsed.as_millis() as u64;

        eprintln!(
            "llm: {} resident after {load_ms} ms ({:.0} MB/s)",
            file_label(&params.path),
            (bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64().max(f64::EPSILON)
        );

        Ok(Self {
            model,
            path: params.path,
            load_ms,
            read: (bytes, elapsed),
        })
    }

    /// A context sized for one cleanup.
    ///
    /// Made per request rather than held alongside the model: the KV cache is
    /// the part of the cost paid per *use*, it is cheap to build — 12 to 20 ms
    /// measured — and a context that outlived a request would hold video memory
    /// through exactly the idle stretch Steno promises to leave the card alone.
    pub fn context(&self, n_ctx: u32, n_batch: u32) -> Result<llama_cpp_2::context::LlamaContext<'_>, String> {
        let params = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch);

        self.model
            .new_context(backend()?, params)
            .map_err(|error| format!("could not create a context for the model ({error})"))
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        // The field drops immediately after this, which is where the video
        // memory actually goes back. Logged because the VRAM promise is a hard
        // requirement and a silent unload is one that cannot be audited from a
        // log file after the fact.
        eprintln!("llm: releasing {}", file_label(&self.path));
    }
}

/// Whether a cleanup could run right now, and what to do if it could not.
///
/// Kept from the Ollama version because the frontend contract did not change,
/// but the shape of the answer did: the failure is now a missing file, not a
/// missing server, so there is no command to type. `remedy` stays in the type
/// for the one case that still has one — a build with no compute backend, where
/// the fix is a reinstall rather than a download.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Availability {
    /// The backend DLLs are present and llama.cpp can start.
    pub reachable: bool,
    pub model_installed: bool,
    /// Absolute path of the model file Steno is looking for, so a user who has
    /// put it somewhere else can see where it expected to find it.
    pub model_path: String,
    pub remedy: Option<String>,
}

pub fn availability(path: &Path) -> Availability {
    // The GPU runtime first, because it outranks the backend modules: a missing
    // cuBLAS takes dictation down as well, so reporting a ggml module problem
    // over it would name the smaller of two faults. Same answer the rest of the
    // process gets — `crate::gpu` computes it once.
    let backend_problem = crate::gpu::blocker()
        .map(|blocker| blocker.one_line())
        .or_else(|| backends::diagnose(backends::locate().as_ref()));

    Availability {
        reachable: backend_problem.is_none(),
        model_installed: path.exists(),
        model_path: path.display().to_string(),
        remedy: backend_problem,
    }
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// llama.cpp reports a load failure as one string for several quite different
/// situations, and the two that matter to a user are worth separating.
fn load_failure(path: &Path, detail: &str) -> String {
    let lower = detail.to_ascii_lowercase();

    if lower.contains("out of memory") || lower.contains("cudamalloc") {
        return format!(
            "not enough video memory to load {}. Close whatever else is using \
             the GPU, or set llm.nGpuLayers in settings.json to keep some layers \
             on the CPU.",
            file_label(path)
        );
    }

    // A truncated download is a plausible and confusing failure: the file is
    // there, so "not downloaded" would be wrong and misleading.
    if lower.contains("magic") || lower.contains("invalid") || lower.contains("gguf") {
        return format!(
            "{} is not a valid GGUF file. It is most likely an incomplete \
             download — delete it and download it again.",
            file_label(path)
        );
    }

    format!("could not load {} ({detail})", file_label(path))
}
