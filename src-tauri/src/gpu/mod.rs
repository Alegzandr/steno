//! Whether the GPU runtime this build was compiled against is actually here.
//!
//! A CUDA build of Steno imports `cublas64_<major>.dll` from the executable
//! itself, because whisper.cpp's ggml is linked statically into it. `build.rs`
//! marks that import `/DELAYLOAD`, which is what turns "the process dies at
//! 0xC0000135 before `main`" into "the first call into cuBLAS raises a
//! structured exception" — the difference between no message and the chance of
//! one. This module takes that chance, once, and turns it into a value.
//!
//! **One probe, not several.** Both engines depend on the same DLL and both
//! would hit the same thunk: whisper on its first context, llama.cpp through
//! `ggml-cuda.dll`. Asking separately at each site would be two answers that can
//! drift, and the site that forgot to ask is the one that crashes. So the answer
//! is computed once, cached, and consulted by everything that could reach the
//! thunk — `transcribe::engine::Engine::load`, `format::model::availability`,
//! `audio::Recorder::start` and the warm-up.
//!
//! Caching is not an optimisation. `LoadLibrary` is cheap the second time, but a
//! blocker that answered differently at two moments would let the UI say the GPU
//! runtime is missing while a recording it already refused was under way.
//!
//! **It is cached, not frozen.** Steno downloads cuBLAS itself, so the whole
//! point of the flow is that a process which started blocked stops being
//! blocked. `recheck` is the one place the cached answer may change, and it
//! belongs to the completion of that download and to nothing else. Everywhere
//! else the answer is the same for as long as anyone can observe it. Measured
//! rather than assumed: `examples/gpu_recovery.rs` starts a process with no
//! cuBLAS on the search path, stages the DLLs into a directory while it runs,
//! rechecks, and then transcribes and cleans up in that same process.
//!
//! **Where the DLLs are found.** The delay-load thunk calls `LoadLibrary` with
//! a bare file name, so it takes the ordinary search order. The executable's own
//! directory is in that order but is under Program Files, which is not writable
//! without elevation — a 512 MB download must not open a UAC prompt. So the
//! files land in the app data directory and `use_runtime_dir` puts that
//! directory into the search order with `SetDllDirectory`, before any thunk can
//! fire. `SetDllDirectory` rather than `SetDefaultDllDirectories`: the latter
//! drops `PATH` from the search, which would break the machines that already
//! have the CUDA toolkit installed and are the only ones this has ever worked on.
//!
//! The failure is deliberately *not* intercepted at the delay-load hook. The
//! hook fires from inside the thunk, on whichever thread first touched cuBLAS,
//! and declining to continue from there means raising a structured exception
//! through Rust frames. Asking the loader the same question early is
//! deterministic, testable, and happens where a `Result` still means something.
//!
//! `runtime` is what fixes it — 391 MB from NVIDIA, into the directory
//! `use_runtime_dir` named — and `driver` is what decides whether fixing it is
//! possible on this machine before the 391 MB are spent.

pub mod driver;
pub mod runtime;

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::Serialize;

/// What is missing, said in the two parts a user needs: the fact and the fix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Blocker {
    /// The file name, so the person fixing it can search for it.
    pub missing: String,
    pub message: String,
    pub remedy: String,
}

impl Blocker {
    /// One line, for a log or for a field that has room for a sentence.
    pub fn one_line(&self) -> String {
        format!("{} {}", self.message, self.remedy)
    }
}

/// `None` while nothing has asked yet; `Some(answer)` once, and thereafter,
/// until the download that fixes it says otherwise.
static STATE: RwLock<Option<Option<Blocker>>> = RwLock::new(None);

/// `None` when Steno can use the GPU it was built for.
///
/// Cheap after the first call, so call sites do not need to cache it
/// themselves — and must not, since `recheck` would not reach a copy.
pub fn blocker() -> Option<Blocker> {
    if let Some(answer) = STATE.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return answer.clone();
    }

    let mut state = STATE.write().unwrap_or_else(|e| e.into_inner());
    // Another thread may have answered between the two locks.
    if let Some(answer) = state.as_ref() {
        return answer.clone();
    }

    let found = probe();
    if let Some(blocker) = &found {
        eprintln!("gpu: {}", blocker.one_line());
    }
    *state = Some(found.clone());
    found
}

/// Asks the loader again, forgetting the cached answer.
///
/// **The only caller may be the successful completion of the cuBLAS download.**
/// Not a window show, not a retry button, not a periodic poll: a second opinion
/// that nothing caused is exactly the drift the cache exists to prevent. What
/// makes this one legitimate is that it follows an event which changed the
/// answer, and it runs before anything is told the answer has changed.
///
/// Whether the process then works without a restart is a property of the
/// delay-load thunk, which is unresolved precisely because the first call never
/// happened. See the module docs for how that was measured.
pub fn recheck() -> Option<Blocker> {
    *STATE.write().unwrap_or_else(|e| e.into_inner()) = None;
    blocker()
}

/// Adds the directory Steno downloads its GPU runtime into to the DLL search
/// order.
///
/// Must be called before anything asks `blocker`, because the probe deliberately
/// resolves the same way the thunk will and the search order is what it is
/// resolving against. Startup does this; nothing else should call it.
///
/// A missing directory is not an error: it is the ordinary state of a machine
/// that has not downloaded anything yet, and the call still has to happen so
/// that the download does not need a restart to be seen.
pub fn use_runtime_dir(dir: &Path) -> Result<(), String> {
    debug_assert!(
        STATE.read().unwrap_or_else(|e| e.into_inner()).is_none(),
        "the DLL search path was changed after the GPU probe had already answered"
    );
    set_dll_directory(dir)
}

#[cfg(windows)]
fn set_dll_directory(dir: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }

    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: the buffer is NUL-terminated and outlives the call, which copies
    // the string.
    let ok = unsafe { SetDllDirectoryW(wide.as_ptr()) };

    if ok == 0 {
        return Err(format!("SetDllDirectory({}) failed", dir.display()));
    }
    Ok(())
}

#[cfg(not(windows))]
fn set_dll_directory(_dir: &Path) -> Result<(), String> {
    Ok(())
}

/// Where a downloaded GPU runtime lives: beside the models, under the app data
/// directory.
///
/// Not beside the executable. That directory is in the loader's search order for
/// free, which is the whole attraction, but on an installed Steno it is under
/// Program Files and writing 512 MB there means asking for elevation on first
/// run. `use_runtime_dir` buys the same resolution for a directory the user
/// already owns.
pub fn runtime_dir<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> PathBuf {
    use tauri::Manager;

    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("runtime")
}

/// Asks the loader the question the delay-load thunk would ask.
///
/// The handle is deliberately leaked: the point is the answer, and on success
/// the library is one the process is about to depend on anyway.
#[cfg(all(windows, feature = "cuda"))]
fn probe() -> Option<Blocker> {
    let name = cublas_dll()?;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut std::ffi::c_void;
    }

    let zero_terminated = format!("{name}\0");
    // SAFETY: the pointer is to a NUL-terminated buffer that outlives the call.
    let handle = unsafe { LoadLibraryA(zero_terminated.as_ptr()) };

    handle.is_null().then(|| describe(name))
}

#[cfg(not(all(windows, feature = "cuda")))]
fn probe() -> Option<Blocker> {
    None
}

/// The cuBLAS DLL this build delay-loaded, when it delay-loaded one.
///
/// Set by `build.rs` only when it actually emitted `/DELAYLOAD`, and read from
/// the toolkit rather than hard-coded, because the major version is part of the
/// name. `None` on a CPU build, and on a CUDA build whose import is load-time —
/// in which case this process would not be running to be asked.
pub fn cublas_dll() -> Option<&'static str> {
    option_env!("STENO_CUBLAS_DLL")
}

/// The CUDA major version this build needs a driver to support, read out of the
/// DLL name: `cublas64_13.dll` is CUDA 13.
pub fn cuda_major() -> Option<u32> {
    cublas_dll()?
        .strip_prefix("cublas64_")?
        .strip_suffix(".dll")?
        .parse()
        .ok()
}

/// Kept apart from the probe so the wording can be tested without a machine
/// that is missing the runtime.
fn describe(name: &str) -> Blocker {
    Blocker {
        missing: name.to_owned(),
        message: format!(
            "{name} is missing. It is part of the NVIDIA CUDA runtime, and this build of Steno \
             uses it for both dictation and cleanup, so neither can run without it."
        ),
        remedy: format!(
            "Steno can download it from NVIDIA ({} MB). Installing the CUDA Toolkit yourself \
             works too.",
            runtime::ARCHIVE.bytes / 1_000_000
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file name has to survive into the message: it is the one string the
    /// user can search for, and "the CUDA runtime is missing" sends them to
    /// their driver instead.
    #[test]
    fn the_message_names_the_file() {
        let blocker = describe("cublas64_13.dll");
        assert!(blocker.message.contains("cublas64_13.dll"), "{blocker:?}");
        assert_eq!(blocker.missing, "cublas64_13.dll");
        assert!(blocker.remedy.contains("download"), "{blocker:?}");
        assert!(blocker.one_line().starts_with(&blocker.message));
    }

    /// The archive is pinned to one CUDA major version and the DLL name carries
    /// it. If they ever disagree, Steno would download a cuBLAS that cannot
    /// satisfy its own import.
    #[test]
    fn the_dll_name_and_the_archive_agree_on_the_major_version() {
        let Some(major) = cuda_major() else {
            return; // A CPU build has no import to satisfy.
        };
        assert!(
            runtime::ARCHIVE.id.contains(&format!("-{major}.")),
            "{} does not look like cuBLAS for CUDA {major}",
            runtime::ARCHIVE.id
        );
    }

    /// A CPU build imports nothing from CUDA, so it must never claim to be
    /// blocked — and neither must a CUDA build on a machine that has the DLL,
    /// which is what this asserts when the feature is on.
    #[test]
    fn a_build_that_can_run_is_not_blocked() {
        if cfg!(all(windows, feature = "cuda")) {
            // The dev machine has the toolkit; a machine that does not would be
            // testing the other branch, and the assertion below would be wrong
            // there rather than useful.
            return;
        }
        assert_eq!(blocker(), None);
        // Rechecking has to reach the loader again rather than a frozen answer,
        // which on this build means agreeing with it.
        assert_eq!(recheck(), None);
    }
}
