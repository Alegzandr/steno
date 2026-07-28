//! Checking that the ggml backend modules are where llama.cpp will look.
//!
//! Building llama.cpp with `dynamic-link` + `dynamic-backends` splits it into
//! two kinds of DLL, and the difference decides what this module is for:
//!
//! - `llama.dll`, `ggml.dll`, `ggml-base.dll` and `llama-common.dll` are
//!   resolved by the Windows loader before `main` runs. If one is missing the
//!   process never starts, Windows says so itself, and no Rust code of ours
//!   gets the chance to have an opinion.
//! - `ggml-cpu-*.dll` and `ggml-cuda.dll` are opened later, by ggml itself, and
//!   their absence is not an error at all. It is a successful load of nothing:
//!   the app starts, the window appears, dictation works, and the first Clean
//!   up fails with `no backends are loaded`.
//!
//! That second failure is the one worth code. It is one missing file away from
//! a working build and it surfaces minutes later, in the one operation the user
//! waited for. So the directory is checked at startup, while there is still a
//! plausible moment to say something, and the message names the files rather
//! than describing them.
//!
//! ## Why they sit beside the executable and nothing here loads them
//!
//! The first attempt shipped them in a `backends/` subdirectory and called
//! `load_backends_from_path` on it at startup. Installed, that build reported
//! loading the CUDA backend from the right place and then failed the first
//! cleanup with `llama_model_load_from_file_impl: no backends are loaded`.
//!
//! Steno links two ggmls: whisper.cpp's, statically, and llama.cpp's, through
//! `ggml-base.dll`. They keep separate registries, and a
//! `ggml_backend_load_all_from_path` called from Rust fills the static one —
//! which nothing then asks. `llama.dll` populates its own registry lazily, from
//! `ggml_backend_load_best`, whose search path is fixed in ggml and is not ours
//! to redirect: the `GGML_BACKEND_DIR` baked in at compile time, then the
//! executable's own directory, then the working directory.
//!
//! On the dev machine `GGML_BACKEND_DIR` points into `target/` and exists, so
//! the broken layout worked — right up until the build tree was renamed, which
//! is what the test that found this did. The fix is to put the modules in the
//! one directory ggml searches on a machine that has no build tree, and to let
//! ggml load them. This module now only checks; it does not load.
//!
//! The other missing-DLL failure is not here. cuBLAS is imported by the
//! executable rather than opened by ggml, it takes dictation down as well as
//! cleanup, and it is answered once for the whole process — see `crate::gpu`.

use std::path::{Path, PathBuf};

/// The backend directory, and how it was found.
pub struct Location {
    pub path: PathBuf,
    /// True when this is the build tree rather than an installed layout. Worth
    /// logging: a dev build that silently used the build tree would hide a
    /// bundling mistake until someone installed it — and did.
    pub from_build_tree: bool,
}

/// Resolves the directory ggml will actually load from.
///
/// The executable's own directory first, because that is where a bundle puts
/// them and where ggml looks second. `BACKENDS_DIR` is
/// `GGML_BACKEND_DIR` — baked in at compile time, pointing into `target/`,
/// right for `cargo run` and meaningless on a user's machine — so it is the
/// fallback. ggml itself searches them in the opposite order; the order here is
/// about which one to *report*, and reporting the installed layout is what
/// catches a bundling mistake on a machine that has both.
pub fn locate() -> Option<Location> {
    if let Some(shipped) = executable_dir() {
        if has_cpu_backend(&shipped) {
            return Some(Location {
                path: shipped,
                from_build_tree: false,
            });
        }
    }

    llama_cpp_2::llama_backend::BACKENDS_DIR
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .map(|path| Location {
            path,
            from_build_tree: true,
        })
}

/// Presence of a CPU variant, not of the directory: the executable's directory
/// always exists, so `is_dir` would make the shipped layout win every time and
/// the build-tree fallback unreachable.
fn has_cpu_backend(directory: &Path) -> bool {
    list_dlls(directory)
        .iter()
        .any(|name| name.starts_with("ggml-cpu-"))
}

fn executable_dir() -> Option<PathBuf> {
    Some(std::env::current_exe().ok()?.parent()?.to_path_buf())
}

/// What is wrong with a backend directory, in the user's terms.
///
/// `None` means it is usable. The strings name files because that is what the
/// person fixing it has to go and find; "the CUDA backend could not be loaded"
/// sends them to their driver.
pub fn diagnose(location: Option<&Location>) -> Option<String> {
    let Some(location) = location else {
        return Some(
            "Steno could not find its ggml-cpu-*.dll. They belong next to the Steno \
             executable. Without them there is no compute backend and cleanup fails \
             with `no backends are loaded`."
                .to_owned(),
        );
    };

    let present = list_dlls(&location.path);

    // At least one CPU variant. Which one is chosen at run time by feature
    // detection, so naming a specific file here would be wrong on every machine
    // but this one.
    if !present.iter().any(|name| name.starts_with("ggml-cpu-")) {
        return Some(format!(
            "No ggml-cpu-*.dll in {}. Steno has no compute backend and cleanup \
             cannot run.",
            location.path.display()
        ));
    }

    // The GPU backend is checked only in a build that was compiled to use one.
    // A CPU build missing ggml-cuda.dll is correct, not broken.
    if cfg!(feature = "cuda") && !present.iter().any(|name| name == "ggml-cuda.dll") {
        return Some(format!(
            "ggml-cuda.dll is missing from {}. This build of Steno was compiled \
             for CUDA, so formatting would fall back to the CPU and take minutes \
             instead of seconds.",
            location.path.display()
        ));
    }

    if cfg!(feature = "vulkan") && !present.iter().any(|name| name == "ggml-vulkan.dll") {
        return Some(format!(
            "ggml-vulkan.dll is missing from {}. This build of Steno was compiled \
             for Vulkan, so formatting would fall back to the CPU and take minutes \
             instead of seconds.",
            location.path.display()
        ));
    }

    None
}

fn list_dlls(directory: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };

    entries
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.to_ascii_lowercase().ends_with(".dll"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location_of(path: &Path) -> Location {
        Location {
            path: path.to_path_buf(),
            from_build_tree: false,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("steno-backends-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        path
    }

    fn touch(directory: &Path, name: &str) {
        std::fs::write(directory.join(name), b"").expect("stub dll");
    }

    #[test]
    fn a_missing_directory_names_what_to_look_for() {
        let message = diagnose(None).expect("no directory is a problem");
        assert!(message.contains("ggml-cpu-"), "{message}");
        assert!(message.contains("executable"), "{message}");
    }

    #[test]
    fn an_empty_directory_is_reported() {
        let directory = scratch("empty");
        let message =
            diagnose(Some(&location_of(&directory))).expect("no CPU backend is a problem");
        assert!(message.contains("ggml-cpu-"), "{message}");
    }

    /// The trap this whole module exists for: the directory is there, the CPU
    /// variants are there, and the one file that makes cleanup fast is not.
    #[test]
    fn a_cpu_only_directory_is_flagged_in_a_gpu_build() {
        let directory = scratch("cpu-only");
        touch(&directory, "ggml-cpu-haswell.dll");

        let message = diagnose(Some(&location_of(&directory)));

        if cfg!(feature = "cuda") {
            let message = message.expect("a CUDA build without ggml-cuda.dll is broken");
            assert!(message.contains("ggml-cuda.dll"), "{message}");
        } else {
            assert!(
                message.is_none(),
                "a CPU build is complete without a GPU backend: {message:?}"
            );
        }
    }

    #[test]
    fn a_complete_directory_says_nothing() {
        let directory = scratch("complete");
        touch(&directory, "ggml-cpu-haswell.dll");
        touch(&directory, "ggml-cpu-x64.dll");
        touch(&directory, "ggml-cuda.dll");
        touch(&directory, "ggml-vulkan.dll");

        assert_eq!(diagnose(Some(&location_of(&directory))), None);
    }

    /// Case comes from the filesystem, not from us.
    #[test]
    fn extensions_are_matched_case_insensitively() {
        let directory = scratch("case");
        touch(&directory, "ggml-cpu-haswell.DLL");
        assert!(list_dlls(&directory).len() == 1);
    }
}
