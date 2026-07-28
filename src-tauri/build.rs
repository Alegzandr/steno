use std::path::{Path, PathBuf};

fn main() {
    stage_llama_dlls();
    stage_vc_runtime();
    delay_load_cublas();
    tauri_build::build()
}

/// Copies the Visual C++ runtime beside the executable.
///
/// Every DLL Steno ships imports `MSVCP140.dll`, `VCRUNTIME140.dll` and
/// `VCRUNTIME140_1.dll`, and so does `steno.exe`. They are not part of Windows —
/// they arrive with the "Visual C++ 2015-2022 Redistributable", which is
/// present on most machines because something else installed it and on none by
/// guarantee. Missing, they fail the same way the llama DLLs did: 0xC0000135
/// before `main`, no message.
///
/// 708 KB, redistribution permitted, and app-local deployment is the documented
/// alternative to running the redistributable installer. There is nothing to
/// weigh. The dependency closure really is these three: they pull in nothing
/// beyond the Universal CRT, which *is* part of Windows 10 and 11.
fn stage_vc_runtime() {
    const NEEDED: [&str; 3] = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"];

    let Some(redist) = vc_redist_dir() else {
        println!(
            "cargo:warning=could not find the Visual C++ redistributable DLLs; a bundle built \
             now would rely on the target machine already having them"
        );
        return;
    };

    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime");
    if let Err(error) = std::fs::create_dir_all(&destination) {
        println!("cargo:warning=could not create {}: {error}", destination.display());
        return;
    }

    for name in NEEDED {
        let from = redist.join(name);
        let to = destination.join(name);
        if same_size(&from, &to) {
            continue;
        }
        if let Err(error) = std::fs::copy(&from, &to) {
            println!("cargo:warning=could not stage {name} from {}: {error}", redist.display());
        }
    }

    println!("cargo:rerun-if-changed={}", redist.display());
}

/// `VC/Redist/MSVC/<version>/x64/Microsoft.VC<n>.CRT`, newest version.
///
/// `VCToolsRedistDir` is set only inside a developer prompt, and cargo is
/// normally run from an ordinary shell, so `vswhere` is the path that actually
/// gets taken here. It ships with every Visual Studio installer since 2017 at a
/// fixed location, which is the one thing about the VS layout that can be
/// hard-coded honestly.
fn vc_redist_dir() -> Option<PathBuf> {
    let root = match std::env::var("VCToolsRedistDir") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir),
        _ => vswhere_install_path()?.join("VC").join("Redist").join("MSVC"),
    };

    let versioned = if root.join("x64").is_dir() {
        root
    } else {
        // Version directories sort lexically in the right order for 14.4x/14.5x.
        let mut versions: Vec<PathBuf> = std::fs::read_dir(&root)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.join("x64").is_dir())
            .collect();
        versions.sort();
        versions.pop()?
    };

    std::fs::read_dir(versioned.join("x64"))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("Microsoft.VC") && name.ends_with(".CRT"))
        })
}

fn vswhere_install_path() -> Option<PathBuf> {
    let program_files = std::env::var("ProgramFiles(x86)").ok()?;
    let vswhere = PathBuf::from(program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");

    let output = std::process::Command::new(vswhere)
        .args(["-latest", "-property", "installationPath"])
        .output()
        .ok()?;

    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// Makes the cuBLAS import lazy instead of load-time.
///
/// whisper.cpp's ggml is linked statically, so `steno.exe` itself imports
/// `cublas64_<major>.dll`. Left alone that is a load-time import: on a machine
/// without the CUDA toolkit the process dies at 0xC0000135 before a line of our
/// code runs, and no startup check can exist because there is no startup.
/// Measured, with the toolkit removed from `PATH`.
///
/// `/DELAYLOAD` moves the resolution to the first call into cuBLAS, which is
/// after `main`, which is what makes a message possible at all. `cublasLt` is
/// imported by `cublas` rather than by us, so it follows without being named.
/// `delayimp` carries the thunk the flag generates.
///
/// This does not decide where the DLLs come from. It only buys the chance to
/// say something.
fn delay_load_cublas() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }

    let Some(name) = cublas_dll_name() else {
        println!(
            "cargo:warning=no cublas64_*.dll found under CUDA_PATH; the cuBLAS import will stay \
             load-time and a machine without the toolkit will fail to start"
        );
        return;
    };

    println!("cargo:rustc-link-arg=/DELAYLOAD:{name}");
    println!("cargo:rustc-link-lib=delayimp");
    println!("cargo:rustc-env=STENO_CUBLAS_DLL={name}");
}

/// The exact file name the linker will have written into the import table.
///
/// Read from the toolkit rather than hard-coded: the major version is part of
/// the name (`cublas64_13.dll` on CUDA 13, `cublas64_12.dll` on 12), and
/// `/DELAYLOAD` silently does nothing when the name does not match an import.
fn cublas_dll_name() -> Option<String> {
    let cuda = PathBuf::from(std::env::var("CUDA_PATH").ok()?);

    for bin in [cuda.join("bin").join("x64"), cuda.join("bin")] {
        let found = std::fs::read_dir(&bin).ok().and_then(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.file_name().into_string().ok())
                .find(|name| name.starts_with("cublas64_") && name.ends_with(".dll"))
        });
        if found.is_some() {
            return found;
        }
    }

    None
}

/// Copies llama.cpp's DLLs into `src-tauri/` so the bundler can ship them.
///
/// They are built by `llama-cpp-sys-2` into its own `OUT_DIR`, whose path
/// contains a metadata hash that changes with the feature set and the compiler
/// version. Tauri's `resources` list is static text in `tauri.conf.json` and
/// cannot name a path like that, so this stages them somewhere stable first.
///
/// The sys crate keeps them in two directories and they fail differently, but
/// they are staged into one, because both have to end up beside the executable:
///
/// - `out/bin` holds `llama.dll`, `ggml.dll`, `ggml-base.dll` and
///   `llama-common.dll`. The Windows loader resolves them before `main`, so a
///   bundle missing them dies with exit code 0xC0000135 and no message at all —
///   measured, on the first installer built for 5.1.
/// - `out/backends` holds `ggml-cpu-*.dll` and `ggml-cuda.dll`, which ggml
///   opens later, from a search path that is not ours to choose. A bundle that
///   put them in a subdirectory installed cleanly, started, and failed the
///   first cleanup with `no backends are loaded` — also measured. See
///   `format::backends`.
///
/// Deliberately not a hard failure. A CPU-only `cargo check` on a machine that
/// has never built the native side has nothing to copy, and refusing to build
/// over that would be worse than the gap. This only makes the bundle possible;
/// it does not certify it.
fn stage_llama_dlls() {
    let Some(out) = sys_out_dir() else {
        println!(
            "cargo:warning=llama-cpp-sys-2 has not been built here; a bundle built now would ship no DLLs"
        );
        return;
    };

    for source in ["bin", "backends"] {
        stage_dir(&out.join(source), "runtime");
    }
}

fn stage_dir(source: &Path, destination_name: &str) {
    if !source.is_dir() {
        println!("cargo:warning={} does not exist; nothing staged into {destination_name}/", source.display());
        return;
    }

    let destination = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(destination_name);
    if let Err(error) = std::fs::create_dir_all(&destination) {
        println!("cargo:warning=could not create {}: {error}", destination.display());
        return;
    }

    let Ok(entries) = std::fs::read_dir(source) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let from = entry.path();
        if from.extension().and_then(|e| e.to_str()) != Some("dll") {
            continue;
        }

        let to = destination.join(&name);
        // Only when it differs: rewriting a 51 MB CUDA DLL on every build makes
        // `cargo check` visibly slower and touches a file the bundler watches.
        if same_size(&from, &to) {
            continue;
        }
        if let Err(error) = std::fs::copy(&from, &to) {
            println!("cargo:warning=could not stage {}: {error}", name.to_string_lossy());
        }
    }

    println!("cargo:rerun-if-changed={}", source.display());
}

fn same_size(from: &Path, to: &Path) -> bool {
    match (std::fs::metadata(from), std::fs::metadata(to)) {
        (Ok(a), Ok(b)) => a.len() == b.len(),
        _ => false,
    }
}

/// Finds `target/<profile>/build/llama-cpp-sys-2-<hash>/out`.
///
/// `OUT_DIR` for *this* crate is a sibling of the one wanted, which is what
/// makes the search a walk up two levels and back down rather than a guess at
/// the target directory — `CARGO_TARGET_DIR` may point anywhere, and the CUDA
/// build here deliberately uses a second one.
fn sys_out_dir() -> Option<PathBuf> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    let build_root = out_dir.parent()?.parent()?;

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(build_root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("llama-cpp-sys-2-"))
        })
        .map(|path| path.join("out"))
        .filter(|path| path.join("bin").is_dir() || path.join("backends").is_dir())
        .collect();

    // More than one hash can survive a feature switch. Newest wins: it is the
    // one this build just produced.
    candidates.sort_by_key(|path| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
    });
    candidates.pop()
}
