//! Does a process that started without cuBLAS work once cuBLAS arrives?
//!
//! The download branch depends on the answer being yes: Steno is meant to
//! launch on a machine with no CUDA runtime, fetch `cublas64_*.dll` itself, and
//! then dictate — without asking for a restart it has no way to make appealing.
//! The delay-load thunk says this should work, since nothing has resolved it
//! yet. Reasoning is not the standard here, so this observes it instead.
//!
//! Run it with the CUDA toolkit removed from `PATH`, so that the only copy of
//! cuBLAS the process can reach is the one this harness stages while it runs:
//!
//! ```text
//! cargo run --release --features cuda --example gpu_recovery \
//!     -- <models-dir> <clip.wav> <buffer.txt> <model.gguf> [cuda-bin-dir]
//! ```
//!
//! The staging directory is a fresh one under the OS temp dir, never the app's
//! own; the DLLs are *read* from an installed toolkit, which is the closest a
//! test can honestly get to a completed download. Both models are read from
//! wherever they already are.
//!
//! Exit code 0 means: blocked at first, not blocked after `gpu::recheck`, and a
//! real transcription and a real cleanup both completed in that same process.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use steno_lib::config::{Settings, WhisperSettings};
use steno_lib::format::cleanup::{transfer, Outcome, Request};
use steno_lib::format::model::{availability, Loaded, Params};
use steno_lib::gpu;
use steno_lib::model;
use steno_lib::transcribe::{engine::Engine, read_clip};

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: gpu_recovery <models-dir> <clip.wav> <buffer.txt> <model.gguf> [cuda-bin-dir]";

    let (Some(models_dir), Some(clip), Some(buffer), Some(gguf)) = (
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
    ) else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let source = args.next().map(PathBuf::from).or_else(cuda_bin_dir);

    // A directory of our own, under the OS temp dir. Never the app data
    // directory: the point is to prove the mechanism, not to install anything
    // into a Steno the user is running.
    let staging = std::env::temp_dir().join("steno-gpu-recovery");
    let _ = std::fs::remove_dir_all(&staging);
    if let Err(error) = std::fs::create_dir_all(&staging) {
        eprintln!("could not create {}: {error}", staging.display());
        std::process::exit(1);
    }
    println!("staging        {}", staging.display());

    // Exactly what `lib.rs` does at startup, and before anything asks: the probe
    // resolves against the search order, so the search order has to be final.
    if let Err(error) = gpu::use_runtime_dir(&staging) {
        eprintln!("{error}");
        std::process::exit(1);
    }

    // 1. Empty directory, no toolkit on PATH: this must report a blocker, or
    //    the run proves nothing and the operator needs to hear why.
    let Some(before) = gpu::blocker() else {
        eprintln!();
        eprintln!("not blocked at startup — cuBLAS is already reachable from this process.");
        eprintln!("Re-run with the CUDA toolkit removed from PATH; otherwise this harness would");
        eprintln!("only be measuring a machine that never needed the download.");
        std::process::exit(1);
    };
    println!("before         blocked on {}", before.missing);

    // 2. Stage the DLLs, which is what a completed download leaves behind.
    let Some(source) = source else {
        eprintln!("no CUDA bin directory given and CUDA_PATH is not set; nothing to stage");
        std::process::exit(2);
    };
    match stage(&source, &staging, &before.missing) {
        Ok(staged) => {
            for (name, bytes) in &staged {
                println!("staged         {name} ({bytes} bytes)");
            }
        }
        Err(error) => {
            eprintln!("could not stage from {}: {error}", source.display());
            std::process::exit(1);
        }
    }

    // 3. The one legitimate invalidation, standing in for the download's
    //    completion.
    let started = Instant::now();
    if let Some(still) = gpu::recheck() {
        eprintln!("still blocked after staging: {}", still.one_line());
        eprintln!("the DLLs are in {} — the search path is what failed", staging.display());
        std::process::exit(1);
    }
    println!("after          clear, in {} ms", started.elapsed().as_millis());
    println!();

    // 4. Real work, in this same process, through the same code the app runs.
    transcribe(&models_dir, &clip);
    clean_up(&buffer, &gguf);

    println!();
    println!("no restart was needed.");
}

/// Runs a clip through the app's own engine.
fn transcribe(models_dir: &Path, clip: &Path) {
    let spec = model::default_spec();
    let path = models_dir.join(spec.id);

    let samples = read_clip(clip).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", clip.display());
        std::process::exit(1);
    });

    let engine = Engine::load(&path, spec.id).unwrap_or_else(|error| {
        eprintln!("could not load {}: {error}", path.display());
        std::process::exit(1);
    });

    let started = Instant::now();
    let transcript = engine.run(&samples, &WhisperSettings::default()).unwrap_or_else(|error| {
        eprintln!("transcription failed: {error}");
        std::process::exit(1);
    });
    let elapsed = started.elapsed().as_millis();

    let seconds = samples.len() as f64 / 16_000.0;
    println!("transcribed    {seconds:.2} s of audio in {elapsed} ms");
    println!("               {:?}", transcript.text);
}

/// Runs a buffer through the app's own cleanup.
fn clean_up(buffer: &Path, gguf: &Path) {
    let text = std::fs::read_to_string(buffer).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", buffer.display());
        std::process::exit(1);
    });

    let state = availability(gguf);
    if !state.reachable || !state.model_installed {
        eprintln!(
            "not ready: {}",
            state.remedy.unwrap_or_else(|| format!("{} is missing", state.model_path))
        );
        std::process::exit(1);
    }

    let settings = Settings::default();
    let loaded = Loaded::load(Params {
        path: gguf.to_path_buf(),
        n_gpu_layers: settings.llm.n_gpu_layers,
    })
    .unwrap_or_else(|error| {
        eprintln!("could not load the model: {error}");
        std::process::exit(1);
    });

    let cancel = AtomicBool::new(false);
    let sink = |_: &str| {};
    let outcome = transfer(&loaded, &Request::from_settings(&settings), &text, &cancel, &sink)
        .unwrap_or_else(|error| {
            eprintln!("cleanup failed: {error}");
            std::process::exit(1);
        });

    match outcome {
        Outcome::Complete(done) => println!(
            "cleaned up     {} prompt tokens, {} out, first token {} ms, {:.1} tok/s",
            done.prompt_tokens, done.output_tokens, done.ttft_ms, done.tokens_per_second
        ),
        Outcome::Cancelled(chars) => {
            eprintln!("cleanup cancelled after {chars} chars");
            std::process::exit(1);
        }
    }
}

/// Copies cuBLAS and the library it pulls in behind it.
///
/// `cublasLt` is imported by `cublas`, not by us, so it is never the file the
/// probe names — and a staging that forgot it would fail at the first call
/// rather than at the probe, which is exactly the failure this harness exists
/// to rule out.
fn stage(source: &Path, staging: &Path, missing: &str) -> Result<Vec<(String, u64)>, String> {
    let entries = std::fs::read_dir(source).map_err(|error| error.to_string())?;

    let mut staged = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let wanted = name == missing || (name.starts_with("cublasLt64_") && name.ends_with(".dll"));
        if !wanted {
            continue;
        }
        let bytes = std::fs::copy(entry.path(), staging.join(&name))
            .map_err(|error| format!("{name}: {error}"))?;
        staged.push((name, bytes));
    }

    if staged.is_empty() {
        return Err(format!("no {missing} there"));
    }
    Ok(staged)
}

fn cuda_bin_dir() -> Option<PathBuf> {
    let cuda = PathBuf::from(std::env::var("CUDA_PATH").ok()?);
    [cuda.join("bin").join("x64"), cuda.join("bin")]
        .into_iter()
        .find(|dir| dir.is_dir())
}
