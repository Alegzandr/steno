//! Drives the real cuBLAS install, from NVIDIA, into a directory of our own.
//!
//! This is the path a new user takes on first launch and the one the developer
//! never takes, because the developer has the toolkit. So it is exercised here:
//! the pinned URL, the published digest, the resumable transfer, the two deflate
//! members and their sizes, and — the part that cannot be reasoned about — the
//! loader finding the result and the process working afterwards.
//!
//! Run it with the CUDA toolkit removed from `PATH`, or it proves nothing:
//!
//! ```text
//! cargo run --release --features cuda --example cublas_install \
//!     -- [<models-dir> <clip.wav> <buffer.txt> <model.gguf>]
//! ```
//!
//! With the four optional arguments it goes on to transcribe and clean up in the
//! same process. Without them it stops once the loader is happy.
//!
//! It downloads into a fresh directory under the OS temp dir, never
//! `%APPDATA%\com.steno.app\runtime`: the app owns that one, and a harness has
//! no business writing there.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use steno_lib::config::{Settings, WhisperSettings};
use steno_lib::format::cleanup::{transfer, Outcome, Request};
use steno_lib::format::model::{availability, Loaded, Params};
use steno_lib::gpu::{self, runtime};
use steno_lib::model;
use steno_lib::model::download::Progress;
use steno_lib::transcribe::{engine::Engine, read_clip};

fn main() {
    let mut args = std::env::args().skip(1);
    let models_dir = args.next().map(PathBuf::from);
    let clip = args.next().map(PathBuf::from);
    let buffer = args.next().map(PathBuf::from);
    let gguf = args.next().map(PathBuf::from);

    let staging = std::env::temp_dir().join("steno-cublas-install");
    println!("staging        {}", staging.display());
    println!("archive        {} ({} bytes)", runtime::ARCHIVE.id, runtime::ARCHIVE.bytes);
    println!("from           {}", runtime::ARCHIVE.url);
    println!("installs       {} bytes", runtime::install_bytes());

    if let Err(error) = gpu::use_runtime_dir(&staging) {
        eprintln!("{error}");
        std::process::exit(1);
    }

    println!(
        "driver         {}",
        gpu::driver::installed()
            .map(|driver| format!(
                "{} · CUDA {}.{}",
                driver.version, driver.cuda_major, driver.cuda_minor
            ))
            .unwrap_or_else(|| "could not be determined".to_owned())
    );

    let Some(blocker) = gpu::blocker() else {
        eprintln!();
        eprintln!("not blocked — cuBLAS is already reachable from this process.");
        eprintln!("Re-run with the CUDA toolkit removed from PATH; otherwise this would only");
        eprintln!("measure a machine that never needed the download.");
        std::process::exit(1);
    };
    println!("before         blocked on {}", blocker.missing);
    println!();

    let cancel = AtomicBool::new(false);
    let report = |progress: Progress| {
        let percent = progress.received_bytes as f64 / progress.total_bytes as f64 * 100.0;
        print!(
            "\r  {percent:5.1}%  {:.1} MB/s      ",
            progress.bytes_per_second as f64 / 1_000_000.0
        );
        let _ = std::io::stdout().flush();
    };
    let stage = |name: &str| println!("\n{name}…");

    let started = Instant::now();
    let outcome = tauri::async_runtime::block_on(runtime::fetch_into(
        &staging, &cancel, &report, &stage,
    ));
    if let Err(error) = outcome {
        eprintln!("\ninstall failed: {error}");
        std::process::exit(1);
    }
    println!("installed      in {:.1} s", started.elapsed().as_secs_f64());

    for entry in std::fs::read_dir(&staging).into_iter().flatten().flatten() {
        println!(
            "               {} ({} bytes)",
            entry.file_name().to_string_lossy(),
            entry.metadata().map(|meta| meta.len()).unwrap_or(0)
        );
    }

    if let Some(still) = gpu::recheck() {
        eprintln!("still blocked after installing: {}", still.one_line());
        std::process::exit(1);
    }
    println!("after          clear");

    let (Some(models_dir), Some(clip), Some(buffer), Some(gguf)) = (models_dir, clip, buffer, gguf)
    else {
        println!();
        println!("the loader is satisfied; no models given, so nothing was run through it.");
        return;
    };

    println!();
    transcribe(&models_dir, &clip);
    clean_up(&buffer, &gguf);
    println!();
    println!("dictation and cleanup both ran on a cuBLAS this process downloaded.");
}

fn transcribe(models_dir: &Path, clip: &Path) {
    let spec = model::default_spec();
    let samples = read_clip(clip).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", clip.display());
        std::process::exit(1);
    });

    let engine = Engine::load(&models_dir.join(spec.id), spec.id).unwrap_or_else(|error| {
        eprintln!("could not load {}: {error}", spec.id);
        std::process::exit(1);
    });

    let started = Instant::now();
    let transcript = engine
        .run(&samples, &WhisperSettings::default())
        .unwrap_or_else(|error| {
            eprintln!("transcription failed: {error}");
            std::process::exit(1);
        });

    println!(
        "transcribed    {:.2} s of audio in {} ms",
        samples.len() as f64 / 16_000.0,
        started.elapsed().as_millis()
    );
    println!("               {:?}", transcript.text);
}

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
    match transfer(&loaded, &Request::from_settings(&settings), &text, &cancel, &sink) {
        Ok(Outcome::Complete(done)) => println!(
            "cleaned up     {} prompt tokens, {} out, first token {} ms, {:.1} tok/s",
            done.prompt_tokens, done.output_tokens, done.ttft_ms, done.tokens_per_second
        ),
        Ok(Outcome::Cancelled(chars)) => {
            eprintln!("cleanup cancelled after {chars} chars");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("cleanup failed: {error}");
            std::process::exit(1);
        }
    }
}
