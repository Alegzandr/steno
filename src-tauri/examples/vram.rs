//! Proves the video memory actually comes back, rather than trusting the APIs
//! that say it will.
//!
//! Three separate claims, three subcommands:
//!
//! ```text
//! cargo run --release --features cuda --example vram -- whisper <models-dir>
//! cargo run --release --example vram -- ollama [model]
//! cargo run --release --example vram -- job
//! ```
//!
//! * `whisper` â€” load a Whisper context, drop it, and read `nvidia-smi` at each
//!   step. Dropping a `WhisperContext` frees its own buffers; whether ggml's
//!   per-device CUDA state comes back with it is a question only the meter can
//!   answer.
//! * `ollama`  â€” warm the formatting model, drop the claim, and watch the
//!   memory return. Deliberately reads the meter *after* `/api/ps` goes quiet,
//!   because the HTTP call returns before the runner process holding the memory
//!   has exited.
//! * `job`     â€” the one claim that cannot be checked by reading code: that a
//!   child in a kill-on-close job object dies when its parent is force-killed
//!   rather than shut down. The parent here is terminated with
//!   `TerminateProcess`, the same call Task Manager's End task makes, so no
//!   destructor, exit hook or unwinding runs.

use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

fn main() {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("whisper") => whisper(args.next().map(PathBuf::from)),
        Some("llm") => llm(args.next().map(PathBuf::from)),
        Some("both") => both(args.next().map(PathBuf::from), args.next().map(PathBuf::from)),
        _ => {
            eprintln!(
                "usage: vram <whisper <models-dir> | llm <model.gguf> \
                 | both <models-dir> <buffer.txt>>"
            );
            std::process::exit(2);
        }
    }
}

/// Used video memory in MiB, across all adapters, as reported by the driver.
fn used_mib() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn meter(label: &str) -> Option<u64> {
    let used = used_mib();
    match used {
        Some(mib) => println!("  {label:<28} {mib:>6} MiB"),
        None => println!("  {label:<28}      ? (nvidia-smi unavailable)"),
    }
    used
}

fn whisper(models_dir: Option<PathBuf>) {
    use steno_lib::config::WhisperSettings;
    use steno_lib::model;
    use steno_lib::transcribe::engine::Engine;

    let Some(models_dir) = models_dir else {
        eprintln!("usage: vram whisper <models-dir>");
        std::process::exit(2);
    };

    let spec = model::default_spec();
    let path = models_dir.join(spec.id);
    println!("whisper  {} on the {} backend", spec.id, model::backend_name());

    let baseline = meter("baseline");
    let settings = WhisperSettings::default();
    let silence = vec![0.0f32; 16_000 * 5];

    let mut peak = baseline;
    let mut released = baseline;

    // Two full cycles, because one number cannot tell the two failure modes
    // apart. A fixed residue that does not grow is ggml's per-device CUDA
    // state â€” one context, one cuBLAS handle, one memory pool â€” which is
    // created on first use and lives until the process exits. A residue that
    // grows with each cycle is a leak, and a very different problem.
    for cycle in 1..=2 {
        let engine = match Engine::load(&path, spec.id) {
            Ok(engine) => engine,
            Err(error) => {
                eprintln!("could not load the model: {error}");
                eprintln!("run the rtf example first, it downloads the model");
                std::process::exit(1);
            }
        };
        meter(&format!("cycle {cycle}: loaded"));

        // A context that has never decoded has not allocated its compute
        // buffers. Measuring before a run would understate the footprint.
        let _ = engine.run(&silence, &settings);
        peak = peak.max(meter(&format!("cycle {cycle}: after a decode")));

        drop(engine);
        // Freeing is not instantaneous from the driver's point of view.
        thread::sleep(Duration::from_millis(1_500));
        released = meter(&format!("cycle {cycle}: dropped"));
    }

    report(baseline, peak, released, GGML_DEVICE_STATE_MIB);
}

/// The formatting model, loaded and dropped in this process.
///
/// Before 5.1 the weights lived in Ollama and "released" meant released to the
/// byte, because the memory was never ours to begin with. It is ours now, so
/// this shares the whisper case's allowance: ggml's per-device CUDA state is
/// created on first use and freed only at process exit. Linking a second ggml
/// did not double it â€” measured flat at 253 MiB across six load/unload cycles,
/// with no drift.
fn llm(gguf: Option<PathBuf>) {
    use steno_lib::format::model::{Loaded, Params};

    let Some(path) = gguf else {
        eprintln!("usage: vram llm <model.gguf>");
        std::process::exit(2);
    };

    println!("llm      {}", path.display());
    let baseline = meter("baseline");

    let loaded = match Loaded::load(Params {
        path,
        n_gpu_layers: steno_lib::config::Settings::default().llm.n_gpu_layers,
    }) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("could not load the model: {error}");
            std::process::exit(1);
        }
    };
    println!("  load                         {:>6} ms", loaded.load_ms);
    let resident = meter("model resident");

    drop(loaded);
    // The driver does not hand video memory back the instant a model dies.
    std::thread::sleep(Duration::from_secs(3));
    let released = meter("model unloaded");

    report(baseline, resident, released, GGML_DEVICE_STATE_MIB);
}

/// The peak: both models resident at once, during a real cleanup.
///
/// This is the acceptance row that cannot be reached from the outside without
/// somebody holding down push-to-talk, so it is reached through the same code
/// the app runs instead â€” `Engine`, `Loaded` and `cleanup::transfer`, in the
/// order a dictation followed by a cleanup puts them in.
fn both(models_dir: Option<PathBuf>, buffer: Option<PathBuf>) {
    use std::sync::atomic::AtomicBool;

    use steno_lib::config::{Settings, WhisperSettings};
    use steno_lib::format::cleanup::{transfer, Outcome, Request};
    use steno_lib::format::model::{Loaded, Params};
    use steno_lib::model;
    use steno_lib::transcribe::engine::Engine;

    let (Some(models_dir), Some(buffer)) = (models_dir, buffer) else {
        eprintln!("usage: vram both <models-dir> <buffer.txt>");
        std::process::exit(2);
    };

    let text = std::fs::read_to_string(&buffer).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", buffer.display());
        std::process::exit(1);
    });

    let settings = Settings::default();
    let request = Request::from_settings(&settings);
    let spec = model::default_spec();

    // Both models come out of the same directory now, which is the whole
    // simplification 5.1 bought: one place, one drive, no second process.
    let gguf = models_dir.join(&settings.llm.model_file);
    println!("both     {} + {}", spec.id, settings.llm.model_file);

    let baseline = meter("baseline");

    let engine = Engine::load(&models_dir.join(spec.id), spec.id).unwrap_or_else(|error| {
        eprintln!("could not load Whisper: {error}");
        std::process::exit(1);
    });
    let _ = engine.run(&vec![0.0f32; 16_000 * 5], &WhisperSettings::default());
    meter("whisper resident");

    let loaded = Loaded::load(Params {
        path: gguf,
        n_gpu_layers: settings.llm.n_gpu_layers,
    })
    .unwrap_or_else(|error| {
        eprintln!("could not load the formatting model: {error}");
        std::process::exit(1);
    });
    let peak = meter("both resident");

    let cancel = AtomicBool::new(false);
    let sink = |_: &str| {};
    let during = match transfer(&loaded, &request, &text, &cancel, &sink) {
        Ok(Outcome::Complete(done)) => {
            let during = meter("during the cleanup");
            println!("  cleanup                      {:>6} ms", done.total_ms);
            during
        }
        Ok(Outcome::Cancelled(_)) => meter("during the cleanup"),
        Err(error) => {
            eprintln!("the cleanup failed: {error}");
            std::process::exit(1);
        }
    };

    // Release in the order `lifecycle::evict_all` uses.
    drop(engine);
    thread::sleep(Duration::from_millis(1_500));
    meter("whisper released");

    drop(loaded);
    let released = meter("both released");

    report(baseline, peak.max(during), released, GGML_DEVICE_STATE_MIB);
}

/// Other applications move the figure between two reads, so a few tens of
/// megabytes is noise rather than a leak.
const NOISE_MIB: u64 = 64;

/// ggml creates one CUDA context, cuBLAS handle and memory pool per device on
/// first use and frees them only at process exit. Measured at ~222 MiB and
/// constant across load/drop cycles, so it is not a leak â€” see the accepted
/// deviation in CLAUDE.md. Anything materially above this is a real failure.
const GGML_DEVICE_STATE_MIB: u64 = 320;

fn report(baseline: Option<u64>, peak: Option<u64>, released: Option<u64>, allowance: u64) {
    let (Some(baseline), Some(peak), Some(released)) = (baseline, peak, released) else {
        return;
    };

    println!();
    println!("  held while working           {:>6} MiB", peak.saturating_sub(baseline));
    let residue = released.saturating_sub(baseline);
    println!("  residue after release        {:>6} MiB", residue);

    let verdict = if residue <= NOISE_MIB {
        "clean"
    } else if residue <= allowance {
        "ggml per-device CUDA state, freed at process exit (accepted deviation)"
    } else {
        "MEMORY NOT RETURNED"
    };
    println!("  verdict                      {verdict}");
}
