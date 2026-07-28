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
//! * `whisper` — load a Whisper context, drop it, and read `nvidia-smi` at each
//!   step. Dropping a `WhisperContext` frees its own buffers; whether ggml's
//!   per-device CUDA state comes back with it is a question only the meter can
//!   answer.
//! * `ollama`  — warm the formatting model, drop the claim, and watch the
//!   memory return. Deliberately reads the meter *after* `/api/ps` goes quiet,
//!   because the HTTP call returns before the runner process holding the memory
//!   has exited.
//! * `job`     — the one claim that cannot be checked by reading code: that a
//!   child in a kill-on-close job object dies when its parent is force-killed
//!   rather than shut down. The parent here is terminated with
//!   `TerminateProcess`, the same call Task Manager's End task makes, so no
//!   destructor, exit hook or unwinding runs.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

fn main() {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("whisper") => whisper(args.next().map(PathBuf::from)),
        Some("ollama") => ollama(args.next()),
        Some("both") => both(args.next().map(PathBuf::from), args.next().map(PathBuf::from)),
        Some("job") => job(),
        Some("job-host") => job_host(),
        _ => {
            eprintln!(
                "usage: vram <whisper <models-dir> | ollama [model] \
                 | both <models-dir> <buffer.txt> | job>"
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
    // state — one context, one cuBLAS handle, one memory pool — which is
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

fn ollama(model: Option<String>) {
    use steno_lib::format::model::Loaded;
    use steno_lib::format::server::{self, Server};

    let endpoint = "http://127.0.0.1:11434";
    let model = model.unwrap_or_else(|| "qwen3:14b".to_owned());

    let models_dir = steno_lib::config::Settings::default().ollama.models_dir;
    let server = Server::ensure(endpoint, models_dir.as_deref());
    println!("ollama   {model} on {endpoint} ({:?})", server.ownership);

    if !server.is_reachable() {
        eprintln!("no server, and none could be started");
        std::process::exit(1);
    }

    if server.is_foreign(&model) {
        eprintln!(
            "{model} was already resident before this harness started. It belongs to \
             something else, so unloading it is not ours to do. Skipping."
        );
        std::process::exit(0);
    }

    let baseline = meter("baseline");

    let loaded = match Loaded::warm(endpoint, &model, "20m", false) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("could not warm the model: {error}");
            std::process::exit(1);
        }
    };
    println!("  cold start                   {:>6} ms", loaded.load_ms);
    let resident = meter("model resident");

    // `Drop` sends keep_alive: 0 and then polls /api/ps until the model is gone.
    drop(loaded);
    let released = meter("model unloaded");

    println!("  still in /api/ps             {:?}", server::model_names(endpoint));
    // No allowance: the weights are in another process, so "released" means
    // released to the byte.
    report(baseline, resident, released, NOISE_MIB);
}

/// The peak: both models resident at once, during a real cleanup.
///
/// This is the acceptance row that cannot be reached from the outside without
/// somebody holding down push-to-talk, so it is reached through the same code
/// the app runs instead — `Engine`, `Loaded` and `cleanup::transfer`, in the
/// order a dictation followed by a cleanup puts them in.
fn both(models_dir: Option<PathBuf>, buffer: Option<PathBuf>) {
    use std::sync::atomic::AtomicBool;

    use steno_lib::config::{Settings, WhisperSettings};
    use steno_lib::format::cleanup::{transfer, Outcome, Request};
    use steno_lib::format::model::Loaded;
    use steno_lib::format::server::Server;
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

    let server = Server::ensure(&request.endpoint, settings.ollama.models_dir.as_deref());
    println!(
        "both     {} + {} ({:?})",
        spec.id, request.model, server.ownership
    );
    if !server.is_reachable() {
        eprintln!("no Ollama server, and none could be started");
        std::process::exit(1);
    }
    let foreign = server.is_foreign(&request.model);

    let baseline = meter("baseline");

    let engine = Engine::load(&models_dir.join(spec.id), spec.id).unwrap_or_else(|error| {
        eprintln!("could not load Whisper: {error}");
        std::process::exit(1);
    });
    let _ = engine.run(&vec![0.0f32; 16_000 * 5], &WhisperSettings::default());
    meter("whisper resident");

    let loaded = Loaded::warm(
        &request.endpoint,
        &request.model,
        &request.keep_alive,
        foreign,
    )
    .unwrap_or_else(|error| {
        eprintln!("could not load the formatting model: {error}");
        std::process::exit(1);
    });
    let peak = meter("both resident");

    let cancel = AtomicBool::new(false);
    let sink = |_: &str| {};
    let during = match transfer(&request, &text, &cancel, &sink) {
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

    if foreign {
        println!(
            "  note                         {} was already resident and was left loaded",
            request.model
        );
    }
}

/// Other applications move the figure between two reads, so a few tens of
/// megabytes is noise rather than a leak.
const NOISE_MIB: u64 = 64;

/// ggml creates one CUDA context, cuBLAS handle and memory pool per device on
/// first use and frees them only at process exit. Measured at ~222 MiB and
/// constant across load/drop cycles, so it is not a leak — see the accepted
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

/// Parent side of the force-kill test.
fn job() {
    let exe = std::env::current_exe().expect("current exe");

    let mut host = Command::new(&exe)
        .arg("job-host")
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the host");

    let stdout = host.stdout.take().expect("piped stdout");
    let mut lines = BufReader::new(stdout).lines();
    let announced = lines
        .next()
        .and_then(|line| line.ok())
        .expect("the host announces its child");

    let child_pid: u32 = announced
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("unexpected announcement {announced:?}"));

    println!("job      host pid {}, child pid {child_pid}", host.id());
    println!("  child alive before kill      {}", alive(child_pid));

    // TerminateProcess, exactly what Task Manager's End task does: no
    // destructors, no exit hooks, no unwinding. Only the kernel closing our
    // handles is left, which is the entire point of the job object.
    host.kill().expect("terminate the host");
    let _ = host.wait();

    thread::sleep(Duration::from_millis(1_000));
    let survived = alive(child_pid);
    println!("  child alive after kill       {survived}");
    println!(
        "  verdict                      {}",
        if survived { "ORPHANED" } else { "killed with the parent" }
    );

    if survived {
        // Do not leave it running just because the test failed.
        let _ = Command::new("taskkill").args(["/PID", &child_pid.to_string(), "/F"]).output();
        std::process::exit(1);
    }
}

/// Child side: builds the job exactly as `format::server` does, puts a
/// long-lived process in it, then waits to be killed.
fn job_host() {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use steno_lib::format::server::job::Job;

        let job = Job::new().expect("create the job object");

        // Stands in for `ollama serve`: something that will not exit on its own.
        let child = Command::new("ping")
            .args(["-n", "3600", "127.0.0.1"])
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn the stand-in child");

        job.adopt(child.as_raw_handle()).expect("assign to the job");

        println!("{}", child.id());
        let _ = std::io::stdout().flush();

        // Deliberately never returns. The parent terminates us.
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }

    #[cfg(not(windows))]
    {
        eprintln!("the job object test is Windows-only");
        std::process::exit(2);
    }
}

fn alive(pid: u32) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output();

    match output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()),
        Err(_) => false,
    }
}
