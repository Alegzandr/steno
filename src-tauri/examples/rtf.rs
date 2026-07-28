//! Measures what transcription actually costs on this machine.
//!
//! Drives the same code the app drives — the same downloader, the same
//! `Engine`, the same Whisper parameters read from the same `WhisperSettings`
//! — so the numbers it prints are the numbers the app will produce. Nothing
//! here is a reimplementation; if a parameter changes in `transcribe::engine`,
//! it changes here too.
//!
//! ```text
//! cargo run --release --example rtf -- <models-dir> <clip.wav> [model-id] [runs]
//! cargo run --release --features cuda --example rtf -- ...
//! ```

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use steno_lib::config::WhisperSettings;
use steno_lib::model::{self, download};
use steno_lib::transcribe::{engine::Engine, filter, read_clip};

fn main() {
    let mut args = std::env::args().skip(1);

    let Some(models_dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: rtf <models-dir> <clip.wav> [model-id] [runs]");
        std::process::exit(2);
    };
    let Some(clip) = args.next().map(PathBuf::from) else {
        eprintln!("usage: rtf <models-dir> <clip.wav> [model-id] [runs]");
        std::process::exit(2);
    };

    let requested = args.next();
    let runs: usize = args
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(3)
        .max(1);

    let spec = match requested {
        Some(id) => model::find(&id).unwrap_or_else(|| {
            eprintln!("unknown model {id:?}");
            eprintln!("known: {:?}", model::CATALOGUE.iter().map(|s| s.id).collect::<Vec<_>>());
            std::process::exit(2);
        }),
        None => model::default_spec(),
    };

    println!("backend        {}", model::backend_name());
    println!(
        "video memory   {}",
        model::detect::dedicated_video_memory()
            .map(|bytes| format!("{} MiB", bytes / (1024 * 1024)))
            .unwrap_or_else(|| "unknown".to_owned())
    );
    println!("model          {} ({} bytes)", spec.id, spec.bytes);

    let path = models_dir.join(spec.id);
    if let Err(error) = ensure(spec, &path) {
        eprintln!("download failed: {error}");
        std::process::exit(1);
    }

    let samples = read_clip(&clip).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", clip.display());
        std::process::exit(1);
    });

    let clip_ms = samples.len() as f64 * 1_000.0 / 16_000.0;
    println!("clip           {:.2} s at {:.1} dBFS", clip_ms / 1_000.0, filter::rms_dbfs(&samples));
    println!();

    let engine = Engine::load(&path, spec.id).unwrap_or_else(|error| {
        eprintln!("could not load the model: {error}");
        std::process::exit(1);
    });
    println!("cold start     {} ms", engine.load_ms);

    let settings = WhisperSettings::default();
    let mut transcript = None;

    for run in 1..=runs {
        let started = Instant::now();
        let result = engine.run(&samples, &settings).unwrap_or_else(|error| {
            eprintln!("transcription failed: {error}");
            std::process::exit(1);
        });
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;

        println!(
            "run {run}          {elapsed:.0} ms   RTF {:.3}   ({:.1}x real time)",
            elapsed / clip_ms,
            clip_ms / elapsed
        );

        if run == runs {
            println!(
                "guards         {} segment(s), {} dropped by no-speech, {} by the denylist",
                result.segment_count, result.dropped_no_speech, result.dropped_denylist
            );
            println!(
                "no-speech      peak p={:.3} against a threshold of {:.2}",
                result.peak_no_speech, settings.no_speech_thold
            );
            transcript = Some(result.text);
        }
    }

    println!();
    println!("{}", transcript.unwrap_or_default());
}

/// Downloads the model if it is not already there, through the app's own
/// resumable, checksum-verified path.
fn ensure(spec: &'static model::ModelSpec, path: &PathBuf) -> Result<(), String> {
    if std::fs::metadata(path).is_ok_and(|meta| meta.len() == spec.bytes) {
        println!("model          already present");
        return Ok(());
    }

    let cancel = AtomicBool::new(false);
    let report = |progress: download::Progress| {
        let percent = progress.received_bytes as f64 / progress.total_bytes as f64 * 100.0;
        print!(
            "\rdownloading    {percent:5.1}%  {:.1} MB/s   ",
            progress.bytes_per_second as f64 / 1_000_000.0
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    };

    let outcome = tauri::async_runtime::block_on(download::transfer(spec, path, &cancel, &report));
    println!();

    outcome.map(|took| println!("downloaded     in {:.1} s", took.as_secs_f64()))
}
