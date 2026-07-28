//! Measures what a cleanup costs: time to first token, and total.
//!
//! Drives `format::cleanup::transfer` — the same function the Clean up button
//! reaches — with the same `Request` built from the same settings, so the
//! numbers describe the app rather than a second implementation of it.
//!
//! ```text
//! cargo run --release --example cleanup -- <buffer.txt> [runs]
//! ```
//!
//! Reports the model's own token counts rather than guessing at them from
//! character counts, so "a two-thousand token buffer" means what the tokeniser
//! says it means.
//!
//! Since 5.1 the model file has to be named, because there is no server to ask
//! where it keeps things:
//!
//! ```text
//! cargo run --release --example cleanup -- <buffer.txt> <model.gguf> [runs]
//! ```

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use steno_lib::config::Settings;
use steno_lib::format::cleanup::{transfer, Outcome, Request};
use steno_lib::format::model::{availability, Loaded, Params};

fn main() {
    let mut args = std::env::args().skip(1);

    let Some(path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cleanup <buffer.txt> <model.gguf> [runs]");
        std::process::exit(2);
    };
    let Some(gguf) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cleanup <buffer.txt> <model.gguf> [runs]");
        std::process::exit(2);
    };
    let runs: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(2).max(1);

    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        std::process::exit(1);
    });

    let settings = Settings::default();
    let request = Request::from_settings(&settings);

    println!("model          {}", gguf.display());
    println!("temperature    {}", request.temperature);
    println!("context        {} tokens", request.n_ctx);
    println!("buffer         {} chars", text.chars().count());

    let state = availability(&gguf);
    if !state.reachable || !state.model_installed {
        eprintln!(
            "not ready: {}",
            state
                .remedy
                .unwrap_or_else(|| format!("{} is missing", state.model_path))
        );
        std::process::exit(1);
    }

    // Loaded explicitly, and held, so the figures below measure generation
    // rather than a model load that the app pays for on window show anyway.
    let loaded = Loaded::load(Params {
        path: gguf,
        n_gpu_layers: settings.llm.n_gpu_layers,
    })
    .unwrap_or_else(|error| {
        eprintln!("could not load the model: {error}");
        std::process::exit(1);
    });
    println!("warm-up        {} ms", loaded.load_ms);
    println!();

    let cancel = AtomicBool::new(false);

    for run in 1..=runs {
        // Deliberately silent: emitting each chunk to a console would measure
        // the terminal as much as the model.
        let sink = |_: &str| {};

        // Every run gets a different first line. llama.cpp caches the prompt by
        // prefix, so sending the same buffer twice makes the second run report
        // a time to first token of about a hundred milliseconds — a cache hit,
        // not a latency. Changing the opening invalidates the whole cache and
        // measures the work the app will actually do.
        let text = format!("séance {run}\n\n{text}");

        let started = Instant::now();
        let outcome = transfer(&loaded, &request, &text, &cancel, &sink).unwrap_or_else(|error| {
            eprintln!("cleanup failed: {error}");
            std::process::exit(1);
        });
        let wall_ms = started.elapsed().as_millis();

        match outcome {
            Outcome::Complete(done) => {
                println!("run {run}");
                println!("  prompt       {} tokens", done.prompt_tokens);
                println!("  output       {} tokens", done.output_tokens);
                println!("  first token  {} ms", done.ttft_ms);
                println!("  total        {} ms  (wall {wall_ms} ms)", done.total_ms);
                println!("  rate         {:.1} tok/s", done.tokens_per_second);

                if run == runs {
                    println!();
                    println!("--- output ---");
                    println!("{}", done.text);
                }
            }
            Outcome::Cancelled(chars) => {
                println!("run {run}: cancelled after {chars} chars");
            }
        }
    }
}
