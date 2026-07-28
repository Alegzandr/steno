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
//! Reports the model's own token counts from the final stream frame rather than
//! guessing at them from character counts, so "a two-thousand token buffer"
//! means what Ollama says it means.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use steno_lib::config::Settings;
use steno_lib::format::cleanup::{transfer, Outcome, Request};
use steno_lib::format::model::{availability, Loaded};
use steno_lib::format::server::Server;

fn main() {
    let mut args = std::env::args().skip(1);

    let Some(path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cleanup <buffer.txt> [runs]");
        std::process::exit(2);
    };
    let runs: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(2).max(1);

    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        std::process::exit(1);
    });

    let settings = Settings::default();
    let request = Request::from_settings(&settings);

    println!("endpoint       {}", request.endpoint);
    println!("model          {}", request.model);
    println!("temperature    {}", request.temperature);
    println!("buffer         {} chars", text.chars().count());

    // Adopt a running server or start one, exactly as the app does. Held for
    // the whole run: dropping it is what stops a server we started, and on
    // Windows the job object takes it down even if this process is killed.
    let server = Server::ensure(&request.endpoint, settings.ollama.models_dir.as_deref());
    println!("server         {:?}", server.ownership);

    let state = availability(&request.endpoint, &request.model);
    if !state.reachable || !state.model_installed {
        eprintln!(
            "not ready: {}",
            state.remedy.unwrap_or_else(|| "unknown".to_owned())
        );
        std::process::exit(1);
    }

    // Warmed explicitly, and held, so the figures below measure generation
    // rather than a model load that the app pays for on window show anyway.
    let loaded = Loaded::warm(&request.endpoint, &request.model, &request.keep_alive, false)
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
        let outcome = transfer(&request, &text, &cancel, &sink).unwrap_or_else(|error| {
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
