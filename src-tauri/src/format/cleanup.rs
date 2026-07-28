//! Streaming a whole-buffer cleanup through the local model.
//!
//! The buffer is an accumulation of dictated bursts, so a cleanup is one long
//! request over everything the user has said so far, not an incremental edit.
//! It streams because a two-thousand-token rewrite takes tens of seconds and
//! watching it arrive is the difference between "working" and "hung".
//!
//! Three things this owes the rest of the app:
//!
//! - **It never hangs silently.** Every failure path ends in a `cleanup-error`
//!   carrying a sentence the user can act on, and where there is a command that
//!   fixes it, the command itself.
//! - **It holds a lease for the whole stream.** Hiding the window mid-cleanup
//!   must not pull the model out from under a request in flight; `Resident`
//!   waits for the lease, and the lease lives until the last token.
//! - **It emits deltas, and the full text again at the end.** The frontend
//!   needs the deltas to render progress and the whole string to apply the
//!   single undo transaction. Rebuilding it from the deltas would work right up
//!   until one gets dropped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::server;
use super::Formatter;
use crate::config::Config;
use crate::lifecycle;

pub const STARTED: &str = "cleanup-started";
pub const DELTA: &str = "cleanup-delta";
pub const COMPLETE: &str = "cleanup-complete";
pub const FAILED: &str = "cleanup-error";
pub const CANCELLED: &str = "cleanup-cancelled";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Started {
    pub model: String,
    pub input_chars: usize,
    /// Whether the model still has to be loaded. Decides whether the UI says
    /// "cleaning up" or "loading the model".
    pub model_cold: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Delta {
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Complete {
    pub text: String,
    /// Request sent to first token on screen. The number that decides whether
    /// the wait feels like a pause or a freeze.
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub tokens_per_second: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Failed {
    pub message: String,
    /// A command to type, when one would fix it.
    pub remedy: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cancelled {
    pub partial_chars: usize,
}

/// Managed state: at most one cleanup at a time, and a flag to stop it.
#[derive(Default)]
pub struct Cleanup {
    running: AtomicBool,
    cancel: AtomicBool,
}

impl Cleanup {
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Asks the running cleanup to stop. Returns whether there was one.
    pub fn cancel(&self) -> bool {
        let running = self.is_running();
        if running {
            self.cancel.store(true, Ordering::Release);
        }
        running
    }
}

/// Starts a cleanup on a background thread.
///
/// Returns as soon as the thread is running: everything after this point
/// arrives as events. The only error it returns directly is the one the caller
/// can do something about — that a cleanup is already in flight.
pub fn spawn<R: Runtime>(app: AppHandle<R>, text: String) -> Result<(), String> {
    let cleanup = app.state::<Arc<Cleanup>>().inner().clone();

    if cleanup
        .running
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("a cleanup is already running".to_owned());
    }
    cleanup.cancel.store(false, Ordering::Release);

    let worker = cleanup.clone();
    let spawned = std::thread::Builder::new()
        .name("steno-cleanup".to_owned())
        .spawn(move || {
            run(&app, text, worker.clone());
            worker.running.store(false, Ordering::Release);
        });

    if let Err(error) = spawned {
        cleanup.running.store(false, Ordering::Release);
        return Err(format!("could not start the cleanup thread ({error})"));
    }

    Ok(())
}

fn run<R: Runtime>(app: &AppHandle<R>, text: String, cleanup: Arc<Cleanup>) {
    lifecycle::touch(app);

    let settings = app.state::<Config>().get();
    let ollama = settings.ollama.clone();
    let request = Request::from_settings(&settings);

    // Acquiring can block for a minute on a cold model, so the UI is told what
    // is happening before it starts rather than after.
    let formatter = app.state::<Formatter>().inner().clone();
    let model_cold = !formatter.is_warm();

    emit(
        app,
        STARTED,
        Started {
            model: ollama.model.clone(),
            input_chars: text.chars().count(),
            model_cold,
        },
    );

    let lease = match formatter.acquire(lifecycle::formatter_loader(app)) {
        Ok(lease) => lease,
        Err(message) => {
            // The model failed to load. `availability` turns that into the
            // command that fixes it, which is nearly always `ollama pull`.
            let remedy = super::model::availability(&ollama.endpoint, &ollama.model).remedy;
            emit(app, FAILED, Failed { message, remedy });
            return;
        }
    };

    let report = |chunk: &str| {
        let _ = app.emit(
            DELTA,
            Delta {
                text: chunk.to_owned(),
            },
        );
    };

    match transfer(&request, &text, &cleanup.cancel, &report) {
        Ok(Outcome::Complete(complete)) => {
            eprintln!(
                "cleanup: {} output tokens in {} ms (first at {} ms, {:.1} tok/s)",
                complete.output_tokens,
                complete.total_ms,
                complete.ttft_ms,
                complete.tokens_per_second
            );
            emit(app, COMPLETE, complete);
        }
        Ok(Outcome::Cancelled(partial_chars)) => {
            eprintln!("cleanup: cancelled after {partial_chars} characters");
            emit(app, CANCELLED, Cancelled { partial_chars });
        }
        Err(message) => {
            let remedy = super::model::availability(&ollama.endpoint, &ollama.model).remedy;
            eprintln!("cleanup: failed ({message})");
            emit(app, FAILED, Failed { message, remedy });
        }
    }

    // The lease is released here, not before: an eviction triggered while the
    // stream was running has been waiting for exactly this.
    drop(lease);
    lifecycle::touch(app);
}

pub enum Outcome {
    Complete(Complete),
    Cancelled(usize),
}

/// The request itself.
///
/// `think` is sent as false because qwen3 is a thinking model and its reasoning
/// trace is neither wanted in the buffer nor worth the tokens: a cleanup is a
/// mechanical rewrite, not a problem to reason about. Servers or models that do
/// not understand the field reject the request outright, so that rejection is
/// caught and the request retried without it rather than surfaced to the user.
pub fn transfer(
    request: &Request,
    text: &str,
    cancel: &AtomicBool,
    report: &(dyn Fn(&str) + Send + Sync),
) -> Result<Outcome, String> {
    match send(request, text, cancel, report, true) {
        Err(Rejected::Thinking) => {
            eprintln!("cleanup: the server rejected `think`, retrying without it");
            send(request, text, cancel, report, false).map_err(|error| error.message())
        }
        Err(other) => Err(other.message()),
        Ok(outcome) => Ok(outcome),
    }
}

/// What settings a cleanup needs, flattened so the request builder takes one
/// argument instead of reaching back into `Settings`.
pub struct Request {
    pub endpoint: String,
    pub model: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub keep_alive: String,
}

impl Request {
    /// The one place a cleanup request is derived from settings, so the app and
    /// the measurement harness cannot disagree about the prompt or the
    /// temperature they are testing.
    pub fn from_settings(settings: &crate::config::Settings) -> Self {
        Self {
            endpoint: settings.ollama.endpoint.clone(),
            model: settings.ollama.model.clone(),
            system_prompt: settings.ollama.system_prompt.clone(),
            temperature: settings.ollama.temperature,
            keep_alive: settings.vram.keep_alive.clone(),
        }
    }
}

enum Rejected {
    /// The `think` field specifically. Recoverable by retrying without it.
    Thinking,
    Other(String),
}

impl Rejected {
    fn message(self) -> String {
        match self {
            Rejected::Thinking => "the server rejected the `think` field".to_owned(),
            Rejected::Other(message) => message,
        }
    }
}

fn send(
    request: &Request,
    text: &str,
    cancel: &AtomicBool,
    report: &(dyn Fn(&str) + Send + Sync),
    think: bool,
) -> Result<Outcome, Rejected> {
    let url = format!("{}/api/generate", request.endpoint.trim_end_matches('/'));

    let mut body = serde_json::json!({
        "model": request.model,
        "system": request.system_prompt,
        "prompt": text,
        "stream": true,
        // Never -1, and never absent: this is what returns the video memory if
        // Steno dies before it can run its own unload.
        "keep_alive": request.keep_alive,
        "options": { "temperature": request.temperature },
    });
    if think {
        body["think"] = serde_json::Value::Bool(false);
    }

    let endpoint = request.endpoint.as_str();

    // The stream has no overall timeout on purpose: a slow model is not a
    // broken one, and a two-thousand-token rewrite on a cold cache legitimately
    // takes minutes. What protects against a genuinely dead server is the
    // connect timeout, which is short.
    server::blocking(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|error| Rejected::Other(format!("could not build an HTTP client ({error})")))?;

        let started = Instant::now();
        let response = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|error| Rejected::Other(unreachable(endpoint, &error)))?;

        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();

            if detail.contains("think") {
                return Err(Rejected::Thinking);
            }
            if detail.contains("not found") || status == reqwest::StatusCode::NOT_FOUND {
                return Err(Rejected::Other(format!(
                    "the model is not installed on {endpoint}"
                )));
            }
            return Err(Rejected::Other(format!(
                "Ollama answered {status} ({})",
                detail.trim()
            )));
        }

        let mut chunks = response.bytes_stream();
        let mut pending = Vec::new();
        let mut collected = String::new();
        let mut ttft_ms = 0u64;
        let mut stats = Stats::default();

        while let Some(chunk) = chunks.next().await {
            if cancel.load(Ordering::Acquire) {
                // Dropping the response closes the connection, which is what
                // actually stops the server generating.
                return Ok(Outcome::Cancelled(collected.chars().count()));
            }

            let chunk =
                chunk.map_err(|error| Rejected::Other(format!("the stream broke ({error})")))?;
            pending.extend_from_slice(&chunk);

            // Ollama writes one JSON object per line, but a chunk boundary can
            // fall anywhere, including mid-object and mid-UTF-8-sequence.
            while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = pending.drain(..=newline).collect();
                let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&line) else {
                    continue;
                };

                if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
                    return Err(Rejected::Other(error.to_owned()));
                }

                if let Some(piece) = parsed.get("response").and_then(|r| r.as_str()) {
                    if !piece.is_empty() {
                        if collected.is_empty() {
                            ttft_ms = started.elapsed().as_millis() as u64;
                        }
                        collected.push_str(piece);
                        report(piece);
                    }
                }

                if parsed.get("done").and_then(|d| d.as_bool()) == Some(true) {
                    stats.absorb(&parsed);
                }
            }
        }

        let total_ms = started.elapsed().as_millis() as u64;

        Ok(Outcome::Complete(Complete {
            text: collected.trim().to_owned(),
            ttft_ms,
            total_ms,
            prompt_tokens: stats.prompt_tokens,
            output_tokens: stats.output_tokens,
            tokens_per_second: stats.rate(),
        }))
    })
}

/// The counters Ollama reports on the final line.
#[derive(Default)]
struct Stats {
    prompt_tokens: u64,
    output_tokens: u64,
    /// Nanoseconds spent generating, which is what the rate should be measured
    /// against — not wall clock, which includes loading the model.
    eval_ns: u64,
}

impl Stats {
    fn absorb(&mut self, parsed: &serde_json::Value) {
        let read = |key: &str| parsed.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
        self.prompt_tokens = read("prompt_eval_count");
        self.output_tokens = read("eval_count");
        self.eval_ns = read("eval_duration");
    }

    fn rate(&self) -> f64 {
        if self.eval_ns == 0 {
            return 0.0;
        }
        self.output_tokens as f64 / (self.eval_ns as f64 / 1_000_000_000.0)
    }
}

fn unreachable(endpoint: &str, error: &reqwest::Error) -> String {
    if error.is_connect() {
        format!("no Ollama server on {endpoint}. Start one with: ollama serve")
    } else if error.is_timeout() {
        format!("Ollama on {endpoint} did not answer in time")
    } else {
        format!("could not reach Ollama on {endpoint} ({error})")
    }
}

fn emit<R: Runtime, P: Serialize + Clone>(app: &AppHandle<R>, event: &str, payload: P) {
    let _ = app.emit(event, payload);
}
