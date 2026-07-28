//! Loading and unloading the formatting model, and proving the video memory
//! actually came back.
//!
//! Ollama has no unload call. What it has is `keep_alive`, and a request
//! carrying `keep_alive: 0` releases the model as soon as it finishes. That is
//! the mechanism. It is not, on its own, evidence: the HTTP response returns
//! before the runner subprocess holding the memory has exited, so a check that
//! reads `nvidia-smi` the instant the call returns will report the old figure
//! and conclude, wrongly, that nothing was freed.
//!
//! So the unload is confirmed rather than assumed: `/api/ps` is polled until
//! the model is gone from the server's own inventory.

use std::time::{Duration, Instant};

use serde::Serialize;

use super::server;

/// Loading nine gigabytes off a cold disk is not fast, and a timeout here
/// would look exactly like a hang.
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(15);

/// How long to keep polling `/api/ps` before reporting that the model outlived
/// its unload.
const RELEASE_CEILING: Duration = Duration::from_secs(10);
const RELEASE_POLL: Duration = Duration::from_millis(100);

/// A claim on a model resident in the Ollama process.
///
/// The weights are not in this process, so this holds no memory of its own.
/// What it holds is the obligation to give them back, discharged in `Drop`.
pub struct Loaded {
    pub endpoint: String,
    pub model: String,
    pub load_ms: u64,
    /// The model was already resident when Steno adopted a running server, so
    /// it belongs to somebody else and must survive our eviction.
    foreign: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Availability {
    pub reachable: bool,
    pub model_installed: bool,
    pub installed_models: Vec<String>,
    /// The exact command to run, when there is one that would help.
    pub remedy: Option<String>,
}

impl Loaded {
    /// Asks Ollama to bring the model into memory.
    ///
    /// Blocking, and slow on a cold start. Runs on a warm-up thread.
    pub fn warm(
        endpoint: &str,
        model: &str,
        keep_alive: &str,
        foreign: bool,
    ) -> Result<Self, String> {
        let started = Instant::now();
        let url = generate_url(endpoint);
        let body = serde_json::json!({
            "model": model,
            "prompt": "",
            "stream": false,
            // Never -1. A finite value is what returns the memory if Steno
            // dies without running the unload below.
            "keep_alive": keep_alive,
        });

        let name = model.to_owned();
        let where_to = endpoint.to_owned();

        server::blocking(async move {
            let client = reqwest::Client::builder()
                .timeout(LOAD_TIMEOUT)
                .build()
                .map_err(|error| format!("could not build an HTTP client ({error})"))?;

            let response = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|error| unreachable_message(&where_to, &error))?;

            if response.status().is_success() {
                return Ok(());
            }

            let status = response.status();
            let detail = response.text().await.unwrap_or_default();

            // The one failure the user can fix in one command, so name it.
            if status == reqwest::StatusCode::NOT_FOUND || detail.contains("not found") {
                // The second sentence is not padding. If Steno started the
                // server itself and the user has moved Ollama's model store,
                // this fires for a model that is installed, and the obvious
                // remedy wastes a nine-gigabyte download to no effect.
                return Err(format!(
                    "the server on {where_to} has no model {name}. If `ollama list` shows it, \
                     Steno started its own server against the wrong model directory — set \
                     ollama.modelsDir in settings.json. Otherwise run: ollama pull {name}"
                ));
            }

            Err(format!("Ollama answered {status} ({})", detail.trim()))
        })?;

        let load_ms = started.elapsed().as_millis() as u64;
        eprintln!("ollama: {model} resident after {load_ms} ms (keep_alive {keep_alive})");

        Ok(Self {
            endpoint: endpoint.to_owned(),
            model: model.to_owned(),
            load_ms,
            foreign,
        })
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        if self.foreign {
            eprintln!(
                "ollama: leaving {} loaded, it was resident before Steno started",
                self.model
            );
            return;
        }

        let started = Instant::now();

        if let Err(error) = request_unload(&self.endpoint, &self.model) {
            eprintln!("ollama: the unload request failed ({error})");
            return;
        }

        match wait_until_released(&self.endpoint, &self.model) {
            true => eprintln!(
                "ollama: {} released after {} ms",
                self.model,
                started.elapsed().as_millis()
            ),
            false => eprintln!(
                "ollama: {} was still resident {} s after the unload request",
                self.model,
                RELEASE_CEILING.as_secs()
            ),
        }
    }
}

/// The unload itself: a request that asks for zero seconds of keep-alive.
fn request_unload(endpoint: &str, model: &str) -> Result<(), String> {
    let url = generate_url(endpoint);
    let body = serde_json::json!({
        "model": model,
        "keep_alive": 0,
    });

    server::blocking(async move {
        let client = reqwest::Client::builder()
            .timeout(UNLOAD_TIMEOUT)
            .build()
            .map_err(|error| format!("could not build an HTTP client ({error})"))?;

        client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("{error}"))?;

        Ok(())
    })
}

/// Polls the server's own inventory until the model is gone from it.
///
/// This is the part that turns "we sent the request" into "the memory is
/// back": Ollama drops the model from `/api/ps` when the runner holding it has
/// exited, which is the same moment the video memory is released.
fn wait_until_released(endpoint: &str, model: &str) -> bool {
    let deadline = Instant::now() + RELEASE_CEILING;

    while Instant::now() < deadline {
        if !server::model_names(endpoint).iter().any(|name| name == model) {
            return true;
        }
        std::thread::sleep(RELEASE_POLL);
    }

    false
}

/// Whether a cleanup could run right now, and what to type if it could not.
pub fn availability(endpoint: &str, model: &str) -> Availability {
    if !server::probe(endpoint) {
        return Availability {
            reachable: false,
            model_installed: false,
            installed_models: Vec::new(),
            remedy: Some("ollama serve".to_owned()),
        };
    }

    // A server that stopped answering between the probe and here is a
    // transient, not an empty machine: say so rather than inventing a `pull`.
    let Some(installed_models) = server::catalogue(endpoint) else {
        return Availability {
            reachable: true,
            model_installed: false,
            installed_models: Vec::new(),
            remedy: None,
        };
    };
    // Ollama reports `qwen3:14b`; a user may well have written `qwen3` in
    // settings.json and pulled the same thing.
    let model_installed = installed_models
        .iter()
        .any(|name| name == model || name.split(':').next() == Some(model));

    Availability {
        reachable: true,
        model_installed,
        installed_models,
        remedy: (!model_installed).then(|| format!("ollama pull {model}")),
    }
}

fn generate_url(endpoint: &str) -> String {
    format!("{}/api/generate", endpoint.trim_end_matches('/'))
}

/// Connection errors are the common case and deserve a message that says what
/// to do, not a wrapped `hyper` string.
fn unreachable_message(endpoint: &str, error: &reqwest::Error) -> String {
    if error.is_connect() {
        format!("no Ollama server on {endpoint}. Start one with: ollama serve")
    } else if error.is_timeout() {
        format!("Ollama on {endpoint} did not answer in time")
    } else {
        format!("could not reach Ollama on {endpoint} ({error})")
    }
}
