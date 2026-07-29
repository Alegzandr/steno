//! Streaming a whole-buffer cleanup through the local model.
//!
//! The buffer is an accumulation of dictated bursts, so a cleanup is one long
//! request over everything the user has said so far, not an incremental edit.
//! It streams because a two-thousand-token rewrite takes tens of seconds and
//! watching it arrive is the difference between "working" and "hung".
//!
//! Four things this owes the rest of the app:
//!
//! - **It never hangs silently.** Every failure path ends in a `cleanup-error`
//!   carrying a sentence the user can act on.
//! - **It holds a lease for the whole stream.** Hiding the window mid-cleanup
//!   must not pull the model out from under a request in flight; `Resident`
//!   waits for the lease, and the lease lives until the last token.
//! - **It emits deltas, and the full text again at the end.** The frontend
//!   needs the deltas to render progress and the whole string to apply the
//!   single undo transaction. Rebuilding it from the deltas would work right up
//!   until one gets dropped.
//! - **The context dies with the request.** The KV cache is video memory, it is
//!   cheap to rebuild, and holding one between cleanups would break the promise
//!   that Steno occupies nothing while it is idle.
//!
//! Before 5.1 this was an HTTP stream and the loop below was Ollama's problem.
//! What replaced it is a token loop, and two details of it are load-bearing:
//! partial UTF-8 must not be emitted (a token is bytes, not a character), and
//! qwen3's reasoning block must not reach the buffer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage};
use llama_cpp_2::sampling::LlamaSampler;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::model::Loaded;
use super::Formatter;
use crate::config::Config;
use crate::lifecycle;

pub const STARTED: &str = "cleanup-started";
pub const DELTA: &str = "cleanup-delta";
pub const COMPLETE: &str = "cleanup-complete";
pub const FAILED: &str = "cleanup-error";
pub const CANCELLED: &str = "cleanup-cancelled";

/// Appended to the user turn to stop qwen3 reasoning before it answers.
///
/// A cleanup is a mechanical rewrite, not a problem to think about: the trace
/// is unwanted in the buffer and it is most of the tokens. This is qwen3's own
/// switch. It is not sufficient on its own — the model still emits an empty
/// `<think></think>` pair — which is what `ThinkFilter` is for.
const NO_THINK: &str = "/no_think";

/// Scratch space for detokenising one token. Comfortably above the longest
/// single piece any byte-level BPE vocabulary produces; the call fails rather
/// than truncates if it is ever not.
const TOKEN_BUFFER: usize = 256;

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
    /// A remedy, when there is one. Since 5.1 this is no longer a command to
    /// type — there is no server to start — but a build with no compute backend
    /// still has an answer worth giving.
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

    // The badge follows `running`, at both ends. Set here rather than inside the
    // thread so a cleanup that is waiting on a nine-gigabyte model load already
    // shows as working.
    let handle = app.clone();
    crate::tray::refresh(&handle);

    let worker = cleanup.clone();
    let spawned = std::thread::Builder::new()
        .name("steno-cleanup".to_owned())
        .spawn({
            let handle = handle.clone();
            move || {
                run(&app, text, worker.clone());
                worker.running.store(false, Ordering::Release);
                crate::tray::refresh(&handle);
            }
        });

    if let Err(error) = spawned {
        cleanup.running.store(false, Ordering::Release);
        crate::tray::refresh(&handle);
        return Err(format!("could not start the cleanup thread ({error})"));
    }

    Ok(())
}

fn run<R: Runtime>(app: &AppHandle<R>, text: String, cleanup: Arc<Cleanup>) {
    lifecycle::touch(app);

    let settings = app.state::<Config>().get();
    if let Some(name) = settings.llm.unknown_prompt() {
        eprintln!(
            "cleanup: llm.prompt is \"{name}\", which is not in llm.prompts; using \
             \"{}\" instead",
            crate::config::PROMPT_FAITHFUL
        );
    }
    let request = Request::from_settings(&settings);
    let model_path = super::model_path(app);

    // Acquiring can block for seconds on a cold model, so the UI is told what
    // is happening before it starts rather than after.
    let formatter = app.state::<Formatter>().inner().clone();
    let model_cold = !formatter.is_warm();

    emit(
        app,
        STARTED,
        Started {
            model: settings.llm.model_file.clone(),
            input_chars: text.chars().count(),
            model_cold,
        },
    );

    let lease = match formatter.acquire(lifecycle::formatter_loader(app)) {
        Ok(lease) => lease,
        Err(message) => {
            let remedy = super::model::availability(&model_path).remedy;
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

    match transfer(&lease, &request, &text, &cleanup.cancel, &report) {
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
            let remedy = super::model::availability(&model_path).remedy;
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

/// What a cleanup needs from settings, flattened so the generator takes one
/// argument instead of reaching back into `Settings`.
pub struct Request {
    pub system_prompt: String,
    pub temperature: f32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub max_output_tokens: u32,
}

impl Request {
    /// The one place a cleanup request is derived from settings, so the app and
    /// the measurement harness cannot disagree about the prompt or the
    /// temperature they are testing.
    pub fn from_settings(settings: &crate::config::Settings) -> Self {
        Self {
            system_prompt: settings.llm.system_prompt().to_owned(),
            temperature: settings.llm.temperature,
            n_ctx: settings.llm.n_ctx,
            n_batch: settings.llm.n_batch,
            max_output_tokens: settings.llm.max_output_tokens,
        }
    }
}

/// Prefill, then generate, reporting each piece of text as it appears.
///
/// Cancellation is checked once per token rather than per batch: the user
/// pressed Esc and the only honest response is to stop at the next opportunity,
/// which for a decode loop is the next iteration.
pub fn transfer(
    loaded: &Loaded,
    request: &Request,
    text: &str,
    cancel: &AtomicBool,
    report: &(dyn Fn(&str) + Send + Sync),
) -> Result<Outcome, String> {
    let prompt = build_prompt(loaded, &request.system_prompt, text)?;

    let tokens = loaded
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| format!("could not tokenise the prompt ({error})"))?;
    let prompt_tokens = tokens.len();

    // Refuse rather than truncate. A cleanup that silently dropped the first
    // half of the buffer would look like a formatting decision, not a failure,
    // and the user would paste it.
    let needed = prompt_tokens as u32 + request.max_output_tokens;
    if needed > request.n_ctx {
        return Err(format!(
            "the buffer is too long for the context window: {prompt_tokens} tokens of input \
             plus room for {} of output exceeds {}. Raise llm.nCtx in settings.json, or clean \
             up in two passes.",
            request.max_output_tokens, request.n_ctx
        ));
    }

    let started = Instant::now();
    let mut context = loaded.context(request.n_ctx, request.n_batch)?;

    let batch_capacity = request.n_batch.max(1) as usize;
    let mut batch = LlamaBatch::new(batch_capacity, 1);

    // Prefill in n_batch-sized chunks. Only the very last token needs logits:
    // asking for them on every position costs a vocab-sized row per token for
    // nothing, and this vocabulary is 151936 wide.
    let last_index = prompt_tokens - 1;
    for (chunk_start, chunk) in tokens.chunks(batch_capacity).enumerate().map(|(i, c)| (i * batch_capacity, c)) {
        batch.clear();
        for (offset, token) in chunk.iter().enumerate() {
            let position = chunk_start + offset;
            batch
                .add(*token, position as i32, &[0], position == last_index)
                .map_err(|error| format!("could not build the prompt batch ({error})"))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| format!("the model failed while reading the prompt ({error})"))?;
    }

    let mut sampler = sampler_for(request.temperature);
    let mut filter = ThinkFilter::default();
    let mut decoder = Utf8Stream::default();

    let mut collected = String::new();
    let mut ttft_ms = 0u64;
    let mut output_tokens = 0u64;
    let mut position = prompt_tokens as i32;
    let mut first_token_at: Option<Instant> = None;

    while output_tokens < u64::from(request.max_output_tokens) {
        if cancel.load(Ordering::Acquire) {
            return Ok(Outcome::Cancelled(collected.chars().count()));
        }

        let token = sampler.sample(&context, -1);
        if loaded.model.is_eog_token(token) {
            break;
        }
        sampler.accept(token);
        output_tokens += 1;

        // `special: false` — control tokens are the model's business, not the
        // buffer's. The one that would otherwise reach the editor is the
        // end-of-turn marker, and it is already handled above.
        let bytes = loaded
            .model
            .token_to_piece_bytes(token, TOKEN_BUFFER, false, None)
            .map_err(|error| format!("could not decode a generated token ({error})"))?;

        if let Some(piece) = decoder.push(&bytes) {
            if let Some(visible) = filter.push(&piece) {
                if !visible.is_empty() {
                    if collected.is_empty() {
                        ttft_ms = started.elapsed().as_millis() as u64;
                        first_token_at = Some(Instant::now());
                    }
                    collected.push_str(&visible);
                    report(&visible);
                }
            }
        }

        batch.clear();
        batch
            .add(token, position, &[0], true)
            .map_err(|error| format!("could not extend the batch ({error})"))?;
        position += 1;

        context
            .decode(&mut batch)
            .map_err(|error| format!("the model failed while generating ({error})"))?;
    }

    let total_ms = started.elapsed().as_millis() as u64;

    // Rate over generation only, not wall clock: the prefill is not generation
    // and including it would report a number that shrinks as the buffer grows.
    let tokens_per_second = first_token_at
        .map(|at| {
            let seconds = at.elapsed().as_secs_f64();
            if seconds > 0.0 {
                output_tokens as f64 / seconds
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    Ok(Outcome::Complete(Complete {
        text: collected.trim().to_owned(),
        ttft_ms,
        total_ms,
        prompt_tokens: prompt_tokens as u64,
        output_tokens,
        tokens_per_second,
    }))
}

/// Greedy at zero, sampled above it.
///
/// Not a style preference: a cleanup is a rewrite of the user's own words, and
/// the default temperature is 0. `LlamaSampler::temp(0.0)` is not the same
/// thing as greedy decoding, so the zero case is spelled out rather than left
/// to the sampler chain to interpret.
fn sampler_for(temperature: f32) -> LlamaSampler {
    if temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }

    LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        // Seeded from the clock: two cleanups of the same buffer at a non-zero
        // temperature are meant to differ, which is the only reason to raise it.
        LlamaSampler::dist(seed()),
    ])
}

fn seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}

/// Wraps the system prompt and the buffer in whatever chat format the GGUF
/// declares.
///
/// Read from the model rather than written here. Steno's model file is
/// user-editable, and a hand-rolled qwen3 template would produce plausible
/// nonsense the moment somebody pointed it at a different GGUF.
fn build_prompt(loaded: &Loaded, system_prompt: &str, text: &str) -> Result<String, String> {
    let template = loaded
        .model
        .chat_template(None)
        .map_err(|error| format!("this model file declares no chat template ({error})"))?;

    let messages = vec![
        LlamaChatMessage::new("system".to_owned(), system_prompt.to_owned())
            .map_err(|error| format!("invalid system prompt ({error})"))?,
        LlamaChatMessage::new("user".to_owned(), format!("{text}\n\n{NO_THINK}"))
            .map_err(|error| format!("the buffer could not be sent to the model ({error})"))?,
    ];

    loaded
        .model
        .apply_chat_template(&template, &messages, true)
        .map_err(|error| format!("could not apply the model's chat template ({error})"))
}

/// Reassembles UTF-8 across token boundaries.
///
/// A token is a sequence of bytes, and a multi-byte character can straddle two
/// of them — routine in French, where every accented letter is two bytes.
/// Emitting a token's bytes as they arrive therefore produces replacement
/// characters in the editor at random intervals. This holds back an incomplete
/// tail until the bytes that finish it turn up.
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);

        let complete_up_to = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(error) => error.valid_up_to(),
        };

        if complete_up_to == 0 {
            return None;
        }

        let head: Vec<u8> = self.pending.drain(..complete_up_to).collect();
        String::from_utf8(head).ok()
    }
}

/// Keeps qwen3's reasoning block out of the buffer.
///
/// `/no_think` stops the model reasoning but not from emitting the tags around
/// the empty result, and a stray `<think></think>` at the head of a cleaned
/// buffer is a bug the user has to delete by hand. This suppresses everything
/// up to and including the first `</think>`, and only when the output opens
/// with `<think>` — so a dictation that genuinely contains the word survives,
/// because it will not be at position zero inside a tag.
///
/// It gives up if the block does not close within a bounded prefix. A model
/// that reasons for two thousand tokens despite `/no_think` is a model whose
/// output the user should see rather than watch be swallowed.
#[derive(Default)]
struct ThinkFilter {
    buffer: String,
    decided: Decision,
}

#[derive(Default, PartialEq)]
enum Decision {
    #[default]
    Undecided,
    Suppressing,
    /// The block has closed but nothing has been emitted yet. The newlines
    /// qwen3 puts after `</think>` arrive as their own tokens, so trimming only
    /// within the piece that contained the closing tag leaves a blank line at
    /// the head of the buffer.
    Trimming,
    PassingThrough,
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// Enough for the empty block plus whitespace, and far short of real reasoning.
const GIVE_UP_AFTER: usize = 512;

impl ThinkFilter {
    fn push(&mut self, piece: &str) -> Option<String> {
        match self.decided {
            Decision::PassingThrough => return Some(piece.to_owned()),
            Decision::Trimming => {
                let trimmed = piece.trim_start();
                if trimmed.is_empty() {
                    return None;
                }
                self.decided = Decision::PassingThrough;
                return Some(trimmed.to_owned());
            }
            Decision::Suppressing | Decision::Undecided => {}
        }

        self.buffer.push_str(piece);

        if self.decided == Decision::Undecided {
            let head = self.buffer.trim_start();

            if !head.is_empty() && !OPEN.starts_with(&head[..head.len().min(OPEN.len())]) {
                // Whatever this is, it is not the opening tag.
                self.decided = Decision::PassingThrough;
                return Some(std::mem::take(&mut self.buffer));
            }

            if head.len() < OPEN.len() {
                return None;
            }
            self.decided = Decision::Suppressing;
        }

        if let Some(end) = self.buffer.find(CLOSE) {
            let rest = self.buffer[end + CLOSE.len()..].trim_start().to_owned();
            self.buffer.clear();
            self.decided = if rest.is_empty() {
                Decision::Trimming
            } else {
                Decision::PassingThrough
            };
            return (!rest.is_empty()).then_some(rest);
        }

        if self.buffer.len() > GIVE_UP_AFTER {
            self.decided = Decision::PassingThrough;
            return Some(std::mem::take(&mut self.buffer));
        }

        None
    }
}

fn emit<R: Runtime, P: Serialize + Clone>(app: &AppHandle<R>, event: &str, payload: P) {
    let _ = app.emit(event, payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(filter: &mut ThinkFilter, pieces: &[&str]) -> String {
        pieces
            .iter()
            .filter_map(|piece| filter.push(piece))
            .collect()
    }

    #[test]
    fn an_empty_think_block_is_removed() {
        let mut filter = ThinkFilter::default();
        let out = drive(&mut filter, &["<think>", "\n\n", "</think>", "\n\n", "## Titre"]);
        assert_eq!(out, "## Titre");
    }

    /// The tags do not arrive whole: they are several tokens.
    #[test]
    fn the_opening_tag_may_be_split_across_tokens() {
        let mut filter = ThinkFilter::default();
        let out = drive(&mut filter, &["<", "th", "ink", ">", "x", "</think>", "bonjour"]);
        assert_eq!(out, "bonjour");
    }

    #[test]
    fn ordinary_output_passes_through_untouched() {
        let mut filter = ThinkFilter::default();
        let out = drive(&mut filter, &["## ", "Objectif", "\n\n- un"]);
        assert_eq!(out, "## Objectif\n\n- un");
    }

    /// The word itself must survive: this is a dictation app for developers and
    /// "think" is a word they say.
    #[test]
    fn the_word_think_in_the_body_is_not_a_tag() {
        let mut filter = ThinkFilter::default();
        let out = drive(&mut filter, &["Le ", "<think> ", "tag ", "existe"]);
        assert_eq!(out, "Le <think> tag existe");
    }

    #[test]
    fn an_unclosed_block_is_eventually_released() {
        let mut filter = ThinkFilter::default();
        let long = "a".repeat(GIVE_UP_AFTER + 1);
        let out = drive(&mut filter, &["<think>", &long]);
        assert!(out.starts_with("<think>"), "the output was swallowed");
        assert!(out.len() > GIVE_UP_AFTER);
    }

    #[test]
    fn split_utf8_is_held_until_it_is_complete() {
        let mut stream = Utf8Stream::default();
        // "é" is 0xC3 0xA9, arriving as two tokens.
        assert_eq!(stream.push(&[0xC3]), None);
        assert_eq!(stream.push(&[0xA9]).as_deref(), Some("é"));
    }

    #[test]
    fn complete_text_passes_straight_through() {
        let mut stream = Utf8Stream::default();
        assert_eq!(stream.push("bonjour".as_bytes()).as_deref(), Some("bonjour"));
    }

    #[test]
    fn a_partial_tail_does_not_hold_back_the_head() {
        let mut stream = Utf8Stream::default();
        let mut bytes = "ok".as_bytes().to_vec();
        bytes.push(0xC3);
        assert_eq!(stream.push(&bytes).as_deref(), Some("ok"));
        assert_eq!(stream.push(&[0xA9]).as_deref(), Some("é"));
    }
}
