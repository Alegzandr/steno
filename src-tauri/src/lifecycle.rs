//! When the two expensive resources are loaded, and when they are given back.
//!
//! Steno runs on machines whose GPU has another job. Holding a Whisper context
//! and a fourteen-billion-parameter model resident between dictations would
//! occupy most of a 16 GB card, and Windows does not refuse an oversubscribed
//! allocation — it spills to system memory and turns into permanent stutter
//! that gets blamed on the graphics driver. So: zero video memory whenever
//! Steno is not actively working.
//!
//! Three triggers release it. Hiding the window, quitting, and going idle. The
//! last one matters most in practice, because Steno is meant to be left open on
//! a second screen: "hidden" and "not in use" are not the same state, and only
//! the second one is the promise.
//!
//! The matching load happens on window show, in the background. By the time
//! push-to-talk is released Whisper has had the whole dictation to load, and
//! the formatting model has until the Clean up button is pressed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::audio::lock;
use crate::config::Config;
use crate::format::model::Loaded;
use crate::format::server::Server;
use crate::format::Formatter;
use crate::model;
use crate::resident::ResidentState;
use crate::transcribe::engine::Engine;
use crate::transcribe::Whisper;

pub const RESOURCE_STATE: &str = "resource-state";

/// How often the idle watcher looks at the clock. Coarse on purpose: the
/// threshold is measured in minutes, and a thread waking four times a minute
/// costs nothing.
const IDLE_TICK: Duration = Duration::from_secs(15);

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceState {
    /// `whisper` or `llm`.
    pub resource: &'static str,
    pub state: ResidentState,
    pub message: Option<String>,
}

/// The Ollama server handle, created once and dropped exactly once.
///
/// Managed separately from the model: the server may be one Steno started, in
/// which case ending it is our responsibility, or one that was already running,
/// in which case touching it is not.
#[derive(Default)]
pub struct Ollama {
    server: Mutex<Option<Server>>,
}

impl Ollama {
    /// Makes sure a server exists, and reports whether `model` was already
    /// resident on it before we arrived.
    ///
    /// Blocking: probes the port, and may start a process and wait for it.
    fn ensure(
        &self,
        endpoint: &str,
        model: &str,
        models_dir: Option<&str>,
    ) -> Result<bool, String> {
        let mut slot = lock(&self.server);

        // A changed endpoint in settings.json means the old handle is stale.
        if slot.as_ref().is_some_and(|s| s.endpoint != endpoint) {
            *slot = None;
        }

        let server = slot.get_or_insert_with(|| Server::ensure(endpoint, models_dir));

        if !server.is_reachable() {
            return Err(format!(
                "no Ollama server on {endpoint} and none could be started. \
                 Start one with: ollama serve"
            ));
        }

        Ok(server.is_foreign(model))
    }

    /// Ends a server Steno started. Leaves an adopted one alone.
    fn shutdown(&self) {
        drop(lock(&self.server).take());
    }
}

/// Tracks whether the window is on screen and when Steno last did anything.
pub struct Activity {
    shown: AtomicBool,
    last: Mutex<Instant>,
}

impl Default for Activity {
    fn default() -> Self {
        Self {
            shown: AtomicBool::new(false),
            last: Mutex::new(Instant::now()),
        }
    }
}

impl Activity {
    fn touch(&self) {
        *lock(&self.last) = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        lock(&self.last).elapsed()
    }
}

/// Records that the user did something. Resets the idle countdown, so a long
/// dictation session never has the models pulled out from under it.
pub fn touch<R: Runtime>(app: &AppHandle<R>) {
    app.state::<Arc<Activity>>().touch();
}

/// The window came on screen. Starts both loads in the background and returns
/// immediately: this runs on the event loop.
pub fn on_show<R: Runtime>(app: &AppHandle<R>) {
    let activity = app.state::<Arc<Activity>>().inner().clone();
    activity.shown.store(true, Ordering::Release);
    activity.touch();

    warm_whisper(app);
    warm_formatter(app);
}

/// The window went off screen. Releases both, on a thread: eviction waits for
/// anything in flight to finish, and the event loop cannot wait for that.
pub fn on_hide<R: Runtime>(app: &AppHandle<R>) {
    app.state::<Arc<Activity>>()
        .shown
        .store(false, Ordering::Release);

    let app = app.clone();
    let spawned = thread::Builder::new()
        .name("steno-evict".to_owned())
        .spawn(move || evict_all(&app, "the window was hidden"));

    if let Err(error) = spawned {
        eprintln!("lifecycle: could not spawn the eviction thread ({error})");
    }
}

/// Steno is quitting. Synchronous, because after this the process is gone and
/// a thread would not finish.
pub fn on_exit<R: Runtime>(app: &AppHandle<R>) {
    evict_all(app, "Steno is quitting");
    app.state::<Arc<Ollama>>().shutdown();
}

fn evict_all<R: Runtime>(app: &AppHandle<R>, why: &str) {
    eprintln!("lifecycle: releasing video memory because {why}");

    // Whisper first: it is ours, it is quick, and it frees the smaller half
    // before the network round trip for the other.
    app.state::<Whisper>().evict();
    emit(app, "whisper", ResidentState::Cold, None);

    app.state::<Formatter>().evict();
    emit(app, "llm", ResidentState::Cold, None);
}

/// Releases both resources after a spell of doing nothing, even with the window
/// still on screen.
pub fn watch_idle<R: Runtime>(app: AppHandle<R>) {
    let spawned = thread::Builder::new()
        .name("steno-idle".to_owned())
        .spawn(move || loop {
            thread::sleep(IDLE_TICK);

            let threshold = app.state::<Config>().get().vram.idle_unload_seconds;
            if threshold == 0 {
                continue;
            }

            let activity = app.state::<Arc<Activity>>();
            if !activity.shown.load(Ordering::Acquire) {
                // Hiding already evicted; nothing to do until it is shown again.
                continue;
            }

            let idle = activity.idle_for();
            if idle < Duration::from_secs(threshold) {
                continue;
            }

            let nothing_resident =
                !app.state::<Whisper>().is_warm() && !app.state::<Formatter>().is_warm();
            if nothing_resident {
                continue;
            }

            evict_all(&app, &format!("nothing has happened for {}s", idle.as_secs()));

            // Without this the watcher would fire again on the next tick and
            // log an eviction with nothing to evict.
            activity.touch();
        });

    if let Err(error) = spawned {
        eprintln!("lifecycle: could not spawn the idle watcher ({error})");
    }
}

fn warm_whisper<R: Runtime>(app: &AppHandle<R>) {
    let whisper = app.state::<Whisper>().inner().clone();
    if whisper.is_warm() {
        return;
    }

    let spec = model::resolve(app);
    let path = model::path(app, spec);

    // Nothing to warm before the first download finishes. Reporting it as cold
    // rather than failed keeps the UI pointing at the download button.
    if !path.exists() {
        emit(app, "whisper", ResidentState::Cold, Some(format!("{} is not downloaded yet", spec.id)));
        return;
    }

    emit(app, "whisper", ResidentState::Loading, None);
    let handle = app.clone();

    whisper.warm(move || {
        let outcome = Engine::load(&path, spec.id);
        match &outcome {
            Ok(engine) => emit(
                &handle,
                "whisper",
                ResidentState::Ready,
                Some(format!("{} on {} in {} ms", engine.model_id, engine.backend, engine.load_ms)),
            ),
            Err(error) => emit(&handle, "whisper", ResidentState::Failed, Some(error.clone())),
        }
        outcome
    });
}

fn warm_formatter<R: Runtime>(app: &AppHandle<R>) {
    if !app.state::<Config>().get().vram.warm_llm_on_show {
        return;
    }

    let formatter = app.state::<Formatter>().inner().clone();
    if formatter.is_warm() {
        return;
    }

    formatter.warm(formatter_loader(app));
}

/// The closure that brings the formatting model into memory, and reports its
/// progress to the UI as it goes.
///
/// Shared by the warm-on-show path and by a cleanup that finds the model cold —
/// pressing Clean up thirty seconds after the idle watcher fired has to load it
/// exactly the same way, including starting a server if there is none. Two
/// copies of this would be two chances to diverge on which one adopts a foreign
/// model, and adopting wrongly means unloading somebody else's weights.
pub fn formatter_loader<R: Runtime>(app: &AppHandle<R>) -> impl FnOnce() -> Result<Loaded, String> {
    let settings = app.state::<Config>().get();
    let ollama = app.state::<Arc<Ollama>>().inner().clone();
    let endpoint = settings.ollama.endpoint;
    let name = settings.ollama.model;
    let models_dir = settings.ollama.models_dir;
    let keep_alive = settings.vram.keep_alive;
    let handle = app.clone();

    move || {
        emit(&handle, "llm", ResidentState::Loading, None);

        // Starting the server belongs inside the closure: it can take seconds,
        // and the closure always runs on a background thread.
        let outcome = ollama
            .ensure(&endpoint, &name, models_dir.as_deref())
            .and_then(|foreign| Loaded::warm(&endpoint, &name, &keep_alive, foreign));

        match &outcome {
            Ok(loaded) => emit(
                &handle,
                "llm",
                ResidentState::Ready,
                Some(format!("{} in {} ms", loaded.model, loaded.load_ms)),
            ),
            Err(error) => emit(&handle, "llm", ResidentState::Failed, Some(error.clone())),
        }
        outcome
    }
}

fn emit<R: Runtime>(
    app: &AppHandle<R>,
    resource: &'static str,
    state: ResidentState,
    message: Option<String>,
) {
    let _ = app.emit(
        RESOURCE_STATE,
        ResourceState {
            resource,
            state,
            message,
        },
    );
}
