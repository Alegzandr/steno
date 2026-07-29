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
//! The matching load happens on window show, in the background, and it is
//! **sequential: Whisper first, then the formatting model**. That ordering is
//! measured, not stylistic. On a cold page cache — the realistic case, since
//! the machine has been doing something else — a window show has to pull about
//! twelve gigabytes off disk, and disk throughput is the binding constraint for
//! both engines. Racing them does not make the total shorter: measured on the
//! dev machine, 13.1 s sequential against 13.4 s concurrent. What it changes is
//! who waits. Whisper alone has a deadline the user can miss, twenty to ninety
//! seconds out when push-to-talk is released; the formatting model is not
//! needed until Clean up is clicked, which is later and unpredictable. Run
//! concurrently, Whisper became ready at 13.3 s instead of 5.1 s — eight
//! seconds of risk bought for nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::audio::lock;
use crate::config::Config;
use crate::format::model::{Loaded, Params};
use crate::format::Formatter;
use crate::model;
use crate::resident::ResidentState;
use crate::storage;
use crate::transcribe::engine::Engine;
use crate::transcribe::Whisper;

pub const RESOURCE_STATE: &str = "resource-state";

/// Emitted at most once per run, and only when there is something wrong worth
/// interrupting for. Not a status channel: silence is the normal case.
pub const STORAGE_ADVISORY: &str = "storage-advisory";

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

/// The window came on screen. Starts the warm-up in the background and returns
/// immediately: this runs on the event loop.
pub fn on_show<R: Runtime>(app: &AppHandle<R>) {
    let activity = app.state::<Arc<Activity>>().inner().clone();
    activity.shown.store(true, Ordering::Release);
    activity.touch();

    // One thread, not two. See the module comment: the ordering is the point,
    // and two independent warm-up threads would be exactly the concurrent case
    // that costs Whisper eight seconds on a cold cache.
    let app = app.clone();
    let spawned = thread::Builder::new()
        .name("steno-warm".to_owned())
        .spawn(move || warm_in_order(&app));

    if let Err(error) = spawned {
        eprintln!("lifecycle: could not spawn the warm-up thread ({error})");
    }
}

/// Whisper, then the formatting model, on this thread.
///
/// `acquire` is what loads: it runs the closure when the slot is cold, waits
/// when another thread is already loading, and hands back a lease. The lease is
/// dropped immediately — nothing here wants to *use* the model, only to have it
/// resident — which leaves the slot Ready and evictable.
fn warm_in_order<R: Runtime>(app: &AppHandle<R>) {
    // Nothing to warm when the GPU runtime is not there. Whisper would refuse
    // anyway — `Engine::load` asks the same question — but llama.cpp would not:
    // ggml simply skips the CUDA module it cannot open and loads nine gigabytes
    // onto the CPU instead, which is minutes of work for a cleanup the user is
    // about to be told cannot run. Both are reported failed so the status bar
    // agrees with the panel the window is showing.
    if let Some(blocker) = crate::gpu::blocker() {
        for resource in ["whisper", "llm"] {
            emit(app, resource, ResidentState::Failed, Some(blocker.one_line()));
        }
        return;
    }

    let whisper = app.state::<Whisper>().inner().clone();
    if let Some(loader) = whisper_loader(app) {
        // The error is already reported to the UI by the loader itself, and a
        // Whisper that will not load must not stop the formatting model from
        // warming: dictation is broken either way, cleanup need not be.
        let _ = whisper.acquire(loader);
    }

    if !app.state::<Config>().get().vram.warm_llm_on_show {
        return;
    }

    let formatter = app.state::<Formatter>().inner().clone();
    let _ = formatter.acquire(formatter_loader(app));
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
}

fn evict_all<R: Runtime>(app: &AppHandle<R>, why: &str) {
    eprintln!("lifecycle: releasing video memory because {why}");

    // Whisper first: it is the smaller half and it is quicker to give back.
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

/// The closure that loads the Whisper context, or `None` when there is nothing
/// to load yet.
///
/// `None` rather than a failing closure for the pre-download case: reporting it
/// as cold rather than failed keeps the UI pointing at the download button
/// instead of showing an error the user cannot act on differently.
fn whisper_loader<R: Runtime>(
    app: &AppHandle<R>,
) -> Option<impl FnOnce() -> Result<Engine, String>> {
    let spec = model::resolve(app);
    let path = model::path(app, spec);

    if !path.exists() {
        emit(
            app,
            "whisper",
            ResidentState::Cold,
            Some(format!("{} is not downloaded yet", spec.id)),
        );
        return None;
    }

    let handle = app.clone();
    Some(move || {
        emit(&handle, "whisper", ResidentState::Loading, None);
        let outcome = Engine::load(&path, spec.id);
        match &outcome {
            Ok(engine) => emit(
                &handle,
                "whisper",
                ResidentState::Ready,
                Some(format!(
                    "{} on {} in {} ms",
                    engine.model_id, engine.backend, engine.load_ms
                )),
            ),
            Err(error) => emit(&handle, "whisper", ResidentState::Failed, Some(error.clone())),
        }
        outcome
    })
}

/// The closure that brings the formatting model into video memory, and reports
/// its progress to the UI as it goes.
///
/// Shared by the warm-on-show path and by a cleanup that finds the model cold —
/// pressing Clean up thirty seconds after the idle watcher fired has to load it
/// exactly the same way. Two copies of this would be two chances to diverge on
/// the load parameters, and one of those parameters is `mmap`, whose wrong
/// value costs a factor of fifty on a mechanical drive without failing.
pub fn formatter_loader<R: Runtime>(app: &AppHandle<R>) -> impl FnOnce() -> Result<Loaded, String> {
    let settings = app.state::<Config>().get();
    let path = crate::format::model_path(app);
    let n_gpu_layers = settings.llm.n_gpu_layers;
    let handle = app.clone();

    move || {
        emit(&handle, "llm", ResidentState::Loading, None);

        let outcome = Loaded::load(Params { path, n_gpu_layers });

        match &outcome {
            Ok(loaded) => {
                // The first real model read is also the only throughput sample
                // Steno ever gets for free, so the storage advisory is settled
                // here rather than by a synthetic benchmark at startup.
                report_storage(&handle, loaded);
                emit(
                    &handle,
                    "llm",
                    ResidentState::Ready,
                    Some(format!("{} ms", loaded.load_ms)),
                );
            }
            Err(error) => emit(&handle, "llm", ResidentState::Failed, Some(error.clone())),
        }
        outcome
    }
}

/// Warns, at most once per run, when the models are on a spinning disk.
///
/// Two sources, and the order is deliberate. The drive is asked first because a
/// seek-penalty flag is a property of the device; the observed read rate is the
/// fallback, because a warm page cache makes a mechanical drive look fast and
/// would let a real problem go unreported on the second load. Asking the device
/// has no such blind spot.
fn report_storage<R: Runtime>(app: &AppHandle<R>, loaded: &Loaded) {
    static REPORTED: std::sync::Once = std::sync::Once::new();

    let directory = loaded.path.parent().unwrap_or(&loaded.path).to_path_buf();
    let (bytes, elapsed) = loaded.read;

    REPORTED.call_once(|| {
        let media = match storage::media_type(&directory) {
            storage::Media::Unknown => {
                let observed = storage::classify_throughput(bytes, elapsed);
                eprintln!(
                    "storage: the drive would not say what it is; judging by the read instead \
                     ({observed:?})"
                );
                observed
            }
            known => {
                eprintln!("storage: {} reports {known:?}", directory.display());
                known
            }
        };

        if let Some(message) = storage::advisory(media) {
            eprintln!("storage: {message}");
            let _ = app.emit(STORAGE_ADVISORY, message);
        }
    });
}

/// Tells the window what is resident, and the tray with it.
///
/// The single funnel for residency changes, which is why the tray is refreshed
/// from here rather than from each of the four call sites: what the status bar
/// shows and what the badge shows are then the same fact, reported once.
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

    crate::tray::report(app, resource, state);
}
