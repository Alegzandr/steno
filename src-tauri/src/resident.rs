//! A resource that is loaded on demand, held while it is in use, and handed
//! back the moment it is not.
//!
//! Both things Steno loads onto the GPU — the Whisper context and the
//! formatting model — have the same shape: expensive to load, worth warming
//! ahead of time, and unacceptable to keep resident while the user is doing
//! something else with their graphics card. This is that shape, once.
//!
//! It is deliberately *not* `audio::capture::WarmUp`. That gate means "the
//! microphone is free again": it carries no value, timing out on it is
//! harmless because you can just open the device anyway, and it latches once.
//! Here the gate means "the resource is loaded", there is no equivalent of
//! opening it anyway, the gate must hand the loaded value to the caller, and
//! the whole cycle repeats every time the window is hidden and shown.
//!
//! Releasing is `Drop` on `T`, not a callback. An eviction can therefore never
//! half-happen. Values are always dropped outside the lock, because freeing
//! nine gigabytes of video memory is not instantaneous and holding the mutex
//! across it would stall every acquire.
//!
//! There is deliberately no `warm` method that loads on a thread of its own.
//! Steno's two resources have to be warmed *in a known order* — see
//! `lifecycle::warm_in_order` — and a fire-and-forget loader per resource is
//! precisely the concurrent arrangement that ordering exists to avoid. Warming
//! is `acquire` on a background thread, with the lease dropped immediately.

use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex};

use serde::Serialize;

use crate::audio::lock;

/// What the UI shows next to a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResidentState {
    Cold,
    Loading,
    Ready,
    Failed,
}

enum Slot<T> {
    Cold,
    Loading,
    /// `leases` counts callers currently using the value. An eviction waits for
    /// it to reach zero rather than pulling the value out from under them.
    Ready { value: Arc<T>, leases: usize },
    Failed(String),
}

pub struct Resident<T> {
    /// Names the resource in log lines and thread names.
    label: &'static str,
    slot: Mutex<Slot<T>>,
    changed: Condvar,
}

impl<T> Resident<T> {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            slot: Mutex::new(Slot::Cold),
            changed: Condvar::new(),
        }
    }

    pub fn state(&self) -> ResidentState {
        match &*lock(&self.slot) {
            Slot::Cold => ResidentState::Cold,
            Slot::Loading => ResidentState::Loading,
            Slot::Ready { .. } => ResidentState::Ready,
            Slot::Failed(_) => ResidentState::Failed,
        }
    }

    /// Whether a load is already done or under way. Lets a caller skip building
    /// the closure at all when there is nothing to do.
    pub fn is_warm(&self) -> bool {
        matches!(&*lock(&self.slot), Slot::Loading | Slot::Ready { .. })
    }

    /// Borrows the value, loading it on this thread if it is cold and waiting
    /// if another thread is already loading it.
    ///
    /// There is no timeout. Unlike the microphone gate there is no degraded
    /// path: without the Whisper context there is no transcription, so waiting
    /// is the only correct behaviour. Callers must therefore never be on the
    /// event loop.
    ///
    /// A failed load is reported once and resets the resource to cold, so the
    /// next attempt retries rather than staying poisoned — the model file may
    /// well have finished downloading in the meantime.
    pub fn acquire<L>(&self, load: L) -> Result<Lease<'_, T>, String>
    where
        L: FnOnce() -> Result<T, String>,
    {
        let mut load = Some(load);
        let mut slot = lock(&self.slot);

        loop {
            match &mut *slot {
                Slot::Ready { value, leases } => {
                    *leases += 1;
                    return Ok(Lease {
                        resident: self,
                        value: Some(value.clone()),
                    });
                }
                Slot::Loading => {
                    slot = self
                        .changed
                        .wait(slot)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                Slot::Failed(message) => {
                    let message = std::mem::take(message);
                    *slot = Slot::Cold;
                    self.changed.notify_all();
                    return Err(message);
                }
                Slot::Cold => {
                    let Some(load) = load.take() else {
                        // Evicted between our load finishing and us re-taking
                        // the lock. Rare, and not worth a second load: the
                        // caller retries.
                        return Err(format!("{} was unloaded while it was loading", self.label));
                    };

                    *slot = Slot::Loading;
                    drop(slot);
                    self.settle(load());
                    slot = lock(&self.slot);
                }
            }
        }
    }

    /// Records the outcome of a load and wakes everyone waiting on it.
    ///
    /// The one way out of `Loading`. Every path that sets that state must reach
    /// here, including the failure paths: a slot left saying `Loading` with
    /// nothing loading it is a permanent hang for the next `acquire`.
    fn settle(&self, outcome: Result<T, String>) {
        let mut slot = lock(&self.slot);

        *slot = match outcome {
            Ok(value) => Slot::Ready {
                value: Arc::new(value),
                leases: 0,
            },
            Err(message) => {
                eprintln!("{}: load failed ({message})", self.label);
                Slot::Failed(message)
            }
        };

        self.changed.notify_all();
    }

    /// Releases the value, waiting for any in-flight user to finish first.
    ///
    /// Blocks, and must never be called by a thread holding a lease on the
    /// same resource: that is a deadlock. The call sites — hiding the window,
    /// quitting, the idle timer — hold none.
    pub fn evict(&self) {
        let mut slot = lock(&self.slot);

        let value = loop {
            match &*slot {
                Slot::Cold => return,
                Slot::Failed(_) => {
                    *slot = Slot::Cold;
                    self.changed.notify_all();
                    return;
                }
                // Do not race the loader: let it finish, then unload what it
                // produced. Interrupting it would leave the slot lying about
                // what is resident.
                Slot::Loading => {
                    slot = self
                        .changed
                        .wait(slot)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                Slot::Ready { leases: 0, .. } => {
                    match std::mem::replace(&mut *slot, Slot::Cold) {
                        Slot::Ready { value, .. } => break value,
                        _ => unreachable!("just matched Ready"),
                    }
                }
                Slot::Ready { .. } => {
                    slot = self
                        .changed
                        .wait(slot)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            }
        };

        self.changed.notify_all();
        drop(slot);

        // Outside the lock. `T::drop` is where the resource is actually given
        // back, and for the formatting model that is a blocking HTTP round
        // trip; holding the mutex across it would stall every acquire.
        drop(value);
        eprintln!("{}: unloaded", self.label);
    }
}

/// A live borrow of a resident value. The resource cannot be evicted while one
/// of these exists.
pub struct Lease<'a, T> {
    resident: &'a Resident<T>,
    /// Always `Some` until `Drop` takes it.
    value: Option<Arc<T>>,
}

impl<T> Deref for Lease<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.as_ref().expect("a lease holds its value until it is dropped")
    }
}

impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        // Release our reference *before* announcing it, so that an evict woken
        // by the notification below holds the last one and runs `T::drop` on
        // its own thread rather than on ours.
        drop(self.value.take());

        let mut slot = lock(&self.resident.slot);
        if let Slot::Ready { leases, .. } = &mut *slot {
            *leases = leases.saturating_sub(1);
        }
        self.resident.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    /// Counts its own drops, standing in for a resource whose release matters.
    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn acquire_loads_once_and_reuses() {
        let loads = Arc::new(AtomicUsize::new(0));
        let resident = Resident::new("test");

        for _ in 0..3 {
            let loads = loads.clone();
            let lease = resident
                .acquire(move || {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(7u32)
                })
                .unwrap_or_else(|error| panic!("load should have succeeded: {error}"));
            assert_eq!(*lease, 7);
        }

        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failed_load_is_retried_by_the_next_caller() {
        let resident = Resident::new("test");

        match resident.acquire(|| Err::<u32, _>("no model file".to_owned())) {
            Ok(_) => panic!("a failing load must not produce a lease"),
            Err(message) => assert_eq!(message, "no model file"),
        }
        assert_eq!(resident.state(), ResidentState::Cold);

        let second = resident
            .acquire(|| Ok(1u32))
            .unwrap_or_else(|error| panic!("the retry should have succeeded: {error}"));
        assert_eq!(*second, 1);
    }

    #[test]
    fn evict_drops_the_value_and_returns_to_cold() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resident = Resident::new("test");

        {
            let drops = drops.clone();
            let _lease = resident
                .acquire(move || Ok(Tracked(drops)))
                .unwrap_or_else(|error| panic!("load should have succeeded: {error}"));
        }

        assert_eq!(drops.load(Ordering::SeqCst), 0, "still resident");
        resident.evict();
        assert_eq!(drops.load(Ordering::SeqCst), 1, "released on evict");
        assert_eq!(resident.state(), ResidentState::Cold);

        // Idempotent: hiding the window twice must not be an error.
        resident.evict();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn evict_waits_for_an_in_flight_lease() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resident = Arc::new(Resident::new("test"));

        let lease_drops = drops.clone();
        let holder = resident.clone();
        let observed = Arc::new(AtomicUsize::new(usize::MAX));

        let seen = observed.clone();
        let worker = thread::spawn(move || {
            let lease = holder
                .acquire(move || Ok(Tracked(lease_drops)))
                .unwrap_or_else(|error| panic!("load should have succeeded: {error}"));
            // Long enough that the evict below is genuinely blocked on us.
            thread::sleep(Duration::from_millis(150));
            seen.store(lease.0.load(Ordering::SeqCst), Ordering::SeqCst);
        });

        // Let the worker take its lease first.
        while !resident.is_warm() {
            thread::sleep(Duration::from_millis(1));
        }

        resident.evict();
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "evict returned before the value was released"
        );

        worker.join().expect("worker finished");
        assert_eq!(
            observed.load(Ordering::SeqCst),
            0,
            "the value was dropped while a lease still pointed at it"
        );
    }

    /// The warm-up path is a background thread calling `acquire` and dropping
    /// the lease at once — see `lifecycle::warm_in_order`. What that relies on
    /// is that a second caller arriving mid-load waits for the first rather
    /// than starting its own, which is what this pins.
    #[test]
    fn a_second_caller_waits_for_the_load_in_progress() {
        let resident = Arc::new(Resident::new("test"));
        let loads = Arc::new(AtomicUsize::new(0));

        let background = {
            let resident = resident.clone();
            let loads = loads.clone();
            thread::spawn(move || {
                let lease = resident
                    .acquire(move || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(50));
                        Ok(42u32)
                    })
                    .unwrap_or_else(|error| panic!("load should have succeeded: {error}"));
                drop(lease);
            })
        };

        // Let the loader claim the slot before racing it.
        while !resident.is_warm() {
            thread::sleep(Duration::from_millis(1));
        }

        let lease = resident
            .acquire(|| Ok(0u32))
            .unwrap_or_else(|error| panic!("acquire should have waited: {error}"));
        assert_eq!(*lease, 42, "acquire returned before the load finished");
        assert_eq!(loads.load(Ordering::SeqCst), 1, "the value was loaded twice");

        background.join().expect("loader finished");
    }
}
