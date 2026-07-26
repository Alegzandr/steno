//! Opening the input device and feeding its audio into a buffer.
//!
//! WASAPI hands us whatever the device is configured for, which is essentially
//! never 16 kHz. Rate conversion happens once at the end, in `resample`; here
//! we only downmix to mono so the buffer stays one channel wide.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use serde::Serialize;

use super::{lock, AudioError, CLIP_THRESHOLD, MAX_DURATION};

/// What the device actually gave us.
#[derive(Clone, Debug)]
pub struct InputConfig {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// One row in the microphone dropdown: the name the user picks by, and whether
/// it is the system default.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InputDevice {
    pub name: String,
    pub is_default: bool,
}

/// The available input devices, with the system default flagged. Names are the
/// only handle the UI has; a device whose description cannot be read (it was
/// unplugged between listing and querying) falls back to its `Display` string.
pub fn enumerate() -> Result<Vec<InputDevice>, AudioError> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().map(|d| describe(&d));

    let mut devices = Vec::new();
    for device in host.input_devices()? {
        let name = describe(&device);
        let is_default = default_name.as_deref() == Some(name.as_str());
        devices.push(InputDevice { name, is_default });
    }

    Ok(devices)
}

/// The saved device if it is still present, otherwise the system default.
/// Matching is by the same name the dropdown shows, so a device that was
/// unplugged since it was chosen simply is not found and we fall through.
fn select_device(host: &cpal::Host, preferred: Option<&str>) -> Option<cpal::Device> {
    if let Some(wanted) = preferred {
        if let Ok(devices) = host.input_devices() {
            if let Some(device) = devices.into_iter().find(|d| describe(d) == wanted) {
                return Some(device);
            }
        }
    }

    host.default_input_device()
}

/// The device name as the user sees it, with a `Display` fallback for a device
/// that has gone away since it was enumerated.
fn describe(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_owned())
        .unwrap_or_else(|_| device.to_string())
}

/// A one-shot gate, opened once the startup warm-up has released the device.
///
/// The warm-up and a push-to-talk session both open the same endpoint, and
/// nothing else connects them: the recorder's `Slot` serialises sessions
/// against each other, not against the warm-up, and the shortcut is armed
/// before the warm-up thread is even spawned. Waiting on this gate from the
/// session thread keeps the two opens apart. It is deliberately not a lock
/// around the device: a lock would be held for the whole dictation, up to
/// `MAX_DURATION`, and turn a harmless overlap into a two-minute block.
pub struct WarmUp {
    /// Fast path, so every recording after the first pays nothing.
    open: AtomicBool,
    latch: Mutex<bool>,
    opened: Condvar,
}

impl Default for WarmUp {
    fn default() -> Self {
        Self::new()
    }
}

impl WarmUp {
    pub fn new() -> Self {
        Self {
            open: AtomicBool::new(false),
            latch: Mutex::new(false),
            opened: Condvar::new(),
        }
    }

    /// Blocks until the warm-up is done, or `ceiling` elapses. Returns how long
    /// it waited. Timing out is not an error: WASAPI opens the endpoint in
    /// shared mode, so a concurrent open degrades to extra latency, never to a
    /// failure, and a stuck warm-up must not disable push-to-talk.
    pub fn wait(&self, ceiling: Duration) -> Duration {
        let started = Instant::now();

        if self.open.load(Ordering::Acquire) {
            return Duration::ZERO;
        }

        let (_latch, timeout) = self
            .opened
            .wait_timeout_while(lock(&self.latch), ceiling, |open| !*open)
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if timeout.timed_out() {
            eprintln!(
                "warm-up: still holding the device after {} ms, opening anyway",
                ceiling.as_millis()
            );
        }

        started.elapsed()
    }

    /// Releases every waiter. Idempotent.
    fn release(&self) {
        *lock(&self.latch) = true;
        self.open.store(true, Ordering::Release);
        self.opened.notify_all();
    }
}

/// Opens the gate however `warm_up` ends, including on a panic or an early
/// return. A gate left shut would cost every recording the full ceiling.
struct ReleaseOnDrop(Arc<WarmUp>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Opens the input device once and immediately closes it, paying the driver's
/// first-open cost (~350 ms cold on this machine) at startup rather than on the
/// user's first recording. Silent: the measured cost and any failure go to
/// stderr, never to the UI. Uses the same device the next recording will, so
/// it warms the driver that matters. Opens `gate` on the way out, whatever
/// happens, so a session waiting to record is released as soon as the device
/// is free.
pub fn warm_up(preferred: Option<String>, gate: Arc<WarmUp>) {
    let _release = ReleaseOnDrop(gate);

    let started = Instant::now();
    match start(preferred.as_deref(), |_| {}) {
        Ok(capture) => {
            drop(capture.stream);
            eprintln!(
                "warm-up: opened and closed the input device in {} ms",
                started.elapsed().as_millis()
            );
        }
        Err(error) => eprintln!("warm-up: skipped, could not open the input device ({error})"),
    }
}

/// A running capture. Dropping `stream` stops the device.
pub struct Capture {
    pub stream: Stream,
    pub config: InputConfig,
    /// Mono f32 at `config.sample_rate`, appended to by the audio callback.
    pub samples: Arc<Mutex<Vec<f32>>>,
    /// Set on the first callback: the device is delivering audio now.
    pub live: Arc<AtomicBool>,
    /// Latched once any sample reaches `CLIP_THRESHOLD`.
    pub clipped: Arc<AtomicBool>,
}

/// Opens an input device at its native configuration and starts capturing.
/// `preferred` is the saved device name; when it is `None`, or names a device
/// that is not currently present, the system default is used instead — the
/// saved name is a hint, never a hard requirement, so unplugging the chosen
/// mic degrades to the default rather than failing. `on_error` runs on cpal's
/// own thread when the stream fails, for instance when the device is unplugged
/// mid-recording.
pub fn start<E>(preferred: Option<&str>, on_error: E) -> Result<Capture, AudioError>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    let host = cpal::default_host();
    let device = select_device(&host, preferred).ok_or(AudioError::NoInputDevice)?;

    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = config.channels;

    // Sized for the hard cap so the callback never allocates and never
    // reallocates, whatever the device rate turns out to be.
    let capacity = MAX_DURATION.as_secs() as usize * sample_rate as usize;
    let samples = Arc::new(Mutex::new(Vec::with_capacity(capacity)));
    let live = Arc::new(AtomicBool::new(false));
    let clipped = Arc::new(AtomicBool::new(false));

    let stream = build(
        &device,
        &config,
        sample_format,
        Sinks {
            samples: samples.clone(),
            live: live.clone(),
            clipped: clipped.clone(),
        },
        on_error,
    )?;
    stream.play()?;

    let device_name = describe(&device);

    Ok(Capture {
        stream,
        config: InputConfig {
            device_name,
            sample_rate,
            channels,
        },
        samples,
        live,
        clipped,
    })
}

/// The shared state the audio callback writes to.
#[derive(Clone)]
struct Sinks {
    samples: Arc<Mutex<Vec<f32>>>,
    live: Arc<AtomicBool>,
    clipped: Arc<AtomicBool>,
}

fn build<E>(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    sinks: Sinks,
    on_error: E,
) -> Result<Stream, AudioError>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    let channels = config.channels;

    macro_rules! typed {
        ($t:ty) => {
            device.build_input_stream(
                config.clone(),
                callback::<$t>(sinks, channels),
                on_error,
                None,
            )
        };
    }

    // Only one arm runs, so moving `sinks` and `on_error` in each is fine.
    let stream = match format {
        SampleFormat::I8 => typed!(i8),
        SampleFormat::I16 => typed!(i16),
        SampleFormat::I32 => typed!(i32),
        SampleFormat::I64 => typed!(i64),
        SampleFormat::U8 => typed!(u8),
        SampleFormat::U16 => typed!(u16),
        SampleFormat::U32 => typed!(u32),
        SampleFormat::U64 => typed!(u64),
        SampleFormat::F32 => typed!(f32),
        SampleFormat::F64 => typed!(f64),
        other => return Err(AudioError::SampleFormat(other)),
    }?;

    Ok(stream)
}

/// The audio callback. No allocation, no I/O, no event emission: it downmixes
/// into the preallocated buffer and sets two flags. Level metering and
/// everything else happen on the session thread.
fn callback<T>(
    sinks: Sinks,
    channels: u16,
) -> impl FnMut(&[T], &cpal::InputCallbackInfo) + Send + 'static
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = channels as usize;
    let scale = 1.0 / channels as f32;

    move |data, _info| {
        sinks.live.store(true, Ordering::Relaxed);

        let Ok(mut buf) = sinks.samples.lock() else {
            return;
        };

        // Never grow past the preallocated capacity: the session thread stops
        // us at MAX_DURATION anyway, this only covers the few milliseconds of
        // overlap.
        let room = buf.capacity() - buf.len();
        let mut clipped = false;

        for frame in data.chunks_exact(channels).take(room) {
            let mut sum = 0.0;
            for &sample in frame {
                let value = f32::from_sample_(sample);
                sum += value;
                clipped |= value.abs() >= CLIP_THRESHOLD;
            }
            buf.push(sum * scale);
        }

        if clipped {
            sinks.clipped.store(true, Ordering::Relaxed);
        }
    }
}
