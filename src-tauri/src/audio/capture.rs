//! Opening the input device and feeding its audio into a buffer.
//!
//! WASAPI hands us whatever the device is configured for, which is essentially
//! never 16 kHz. Rate conversion happens once at the end, in `resample`; here
//! we only downmix to mono so the buffer stays one channel wide.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use super::{AudioError, CLIP_THRESHOLD, MAX_DURATION};

/// What the device actually gave us.
#[derive(Clone, Debug)]
pub struct InputConfig {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
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

/// Opens the default input device at its native configuration and starts
/// capturing. `on_error` runs on cpal's own thread when the stream fails, for
/// instance when the device is unplugged mid-recording.
pub fn start<E>(on_error: E) -> Result<Capture, AudioError>
where
    E: FnMut(cpal::Error) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(AudioError::NoInputDevice)?;

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

    let device_name = device
        .description()
        .map(|d| d.name().to_owned())
        .unwrap_or_else(|_| device.to_string());

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
