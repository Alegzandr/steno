//! WAV output. 16 kHz mono 16-bit PCM, in the OS temp dir and nowhere else.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hound::{SampleFormat, WavSpec, WavWriter};

use super::{AudioError, TARGET_RATE};

/// Full scale for 16-bit PCM. 32767 rather than 32768: with 32768 a sample of
/// exactly +1.0 lands one past `i16::MAX` and wraps to `i16::MIN`, turning the
/// loudest peaks into full-amplitude clicks. Scaling by 32767 keeps the mapping
/// symmetric at the cost of one unused code at the bottom of the range.
const FULL_SCALE: f32 = 32_767.0;

fn spec() -> WavSpec {
    WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

/// Converts to 16-bit PCM and writes to `path`.
fn write(path: &Path, samples: &[f32]) -> Result<(), AudioError> {
    let mut writer = WavWriter::create(path, spec())?;

    for &sample in samples {
        // Clamp first: resampling can overshoot slightly past full scale
        // around sharp transients, and `as i16` on an out-of-range float
        // saturates silently rather than wrapping, which would hide it.
        let scaled = sample.clamp(-1.0, 1.0) * FULL_SCALE;
        writer.write_sample(scaled.round() as i16)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Writes the clip into the OS temp dir and returns its path.
pub fn write_temp(samples: &[f32]) -> Result<PathBuf, AudioError> {
    let path = std::env::temp_dir().join(file_name());
    write(&path, samples)?;
    Ok(path)
}

/// Process id plus nanoseconds: unique enough that two recordings, or two
/// running copies of Steno, never collide on the same file.
fn file_name() -> String {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    format!("steno-{}-{stamp}.wav", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(samples: &[f32]) -> Vec<i16> {
        let path = std::env::temp_dir().join(format!("test-{}", file_name()));
        write(&path, samples).expect("write");

        let reader = hound::WavReader::open(&path).expect("open");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, TARGET_RATE);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, SampleFormat::Int);

        let read = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .expect("samples");

        std::fs::remove_file(&path).expect("cleanup");
        read
    }

    #[test]
    fn clamps_instead_of_wrapping() {
        // Without the clamp, +1.5 scaled by 32768 wraps to a large negative
        // value: the loudest part of a clip would come back as a click.
        let read = roundtrip(&[1.5, -1.5, 1.0, -1.0, 0.0]);
        assert_eq!(read, vec![32_767, -32_767, 32_767, -32_767, 0]);
    }

    #[test]
    fn scales_by_full_scale() {
        let read = roundtrip(&[0.5, -0.5]);
        assert_eq!(read, vec![16_384, -16_384]);
    }

    #[test]
    fn writes_a_readable_header_for_an_empty_clip() {
        assert!(roundtrip(&[]).is_empty());
    }
}
