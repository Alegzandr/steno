//! What kind of drive the models are sitting on.
//!
//! This is a performance question with a fifty-fold answer and no visible
//! symptom. Measured on the dev machine, the same 9.3 GB GGUF loads in 1.1 s
//! from the NVMe and 56.4 s from a 7200 rpm SATA disk, because the load is
//! bounded by sequential read throughput — 1288 MB/s against 162 MB/s — and
//! nothing else. Neither number is visible to the user, and relocating an app
//! data directory onto the big spare drive is an ordinary thing to do. So Steno
//! looks, once, and says so.
//!
//! Two routes, in order of confidence:
//!
//! 1. Ask the device. `IOCTL_STORAGE_QUERY_PROPERTY` with
//!    `StorageDeviceSeekPenaltyProperty` returns a flag the driver sets, which
//!    is exactly the distinction that matters here.
//! 2. Watch the clock. The seek-penalty descriptor is optional and comes back
//!    empty behind some RAID controllers, USB bridges and network redirectors,
//!    and a guess dressed up as a fact is worse than no answer. When the device
//!    will not say, `classify_throughput` judges the first real model read
//!    instead — an observation about this machine, not an inference about it.

use std::path::Path;
use std::time::Duration;

/// Below this, a sequential read is not coming off flash. The gap it sits in is
/// wide: measured spinning throughput here was 162 MB/s and measured NVMe
/// throughput 1288 MB/s, and even an early SATA SSD clears 400 MB/s. It is set
/// nearer the slow end so that a busy or thermally throttled SSD is called
/// unknown rather than mislabelled.
const SPINNING_CEILING_MB_S: f64 = 250.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Media {
    /// No seek penalty: flash.
    Solid,
    /// Seek penalty: a platter, and a first model load measured in tens of
    /// seconds.
    Spinning,
    /// The device declined to say. Not a failure — see `classify_throughput`.
    Unknown,
}

/// Judges a completed read rather than asking the device about itself.
///
/// The fallback when `media_type` returns `Unknown`. Deliberately refuses to
/// call anything solid: a fast read proves the read was fast, which is all
/// Steno needs to stay quiet, and returning `Unknown` there keeps the warning
/// as the only claim this module ever makes out loud.
pub fn classify_throughput(bytes: u64, elapsed: Duration) -> Media {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 || bytes == 0 {
        return Media::Unknown;
    }

    let mb_s = (bytes as f64 / 1_000_000.0) / seconds;
    if mb_s < SPINNING_CEILING_MB_S {
        Media::Spinning
    } else {
        Media::Unknown
    }
}

/// The sentence the user sees, or `None` when there is nothing worth saying.
///
/// It carries the measured consequence rather than a caution, because "your
/// models are on a slow drive" invites the reasonable reply "so what". A minute
/// against a few seconds is a decision the user can act on.
pub fn advisory(media: Media) -> Option<String> {
    match media {
        Media::Spinning => Some(
            "The models are on a mechanical drive. The first load after starting \
             Steno will take about a minute instead of a few seconds. Moving the \
             models directory to an SSD fixes it."
                .to_owned(),
        ),
        Media::Solid | Media::Unknown => None,
    }
}

/// Free space on the volume holding `path`, in bytes.
///
/// `None` means the question could not be answered, which is not the same as
/// zero and must never be shown as "no space": a download refused on the
/// strength of a failed query is worse than one that runs out of disk and says
/// so. The nearest existing ancestor is asked, because the directory a download
/// is about to create does not exist yet.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    free_on_volume(existing)
}

#[cfg(target_os = "windows")]
fn free_on_volume(directory: &Path) -> Option<u64> {
    windows_impl::free_bytes(directory)
}

#[cfg(not(target_os = "windows"))]
fn free_on_volume(_directory: &Path) -> Option<u64> {
    None
}

#[cfg(target_os = "windows")]
pub fn media_type(path: &Path) -> Media {
    windows_impl::seek_penalty(path)
}

#[cfg(not(target_os = "windows"))]
pub fn media_type(_path: &Path) -> Media {
    // Portable on paper, unverified in fact. MVP verification is Windows only,
    // so rather than write a Linux path nobody has run, this defers to the
    // throughput fallback, which needs no platform support at all.
    Media::Unknown
}

#[cfg(target_os = "windows")]
mod windows_impl {
    use super::Media;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Storage::FileSystem::{
        CreateFileW, GetDiskFreeSpaceExW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE,
        OPEN_EXISTING,
    };
    use windows::Win32::System::Ioctl::{
        DEVICE_SEEK_PENALTY_DESCRIPTOR, IOCTL_STORAGE_QUERY_PROPERTY, PropertyStandardQuery,
        STORAGE_PROPERTY_QUERY, StorageDeviceSeekPenaltyProperty,
    };
    use windows::Win32::System::IO::DeviceIoControl;

    /// The caller's own free space, not the volume's.
    ///
    /// Under a disk quota those differ, and the one that decides whether a
    /// 391 MB download completes is the caller's.
    pub fn free_bytes(directory: &Path) -> Option<u64> {
        let wide: Vec<u16> = directory
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut available = 0u64;
        // SAFETY: the buffer is NUL-terminated and outlives the call.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                PCWSTR(wide.as_ptr()),
                Some(&mut available),
                None,
                None,
            )
        };

        ok.ok().map(|()| available)
    }

    pub fn seek_penalty(path: &Path) -> Media {
        let Some(volume) = device_path(path) else {
            return Media::Unknown;
        };

        // Zero desired access on purpose. Metadata IOCTLs are answered without
        // it, and asking for read access on a volume needs administrator rights
        // — which would turn "we could not tell" into "we could not tell,
        // unless elevated", for no gain.
        let handle = unsafe {
            CreateFileW(
                PCWSTR(volume.as_ptr()),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                None,
            )
        };

        let Ok(handle) = handle else {
            return Media::Unknown;
        };

        let media = query(handle);
        unsafe {
            let _ = CloseHandle(handle);
        }
        media
    }

    fn query(handle: HANDLE) -> Media {
        let request = STORAGE_PROPERTY_QUERY {
            PropertyId: StorageDeviceSeekPenaltyProperty,
            QueryType: PropertyStandardQuery,
            AdditionalParameters: [0],
        };
        let mut descriptor = DEVICE_SEEK_PENALTY_DESCRIPTOR::default();
        let mut returned = 0u32;

        let ok = unsafe {
            DeviceIoControl(
                handle,
                IOCTL_STORAGE_QUERY_PROPERTY,
                Some(std::ptr::addr_of!(request).cast()),
                std::mem::size_of::<STORAGE_PROPERTY_QUERY>() as u32,
                Some(std::ptr::addr_of_mut!(descriptor).cast()),
                std::mem::size_of::<DEVICE_SEEK_PENALTY_DESCRIPTOR>() as u32,
                Some(&mut returned),
                None,
            )
        };

        // A short reply is not a seek-penalty answer. Some drivers succeed and
        // fill in only the header, and reading `IncursSeekPenalty` out of that
        // would be reading uninitialised memory and calling it a measurement.
        if ok.is_err() || (returned as usize) < std::mem::size_of::<DEVICE_SEEK_PENALTY_DESCRIPTOR>()
        {
            return Media::Unknown;
        }

        match descriptor.IncursSeekPenalty {
            true => Media::Spinning,
            false => Media::Solid,
        }
    }

    /// `C:\Users\…\models` becomes `\\.\C:`, NUL-terminated and wide.
    ///
    /// Returns `None` for anything that is not a plain drive letter — a UNC
    /// share has no local device to interrogate, and the throughput fallback is
    /// the honest answer for it anyway.
    fn device_path(path: &Path) -> Option<Vec<u16>> {
        let text = path.to_str()?;
        let mut chars = text.chars();
        let letter = chars.next()?;
        if chars.next() != Some(':') || !letter.is_ascii_alphabetic() {
            return None;
        }

        Some(
            format!(r"\\.\{}:", letter.to_ascii_uppercase())
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn a_slow_read_is_called_spinning() {
        // 9.3 GB in 56 s: the measured HDD case.
        assert_eq!(
            classify_throughput(9_276_184_896, Duration::from_secs(56)),
            Media::Spinning
        );
    }

    #[test]
    fn a_fast_read_makes_no_claim() {
        // 9.3 GB in 7.2 s: the measured NVMe case. Fast enough to stay quiet,
        // not evidence of what the device is.
        assert_eq!(
            classify_throughput(9_276_184_896, Duration::from_millis(7_200)),
            Media::Unknown
        );
    }

    #[test]
    fn a_zero_length_read_is_not_evidence() {
        assert_eq!(classify_throughput(0, Duration::from_secs(1)), Media::Unknown);
        assert_eq!(classify_throughput(1000, Duration::ZERO), Media::Unknown);
    }

    /// Asks the real drives, and prints what they said.
    ///
    /// Ignored because the answer is a property of the machine, not of the
    /// code: there is nothing to assert that would hold on another one. It
    /// exists so the question "did the device actually answer, or is the
    /// throughput fallback doing the work?" can be settled by running something
    /// rather than by reasoning about it.
    ///
    /// `cargo test --lib -- --ignored --nocapture storage`
    #[test]
    #[ignore = "reports this machine's drives; nothing to assert"]
    fn report_this_machines_drives() {
        for letter in ["C", "D", "F"] {
            let path = PathBuf::from(format!(r"{letter}:\"));
            if !path.exists() {
                continue;
            }
            println!("{letter}: {:?}", media_type(&path));
        }
    }

    #[test]
    fn only_spinning_says_anything() {
        assert!(advisory(Media::Spinning).is_some());
        assert!(advisory(Media::Solid).is_none());
        assert!(advisory(Media::Unknown).is_none());
    }
}
