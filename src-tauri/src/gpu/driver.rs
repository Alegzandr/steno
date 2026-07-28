//! Which CUDA version the installed NVIDIA driver can actually run.
//!
//! cuBLAS is not self-contained: it is a user-mode library that talks to the
//! CUDA driver inside the display driver, and the pinned archive is CUDA 13. A
//! machine on an older driver would download 391 MB and then fail to load them,
//! which is the one failure worth spending a millisecond to avoid.
//!
//! **The version table is deliberately not used.** CUDA 13.x wants driver r580
//! or newer, and a lookup from "581.42" to "CUDA 13" is a table that goes stale
//! and a string that is not always a number. NVML answers the question directly
//! instead: `nvmlSystemGetCudaDriverVersion_v2` reports the highest CUDA version
//! the installed driver supports, encoded as `major * 1000 + minor * 10`. That
//! is the driver's own statement about itself, and it already accounts for the
//! minor-version compatibility that makes any r580+ driver run any CUDA 13.x.
//!
//! `nvml.dll` ships with the *display driver*, not the toolkit — it is in
//! `System32` on this machine, which has no toolkit on `PATH` — so this works on
//! exactly the machines it needs to. It is opened with `LoadLibrary` rather than
//! linked, because a machine with no NVIDIA driver at all must get an answer of
//! "cannot tell" rather than a process that will not start.
//!
//! **Unknown is not "too old".** Every failure here — no NVML, an NVML that
//! errors, a version that does not parse — yields `None`, and `None` never
//! blocks the download. Refusing to fetch cuBLAS because a query failed would
//! turn a diagnostic into an outage.

use serde::Serialize;

/// What the installed driver says about itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Driver {
    /// As NVML prints it, e.g. `"610.74"`. Shown, never parsed.
    pub version: String,
    /// Highest CUDA major version this driver supports.
    pub cuda_major: u32,
    pub cuda_minor: u32,
}

impl Driver {
    /// Whether this driver can run a cuBLAS built for CUDA `major`.
    ///
    /// A newer driver runs an older CUDA — that direction is guaranteed and is
    /// the ordinary case. The comparison is one-sided for that reason.
    pub fn supports(&self, major: u32) -> bool {
        self.cuda_major >= major
    }

    /// Why the download would be wasted, said with both numbers in it.
    pub fn too_old_for(&self, major: u32) -> String {
        format!(
            "The NVIDIA driver installed here is {} and supports CUDA up to {}.{}. Steno needs \
             CUDA {major}, so the {} MB download would not load. Update the NVIDIA display driver \
             first — driver 580 or newer — from nvidia.com.",
            self.version,
            self.cuda_major,
            self.cuda_minor,
            super::runtime::ARCHIVE.bytes / 1_000_000
        )
    }
}

/// The installed driver, or `None` when the question could not be answered.
#[cfg(windows)]
pub fn installed() -> Option<Driver> {
    nvml::query()
}

#[cfg(not(windows))]
pub fn installed() -> Option<Driver> {
    None
}

#[cfg(windows)]
mod nvml {
    use super::Driver;
    use std::ffi::c_void;

    type Init = unsafe extern "system" fn() -> i32;
    type Shutdown = unsafe extern "system" fn() -> i32;
    type CudaVersion = unsafe extern "system" fn(*mut i32) -> i32;
    type DriverVersion = unsafe extern "system" fn(*mut u8, u32) -> i32;

    /// NVML's own buffer size for the version string.
    const VERSION_BUFFER: usize = 80;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    pub fn query() -> Option<Driver> {
        // SAFETY: both strings are NUL-terminated literals, and every symbol is
        // called with the signature NVML documents for it.
        unsafe {
            let module = LoadLibraryA(c"nvml.dll".as_ptr().cast());
            if module.is_null() {
                return None;
            }

            let init: Init = std::mem::transmute(symbol(module, c"nvmlInit_v2")?);
            let cuda: CudaVersion =
                std::mem::transmute(symbol(module, c"nvmlSystemGetCudaDriverVersion_v2")?);
            let version: DriverVersion =
                std::mem::transmute(symbol(module, c"nvmlSystemGetDriverVersion")?);
            let shutdown: Shutdown = std::mem::transmute(symbol(module, c"nvmlShutdown")?);

            if init() != 0 {
                return None;
            }

            let driver = read(cuda, version);

            // Paired with the init whatever happened above: NVML counts its
            // initialisations, and leaving one outstanding leaves a handle to
            // the driver open for the life of the process.
            shutdown();
            driver
        }
    }

    /// # Safety
    /// The two function pointers must be the NVML symbols they were resolved as.
    unsafe fn read(cuda: CudaVersion, version: DriverVersion) -> Option<Driver> {
        let mut encoded = 0i32;
        if unsafe { cuda(&mut encoded) } != 0 || encoded <= 0 {
            return None;
        }

        let mut buffer = [0u8; VERSION_BUFFER];
        let printed = unsafe { version(buffer.as_mut_ptr(), VERSION_BUFFER as u32) };

        let text = if printed == 0 {
            let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(0);
            String::from_utf8_lossy(&buffer[..end]).into_owned()
        } else {
            String::new()
        };

        Some(Driver {
            // NVML encodes the CUDA version as major * 1000 + minor * 10.
            cuda_major: (encoded as u32) / 1000,
            cuda_minor: ((encoded as u32) % 1000) / 10,
            version: if text.is_empty() { "unknown".to_owned() } else { text },
        })
    }

    /// # Safety
    /// `module` must be a live module handle.
    unsafe fn symbol(module: *mut c_void, name: &std::ffi::CStr) -> Option<*mut c_void> {
        let found = unsafe { GetProcAddress(module, name.as_ptr().cast()) };
        (!found.is_null()).then_some(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_newer_driver_runs_an_older_cuda() {
        let driver = Driver {
            version: "610.74".to_owned(),
            cuda_major: 14,
            cuda_minor: 0,
        };
        assert!(driver.supports(13));
        assert!(driver.supports(14));
        assert!(!driver.supports(15));
    }

    #[test]
    fn the_refusal_carries_both_versions_and_the_saved_download() {
        let driver = Driver {
            version: "551.86".to_owned(),
            cuda_major: 12,
            cuda_minor: 4,
        };
        let said = driver.too_old_for(13);
        assert!(said.contains("551.86"), "{said}");
        assert!(said.contains("12.4"), "{said}");
        assert!(said.contains("CUDA 13"), "{said}");
        assert!(said.contains("391 MB"), "{said}");
    }

    /// Reports what this machine's driver says. Ignored: the answer is a
    /// property of the machine, and there is nothing to assert that would hold
    /// on another one.
    ///
    /// `cargo test --lib --features cuda -- --ignored --nocapture driver`
    #[test]
    #[ignore = "reports this machine's NVIDIA driver; nothing to assert"]
    fn report_this_machines_driver() {
        println!("{:?}", installed());
    }
}
