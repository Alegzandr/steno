//! Fetching cuBLAS from NVIDIA, on the first launch that needs it.
//!
//! A CUDA build of Steno imports `cublas64_<major>.dll` and cannot dictate or
//! clean up without it. Shipping it would put 492 MB into a 79 MB installer for
//! every user, most of whom already have the CUDA toolkit; so it is downloaded,
//! once, the same way the models are.
//!
//! **Nothing here is hosted by us.** NVIDIA publishes versioned redistributable
//! archives with a `sha256` in a manifest, which is the same contract Hugging
//! Face offers and the one `model::download` already consumes — resumable,
//! digest-checked, and put in place only once the digest matches. The URL is
//! pinned to one build; it is never a "latest" that can change underneath a
//! checksum. The CUDA EULA's Attachment A lists `cublas.dll, cublasLt.dll` as
//! redistributable and §2.3 permits unzipping, and the archive's own `LICENSE`
//! is installed beside the DLLs.
//!
//! **Two members, not the archive.** `by_name` seeks through the central
//! directory, so `nvblas64_13.dll`, the headers and the import libraries are
//! never inflated. What lands is 512 MB out of a 391 MB download, and the zip is
//! deleted the moment the DLLs are in place — a transient peak of ~903 MB, which
//! is what the free-space check is measured against.
//!
//! **`cublasLt` is the trap.** It is imported by `cublas`, not by us, so it is
//! never the file the probe names. An install that forgot it would satisfy the
//! probe and fail at the first call — after the point where anything can be
//! reported. It is therefore extracted *first*, and `cublas64` last: the file
//! whose presence answers the question appears only once the file it depends on
//! is already there.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};

use super::driver::Driver;
use crate::model::download::{self, Downloads};
use crate::model::ModelSpec;
use crate::storage;

/// Progress the model downloader cannot report: it ends when the zip lands, and
/// half a gigabyte of inflation happens after that.
pub const INSTALL_STAGE: &str = "cublas-install-stage";

/// libcublas as published for CUDA 13, pinned to one build.
///
/// Size and digest are the values in NVIDIA's own `redistrib_13.3.0.json`,
/// confirmed against a `HEAD` on the file itself. `Accept-Ranges: bytes` is what
/// makes the resumable download work.
pub const ARCHIVE: ModelSpec = ModelSpec {
    id: "libcublas-windows-x86_64-13.5.1.27-archive.zip",
    url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/\
          libcublas-windows-x86_64-13.5.1.27-archive.zip",
    bytes: 391_055_517,
    sha256: "c946e1c825e05895747a95ed4fee18030b08052c09783b9b7b19818fd2e31f58",
    label: "NVIDIA cuBLAS runtime",
};

/// One file taken out of the archive.
struct Member {
    /// Matched as a suffix, because the archive puts everything under a
    /// directory named after its own version.
    path_suffix: &'static str,
    install_as: &'static str,
    /// Uncompressed size, from the archive's central directory. Checked twice:
    /// against what the entry claims, and against what was actually written.
    bytes: u64,
}

/// Extracted in this order, deliberately. See the module docs: the dependency
/// lands before the file that needs it, and the file the probe looks for is
/// last.
const MEMBERS: [Member; 3] = [
    Member {
        path_suffix: "/LICENSE",
        install_as: "LICENSE-cublas.txt",
        bytes: 68_070,
    },
    Member {
        path_suffix: "/bin/x64/cublasLt64_13.dll",
        install_as: "cublasLt64_13.dll",
        bytes: 460_301_424,
    },
    Member {
        path_suffix: "/bin/x64/cublas64_13.dll",
        install_as: "cublas64_13.dll",
        bytes: 51_870_320,
    },
];

/// What ends up on disk once the zip is gone.
pub const fn install_bytes() -> u64 {
    let mut total = 0;
    let mut index = 0;
    while index < MEMBERS.len() {
        total += MEMBERS[index].bytes;
        index += 1;
    }
    total
}

/// Everything the panel needs to explain itself, decided in one place so the
/// screen and the install cannot disagree about whether it can proceed.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// This build is blocked on cuBLAS. `false` on a CPU build and on a machine
    /// that already has the toolkit.
    pub needed: bool,
    pub missing: Option<String>,
    pub directory: String,
    pub archive_id: &'static str,
    pub archive_bytes: u64,
    /// What the two DLLs and the licence occupy afterwards.
    pub install_bytes: u64,
    /// Free space needed to get from here to done, counting the transient peak
    /// and whatever a previous attempt already fetched.
    pub required_bytes: u64,
    pub free_bytes: Option<u64>,
    /// A resumable partial download is on disk.
    pub partial_bytes: u64,
    /// The verified archive is already here; only extraction is left.
    pub archive_present: bool,
    pub driver: Option<Driver>,
    /// Why the download must not be offered, if there is a reason. `None` means
    /// the button is safe to show.
    pub obstacle: Option<String>,
}

/// Where the DLLs go, and the directory `gpu::use_runtime_dir` put on the
/// loader's search path at startup.
pub fn directory<R: Runtime>(app: &AppHandle<R>) -> PathBuf {
    super::runtime_dir(app)
}

pub fn status<R: Runtime>(app: &AppHandle<R>) -> Status {
    let blocker = super::blocker();
    let directory = directory(app);
    let archive = directory.join(ARCHIVE.id);

    let archive_present = fs::metadata(&archive).is_ok_and(|meta| meta.len() == ARCHIVE.bytes);
    let partial_bytes = fs::metadata(download::partial_path(&archive))
        .map(|meta| meta.len())
        .unwrap_or(0);

    let to_fetch = if archive_present {
        0
    } else {
        ARCHIVE.bytes.saturating_sub(partial_bytes)
    };
    let required_bytes = to_fetch + install_bytes();

    let free_bytes = storage::free_bytes(&directory);
    let driver = super::driver::installed();

    let obstacle = blocker.as_ref().and_then(|_| {
        first_obstacle(driver.as_ref(), free_bytes, required_bytes)
    });

    Status {
        needed: blocker.is_some(),
        missing: blocker.map(|blocker| blocker.missing),
        directory: directory.to_string_lossy().into_owned(),
        archive_id: ARCHIVE.id,
        archive_bytes: ARCHIVE.bytes,
        install_bytes: install_bytes(),
        required_bytes,
        free_bytes,
        partial_bytes,
        archive_present,
        driver,
        obstacle,
    }
}

/// The driver first, then the disk: an out-of-date driver makes the space
/// question moot, and saying both at once reads as two problems when there is
/// one thing to do.
fn first_obstacle(
    driver: Option<&Driver>,
    free_bytes: Option<u64>,
    required_bytes: u64,
) -> Option<String> {
    if let (Some(driver), Some(major)) = (driver, super::cuda_major()) {
        if !driver.supports(major) {
            return Some(driver.too_old_for(major));
        }
    }

    // `None` is "could not tell", never "no space". A download refused on a
    // failed query is worse than one that runs out of disk and says so.
    if let Some(free) = free_bytes {
        if free < required_bytes {
            return Some(format!(
                "This needs {} MB free — {} MB to download and {} MB to unpack, before the \
                 archive is deleted — and there are {} MB. Free some space, or move the Steno \
                 app data directory to a larger drive.",
                required_bytes / 1_000_000,
                ARCHIVE.bytes / 1_000_000,
                install_bytes() / 1_000_000,
                free / 1_000_000
            ));
        }
    }

    None
}

/// Downloads the archive, extracts the two DLLs, and rechecks.
///
/// Every failure leaves the app believing exactly what is true: the DLLs are
/// written under a `.part` name and renamed, a partial extraction removes every
/// member it installed, and the answer to "can Steno use the GPU" is re-asked
/// from the loader rather than assumed from the fact that a download finished.
pub async fn install<R: Runtime>(
    app: AppHandle<R>,
    downloads: &Downloads,
) -> Result<Status, String> {
    let Some(blocker) = super::blocker() else {
        return Err("cuBLAS is already available; there is nothing to download".to_owned());
    };

    let before = status(&app);
    if let Some(obstacle) = before.obstacle {
        return Err(obstacle);
    }

    let directory = directory(&app);

    let cancel = downloads.begin()?;
    let _end = download::EndOnDrop(downloads);

    let emitter = app.clone();
    let report = move |progress: download::Progress| {
        let _ = emitter.emit(download::DOWNLOAD_PROGRESS, progress);
    };
    let announcer = app.clone();
    let stage = move |name: &str| {
        let _ = announcer.emit(INSTALL_STAGE, name);
    };

    fetch_into(&directory, &cancel, &report, &stage).await?;

    stage("checking");
    if let Some(still) = super::recheck() {
        return Err(format!(
            "{} was installed into {}, and {} is still not loadable. {}",
            ARCHIVE.id,
            directory.display(),
            blocker.missing,
            still.message
        ));
    }

    Ok(status(&app))
}

/// The install itself, with no dependency on Tauri.
///
/// Split out for the same reason `download::transfer` is: a path that only ever
/// runs on a user's first launch, on a machine that by definition is not the
/// developer's, is a path nobody has tested. `examples/cublas_install.rs` drives
/// this one against a temp directory, so the URL, the digest, the deflate
/// members and their sizes are exercised without touching the app's own data
/// directory.
pub async fn fetch_into(
    directory: &Path,
    cancel: &std::sync::atomic::AtomicBool,
    report: &(dyn Fn(download::Progress) + Send + Sync),
    stage: &(dyn Fn(&str) + Send + Sync),
) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {} ({error})", directory.display()))?;
    let archive = directory.join(ARCHIVE.id);

    // The digest was checked before the file was put in place, so a full-size
    // archive is one this machine has already verified. Re-fetching 391 MB to
    // re-prove that would be the most expensive way to say nothing.
    let present = fs::metadata(&archive).is_ok_and(|meta| meta.len() == ARCHIVE.bytes);
    if !present {
        stage("downloading");
        download::transfer(&ARCHIVE, &archive, cancel, report).await?;
    }

    stage("extracting");
    let into = directory.to_path_buf();
    let source = archive.clone();
    tauri::async_runtime::spawn_blocking(move || extract(&source, &into))
        .await
        .map_err(|error| format!("the extraction task failed ({error})"))??;

    // Only now: while it existed, it was the thing that let a failed extraction
    // be retried without another 391 MB.
    let _ = fs::remove_file(&archive);

    Ok(())
}

/// Pulls the members out, each through a `.part` name.
///
/// A DLL is mapped by the loader, not parsed by us: a truncated one is not
/// rejected, it is a module with a plausible header and missing pages. So no
/// member appears under its real name until it is complete and its size matches
/// what the archive said it would be.
fn extract(archive: &Path, into: &Path) -> Result<(), String> {
    let file = File::open(archive)
        .map_err(|error| format!("could not open {} ({error})", archive.display()))?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|error| format!("{} is not a readable zip ({error})", ARCHIVE.id))?;

    let names: Vec<String> = zip.file_names().map(str::to_owned).collect();
    let mut installed: Vec<PathBuf> = Vec::new();

    for member in &MEMBERS {
        match one(&mut zip, &names, member, into) {
            Ok(path) => installed.push(path),
            Err(error) => {
                // Not a partial install left behind for the next launch to
                // misread. Everything this attempt wrote goes.
                for path in &installed {
                    let _ = fs::remove_file(path);
                }
                let _ = fs::remove_file(into.join(format!("{}.part", member.install_as)));
                return Err(error);
            }
        }
    }

    Ok(())
}

fn one<Reader: std::io::Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<Reader>,
    names: &[String],
    member: &Member,
    into: &Path,
) -> Result<PathBuf, String> {
    let name = names
        .iter()
        .find(|name| name.ends_with(member.path_suffix))
        .ok_or_else(|| format!("{} holds no {}", ARCHIVE.id, member.path_suffix))?;

    let mut entry = zip
        .by_name(name)
        .map_err(|error| format!("could not read {name} ({error})"))?;

    if entry.size() != member.bytes {
        return Err(format!(
            "{name} is {} bytes in this archive, expected {}",
            entry.size(),
            member.bytes
        ));
    }

    let staged = into.join(format!("{}.part", member.install_as));
    let mut out = File::create(&staged)
        .map_err(|error| format!("could not create {} ({error})", staged.display()))?;

    let written = std::io::copy(&mut entry, &mut out)
        .map_err(|error| format!("could not write {} ({error})", staged.display()))?;
    out.sync_all()
        .map_err(|error| format!("could not flush {} ({error})", staged.display()))?;
    drop(out);

    if written != member.bytes {
        return Err(format!(
            "{} came out {written} bytes, expected {}",
            member.install_as, member.bytes
        ));
    }

    let destination = into.join(member.install_as);
    fs::rename(&staged, &destination)
        .map_err(|error| format!("could not move {} into place ({error})", member.install_as))?;

    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transient peak the free-space check is written against. If either
    /// figure moves, the number in the message moves with it.
    #[test]
    fn the_peak_is_the_archive_plus_what_it_unpacks_to() {
        assert_eq!(install_bytes(), 512_239_814);
        assert_eq!(ARCHIVE.bytes + install_bytes(), 903_295_331);
    }

    /// cuBLAS is useless without the library it imports, and that library is
    /// never the file the probe names.
    #[test]
    fn cublas_lt_is_installed_before_cublas() {
        let names: Vec<&str> = MEMBERS.iter().map(|member| member.install_as).collect();
        let lt = names.iter().position(|name| name.contains("cublasLt")).unwrap();
        let cublas = names
            .iter()
            .position(|name| name.starts_with("cublas64_"))
            .unwrap();
        assert!(lt < cublas, "{names:?}");
    }

    /// A licence that is not shipped beside the binaries it covers is a licence
    /// that was not shipped.
    #[test]
    fn the_licence_comes_along() {
        assert!(MEMBERS.iter().any(|member| member.path_suffix.ends_with("LICENSE")));
    }

    #[test]
    fn an_unanswerable_space_query_does_not_block_the_download() {
        assert_eq!(first_obstacle(None, None, u64::MAX), None);
    }

    #[test]
    fn too_little_space_says_both_numbers() {
        let said = first_obstacle(None, Some(100_000_000), 903_295_331).unwrap();
        assert!(said.contains("903 MB"), "{said}");
        assert!(said.contains("100 MB"), "{said}");
    }

    #[test]
    fn a_driver_that_cannot_run_this_cuda_outranks_the_disk() {
        let old = Driver {
            version: "551.86".to_owned(),
            cuda_major: 12,
            cuda_minor: 4,
        };
        let Some(major) = super::super::cuda_major() else {
            return; // CPU build: nothing to be too old for.
        };
        assert!(major >= 13);
        let said = first_obstacle(Some(&old), Some(0), u64::MAX).unwrap();
        assert!(said.contains("551.86"), "{said}");
    }
}
