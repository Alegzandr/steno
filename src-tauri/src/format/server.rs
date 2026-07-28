//! Who owns the Ollama server, and how a server Steno started is guaranteed to
//! die with it.
//!
//! Two rules. Steno never kills a server it did not start: port 11434 is a
//! shared resource and something else on the machine may be mid-generation on
//! it. And a server Steno *did* start never outlives Steno, because an orphaned
//! Ollama holding nine gigabytes of video memory is precisely the failure this
//! whole subsystem exists to prevent.
//!
//! The second rule cannot be kept in Rust. `Drop` does not run on
//! `TerminateProcess`, and neither does a Ctrl-C handler or a Tauri exit hook,
//! so anyone ending Steno from the Task Manager would leak the server. On
//! Windows the guarantee comes from a job object instead: the kernel closes our
//! handles when the process dies, however it dies, and closing the last handle
//! to a job marked `KILL_ON_JOB_CLOSE` kills everything inside it. It is
//! enforced outside our code, so it does not depend on our code running.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde::Serialize;

/// How long to wait for an answer before deciding nothing is listening.
const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// A freshly spawned server needs a moment before it accepts connections.
const STARTUP_CEILING: Duration = Duration::from_secs(20);
const STARTUP_POLL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ownership {
    /// Something was already listening. Left completely alone on exit.
    Adopted,
    /// We started it, and we are responsible for ending it.
    Spawned,
    /// Nothing is listening and we could not start one.
    Absent,
}

/// The server Steno is talking to.
pub struct Server {
    pub endpoint: String,
    pub ownership: Ownership,
    /// Models that were already resident when we adopted the server. We never
    /// unload these: they belong to whoever loaded them.
    pub foreign_models: Vec<String>,
    child: Option<Child>,
    #[cfg(windows)]
    _job: Option<job::Job>,
}

impl Server {
    /// Finds a server or starts one.
    ///
    /// `models_dir` is passed to a server we start, and ignored when adopting
    /// one: an existing server already knows where its models are.
    ///
    /// Blocking. Called from the startup warm-up thread, never the event loop.
    pub fn ensure(endpoint: &str, models_dir: Option<&str>) -> Self {
        if probe(endpoint) {
            let foreign_models = model_names(endpoint);
            if !foreign_models.is_empty() {
                eprintln!(
                    "ollama: adopting the running server; {} model(s) already loaded and left alone",
                    foreign_models.len()
                );
            } else {
                eprintln!("ollama: adopting the server already listening on {endpoint}");
            }

            return Self {
                endpoint: endpoint.to_owned(),
                ownership: Ownership::Adopted,
                foreign_models,
                child: None,
                #[cfg(windows)]
                _job: None,
            };
        }

        match spawn(models_dir) {
            Ok((child, job)) => {
                let started = std::time::Instant::now();
                while started.elapsed() < STARTUP_CEILING {
                    // Readiness is `/api/tags`, not `/api/version`. A freshly
                    // spawned server answers its version well before it has
                    // scanned the model directory, and a caller that asks in
                    // that window is told the model is not installed — so the
                    // one thing it offers a first-time user is a command to
                    // re-pull nine gigabytes they already have. Measured on
                    // this machine: version answered at 5.7 s, tags later.
                    if probe(endpoint) && catalogue(endpoint).is_some() {
                        eprintln!(
                            "ollama: started a server in {} ms",
                            started.elapsed().as_millis()
                        );
                        return Self {
                            endpoint: endpoint.to_owned(),
                            ownership: Ownership::Spawned,
                            foreign_models: Vec::new(),
                            child: Some(child),
                            #[cfg(windows)]
                            _job: job,
                        };
                    }
                    std::thread::sleep(STARTUP_POLL);
                }

                eprintln!("ollama: started a server but it never answered on {endpoint}");
                Self {
                    endpoint: endpoint.to_owned(),
                    ownership: Ownership::Spawned,
                    foreign_models: Vec::new(),
                    child: Some(child),
                    #[cfg(windows)]
                    _job: job,
                }
            }
            Err(error) => {
                eprintln!("ollama: no server on {endpoint} and none could be started ({error})");
                Self {
                    endpoint: endpoint.to_owned(),
                    ownership: Ownership::Absent,
                    foreign_models: Vec::new(),
                    child: None,
                    #[cfg(windows)]
                    _job: None,
                }
            }
        }
    }

    /// Whether this model was already loaded by somebody else, and must
    /// therefore never be unloaded by us.
    pub fn is_foreign(&self, model: &str) -> bool {
        self.foreign_models.iter().any(|name| name == model)
    }

    pub fn is_reachable(&self) -> bool {
        self.ownership != Ownership::Absent
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };

        // The clean path. The job object below is the backstop for every way
        // this does not get to run.
        eprintln!("ollama: stopping the server we started");
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Whether anything answers the Ollama API on `endpoint`.
pub fn probe(endpoint: &str) -> bool {
    version(endpoint).is_some()
}

pub fn version(endpoint: &str) -> Option<String> {
    let url = format!("{}/api/version", endpoint.trim_end_matches('/'));

    blocking(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(PROBE_TIMEOUT)
            .timeout(PROBE_TIMEOUT)
            .build()
            .ok()?;

        let response = client.get(&url).send().await.ok()?;
        let body: serde_json::Value = response.json().await.ok()?;
        body.get("version")?.as_str().map(|s| s.to_owned())
    })
}

/// Models pulled onto this machine, resident or not.
///
/// `None` means the server did not answer, which is not the same as an empty
/// catalogue and must never be reported as "you have no models": that is how a
/// user with the model already pulled gets told to pull it again.
pub fn catalogue(endpoint: &str) -> Option<Vec<String>> {
    let url = format!("{}/api/tags", endpoint.trim_end_matches('/'));

    blocking(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(PROBE_TIMEOUT)
            .timeout(Duration::from_secs(10))
            .build()
            .ok()?;

        let response = client.get(&url).send().await.ok()?;
        let body: serde_json::Value = response.json().await.ok()?;

        Some(
            body.get("models")?
                .as_array()?
                .iter()
                .filter_map(|entry| entry.get("name")?.as_str().map(|s| s.to_owned()))
                .collect(),
        )
    })
}

/// Names of the models currently held in memory by the server.
pub fn model_names(endpoint: &str) -> Vec<String> {
    let url = format!("{}/api/ps", endpoint.trim_end_matches('/'));

    blocking(async move {
        let client = reqwest::Client::builder()
            .connect_timeout(PROBE_TIMEOUT)
            .timeout(Duration::from_secs(5))
            .build()
            .ok()?;

        let response = client.get(&url).send().await.ok()?;
        let body: serde_json::Value = response.json().await.ok()?;

        Some(
            body.get("models")?
                .as_array()?
                .iter()
                .filter_map(|entry| entry.get("name")?.as_str().map(|s| s.to_owned()))
                .collect(),
        )
    })
    .unwrap_or_default()
}

/// Runs a future to completion from a synchronous context.
///
/// On its own thread, always. `block_on` panics when it is called from inside
/// the async runtime, and some of these calls happen in `Drop`, which can run
/// on any thread at all.
///
/// Deliberately not `'static`: the thread is scoped, so the future may borrow
/// from the caller's frame. That is what lets the cleanup stream report through
/// a borrowed callback instead of having to own an `AppHandle`, which in turn
/// is what lets the measurement harness drive the same code the app drives.
pub(crate) fn blocking<T, F>(future: F) -> T
where
    T: Send,
    F: std::future::Future<Output = T> + Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| tauri::async_runtime::block_on(future))
            .join()
            .unwrap_or_else(|_| panic!("the blocking HTTP helper panicked"))
    })
}

/// Starts `ollama serve`, inside a job object where one is available.
///
/// `OLLAMA_MODELS` is set only when configured. Ollama's desktop app keeps the
/// model directory in its own settings and passes it to the server it starts,
/// so a server started here defaults to `%USERPROFILE%\.ollama\models` and sees
/// nothing at all if the user moved the store. Setting it blindly would be
/// worse than not setting it, hence `settings.json` rather than a guess.
#[cfg(windows)]
fn spawn(models_dir: Option<&str>) -> Result<(Child, Option<job::Job>), String> {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;

    /// No console window flashing up behind the mini editor.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let job = match job::Job::new() {
        Ok(job) => Some(job),
        Err(error) => {
            // Not fatal, but it does mean the only thing standing between a
            // force-kill and an orphaned 9 GB server is `keep_alive`.
            eprintln!("ollama: could not create a job object ({error}); the server will rely on the clean shutdown path alone");
            None
        }
    };

    let mut command = Command::new(executable());
    command
        .arg("serve")
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if let Some(dir) = models_dir {
        eprintln!("ollama: starting a server with OLLAMA_MODELS={dir}");
        command.env("OLLAMA_MODELS", dir);
    }

    let child = command
        .spawn()
        .map_err(|error| format!("could not run `ollama serve` ({error})"))?;

    if let Some(job) = job.as_ref() {
        if let Err(error) = job.adopt(child.as_raw_handle()) {
            eprintln!("ollama: could not put the server in the job object ({error})");
        }
    }

    Ok((child, job))
}

#[cfg(not(windows))]
fn spawn(models_dir: Option<&str>) -> Result<(Child, Option<()>), String> {
    let mut command = Command::new(executable());
    command
        .arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if let Some(dir) = models_dir {
        command.env("OLLAMA_MODELS", dir);
    }

    let child = command
        .spawn()
        .map_err(|error| format!("could not run `ollama serve` ({error})"))?;

    Ok((child, None))
}

/// `ollama` from the PATH, falling back to the per-user install location the
/// Windows installer uses, which is not always on the PATH of a process
/// started from Explorer.
fn executable() -> std::path::PathBuf {
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            let candidate = std::path::Path::new(&local)
                .join("Programs")
                .join("Ollama")
                .join("ollama.exe");
            if candidate.exists() {
                return candidate;
            }
        }
    }

    std::path::PathBuf::from("ollama")
}

// Public so the acceptance harness in `examples/` can prove the kill-on-close
// behaviour against a real force-kill, which is the one claim here that cannot
// be established by reading the code.
#[cfg(windows)]
pub mod job {
    use std::ffi::c_void;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// A kill-on-close job object.
    ///
    /// The handle is deliberately never inherited and never duplicated: the
    /// whole guarantee rests on ours being the last one, so that our process
    /// ending closes it and takes the server with it.
    pub struct Job(HANDLE);

    // SAFETY: a job object handle is a kernel handle. It is not bound to the
    // thread that created it, every operation on it is internally
    // synchronised, and this type never hands the raw handle out. The pointer
    // inside `HANDLE` is what makes the auto traits opt out, not any real
    // thread affinity.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn new() -> Result<Self, String> {
            // SAFETY: a fresh unnamed job object, configured through a fully
            // initialised struct of the size we declare.
            unsafe {
                let handle = CreateJobObjectW(None, None)
                    .map_err(|error| format!("CreateJobObjectW failed ({error})"))?;

                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                let result = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );

                if let Err(error) = result {
                    let _ = CloseHandle(handle);
                    return Err(format!("SetInformationJobObject failed ({error})"));
                }

                Ok(Self(handle))
            }
        }

        /// Puts a process in the job. Anything it starts afterwards joins too,
        /// which is what covers the `ollama runner` subprocess that actually
        /// holds the video memory.
        pub fn adopt(&self, process: *mut c_void) -> Result<(), String> {
            // SAFETY: `process` is a live process handle owned by the `Child`
            // the caller still holds, and `self.0` is our own job handle.
            unsafe {
                AssignProcessToJobObject(self.0, HANDLE(process))
                    .map_err(|error| format!("AssignProcessToJobObject failed ({error})"))
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // Closing the last handle is what kills the members.
            // SAFETY: we created this handle and never handed it out.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}
