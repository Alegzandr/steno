//! A tiny persisted settings file next to the app data dir. Only the chosen
//! input device lives here today; phase 5 folds it into the real settings
//! surface. Kept deliberately small and forgiving: a missing or corrupt file
//! is not an error, it just means "start from defaults".

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

use crate::audio::lock;

/// The on-disk shape. `#[serde(default)]` so a file written by an older build,
/// missing a field, still loads.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// Name of the preferred input device, or `None` for the system default.
    pub input_device: Option<String>,
}

/// Tauri managed state: the resolved file path plus the in-memory settings.
pub struct Config {
    path: PathBuf,
    settings: Mutex<Settings>,
}

impl Config {
    /// Loads `settings.json` from the app config dir. A missing file, an
    /// unreadable one, or malformed JSON all fall back to defaults rather than
    /// failing startup; the next save rewrites the file cleanly.
    pub fn load<R: Runtime>(app: &AppHandle<R>) -> Self {
        let path = app
            .path()
            .app_config_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("settings.json");

        let settings = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();

        Self {
            path,
            settings: Mutex::new(settings),
        }
    }

    /// The saved input-device name, if any.
    pub fn input_device(&self) -> Option<String> {
        lock(&self.settings).input_device.clone()
    }

    /// Persists a new device choice. `None` clears it back to the system
    /// default.
    pub fn set_input_device(&self, name: Option<String>) -> Result<(), String> {
        let snapshot = {
            let mut settings = lock(&self.settings);
            settings.input_device = name;
            settings.clone()
        };
        self.save(&snapshot).map_err(|error| error.to_string())
    }

    fn save(&self, settings: &Settings) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(settings).expect("Settings always serializes");
        fs::write(&self.path, json)
    }
}
