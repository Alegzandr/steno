//! `settings.json`, the one file the user is expected to edit by hand.
//!
//! Every read goes to disk. That is the whole point: the model name, the system
//! prompt, the vocabulary and the denylist are all things you change while
//! Steno is running, and none of them may need a rebuild or even a restart. The
//! cost is a few kilobytes of JSON parsed per user action, which is nothing next
//! to opening a microphone or running Whisper.
//!
//! Writes are read-modify-write under a mutex, so a save from the UI cannot
//! clobber a field you edited in the file a second earlier.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

use crate::audio::lock;

/// The default cleanup instruction. Copied into `settings.json` at first launch
/// and never read from here again, so editing the file wins.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You restructure a developer's dictated brainstorm into a clean Markdown prompt.

The input is raw French speech-to-text: no punctuation, filler words, false
starts, self-corrections, and sentences that restart mid-thought.

Rules:
- Fix punctuation, capitalization and obvious transcription errors.
- Remove fillers (euh, enfin, bah, repeated \"du coup\", \"en fait\"), false
  starts, and abandoned sentences. When the speaker corrects themselves, keep
  only the corrected version.
- Structure the result: `##` headings for distinct topics, bullet lists for
  enumerations, fenced code blocks for code, commands, paths and identifiers.
- Treat \"à la ligne\", \"nouveau paragraphe\", \"titre deux\", \"point\", \"virgule\",
  \"ouvre une liste\" as formatting instructions, not as text.
- Preserve every idea and every technical term exactly as spoken. Do not
  rephrase for style, do not summarise, do not add anything, do not answer
  the content.
- Output only the Markdown, with no preamble and no wrapping code fence.";

/// The on-disk shape. `#[serde(default)]` at every level so a file written by
/// an older build, or one you trimmed by hand, still loads.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// Name of the preferred input device, or `None` for the system default.
    pub input_device: Option<String>,
    pub model: ModelSettings,
    pub whisper: WhisperSettings,
    pub ollama: OllamaSettings,
    pub vram: VramSettings,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelSettings {
    /// Filename of the Whisper model to use, for example
    /// `ggml-large-v3-turbo-q5_0.bin`. `None` means "not decided yet": the
    /// first launch probes the GPU, picks one, and writes it here. Editing this
    /// field is how you override that choice.
    pub id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WhisperSettings {
    /// Forced, never auto-detected: language detection is unreliable on the
    /// short clips push-to-talk produces.
    pub language: String,
    /// Segments Whisper marks as this likely to be silence are dropped.
    pub no_speech_thold: f32,
    /// Clips quieter than this never reach Whisper at all. dBFS, so negative.
    pub rms_floor_dbfs: f32,
    /// Terms fed to Whisper as `initial_prompt`, to bias it towards the
    /// jargon that gets dictated. See `transcribe::prompt` for the length cap.
    pub vocabulary: Vec<String>,
    /// Whether the vocabulary biases every 30-second window or only the first.
    /// Defaults to true: a 90-second brainstorm is three windows, and the
    /// technical terms are as likely to land in the third as in the first.
    /// Needs the vendored whisper-rs patch — see CLAUDE.md.
    pub carry_initial_prompt: bool,
    /// Whole segments matching one of these, ignoring case, surrounding space
    /// and trailing punctuation, are dropped. Whisper emits them on silence.
    pub hallucinations: Vec<String>,
    /// Decoder threads. `None` uses the physical core count.
    pub threads: Option<u16>,
}

impl Default for WhisperSettings {
    fn default() -> Self {
        Self {
            language: "fr".to_owned(),
            no_speech_thold: 0.6,
            rms_floor_dbfs: -50.0,
            vocabulary: [
                "Tauri",
                "Rust",
                "cargo",
                "npm",
                "TypeScript",
                "React",
                "whisper",
                "Ollama",
                "endpoint",
                "middleware",
                "refactor",
                "JSON",
                "API",
                "CLI",
            ]
            .iter()
            .map(|term| (*term).to_owned())
            .collect(),
            carry_initial_prompt: true,
            hallucinations: [
                // Measured, not guessed: ten seconds of digital silence and ten
                // seconds of -40 dBFS noise both come back as exactly "Merci."
                // from large-v3. It is the single most likely thing Whisper
                // invents on French near-silence, and with `no_speech_thold`
                // inoperative (see transcribe::filter) the denylist is the only
                // guard that catches it above the RMS floor.
                //
                // These are matched against the whole transcription, never a
                // segment inside one, so a real dictation that ends on "merci"
                // keeps it. The only cost is that a burst containing nothing
                // but "Merci." is dropped — the right trade for a tool that
                // structures technical brainstorms, and a one-word edit to
                // undo.
                "Merci",
                "Merci beaucoup",
                "Merci à vous",
                "Sous-titres réalisés par la communauté d'Amara.org",
                "Sous-titres réalisés para la communauté d'Amara.org",
                "Sous-titres réalisés par l'Amara.org",
                "Sous-titrage Société Radio-Canada",
                "Sous-titrage ST' 501",
                "Merci d'avoir regardé cette vidéo",
                "Merci d'avoir regardé cette vidéo !",
                "Merci à tous et à bientôt",
                "Abonnez-vous",
                "❤️ par SousTitreur.com",
                "www.amara.org",
            ]
            .iter()
            .map(|phrase| (*phrase).to_owned())
            .collect(),
            threads: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OllamaSettings {
    pub endpoint: String,
    pub model: String,
    pub system_prompt: String,
    pub temperature: f32,
    /// Where Ollama keeps its models, passed as `OLLAMA_MODELS` to a server
    /// Steno starts itself. `None` lets Ollama use its own default.
    ///
    /// Only needed if you have moved the store off the system drive, which is
    /// common on a machine with a small C:. The Ollama desktop app remembers
    /// that location in its own settings and hands it to the server *it*
    /// starts; a bare `ollama serve` does not inherit it and falls back to
    /// `%USERPROFILE%\.ollama\models`. The symptom is Steno reporting that a
    /// model you have installed is missing, and only when Steno had to start
    /// the server — adopting a running one never hits it.
    pub models_dir: Option<String>,
}

impl Default for OllamaSettings {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:11434".to_owned(),
            model: "qwen3:14b".to_owned(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_owned(),
            temperature: 0.0,
            models_dir: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VramSettings {
    /// Both models are evicted after this long with no recording and no
    /// cleanup, even with the window on screen. Steno is meant to be left open
    /// on a second screen; holding twelve gigabytes while it sits there idle is
    /// the failure this prevents. Zero disables idle eviction.
    pub idle_unload_seconds: u64,
    /// What Ollama is told when the model is warmed. Deliberately finite: if
    /// Steno dies without running its unload path, this is what eventually
    /// gives the video memory back.
    pub keep_alive: String,
    /// Whether showing the window warms the formatting model too, or only
    /// Whisper. Turn this off to trade a slower first cleanup for a smaller
    /// resident footprint.
    pub warm_llm_on_show: bool,
}

impl Default for VramSettings {
    fn default() -> Self {
        Self {
            idle_unload_seconds: 300,
            keep_alive: "20m".to_owned(),
            warm_llm_on_show: true,
        }
    }
}

/// Tauri managed state: where the file lives, plus a mutex that serialises
/// read-modify-write cycles against each other.
pub struct Config {
    path: PathBuf,
    writing: Mutex<()>,
}

impl Config {
    /// Resolves the path and makes sure a complete file exists.
    ///
    /// Writing the full defaults out at first launch is not a nicety: an empty
    /// or absent `settings.json` is a file nobody can edit, and every knob in
    /// here is meant to be edited by hand.
    pub fn load<R: Runtime>(app: &AppHandle<R>) -> Self {
        let path = app
            .path()
            .app_config_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("settings.json");

        let config = Self {
            path,
            writing: Mutex::new(()),
        };

        if !config.path.exists() {
            if let Err(error) = config.write(&Settings::default()) {
                eprintln!("settings: could not write the initial file ({error})");
            }
        }

        config
    }

    /// The current settings, read from disk. A missing file, an unreadable one,
    /// or malformed JSON all yield defaults rather than an error: a typo in the
    /// file must not stop you from dictating.
    pub fn get(&self) -> Settings {
        match fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
                eprintln!(
                    "settings: {} is not valid JSON ({error}), using defaults",
                    self.path.display()
                );
                Settings::default()
            }),
            Err(_) => Settings::default(),
        }
    }

    /// The saved input-device name, if any.
    pub fn input_device(&self) -> Option<String> {
        self.get().input_device
    }

    /// Persists a new device choice. `None` clears it back to the system
    /// default.
    pub fn set_input_device(&self, name: Option<String>) -> Result<(), String> {
        self.update(|settings| settings.input_device = name)
    }

    /// Records the model the first-launch probe settled on, leaving an explicit
    /// user choice alone.
    pub fn set_model_id(&self, id: &str) -> Result<(), String> {
        let id = id.to_owned();
        self.update(|settings| settings.model.id = Some(id))
    }

    /// Read-modify-write. Re-reads the file inside the lock so a field you
    /// edited by hand between two saves survives.
    fn update(&self, change: impl FnOnce(&mut Settings)) -> Result<(), String> {
        let _writing = lock(&self.writing);
        let mut settings = self.get();
        change(&mut settings);
        self.write(&settings).map_err(|error| error.to_string())
    }

    fn write(&self, settings: &Settings) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(settings).expect("Settings always serializes");
        fs::write(&self.path, json)
    }
}
