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

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

use crate::audio::lock;

/// Names of the two shipped cleanup instructions, as they appear in
/// `llm.prompt`.
///
/// Two, not one, because which of them is better is an empirical question about
/// real dictations rather than a design decision: `faithful` refuses to add
/// anything and produces flatter output, `structured` is the phase 4 wording
/// that shapes more freely and occasionally invents a heading. Both ship, both
/// are visible in `settings.json`, and switching between them is a one-word
/// edit with no restart.
pub const PROMPT_FAITHFUL: &str = "faithful";
pub const PROMPT_STRUCTURED: &str = "structured";

/// Where a system prompt carried over from a pre-5.1 `settings.json` is put.
///
/// Not a third mode — nothing in the code knows this name — just the place the
/// migration puts text it must not throw away. See `migrate_legacy`.
pub const PROMPT_CUSTOM: &str = "custom";

/// The default cleanup instruction. Written into `settings.json` at first
/// launch, where editing it wins; read from here only when the file has no
/// entry under that name.
pub const FAITHFUL_SYSTEM_PROMPT: &str = "\
You restructure a developer's dictated brainstorm into a clean Markdown prompt.

The input is raw French speech-to-text: no punctuation, filler words, false
starts, self-corrections, and sentences that restart mid-thought.

You are a formatter, not a co-author. The output says what the input said, in
a cleaner shape. It says nothing the input did not say.

Rules:
- Fix punctuation, capitalization and obvious transcription errors.
- Remove fillers (euh, enfin, bah, repeated \"du coup\", \"en fait\"), false
  starts, and abandoned sentences. When the speaker corrects themselves, keep
  only the corrected version.
- Structure the result: `##` headings for distinct topics, bullet lists for
  enumerations, fenced code blocks for code, commands, paths and identifiers.
  Headings are lifted, not written: every word of a heading must already
  appear in the sentences beneath it, and a section with no such phrase gets
  no heading at all. Default to no headings; a short dictation on one subject
  is a list, not a document.
- Treat \"à la ligne\", \"nouveau paragraphe\", \"titre deux\", \"point\", \"virgule\",
  \"ouvre une liste\" as formatting instructions, not as text.
- Add nothing. No examples, no illustrations, no parenthetical expansions, no
  clarifications, no definitions, no suggestions, no closing summary. If the
  speaker named a category without listing what is in it, you do not list it
  either. Every noun in the output was spoken in the input.
- Keep the speaker's own verbs, and the person they spoke in. A bullet list is
  a change of layout, not a change of voice: \"je voudrais un site d'agence\"
  becomes a bullet reading \"je voudrais un site d'agence\", never \"créer un
  site d'agence\".
- Vague stays vague. An underspecified requirement is itself information about
  what the speaker has and has not decided; filling the gap destroys it.
- Keep the framing. If the input says what the text is, who or what it is for,
  or what should be done with it, that statement survives into the output. It
  is the most important sentence there, not preamble to be trimmed.
- Say each thing once. One idea goes under one heading; never restate it in a
  second section for symmetry or emphasis.
- Preserve every idea and every technical term exactly as spoken. Do not
  rephrase for style, do not summarise, do not answer the content.
- Output only the Markdown, with no preamble and no wrapping code fence.";

/// The phase 4 wording, kept verbatim so an A/B against `faithful` compares two
/// things that actually existed rather than a reconstruction of one of them.
///
/// It is the shorter and looser of the two: it asks for structure without
/// saying where structure may come from, which is what lets it write a heading
/// the speaker never said. That was the observation `faithful` was written to
/// answer; whether the cure costs more than the disease is the thing being
/// measured.
pub const STRUCTURED_SYSTEM_PROMPT: &str = "\
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
    /// Renamed from `ollama` in phase 5.1, when the formatting model moved into
    /// this process. A file written before 5.1 is rewritten on first launch by
    /// `migrate_legacy` rather than being read past — see it for what survives
    /// and what cannot.
    pub llm: LlmSettings,
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
                // Dictated in the very first real session and transcribed as
                // "Cloud Code". Two ordinary words that Whisper has no reason
                // to prefer over the acoustically closer pair, so it needs the
                // bias rather than a spelling fix downstream.
                "Claude Code",
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
pub struct LlmSettings {
    /// File name inside Steno's models directory, not a path. Keeping it a bare
    /// name is what stops a settings file from pointing the loader at an
    /// arbitrary location on disk, and it keeps the storage advisory honest:
    /// there is exactly one directory whose drive has to be checked.
    pub model_file: String,
    /// Which entry of `prompts` is in force. `faithful` or `structured` out of
    /// the box; a name that is not in the map falls back to `faithful` and says
    /// so once, rather than quietly cleaning up with the wrong instruction.
    pub prompt: String,
    /// Every cleanup instruction, by name. Both shipped prompts are written out
    /// at first launch so switching is an edit rather than a search through the
    /// source, and editing one in place is the supported way to write your own.
    pub prompts: BTreeMap<String, String>,
    pub temperature: f32,
    /// Context window for one cleanup. The whole buffer plus its rewrite has to
    /// fit: this is an accumulation buffer, so it grows across a session, and a
    /// cleanup that silently truncated the input would be worse than one that
    /// refused. 8192 holds a long dictation and its output; the model itself
    /// trained to 40960 and can go further if a machine has the memory for the
    /// KV cache.
    pub n_ctx: u32,
    /// Prefill batch size. 512 is what Ollama uses for this model.
    pub n_batch: u32,
    /// Layers to put on the GPU. 999 means all of them and is clamped to what
    /// the model has. Lower it to trade speed for video memory on a small card.
    pub n_gpu_layers: u32,
    /// Hard stop on generated tokens, so a model that starts repeating itself
    /// cannot run until the context is full. A cleanup outputs roughly as much
    /// as it was given, so this only ever fires on a degenerate loop.
    pub max_output_tokens: u32,
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            model_file: crate::format::DEFAULT_MODEL_FILE.to_owned(),
            prompt: PROMPT_FAITHFUL.to_owned(),
            prompts: shipped_prompts(),
            temperature: 0.0,
            n_ctx: 8192,
            n_batch: 512,
            n_gpu_layers: 999,
            max_output_tokens: 4096,
        }
    }
}

fn shipped_prompts() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            PROMPT_FAITHFUL.to_owned(),
            FAITHFUL_SYSTEM_PROMPT.to_owned(),
        ),
        (
            PROMPT_STRUCTURED.to_owned(),
            STRUCTURED_SYSTEM_PROMPT.to_owned(),
        ),
    ])
}

impl LlmSettings {
    /// The instruction a cleanup should actually run with.
    ///
    /// Falls back through the map to the built-in text of the same name, so a
    /// `settings.json` that was trimmed down to one entry still switches: the
    /// map is where you *override* a prompt, not the only place one exists.
    pub fn system_prompt(&self) -> &str {
        if let Some(text) = self.prompts.get(&self.prompt) {
            return text;
        }

        match self.prompt.as_str() {
            PROMPT_STRUCTURED => STRUCTURED_SYSTEM_PROMPT,
            _ => FAITHFUL_SYSTEM_PROMPT,
        }
    }

    /// The selected name, when it resolves to nothing.
    ///
    /// Separate from `system_prompt` because the fallback has to be silent at
    /// the point of use — a cleanup must still run — and loud once, where
    /// somebody will read it. A typo in `llm.prompt` is otherwise invisible:
    /// the cleanup works, it is just not the one you asked for.
    pub fn unknown_prompt(&self) -> Option<&str> {
        let known = self.prompts.contains_key(&self.prompt)
            || matches!(self.prompt.as_str(), PROMPT_FAITHFUL | PROMPT_STRUCTURED);
        (!known).then_some(self.prompt.as_str())
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
    /// Whether showing the window warms the formatting model too, or only
    /// Whisper. Turn this off to trade a slower first cleanup for a smaller
    /// resident footprint.
    pub warm_llm_on_show: bool,
}

impl Default for VramSettings {
    fn default() -> Self {
        Self {
            idle_unload_seconds: 300,
            warm_llm_on_show: true,
        }
    }
}

/// The pre-5.1 `ollama` block, as far as migration cares about it.
///
/// Every field optional: this parses a file somebody may have trimmed, and a
/// missing key must mean "nothing to carry", not "the whole block is
/// unreadable".
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LegacyOllama {
    system_prompt: Option<String>,
    temperature: Option<f32>,
    endpoint: Option<String>,
    model: Option<String>,
    models_dir: Option<String>,
}

/// Rewrites a pre-5.1 settings document, in place, and says what it did.
///
/// Returns an empty log when there is no `ollama` block, which is the normal
/// case on every launch after the first.
///
/// Silently defaulting was the original plan and it was wrong. `llm.prompt`
/// selects between two instructions now, so a user who had edited their system
/// prompt would lose it *and* gain a new knob in the same launch — the exact
/// shape of a setting that goes missing and never gets explained. So the text
/// is carried across under `prompts.custom` and selected, rather than being
/// compared against the shipped wording and discarded when it happens to match
/// something.
///
/// Three fields cannot be carried at all — `endpoint`, `model` and `modelsDir`
/// described a server that no longer exists — and each is named in the log
/// rather than disappearing quietly.
fn migrate_legacy(document: &mut serde_json::Value) -> Vec<String> {
    let Some(root) = document.as_object_mut() else {
        return Vec::new();
    };
    let Some(legacy) = root.remove("ollama") else {
        return Vec::new();
    };

    let mut log = vec!["settings: migrating the pre-5.1 `ollama` block".to_owned()];

    let legacy: LegacyOllama = match serde_json::from_value(legacy) {
        Ok(legacy) => legacy,
        Err(error) => {
            log.push(format!(
                "settings: the `ollama` block could not be read ({error}); it has been \
                 removed and nothing was carried across"
            ));
            return log;
        }
    };

    let llm = root
        .entry("llm")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let Some(llm) = llm.as_object_mut() else {
        log.push(
            "settings: `llm` is not an object, so nothing could be carried into it".to_owned(),
        );
        return log;
    };

    if let Some(text) = legacy.system_prompt {
        let prompts = llm
            .entry("prompts")
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        match prompts.as_object_mut() {
            Some(prompts) => {
                // Both shipped prompts go in alongside. Writing only `custom`
                // would leave a file whose `prompts` map has one entry and
                // whose `prompt` could be set to two names that appear nowhere
                // in it — switchable, because `system_prompt` falls back to the
                // built-in text, but not discoverable, which is most of what Z
                // is for.
                for (name, shipped) in shipped_prompts() {
                    prompts
                        .entry(name)
                        .or_insert(serde_json::Value::String(shipped));
                }
                prompts.insert(PROMPT_CUSTOM.to_owned(), serde_json::Value::String(text));
                llm.insert(
                    "prompt".to_owned(),
                    serde_json::Value::String(PROMPT_CUSTOM.to_owned()),
                );
                log.push(format!(
                    "settings: `ollama.systemPrompt` carried across as \
                     `llm.prompts.{PROMPT_CUSTOM}`, and `llm.prompt` now selects it. \
                     Set it to \"{PROMPT_FAITHFUL}\" or \"{PROMPT_STRUCTURED}\" for a \
                     shipped prompt."
                ));
            }
            None => log.push(
                "settings: `llm.prompts` is not an object, so `ollama.systemPrompt` \
                 could not be carried across and has been lost"
                    .to_owned(),
            ),
        }
    }

    if let Some(temperature) = legacy.temperature {
        match serde_json::Number::from_f64(f64::from(temperature)) {
            Some(number) => {
                llm.insert("temperature".to_owned(), serde_json::Value::Number(number));
                log.push(format!(
                    "settings: `ollama.temperature` ({temperature}) carried across as \
                     `llm.temperature`"
                ));
            }
            None => log.push(format!(
                "settings: `ollama.temperature` was {temperature}, which is not a \
                 number that can be written back; `llm.temperature` keeps its default"
            )),
        }
    }

    for (name, value) in [
        ("endpoint", legacy.endpoint),
        ("model", legacy.model),
        ("modelsDir", legacy.models_dir),
    ] {
        if let Some(value) = value {
            log.push(format!(
                "settings: `ollama.{name}` ({value}) has been dropped — it described \
                 the Ollama server, and the formatting model now runs inside Steno"
            ));
        }
    }

    log
}

enum Stored {
    Missing,
    Parsed(Settings),
    Corrupt(String),
}

/// Drops a UTF-8 byte order mark.
///
/// This file is meant to be edited by hand, and on Windows that means it will
/// sometimes be edited by something that writes a BOM — Notepad's older
/// "UTF-8 with BOM", PowerShell 5.1's `Out-File -Encoding utf8`, a few
/// editors' defaults. JSON has no place for those three bytes, so serde_json
/// rejects the whole document at column 1. Accepting them costs one comparison
/// and turns a mystifying failure into no failure at all.
fn without_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes)
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
        } else {
            config.migrate();
        }

        config
    }

    /// Applies `migrate_legacy` to the file on disk, once, at startup.
    ///
    /// Rewrites through `Settings` rather than saving the patched JSON, so the
    /// file gains every field added since it was written — including both
    /// shipped prompts, which is the point of doing this before the user ever
    /// looks for them. The cost is that keys Steno does not know about are
    /// dropped; the only ones that ever existed were in the `ollama` block this
    /// is removing on purpose.
    fn migrate(&self) {
        let _writing = lock(&self.writing);

        let Ok(bytes) = fs::read(&self.path) else {
            return;
        };
        let Ok(mut document) = serde_json::from_slice::<serde_json::Value>(without_bom(&bytes))
        else {
            // A malformed file is left exactly as it is. `get` already falls
            // back to the defaults for the session, and rewriting would destroy
            // whatever the user was in the middle of typing.
            return;
        };

        let log = migrate_legacy(&mut document);
        if log.is_empty() {
            return;
        }

        match serde_json::from_value::<Settings>(document) {
            Ok(settings) => match self.write(&settings) {
                Ok(()) => {
                    for line in log {
                        eprintln!("{line}");
                    }
                }
                Err(error) => eprintln!(
                    "settings: the pre-5.1 file could not be rewritten ({error}); it is \
                     unchanged and the `ollama` block is still being ignored"
                ),
            },
            Err(error) => eprintln!(
                "settings: the migrated file did not parse ({error}); {} is unchanged",
                self.path.display()
            ),
        }
    }

    /// The current settings, read from disk. A missing file, an unreadable one,
    /// or malformed JSON all yield defaults rather than an error: a typo in the
    /// file must not stop you from dictating.
    pub fn get(&self) -> Settings {
        match self.read() {
            Stored::Parsed(settings) => settings,
            Stored::Missing | Stored::Corrupt(_) => Settings::default(),
        }
    }

    /// The three states a settings file can be in, kept apart because `update`
    /// has to treat the third one as a refusal.
    ///
    /// Found the hard way: a file with a UTF-8 BOM parses as nothing, `get`
    /// hands back the defaults as designed, and then the next save — the
    /// first-launch model probe, which runs seconds later — writes those
    /// defaults over the file. One unreadable byte at the front, and the
    /// vocabulary, the denylist and the system prompt are gone with no message
    /// beyond a parse warning that reads like it was handled.
    fn read(&self) -> Stored {
        let Ok(bytes) = fs::read(&self.path) else {
            return Stored::Missing;
        };

        match serde_json::from_slice(without_bom(&bytes)) {
            Ok(settings) => Stored::Parsed(settings),
            Err(error) => {
                eprintln!(
                    "settings: {} is not valid JSON ({error}), using defaults for this \
                     session and refusing to overwrite it",
                    self.path.display()
                );
                Stored::Corrupt(error.to_string())
            }
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

        let mut settings = match self.read() {
            Stored::Parsed(settings) => settings,
            Stored::Missing => Settings::default(),
            // Refusing loses the change. Writing loses the file. The change is
            // one field the caller can make again; the file is everything the
            // user has ever configured.
            Stored::Corrupt(error) => {
                return Err(format!(
                    "{} could not be parsed ({error}), so it has been left alone \
                     rather than overwritten. Fix the JSON and try again.",
                    self.path.display()
                ))
            }
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_document(prompt: &str) -> serde_json::Value {
        serde_json::json!({
            "inputDevice": null,
            "model": { "id": "ggml-large-v3.bin" },
            "ollama": {
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen3:14b",
                "systemPrompt": prompt,
                "temperature": 0.4,
                "modelsDir": r"D:\.ollama\models"
            },
            "vram": { "idleUnloadSeconds": 300 }
        })
    }

    /// The failure FF exists to prevent: an edited prompt disappearing on the
    /// launch that introduced a way to choose between prompts.
    #[test]
    fn an_edited_system_prompt_survives_the_migration() {
        let mut document = legacy_document("Tu écris en français, toujours.");
        let log = migrate_legacy(&mut document);

        let settings: Settings = serde_json::from_value(document).expect("migrated settings");
        assert_eq!(settings.llm.prompt, PROMPT_CUSTOM);
        assert_eq!(settings.llm.system_prompt(), "Tu écris en français, toujours.");
        // The migrated file must show both alternatives, not just the text it
        // rescued: a `prompts` map with one entry is a map nobody switches out
        // of.
        assert!(settings.llm.prompts.contains_key(PROMPT_FAITHFUL));
        assert!(settings.llm.prompts.contains_key(PROMPT_STRUCTURED));
        assert!(
            log.iter().any(|line| line.contains(PROMPT_CUSTOM)),
            "the migration must say where the prompt went: {log:?}"
        );
    }

    /// Even an unedited one. Recognising the shipped wording and dropping it
    /// would be a guess about intent, and the cost of being wrong is a silently
    /// changed prompt — the exact thing being fixed.
    #[test]
    fn the_phase_four_prompt_is_carried_too() {
        let mut document = legacy_document(STRUCTURED_SYSTEM_PROMPT);
        migrate_legacy(&mut document);

        let settings: Settings = serde_json::from_value(document).expect("migrated settings");
        assert_eq!(settings.llm.prompt, PROMPT_CUSTOM);
        assert_eq!(settings.llm.system_prompt(), STRUCTURED_SYSTEM_PROMPT);
    }

    #[test]
    fn the_temperature_is_carried_and_the_dead_fields_are_named() {
        let mut document = legacy_document("anything");
        let log = migrate_legacy(&mut document);

        let settings: Settings = serde_json::from_value(document.clone()).expect("migrated");
        assert_eq!(settings.llm.temperature, 0.4);
        assert!(
            document.get("ollama").is_none(),
            "the dead block must be gone"
        );

        for field in ["endpoint", "model", "modelsDir"] {
            assert!(
                log.iter().any(|line| line.contains(&format!("ollama.{field}"))),
                "{field} was dropped without saying so: {log:?}"
            );
        }
    }

    #[test]
    fn a_file_with_no_ollama_block_is_left_alone() {
        let mut document = serde_json::json!({ "llm": { "prompt": PROMPT_STRUCTURED } });
        assert!(migrate_legacy(&mut document).is_empty());
    }

    /// Both prompts are written out, so choosing between them is an edit rather
    /// than a search through the source.
    #[test]
    fn a_fresh_file_ships_both_prompts() {
        let settings = Settings::default();
        assert_eq!(settings.llm.prompt, PROMPT_FAITHFUL);
        assert_eq!(settings.llm.system_prompt(), FAITHFUL_SYSTEM_PROMPT);
        assert!(settings.llm.prompts.contains_key(PROMPT_STRUCTURED));
        assert_eq!(settings.llm.unknown_prompt(), None);
    }

    #[test]
    fn selecting_structured_selects_the_phase_four_wording() {
        let mut settings = Settings::default();
        settings.llm.prompt = PROMPT_STRUCTURED.to_owned();
        assert_eq!(settings.llm.system_prompt(), STRUCTURED_SYSTEM_PROMPT);
    }

    /// A trimmed `prompts` map still switches: the map overrides the built-in
    /// text, it is not the only copy of it.
    #[test]
    fn a_shipped_name_resolves_without_a_map_entry() {
        let mut settings = Settings::default();
        settings.llm.prompts.clear();
        settings.llm.prompt = PROMPT_STRUCTURED.to_owned();

        assert_eq!(settings.llm.system_prompt(), STRUCTURED_SYSTEM_PROMPT);
        assert_eq!(settings.llm.unknown_prompt(), None);
    }

    fn scratch_config(name: &str, contents: &[u8]) -> Config {
        let path = std::env::temp_dir().join(format!("steno-settings-{name}.json"));
        std::fs::write(&path, contents).expect("scratch settings");
        Config {
            path,
            writing: Mutex::new(()),
        }
    }

    /// Notepad and PowerShell 5.1 both write one; JSON has nowhere to put it.
    #[test]
    fn a_byte_order_mark_is_tolerated() {
        let mut contents = vec![0xEF, 0xBB, 0xBF];
        contents.extend_from_slice(br#"{"llm":{"temperature":0.7}}"#);

        let config = scratch_config("bom", &contents);
        assert_eq!(config.get().llm.temperature, 0.7);
    }

    /// The loss FF is really about: a file Steno cannot read is a file Steno
    /// must not write.
    #[test]
    fn an_unparseable_file_is_never_overwritten() {
        let config = scratch_config("corrupt", b"{ this is not json");

        let result = config.set_model_id("ggml-large-v3.bin");

        assert!(result.is_err(), "a save over a broken file must be refused");
        assert_eq!(
            std::fs::read(&config.path).expect("still there"),
            b"{ this is not json"
        );
    }

    #[test]
    fn a_typo_falls_back_to_faithful_and_is_reported() {
        let mut settings = Settings::default();
        settings.llm.prompt = "faithfull".to_owned();

        assert_eq!(settings.llm.system_prompt(), FAITHFUL_SYSTEM_PROMPT);
        assert_eq!(settings.llm.unknown_prompt(), Some("faithfull"));
    }
}
