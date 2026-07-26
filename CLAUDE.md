# Steno

Local-first voice dictation app. Push-to-talk capture, on-device Whisper
transcription, local LLM formatting into clean Markdown, displayed in a
floating mini-editor. Output is copied to clipboard as Markdown, intended
to be pasted into LLM prompts.

## Locked stack

Do not substitute any of these without asking first.

- Shell: Tauri 2 (Rust core, TypeScript frontend, Vite)
- Frontend: React + TypeScript, CodeMirror 6 with the Markdown language mode
- Audio capture: `cpal`, 16 kHz mono f32, ring buffer
- Transcription: `whisper-rs` (whisper.cpp bindings), model
  `ggml-large-v3-turbo-q5_0.bin`. CPU by default; CUDA and Vulkan behind
  opt-in cargo features. Dev machine is Windows x64 with an Nvidia GPU.
  macOS/Metal and Linux branches may be written but are not testable here.
- Formatting: Ollama HTTP API at http://localhost:11434, streaming,
  default model `qwen3:8b`
- Storage: `tauri-plugin-sql` (SQLite)
- Plugins: `tauri-plugin-global-shortcut`, `tauri-plugin-clipboard-manager`

## Hard constraints

- Fully offline after the initial model download. No cloud API, no telemetry,
  no analytics, no crash reporting. Never propose OpenAI, Anthropic, Deepgram
  or any hosted transcription/LLM endpoint.
- Audio never leaves the machine and is never written to disk outside the OS
  temp dir. Temp WAV files are deleted immediately after transcription.
- Primary dictation language is French. UI language is English.
- The editor shows raw Markdown syntax. Never introduce a WYSIWYG editor.
- Bundle size matters. No Electron, no heavy UI framework, no CSS library
  beyond plain CSS or CSS modules.
- WASAPI input devices almost never expose a 16 kHz config. Capture at the
  device's native rate and resample to 16 kHz mono in Rust. Never assume
  `SupportedStreamConfig` will offer 16000 Hz.

## UX invariants

- Window: 400x600, frameless, always-on-top, no dock/taskbar icon, does not
  steal focus from the app the user was typing in.
- Global shortcut is push-to-talk: hold to record, release to transcribe.
- Raw transcript appears in the editor first, in a muted color, as soon as
  Whisper finishes. The LLM-formatted version then streams in and replaces
  it, in normal color. Never make the user wait for the LLM to see something.
- Cmd/Ctrl+Enter copies the editor contents and hides the window.
- Esc cancels an in-flight recording or generation.

## Working agreement

- Build one phase at a time. Do not start a phase before I ask for it.
- After each phase: `cargo check`, `npm run build`, and confirm the app
  launches before summarizing.
- Keep Rust commands in `src-tauri/src/commands/`, one module per domain
  (audio, transcribe, format, history).
- No mock data, no placeholder implementations, no "TODO: implement later"
  in code you present as done. If something can't be finished, say so.

## Known dependency trap

`cpal` accepts `windows` and `windows-core` in the range >=0.61,<=0.62 as two
independent ranges. Tauri pins `windows` to 0.61, nothing constrains
`windows-core`, so Cargo resolves 0.62 and the WASAPI backend fails to
compile. `windows-core` is pinned to 0.61.2 in Cargo.toml and Cargo.lock.
Never run an untargeted `cargo update`. If the WASAPI backend suddenly stops
compiling, check this first.

## Agent conduct

Never inject synthetic keyboard or mouse events into the live desktop
session without asking first. Other applications are running and may be
foreground. If a test requires real input, tell me and I will perform it.
