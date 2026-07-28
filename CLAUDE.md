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
- Transcription: `whisper-rs` (whisper.cpp bindings). The model is chosen at
  first launch and recorded in settings.json: `ggml-large-v3.bin` when a GPU
  backend is compiled in *and* the adapter reports at least 12 GB,
  `ggml-large-v3-turbo-q5_0.bin` otherwise. CPU by default; CUDA and Vulkan
  behind opt-in cargo features. Dev machine is Windows x64 with an Nvidia GPU.
  macOS/Metal and Linux branches may be written but are not testable here.
- Formatting: Ollama HTTP API at http://localhost:11434, streaming,
  default model `qwen3:14b`
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
- Raw transcript appears in the editor as soon as Whisper finishes. Never make
  the user wait for the LLM to see something. Formatting is not automatic — see
  "Product framing": cleanup is a manual, whole-buffer action. While it streams,
  the incoming text is shown in a muted color *over* the buffer and enters it
  only when complete, so the buffer never holds a half-generated state.
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

## Verification platform

Linux/container verification does not validate this project. The target is
Windows/WASAPI. A green `cargo check` in a container is necessary, never
sufficient. Any change touching `audio/capture.rs`, `cpal`, or the `windows` /
`windows-core` dependencies must be re-verified on the Windows dev machine
before being reported as done. State explicitly in every report which platform
each check ran on.

On Linux, cpal compiles the ALSA backend: the WASAPI backend is never built,
and the `windows-core` resolution trap below cannot manifest. ALSA device
enumeration proves nothing about WASAPI device naming or behaviour.

## Build prerequisites

`whisper-rs-sys` compiles whisper.cpp with CMake and generates its bindings
with bindgen, so the *default CPU build* needs two tools beyond the Rust
toolchain. Neither is optional, and neither ships with Visual Studio in a form
the build can find.

| Tool | Why | Install |
| --- | --- | --- |
| CMake, on PATH | builds whisper.cpp | `winget install Kitware.CMake` |
| LLVM (libclang) | bindgen | `winget install LLVM.LLVM` |
| CUDA Toolkit 12+ | `--features cuda` only | `winget install Nvidia.CUDA` |
| Ollama | formatting | `winget install Ollama.Ollama`, then `ollama pull qwen3:14b` |

Verified against CMake 4.4.0, LLVM 22.1.8, CUDA 13.3 and Ollama 0.32.4.

The CUDA build additionally needs `CUDA_PATH_V<major>_<minor>` in the
environment, not just `CUDA_PATH`: the MSBuild integration reads the versioned
variable and fails with an empty `CudaToolkitDir` without it. Both are set
machine-wide by the installer, so this only bites a shell that inherited a
stale environment.

Set `CMAKE_CUDA_ARCHITECTURES` to the target card (`89` for Ada / RTX 40xx).
The build script forwards any `CMAKE_*` variable to CMake. Without it ggml
builds for every architecture it knows, which is slow and can fail outright
against a CUDA release that has dropped the older ones.

Ollama's model store is not always `%USERPROFILE%\.ollama\models`. The desktop
app remembers a relocated store in its own settings database and passes it to
the server *it* starts; a bare `ollama serve` does not inherit that and falls
back to the default, which on a relocated setup is an empty directory. On this
dev machine the store is `D:\.ollama\models`.

That only bites when Steno has to start the server itself — adopting a running
one is unaffected — and the symptom is a model you demonstrably have being
reported as missing, with `ollama pull` offered as the fix. Set
`ollama.modelsDir` in settings.json; it is passed as `OLLAMA_MODELS` to a
spawned server and ignored when adopting one. Never guess the path in code.

Keep the CUDA build in its own target directory (`CARGO_TARGET_DIR=target-cuda`,
already gitignored) so switching features does not rebuild whisper.cpp each
time.

## Known dependency trap

`cpal` accepts `windows` and `windows-core` in the range >=0.61,<=0.62 as two
independent ranges. Tauri pins `windows` to 0.61, nothing constrains
`windows-core`, so Cargo resolves 0.62 and the WASAPI backend fails to
compile. `windows-core` is pinned to 0.61.2 in Cargo.toml and Cargo.lock.
Never run an untargeted `cargo update`. If the WASAPI backend suddenly stops
compiling, check this first.

## Vendored whisper-rs patch

`src-tauri/vendor/whisper-rs` is whisper-rs 0.16.0 with one three-line addition:
`FullParams::set_carry_initial_prompt`, which the published crate does not
expose. `Cargo.toml` points at it with `path = "vendor/whisper-rs"`.

Without the flag, `initial_prompt` biases only the first 30-second window, so
the custom vocabulary stops applying after thirty seconds of a dictation. **The
failure mode is silent**: no compile error, no warning, just worse transcription
of technical terms in everything past the opening window. Treat a
`carry_initial_prompt` regression as a correctness bug, not a tuning issue.

Three things keep it from disappearing:

- `transcribe::engine` calls the setter unconditionally, so reverting to the
  crates.io crate is a compile error.
- `tests/vendored_whisper_rs.rs` fails with an explanatory message if the patch,
  the `path` dependency, or the crates.io `whisper-rs-sys` is lost.
- `vendor/whisper-rs/PATCHES.md` carries the diff and the reason.

`whisper-rs-sys` is deliberately *not* vendored — it comes from crates.io as it
does for the published crate, so this costs no extra build time and whisper.cpp
is not rebuilt. Never give that dependency a `path`.

Delete `vendor/` and go back to crates.io as soon as an upstream release carries
the setter.

## Measured behaviour, not assumed

Findings from phase 3 on the dev machine (RTX 4080 SUPER, Ryzen 7 9850X3D,
whisper.cpp 1.8.3, Ollama 0.32.4). Re-measure with `examples/rtf.rs` and
`examples/vram.rs` rather than trusting these numbers after a dependency bump.

- **`no_speech_thold` does not work.** whisper.cpp derives the per-segment
  no-speech probability from the `<|nospeech|>` token after the first decode,
  and on large-v3 it reads 0.000 — on noise, and on ten seconds of digital
  silence alike. Internally that value only feeds the rolling context; it never
  suppresses a segment. The guard is kept wired because it costs nothing and
  will start working if upstream fixes it, but nothing may depend on it. The
  denylist is the only guard that catches a hallucination above the RMS floor.
- **Silence produces `Merci.`** — reliably, from both digital silence and
  -40 dBFS noise, which clears the default floor. It is in the shipped
  denylist for that reason.
- **CPU transcription is not viable.** large-v3-turbo-q5_0 runs at RTF 3.2 on
  eight cores; the same model on CUDA runs at 0.012. The CPU build exists as a
  build that works without a toolkit, not as a usable dictation path.
- **Dropping a `WhisperContext` leaves ~222 MiB of video memory held.** It is
  ggml's per-device CUDA state, created on first use and freed only at process
  exit. It is constant across load/unload cycles, so it is not a leak. Ollama,
  by contrast, returns to the byte.
- **Ollama's unload is asynchronous.** The HTTP call returns before the runner
  process holding the memory has exited, so `/api/ps` must be polled; reading
  `nvidia-smi` straight after the response reports the old figure.
- **qwen3:14b costs 73 s to load from cold disk, 2.4 s from the page cache**,
  and holds 9.5 GB. Idle eviction only drops it from video memory, so a
  re-warm within a session pays the 2.4 s, not the 73 s.

## Agent conduct

Never inject synthetic keyboard or mouse events into the live desktop
session without asking first. Other applications are running and may be
foreground. If a test requires real input, tell me and I will perform it.

## Product framing

Steno is a sidecar for developers writing prompts. The user keeps it open on
a second screen, dictates a long messy brainstorm in several bursts, hits one
button to structure it, and pastes the result into an LLM. Optimise for that
loop, not for prose dictation.

Consequences:

- The editor is an accumulation buffer. Each recording appends at the cursor,
  it never replaces the buffer.
- Cleanup is manual, triggered by a button, and operates on the whole buffer.
- Cleanup must be a single undo transaction. One Ctrl+Z restores the exact
  pre-cleanup text.
- Dictated content is technical: identifiers, CLI flags, file paths, library
  names. Custom vocabulary is a core feature, not a nicety.

## MVP scope

Windows only for verification. Keep the code portable (no Windows-only APIs
outside existing #[cfg] blocks) but do not attempt to validate macOS or Linux.
Cross-platform is post-MVP.

Default formatting model is qwen3:14b via Ollama. The model name and the
system prompt both live in settings.json and are user-editable. Changing
either must never require a rebuild.

## VRAM discipline

This runs on dual-use machines: the same GPU plays games. Steno must hold
zero VRAM when it is not actively working. This is a hard requirement, not a
preference. On Windows, oversubscribed VRAM spills to shared system memory
and produces permanent stutter rather than a clean failure, so the symptom is
misattributed to drivers.

- Never use keep_alive: -1. The formatting model is unloaded when Steno hides
  or quits.
- The WhisperContext is dropped on the same trigger. It is not kept resident
  for the lifetime of the process.
- Both are warmed on window show, in the background, so the load cost is
  absorbed while the user is speaking rather than paid at the point of use.
- Steno only kills an Ollama process it started itself. A pre-existing
  instance on 11434 is used and left alone.
- Acceptance: nvidia-smi reports the same used-memory figure before launch and
  after quit.

Accepted deviation: ggml's per-device CUDA state (~222 MiB) persists from
first whisper use until process exit. It is not a leak and post-quit VRAM is
zero. A whisper sidecar process would remove it at the cost of IPC and a
process cold start. Judged not worth it. Do not implement one without asking.
