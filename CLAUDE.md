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
- Formatting: `llama-cpp-2` (llama.cpp bindings), in-process, streaming,
  default model `Qwen3-14B-Q4_K_M.gguf` downloaded from Hugging Face on first
  use. There is no server, no HTTP, no port. Ollama was the phase-4 engine and
  was removed in 5.1 — do not reintroduce it, and do not treat any surviving
  mention of `11434`, `keep_alive` or a job object as current.
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

`whisper-rs-sys` and `llama-cpp-sys-2` both compile their C++ with CMake and
generate their bindings with bindgen, so the *default CPU build* needs two tools
beyond the Rust toolchain. Neither is optional, and neither ships with Visual
Studio in a form the build can find.

| Tool | Why | Install |
| --- | --- | --- |
| CMake, on PATH | builds whisper.cpp and llama.cpp | `winget install Kitware.CMake` |
| LLVM (libclang) | bindgen | `winget install LLVM.LLVM` |
| CUDA Toolkit 12+ | `--features cuda` only | `winget install Nvidia.CUDA` |

Verified against CMake 4.4.0, LLVM 22.1.8 and CUDA 13.3. Nothing has to be
installed for formatting: the model is a file Steno downloads, and llama.cpp is
linked into the process.

The CUDA build additionally needs `CUDA_PATH_V<major>_<minor>` in the
environment, not just `CUDA_PATH`: the MSBuild integration reads the versioned
variable and fails with an empty `CudaToolkitDir` without it. Both are set
machine-wide by the installer, so this only bites a shell that inherited a
stale environment.

Set `CMAKE_CUDA_ARCHITECTURES` to the target card (`89` for Ada / RTX 40xx).
The build script forwards any `CMAKE_*` variable to CMake. Without it ggml
builds for every architecture it knows, which is slow and can fail outright
against a CUDA release that has dropped the older ones.

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

## Two ggmls in one process, and where the DLLs go

`llama-cpp-2` is built with `dynamic-link` + `dynamic-backends`, so llama.cpp
arrives as DLLs while whisper.cpp is still linked statically. That means Steno
contains **two independent ggml backend registries** and they do not see each
other. Everything below follows from that and was established by installing a
bundle, not by reading documentation.

Fifteen DLLs must sit **flat, beside the executable**. Two kinds, failing two
different ways:

- `llama.dll`, `ggml.dll`, `ggml-base.dll`, `llama-common.dll` are resolved by
  the Windows loader before `main`. Missing one gives exit code `0xC0000135`
  and no message whatsoever.
- `ggml-cpu-*.dll` (nine variants) and `ggml-cuda.dll` are opened later by ggml
  itself. Missing them is not an error: the app starts, dictation works, and the
  first Clean up fails with `no backends are loaded`.

**Never call `load_backends_from_path` from Rust.** It fills whisper.cpp's
static registry, which nothing then consults. `llama.dll` populates its own
registry through `ggml_backend_load_best`, whose search path is fixed in ggml
and is not ours to redirect: the compile-time `GGML_BACKEND_DIR`, then the
executable's directory, then the working directory. **No `backends/`
subdirectory is ever searched.** `format::backends` therefore only checks and
reports; it does not load.

The trap that caught this: `GGML_BACKEND_DIR` points into `target/` on a dev
machine, so a bundle with the DLLs in the wrong place works here and only here.
A layout change is verified by renaming the build tree away and running the
*installed* app. `build.rs` stages `llama-cpp-sys-2`'s `out/bin` and
`out/backends` into one `src-tauri/runtime/`, which `tauri.conf.json` maps with
`"runtime/*.dll": ""`.

**Open question, not yet decided.** A CUDA build imports `cublas64_*.dll` at
load time — from the executable itself, not only from `ggml-cuda.dll` — and it
is not shipped. With the toolkit off `PATH`, measured: exit `0xC0000135` before
`main`. Adding `cublas64_13.dll` and `cublasLt64_13.dll` alone is sufficient and
they cost 492 MB against a 79 MB installer. `cudart` is statically linked and is
not a problem. Ship, static-link, or detect and tell the user — undecided; do
not pick one unilaterally.

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

Findings on the dev machine (RTX 4080 SUPER, Ryzen 7 9850X3D, whisper.cpp 1.8.3,
llama.cpp via `llama-cpp-2`). Re-measure with `examples/rtf.rs`,
`examples/vram.rs` and `examples/cleanup.rs` rather than trusting these numbers
after a dependency bump.

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
  exit. It is constant across load/unload cycles, so it is not a leak. Dropping
  the llama.cpp model, by contrast, returns to the byte.
- **`mmap` is off, deliberately.** It is llama.cpp's default and it is wrong
  here: every byte of the file is destined for video memory, so mapping buys
  nothing but a page fault per tensor, taken in tensor order rather than file
  order. Measured with the file fully cached: 2784 ms with mmap against 1098 ms
  for the explicit read. Do not "fix" `with_use_mmap(false)` back to the
  default.
- **Model load is bounded by sequential read throughput, and the spread is
  fifty-fold.** The same 9 GB GGUF loads in 1.1 s from the NVMe and 56.4 s from
  a 7200 rpm SATA disk (1288 MB/s against 162 MB/s). Neither figure is visible
  to a user who has relocated the app data directory onto the big spare drive,
  so `storage.rs` asks the device (`IOCTL_STORAGE_QUERY_PROPERTY`,
  `StorageDeviceSeekPenaltyProperty`) and, when the descriptor comes back empty
  — optional, and absent behind some RAID controllers, USB bridges and network
  redirectors — judges the first real read instead. It warns; it never claims a
  drive is fast.
- **Qwen3-14B-Q4_K_M, official Hugging Face file**: 9,001,752,960 bytes,
  sha256 `500a8806…`. Resident in 1191 ms warm, first token 311–336 ms, 878
  prompt tokens, ~69.5 tok/s. The Ollama blob of the same quantisation gave
  67.8 tok/s with identical tokenisation, so the producer does not matter.

## Agent conduct

Never inject synthetic keyboard or mouse events into the live desktop
session without asking first. Other applications are running and may be
foreground. If a test requires real input, tell me and I will perform it.

Never write to my real configuration or data files from a test harness. That
means `%APPDATA%\com.steno.app\settings.json`, the history database, the model
directory — anything the running app owns. Use a temp directory, or a copy, and
point the code under test at it. This has already cost a live settings.json: a
harness rewrote it to construct a pre-5.1 fixture, wrote a UTF-8 BOM by
accident, and the app's next save overwrote the file with defaults. The bug it
exposed was real and is fixed, but the fixture had no business being there.
Reading a real file to copy it is fine; writing one is not.

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

Default formatting model is `Qwen3-14B-Q4_K_M.gguf`, loaded in-process by
llama.cpp. The model path and the system prompts both live in settings.json and
are user-editable. Changing either must never require a rebuild.

Two system prompts ship: `faithful` (the default — punctuation, paragraphs, no
reorganisation) and `structured` (the phase-4 wording, which imposes headings
and lists). `llm.prompt` selects by name from `llm.prompts`. A name absent from
the map falls back to the compiled text of the same name, an unknown name falls
back to `faithful` and says so once. `custom` is not a third mode: it is where
the pre-5.1 settings migration deposits a system prompt it refuses to discard.
Do not add a third mode without asking.

## VRAM discipline

This runs on dual-use machines: the same GPU plays games. Steno must hold
zero VRAM when it is not actively working. This is a hard requirement, not a
preference. On Windows, oversubscribed VRAM spills to shared system memory
and produces permanent stutter rather than a clean failure, so the symptom is
misattributed to drivers.

- The formatting model is dropped when Steno hides or quits. It lives in this
  process, so this is an ordinary Rust lifetime — `Drop` on a struct field, not
  a protocol. `LlamaBackend::init` stays for the life of the process (it is a
  global one-shot); the model is what holds the nine gigabytes.
- The WhisperContext is dropped on the same trigger. It is not kept resident
  for the lifetime of the process.
- Both are warmed on window show, in the background, so the load cost is
  absorbed while the user is speaking rather than paid at the point of use.
- The llama.cpp context is built per request, not held beside the model: the KV
  cache is the per-use cost, it takes 12–20 ms to build, and a context that
  outlived a request would hold video memory through exactly the idle stretch
  Steno promises to leave the card alone.
- Acceptance: nvidia-smi reports the same used-memory figure before launch and
  after quit.

Accepted deviation: ggml's per-device CUDA state (~222 MiB) persists from
first whisper use until process exit. It is not a leak and post-quit VRAM is
zero. A whisper sidecar process would remove it at the cost of IPC and a
process cold start. Judged not worth it. Do not implement one without asking.
