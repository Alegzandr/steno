# Local patches

This is whisper-rs 0.16.0 from crates.io, unmodified except for what is listed
here. `whisper-rs-sys` is **not** vendored: this crate depends on it from
crates.io exactly as the published one does, so whisper.cpp is never rebuilt on
account of this directory.

Re-vendoring means downloading 0.16.0 again and re-applying every patch below.

## 1. `set_carry_initial_prompt`

**File:** `src/whisper_params.rs`, immediately after `set_initial_prompt`.

**Why:** `whisper_full_params.carry_initial_prompt` exists in whisper.cpp (added
in 1.7.2) and in the generated bindings of `whisper-rs-sys` 0.15, but
whisper-rs 0.16 exposes no setter for it. Without the flag, `initial_prompt`
biases the first 30-second window and no other, so Steno's custom vocabulary
stops applying two thirds of the way through a 90-second brainstorm.

**Why patch rather than work around:** `FullParams.fp` is `pub(crate)` and
`WhisperState.ptr` is private, so there is no seam to reach the field from
outside. Reaching it without this patch would mean reimplementing context
creation, state creation, `full()` and segment iteration on raw `-sys` calls —
replacing whisper-rs rather than extending it.

**The change:**

```rust
    pub fn set_carry_initial_prompt(&mut self, carry_initial_prompt: bool) {
        self.fp.carry_initial_prompt = carry_initial_prompt;
    }
```

plus its doc comment. Nothing else in the file is touched, and the shape is
copied from the neighbouring `set_suppress_nst` so the diff is trivially
reviewable.

**Upstream:** intended for <https://codeberg.org/tazz4843/whisper-rs>. Drop this
patch, and the whole `vendor/` directory, once a release carries the setter.
