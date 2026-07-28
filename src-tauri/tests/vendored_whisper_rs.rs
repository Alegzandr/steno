//! Guards the vendored whisper-rs patch against a silent disappearance.
//!
//! The patch is three lines in a crate nobody reads, and losing it does not
//! break the build in any obvious way — it degrades transcription after the
//! first thirty seconds of every clip, which looks like the model having a bad
//! day. The failure has no error message and no stack trace, so it needs a test
//! that produces one.
//!
//! Two layers, because each catches what the other cannot:
//!
//! 1. `transcribe::engine` calls `set_carry_initial_prompt` unconditionally, so
//!    reverting to the published crate is a *compile* error. That is the real
//!    guard and it needs no test.
//! 2. These assertions, for the case the compile error gets "fixed" by deleting
//!    the call site, or by a dependency refresh that rewrites the manifest.
//!
//! What this does NOT cover: it reads source text, so it proves the patch is
//! present, never that it works. That whisper.cpp honours the flag alongside
//! `no_context` is a behavioural question, answered by measurement — see the
//! `carry` example and the "Vendored whisper-rs patch" section of CLAUDE.md.

use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("could not read {} ({error})", path.display()))
}

#[test]
fn the_carry_initial_prompt_setter_is_still_patched_in() {
    let params = read("vendor/whisper-rs/src/whisper_params.rs");

    assert!(
        params.contains("pub fn set_carry_initial_prompt(&mut self, carry_initial_prompt: bool)"),
        "The vendored whisper-rs has lost the `set_carry_initial_prompt` patch.\n\
         Without it the custom vocabulary biases only the first 30-second window \
         and long dictations quietly lose their technical terms.\n\
         Re-apply it in vendor/whisper-rs/src/whisper_params.rs — see the \
         \"Vendored whisper-rs patch\" section of CLAUDE.md."
    );

    assert!(
        params.contains("self.fp.carry_initial_prompt = carry_initial_prompt;"),
        "`set_carry_initial_prompt` exists but no longer assigns the field."
    );
}

#[test]
fn whisper_rs_is_taken_from_the_vendored_copy() {
    let manifest = read("Cargo.toml");

    let declaration = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("whisper-rs = "))
        .unwrap_or_else(|| {
            panic!("Cargo.toml no longer declares whisper-rs at all");
        });

    assert!(
        declaration.contains("path = \"vendor/whisper-rs\""),
        "whisper-rs is no longer the vendored copy:\n  {declaration}\n\
         The published 0.16 has no `set_carry_initial_prompt`, so this reverts \
         the vocabulary to the first window only. See CLAUDE.md."
    );
}

#[test]
fn the_vendored_copy_does_not_drag_whisper_cpp_with_it() {
    // The whole point of vendoring only the safe wrapper: `whisper-rs-sys`
    // still comes from crates.io, so this patch costs nothing at build time.
    // A `path` on that dependency would mean vendoring whisper.cpp too, and a
    // full CMake rebuild on every checkout.
    let manifest = read("vendor/whisper-rs/Cargo.toml");
    let sys = manifest
        .split("[dependencies.whisper-rs-sys]")
        .nth(1)
        .expect("the vendored crate no longer depends on whisper-rs-sys")
        .split("\n[")
        .next()
        .unwrap_or_default()
        .to_owned();

    assert!(
        !sys.contains("path"),
        "vendor/whisper-rs now points whisper-rs-sys at a path:{sys}\n\
         That vendors whisper.cpp as well and rebuilds it from source. Keep the \
         sys crate on crates.io."
    );
}
