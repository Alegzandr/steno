//! Measures whether the custom vocabulary survives past the first 30-second
//! window, and whether it does so alongside `no_context(true)`.
//!
//! The two flags are not obviously compatible: `carry_initial_prompt` prepends
//! the initial prompt to every window's prompt, `no_context` clears the decoded
//! text carried between windows, and both write to the same prompt buffer in
//! whisper.cpp. Whether the first survives the second is a question about an
//! implementation, so it is answered by running it rather than by reading it.
//!
//! Feed it a clip whose technical terms all fall in the final thirty seconds.
//! `tools/carry-passage.fr.txt` is built for exactly that: a minute of ordinary
//! French with no jargon at all, then half a minute of nothing but jargon.
//!
//! ```text
//! cargo run --release --features cuda --example carry -- <models-dir> <clip.wav> [model-id]
//! ```
//!
//! Both runs use one `Engine` and one decode of the same samples, so the only
//! difference between them is the flag.

use std::path::PathBuf;
use std::time::Instant;

use steno_lib::config::WhisperSettings;
use steno_lib::model;
use steno_lib::transcribe::{engine::Engine, read_clip};

fn main() {
    let mut args = std::env::args().skip(1);

    let (Some(models_dir), Some(clip)) = (
        args.next().map(PathBuf::from),
        args.next().map(PathBuf::from),
    ) else {
        eprintln!("usage: carry <models-dir> <clip.wav> [model-id]");
        std::process::exit(2);
    };

    let spec = match args.next() {
        Some(id) => model::find(&id).unwrap_or_else(|| {
            eprintln!("unknown model {id:?}");
            std::process::exit(2);
        }),
        None => model::default_spec(),
    };

    let samples = read_clip(&clip).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", clip.display());
        std::process::exit(1);
    });
    let seconds = samples.len() as f64 / 16_000.0;

    println!("backend        {}", model::backend_name());
    println!("model          {}", spec.id);
    println!("clip           {:.1} s  ({:.0} windows)", seconds, (seconds / 30.0).ceil());

    let engine = Engine::load(&models_dir.join(spec.id), spec.id).unwrap_or_else(|error| {
        eprintln!("could not load the model: {error}");
        std::process::exit(1);
    });

    let settings = WhisperSettings::default();
    println!("vocabulary     {} terms", settings.vocabulary.len());
    println!();

    let off = run(&engine, &samples, &settings, false);
    let on = run(&engine, &samples, &settings, true);

    report(&settings.vocabulary, &off, &on);
}

struct Run {
    carry: bool,
    text: String,
    ms: u128,
}

fn run(engine: &Engine, samples: &[f32], base: &WhisperSettings, carry: bool) -> Run {
    let mut settings = base.clone();
    settings.carry_initial_prompt = carry;

    let started = Instant::now();
    let transcript = engine.run(samples, &settings).unwrap_or_else(|error| {
        eprintln!("transcription failed: {error}");
        std::process::exit(1);
    });

    Run {
        carry,
        text: transcript.text,
        ms: started.elapsed().as_millis(),
    }
}

/// Prints both transcripts and, more usefully, which vocabulary terms each one
/// actually recovered.
///
/// Two scores, because the lenient one is misleading on its own. `approx` is a
/// case-insensitive substring test: it counts `ruste` as a hit for `Rust` and
/// `NPM` as a hit for `npm`, which is exactly the kind of near-miss the custom
/// vocabulary exists to eliminate. `exact` requires the term verbatim, case
/// included, as a whole word — the spelling that can be pasted into a prompt
/// without being fixed by hand. Only `exact` answers the question this
/// measurement was built for.
fn report(vocabulary: &[String], off: &Run, on: &Run) {
    for run in [off, on] {
        println!("=== carry_initial_prompt = {} ({} ms) ===", run.carry, run.ms);
        println!("{}", run.text);
        println!();
    }

    // A control: the flag must not touch the first window at all. If the two
    // runs start to differ early, something other than the prompt changed and
    // no conclusion below is safe.
    let shared = common_prefix(&off.text, &on.text);
    println!(
        "identical prefix   {} of {} chars — the two runs diverge only after this point",
        shared,
        off.text.chars().count()
    );
    println!();

    println!("{:<14} {:^17} {:^17}", "", "carry off", "carry on");
    println!("{:<14} {:^8} {:^8} {:^8} {:^8}", "term", "exact", "approx", "exact", "approx");
    println!("{}", "-".repeat(50));

    let mut totals = [0usize; 4];
    for term in vocabulary {
        let hits = [
            exact(&off.text, term),
            approx(&off.text, term),
            exact(&on.text, term),
            approx(&on.text, term),
        ];
        for (total, hit) in totals.iter_mut().zip(hits) {
            *total += usize::from(hit);
        }

        // A term both runs spelled correctly says nothing about the flag.
        if !(hits[0] && hits[2]) {
            println!(
                "{:<14} {:^8} {:^8} {:^8} {:^8}",
                term,
                mark(hits[0]),
                mark(hits[1]),
                mark(hits[2]),
                mark(hits[3])
            );
        }
    }

    println!("{}", "-".repeat(50));
    let n = vocabulary.len();
    println!(
        "{:<14} {:^8} {:^8} {:^8} {:^8}",
        "recovered",
        format!("{}/{n}", totals[0]),
        format!("{}/{n}", totals[1]),
        format!("{}/{n}", totals[2]),
        format!("{}/{n}", totals[3])
    );
}

fn mark(hit: bool) -> &'static str {
    if hit {
        "yes"
    } else {
        "-"
    }
}

/// The term verbatim, case included, not glued to a longer word.
///
/// Word-bounded rather than a bare `contains` so `Rust` does not match `ruste`,
/// which is the specific failure the vocabulary is supposed to prevent. The
/// cost is that a French inflection of a borrowed verb — `refactorer` for
/// `refactor` — counts as a miss here and only shows up in the approximate
/// column.
fn exact(haystack: &str, term: &str) -> bool {
    haystack.match_indices(term).any(|(at, _)| {
        let before = haystack[..at].chars().next_back();
        let after = haystack[at + term.len()..].chars().next();
        let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
        boundary(before) && boundary(after)
    })
}

/// Case-insensitive substring: the term is in there somewhere, in some shape.
fn approx(haystack: &str, term: &str) -> bool {
    haystack.to_lowercase().contains(&term.to_lowercase())
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}
