//! The three guards against Whisper inventing speech that was never spoken.
//!
//! On silent or near-silent audio Whisper does not return nothing. It returns
//! whatever filled that acoustic space most often in its training data, which
//! for French is overwhelmingly subtitle boilerplate: *Sous-titres réalisés par
//! la communauté d'Amara.org*, *Merci d'avoir regardé cette vidéo*. It comes
//! back confident and well punctuated, so nothing downstream can tell it apart
//! from a real transcript.
//!
//! Three guards, at three different levels, because each catches cases the
//! others miss:
//!
//! 1. **RMS floor**, before Whisper runs at all. Cheapest, and the only one
//!    that also saves the compute. Misses quiet-but-real dictation if set too
//!    high, and misses hallucinations triggered by background noise that is
//!    loud but wordless.
//! 2. **`no_speech_thold`**, per segment, using Whisper's own estimate that a
//!    segment is silence. **Measured to be inoperative** with whisper.cpp
//!    1.8.3: the probability is the likelihood of the `<|nospeech|>` token
//!    after the first decode, and on large-v3 it reads 0.000 — on noise, and
//!    on ten seconds of digital silence alike. Whisper does not merely fail to
//!    notice silence, it reports maximum confidence that silence is speech.
//!    The wiring is kept because it costs nothing and starts working the day
//!    upstream fixes it, but nothing may be assumed to rest on it.
//! 3. **Denylist**, on the text itself. Consequently the *only* guard that
//!    catches a hallucination over audio loud enough to clear the floor, and
//!    the only one the user can extend without a rebuild. It is scoped to the
//!    whole transcription, never to a segment inside one: see
//!    [`is_wholly_denied`].

use serde::Serialize;

/// Which guard rejected a clip or a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Guard {
    RmsFloor,
    NoSpeech,
    Denylist,
}

impl Guard {
    pub fn as_str(self) -> &'static str {
        match self {
            Guard::RmsFloor => "rms-floor",
            Guard::NoSpeech => "no-speech",
            Guard::Denylist => "denylist",
        }
    }
}

/// Floor for a fully digital-silent buffer, so the reported figure stays a
/// number the UI can format.
const SILENCE_DBFS: f32 = -120.0;

/// Root mean square of the clip, in dBFS. Zero for digital silence maps to
/// `SILENCE_DBFS` rather than negative infinity.
pub fn rms_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return SILENCE_DBFS;
    }

    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    let rms = (sum / samples.len() as f64).sqrt();

    if rms <= 0.0 {
        return SILENCE_DBFS;
    }

    (20.0 * rms.log10()).max(f64::from(SILENCE_DBFS)) as f32
}

/// Whether a whole *transcription* is nothing but boilerplate.
///
/// The scope is the point of this function, and the denylist must only ever be
/// applied through it. The guard may discard a clip; it may never edit one. A
/// real dictation that happens to end on "merci" keeps that word, because the
/// dictation as a whole is not a denylist entry.
///
/// Two shapes count as "the entire content", because Whisper does not segment
/// silence consistently between runs:
///
/// - the segments joined back together reduce to one entry — Whisper split a
///   single boilerplate phrase across a segment boundary;
/// - every segment reduces to an entry on its own — Whisper repeated the
///   boilerplate rather than splitting it.
///
/// Anything else is kept in full, including a clip where only some segments
/// match.
pub fn is_wholly_denied(segments: &[String], denylist: &[String]) -> bool {
    if segments.is_empty() {
        return false;
    }

    is_denied(&segments.join(" "), denylist)
        || segments.iter().all(|segment| is_denied(segment, denylist))
}

/// Whether one string, on its own, is one of the known hallucinations.
///
/// A building block for [`is_wholly_denied`], which owns the scoping rule.
/// Equality after normalisation, deliberately not a substring test: `Merci` and
/// `Abonnez-vous` are things a developer genuinely dictates, and a substring
/// rule would delete the sentence containing them.
fn is_denied(text: &str, denylist: &[String]) -> bool {
    let candidate = normalise(text);
    if candidate.is_empty() {
        return false;
    }

    denylist
        .iter()
        .any(|phrase| normalise(phrase) == candidate)
}

/// Case, accent-neutral apostrophes, surrounding punctuation and repeated
/// whitespace all vary between Whisper runs of the same hallucination, so none
/// of them may decide a match.
fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;

    for ch in text.chars() {
        // U+2019 and friends: Whisper picks whichever apostrophe it feels like.
        let ch = match ch {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' => '\'',
            other => other,
        };

        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }

        if pending_space {
            out.push(' ');
            pending_space = false;
        }

        for lower in ch.to_lowercase() {
            out.push(lower);
        }
    }

    // Trailing sentence punctuation only. Interior punctuation is meaningful:
    // `d'Amara.org` must keep both marks.
    out.trim_end_matches(['.', '!', '?', ',', ';', ':', ' '])
        .trim_start_matches(['.', '!', '?', ',', ';', ':', ' ', '-', '\u{2014}'])
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn denylist() -> Vec<String> {
        crate::config::WhisperSettings::default().hallucinations
    }

    #[test]
    fn digital_silence_is_far_below_any_sane_floor() {
        assert_eq!(rms_dbfs(&[0.0; 16_000]), SILENCE_DBFS);
        assert_eq!(rms_dbfs(&[]), SILENCE_DBFS);
    }

    #[test]
    fn full_scale_is_zero_dbfs() {
        // A square wave at full scale has an RMS of exactly 1.0.
        let square: Vec<f32> = (0..16_000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!((rms_dbfs(&square) - 0.0).abs() < 0.001);
    }

    #[test]
    fn a_quiet_room_lands_below_the_default_floor() {
        // -60 dBFS: a gated microphone's idea of silence.
        let quiet: Vec<f32> = (0..16_000)
            .map(|i| if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let level = rms_dbfs(&quiet);
        assert!(level < -50.0, "expected below the default floor, got {level}");
    }

    #[test]
    fn catches_the_amara_hallucination_however_it_is_punctuated() {
        let list = denylist();
        for variant in [
            "Sous-titres réalisés par la communauté d'Amara.org",
            "sous-titres réalisés par la communauté d'Amara.org.",
            "  Sous-titres réalisés par la communauté d\u{2019}Amara.org  ",
            "Sous-titres  réalisés par la communauté d'Amara.org!",
        ] {
            assert!(is_denied(variant, &list), "missed {variant:?}");
        }
    }

    #[test]
    fn leaves_real_dictation_alone() {
        let list = denylist();
        for real in [
            "Merci de vérifier le endpoint avant de merger",
            "On abonne le composant au store",
            "Il faut regarder cette vidéo de la conf Rust",
            "",
        ] {
            assert!(!is_denied(real, &list), "wrongly denied {real:?}");
        }
    }

    #[test]
    fn catches_the_bare_merci_that_silence_actually_produces() {
        // What large-v3 returns for ten seconds of digital silence, measured.
        let list = denylist();
        for variant in ["Merci.", "merci", " Merci ! "] {
            assert!(is_denied(variant, &list), "missed {variant:?}");
        }
    }

    #[test]
    fn a_dictation_that_ends_on_merci_keeps_it() {
        // The regression this scoping exists for. "Merci" is a denylist entry
        // and the last segment matches it exactly, but the transcription as a
        // whole is real speech, so nothing may be dropped.
        let list = denylist();
        let segments = [
            "On refactor le middleware avant de merger".to_owned(),
            "Merci.".to_owned(),
        ];
        assert!(!is_wholly_denied(&segments, &list));
    }

    #[test]
    fn boilerplate_split_across_segments_is_still_caught() {
        // Whisper does not put the segment boundary in the same place twice.
        let list = denylist();
        let segments = [
            "Sous-titres réalisés par la".to_owned(),
            "communauté d'Amara.org".to_owned(),
        ];
        assert!(is_wholly_denied(&segments, &list));
    }

    #[test]
    fn boilerplate_repeated_across_segments_is_still_caught() {
        let list = denylist();
        let segments = ["Merci.".to_owned(), "Merci beaucoup".to_owned()];
        assert!(is_wholly_denied(&segments, &list));

        // And the single-segment case silence actually produces.
        assert!(is_wholly_denied(&["Merci.".to_owned()], &list));
    }

    #[test]
    fn an_empty_transcription_is_not_a_denylist_hit() {
        // Nothing to drop, and the caller must be able to tell "the denylist
        // emptied this" from "Whisper returned nothing".
        assert!(!is_wholly_denied(&[], &denylist()));
    }

    #[test]
    fn a_denied_phrase_inside_a_sentence_is_kept() {
        // Whole-segment matching: the guard must not eat a real sentence that
        // happens to quote one of the phrases.
        let list = denylist();
        assert!(!is_denied(
            "Le modèle sort systématiquement Merci d'avoir regardé cette vidéo sur du silence",
            &list
        ));
    }
}
