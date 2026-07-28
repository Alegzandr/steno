//! Turning the custom vocabulary into Whisper's `initial_prompt`.
//!
//! Whisper has no vocabulary API. The only way to bias it towards jargon is to
//! prepend text it treats as the transcript so far, so the decoder's language
//! model has already "seen" the words you are about to say. It works well for
//! spellings it would otherwise mangle — `cargo` as *Cargo*, `Tauri` as
//! *Taury*, `npm` as *N.P.M.* — and it costs nothing at run time.

/// How much of Whisper's prompt budget the vocabulary may use.
///
/// whisper.cpp keeps at most `n_text_ctx / 2` tokens of prompt, which is 224
/// for every large model, and it keeps the *last* ones: overflow does not
/// error, it silently drops the beginning of your list. The cap is set below
/// that so truncation is our decision and can be reported.
const TOKEN_BUDGET: usize = 200;

/// Rough tokens-per-character for Whisper's multilingual BPE on the kind of
/// text this list holds. French words run about 3.5 characters per token;
/// identifiers like `tauri-plugin-global-shortcut` fragment much harder, near
/// 2. The pessimistic figure is the useful one — overshooting the budget is
/// silent, undershooting merely wastes a little bias.
const CHARS_PER_TOKEN: f32 = 2.5;

/// The practical ceiling, in terms, before quality degrades rather than merely
/// truncating. Reported in the UI and in the docs; not enforced.
pub const RECOMMENDED_MAX_TERMS: usize = 80;

/// Builds the prompt, and reports how many terms had to be left out.
///
/// Terms are joined as a plain comma-separated sentence rather than a list of
/// bare words: Whisper conditions on it as ordinary text, and text that looks
/// like a sentence biases spelling without pulling the transcript towards
/// list-shaped output.
pub fn build(vocabulary: &[String]) -> Prompt {
    let mut used = Vec::new();
    let mut chars = 0usize;
    let budget_chars = (TOKEN_BUDGET as f32 * CHARS_PER_TOKEN) as usize;

    for term in vocabulary {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }

        // ", " between terms, and the trailing full stop.
        let cost = term.chars().count() + 2;
        if chars + cost > budget_chars {
            break;
        }

        chars += cost;
        used.push(term);
    }

    let dropped = vocabulary.iter().filter(|t| !t.trim().is_empty()).count() - used.len();
    if dropped > 0 {
        eprintln!(
            "vocabulary: {dropped} term(s) did not fit in Whisper's {TOKEN_BUDGET}-token \
             prompt budget and were left out"
        );
    }

    let text = if used.is_empty() {
        String::new()
    } else {
        format!("{}.", used.join(", "))
    };

    Prompt {
        text,
        used: used.len(),
        dropped,
    }
}

#[derive(Clone, Debug)]
pub struct Prompt {
    pub text: String,
    pub used: usize,
    pub dropped: usize,
}

impl Prompt {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(list: &[&str]) -> Vec<String> {
        list.iter().map(|t| (*t).to_owned()).collect()
    }

    #[test]
    fn joins_terms_into_a_sentence() {
        let prompt = build(&terms(&["Tauri", "cargo", "npm"]));
        assert_eq!(prompt.text, "Tauri, cargo, npm.");
        assert_eq!(prompt.used, 3);
        assert_eq!(prompt.dropped, 0);
    }

    #[test]
    fn an_empty_vocabulary_yields_no_prompt() {
        let prompt = build(&[]);
        assert!(prompt.is_empty());

        // Blank entries are a normal consequence of hand-editing JSON.
        let prompt = build(&terms(&["", "   "]));
        assert!(prompt.is_empty());
        assert_eq!(prompt.dropped, 0);
    }

    #[test]
    fn truncates_rather_than_overflowing_the_budget() {
        let long = terms(&["supercalifragilistic"; 200]);
        let prompt = build(&long);

        assert!(prompt.dropped > 0, "a 200-term list must not fit");
        assert!(
            prompt.text.chars().count() <= (TOKEN_BUDGET as f32 * CHARS_PER_TOKEN) as usize + 1,
            "prompt overflowed the character budget"
        );
        assert_eq!(prompt.used + prompt.dropped, 200);
    }

    #[test]
    fn the_seeded_vocabulary_fits_comfortably() {
        let seeded = crate::config::WhisperSettings::default().vocabulary;
        let prompt = build(&seeded);
        assert_eq!(prompt.dropped, 0, "the shipped defaults must not truncate");
    }
}
