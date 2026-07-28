//! The formatting model may reshape the dictation. It may not add to it.
//!
//! This exists because of a specific, observed failure on the first real
//! dictation: asked to structure a brainstorm about a demo app, the model
//! emitted "(exemple : site web, application, démo interactive)" — a list of
//! artefact kinds the speaker never gave. It also dropped the sentence saying
//! the text was a prompt for Claude Code, and restated "produire un artefact
//! en sortie" under two headings.
//!
//! The check is deliberately crude and mechanical: every content word in the
//! output must be traceable to a content word in the input. It cannot judge
//! meaning, so it catches invention and misses paraphrase.
//!
//! What this does NOT cover:
//!
//! - **Omission.** A model that deletes half the dictation passes every
//!   assertion here. Dropped framing — one of the three observed faults — is
//!   invisible to a word-addition check.
//!   `framing_survives` covers the one case we can state exactly.
//! - **Duplication.** Restating an idea under a second heading reuses input
//!   words and is therefore grounded. `no_idea_stated_twice` is a targeted
//!   substring check, not a general one.
//! - **The live model**, unless you ask for it. `live_cleanup_adds_nothing` is
//!   `#[ignore]`d: it needs the formatting model on disk and a GPU. The unit
//!   tests below exercise the detector against captured text, which proves the
//!   detector works and proves nothing about today's model.

/// Lowercased, unaccented, punctuation-free. Markdown syntax falls out of this
/// for free: `##`, `-`, `*` and backticks are not alphanumeric.
fn normalise(word: &str) -> String {
    word.chars()
        .filter_map(|c| {
            let c = c.to_lowercase().next().unwrap_or(c);
            Some(match c {
                'à' | 'â' | 'ä' | 'á' => 'a',
                'é' | 'è' | 'ê' | 'ë' => 'e',
                'î' | 'ï' | 'í' => 'i',
                'ô' | 'ö' | 'ó' => 'o',
                'ù' | 'û' | 'ü' | 'ú' => 'u',
                'ç' => 'c',
                other if other.is_alphanumeric() => other,
                _ => return None,
            })
        })
        .collect()
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
        .flat_map(|chunk| chunk.split(['\'', '-']))
        .map(normalise)
        .filter(|word| !word.is_empty())
        .collect()
}

/// French function words, plus the handful of English ones a technical
/// dictation drags in. A word on this list is never reported as an addition:
/// grammar is the model's to fix.
const FUNCTION_WORDS: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux", "et", "ou", "ni", "mais",
    "donc", "or", "car", "que", "qui", "quoi", "dont", "ou", "si", "ne", "pas", "plus", "moins",
    "tres", "trop", "peu", "tout", "tous", "toute", "toutes", "ce", "cet", "cette", "ces", "celui",
    "celle", "ceux", "il", "elle", "ils", "elles", "je", "j", "tu", "nous", "vous", "on", "me",
    "te", "se", "s", "lui", "leur", "leurs", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa",
    "ses", "notre", "nos", "votre", "vos", "en", "y", "a", "dans", "sur", "sous", "avec", "sans",
    "pour", "par", "vers", "chez", "entre", "apres", "avant", "pendant", "depuis", "jusqu",
    "jusque", "comme", "aussi", "encore", "deja", "toujours", "jamais", "meme", "autre", "autres",
    "etre", "est", "sont", "etait", "etaient", "sera", "seront", "soit", "ete", "avoir", "ai",
    "as", "ont", "avait", "avaient", "aura", "auront", "aurait", "faire", "fait", "faut", "peut",
    "peuvent", "pouvoir", "doit", "doivent", "devoir", "veut", "veux", "vouloir", "voudrais",
    "aurait", "il", "y", "the", "of", "to", "and", "in", "for", "is", "it", "with",
];

fn is_function_word(word: &str) -> bool {
    word.len() < 3 || FUNCTION_WORDS.contains(&word)
}

/// One word is grounded in another when they are the same word up to a short
/// suffix: plural, gender, conjugation.
///
/// Two clauses, because French inflection is not one shape. Plurals and
/// genders extend the word — `site`/`sites` — so a prefix relation with at
/// most three extra characters covers them. Conjugation *replaces* the ending
/// — `produise`/`produire` — so a long shared stem with a short tail on each
/// side covers that.
///
/// Both clauses are deliberately tight enough to still reject the inventions
/// that merely start the same way: `interactive`/`interface` shares five
/// characters but neither tail is short, `démo`/`démontrer` and
/// `application`/`app` share too few. Expanding an abbreviation the speaker
/// used is an addition, not an inflection.
fn same_word(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }

    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if short.len() >= 3 && long.starts_with(short) && long.len() - short.len() <= 3 {
        return true;
    }

    let stem = a
        .chars()
        .zip(b.chars())
        .take_while(|(x, y)| x == y)
        .count();
    stem >= 5 && a.len() - stem <= 3 && b.len() - stem <= 3
}

/// Content words present in `output` that nothing in `input` accounts for.
///
/// `vocabulary` is the configured Whisper vocabulary: those terms are the
/// spellings the model is *expected* to correct towards, so "Cloud Code"
/// becoming "Claude Code" is a fix, not an invention.
fn additions(input: &str, output: &str, vocabulary: &[&str]) -> Vec<String> {
    let source: Vec<String> = tokens(input);
    let allowed: Vec<String> = vocabulary.iter().flat_map(|term| tokens(term)).collect();

    let mut found: Vec<String> = Vec::new();
    for word in tokens(output) {
        if is_function_word(&word) || found.contains(&word) {
            continue;
        }
        let grounded = source.iter().any(|s| same_word(s, &word))
            || allowed.iter().any(|a| same_word(a, &word));
        if !grounded {
            found.push(word);
        }
    }
    found
}

/// The dictation that produced the observed failure, exactly as Whisper
/// transcribed it — "Cloud Code" included.
const RAW: &str = "Salut, je voudrais qu'on travaille sur le prompt Cloud Code d'une app qui \
aurait pour but de démontrer les capacités frontend du nouveau modèle Opus 5. Je voudrais un \
site d'agence où le focus est mis sur les animations, la 3D, et j'aimerais voir ses capacités. \
Donc peu importe la stack, il faut qu'il produise un artefact en sortie.";

/// The configured vocabulary, as far as this test needs it.
const VOCABULARY: &[&str] = &[
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
    "Claude Code",
];

#[test]
fn the_observed_failure_is_caught() {
    // Reconstructed from the first real session: the parenthetical the model
    // invented, in the sentence it invented it in.
    let bad = "## Objectif\n\nDémontrer les capacités frontend du modèle Opus 5 en produisant \
un artefact en sortie (exemple : site web, application, démo interactive).";

    let added = additions(RAW, bad, VOCABULARY);
    assert!(
        added.contains(&"exemple".to_string()),
        "the invented example marker was not caught; got {added:?}"
    );
    assert!(
        added.contains(&"interactive".to_string()),
        "`interactive` is nowhere in the input and must be reported; got {added:?}"
    );
}

#[test]
fn a_faithful_cleanup_passes() {
    let good = "## Prompt Claude Code\n\nJe voudrais qu'on travaille sur le prompt Claude Code \
d'une app qui doit démontrer les capacités frontend du nouveau modèle Opus 5.\n\n## Site \
d'agence\n\n- Focus sur les animations et la 3D.\n- Peu importe la stack.\n- Il faut qu'il \
produise un artefact en sortie.";

    let added = additions(RAW, good, VOCABULARY);
    assert!(
        added.is_empty(),
        "a faithful cleanup was reported as adding {added:?}"
    );
}

#[test]
fn correcting_towards_the_vocabulary_is_not_an_addition() {
    // "Cloud Code" in, "Claude Code" out. The vocabulary is what makes that a
    // correction rather than an invention, and without it this would fail.
    let added = additions("le prompt Cloud Code", "Le prompt Claude Code.", VOCABULARY);
    assert!(added.is_empty(), "got {added:?}");

    let unlisted = additions("le prompt Cloud Code", "Le prompt Claude Code.", &[]);
    assert_eq!(unlisted, vec!["claude".to_string()]);
}

#[test]
fn inflection_is_not_an_addition_but_expansion_is() {
    assert!(additions("une animation", "Des animations.", &[]).is_empty());
    assert!(additions("les capacités", "La capacité.", &[]).is_empty());

    // `app` -> `application` starts the same and still says more than the
    // speaker did.
    assert_eq!(
        additions("une app", "Une application.", &[]),
        vec!["application".to_string()]
    );
    // `démontrer` must not launder `démo` into the output.
    assert_eq!(
        additions("démontrer les capacités", "Une démo.", &[]),
        vec!["demo".to_string()]
    );

    // Conjugation replaces the ending rather than extending it, so the prefix
    // rule alone would report this as an addition. It is not one.
    assert!(additions("qu'il produise", "Produire.", &[]).is_empty());
    assert!(additions("je voudrais", "Je voudrais.", &[]).is_empty());

    // A shared stem is not enough on its own.
    assert_eq!(
        additions("une interface", "Interactive.", &[]),
        vec!["interactive".to_string()]
    );
}

#[test]
fn markdown_syntax_is_not_content() {
    let added = additions(
        "il faut qu'il produise un artefact en sortie",
        "## Sortie\n\n- Il faut qu'il produise un artefact en sortie.\n\n```\nartefact\n```",
        &[],
    );
    assert!(added.is_empty(), "got {added:?}");
}

/// The framing sentence — what the text is and who it is for — was dropped by
/// the model on the first real dictation. An addition check cannot see that,
/// so it gets its own assertion.
fn framing_survives(output: &str) -> bool {
    let normalised = tokens(output);
    let has = |word: &str| normalised.iter().any(|w| same_word(w, word));
    has("prompt") && (has("claude") || has("cloud"))
}

#[test]
fn dropped_framing_is_detected() {
    assert!(framing_survives("## Prompt Claude Code\n\nUne app de démonstration."));
    assert!(!framing_survives("## Objectif\n\nUne app de démonstration."));
}

/// "produire un artefact en sortie" appeared under two headings. Restating an
/// idea reuses the speaker's words, so only a direct check sees it.
fn repeated_phrases(output: &str) -> Vec<String> {
    const WATCHED: &[&str] = &["artefact en sortie", "capacites frontend"];
    let flat = tokens(output).join(" ");
    WATCHED
        .iter()
        .filter(|phrase| flat.matches(*phrase).count() > 1)
        .map(|phrase| (*phrase).to_string())
        .collect()
}

#[test]
fn no_idea_stated_twice() {
    let duplicated = "## Objectif\n\nProduire un artefact en sortie.\n\n## Livrable\n\nIl doit \
produire un artefact en sortie.";
    assert_eq!(repeated_phrases(duplicated), vec!["artefact en sortie"]);
    assert!(repeated_phrases("Il faut produire un artefact en sortie.").is_empty());
}

/// The real thing. Ignored by default: it loads several gigabytes of weights
/// and needs a GPU, which `cargo test` must not assume.
///
/// Run with `cargo test --features cuda -- --ignored --nocapture`.
#[test]
#[ignore = "loads the formatting model; run explicitly"]
fn live_cleanup_adds_nothing() {
    use std::sync::atomic::AtomicBool;

    let settings = steno_lib::config::Settings::default();
    let request = steno_lib::format::cleanup::Request::from_settings(&settings);
    let cancel = AtomicBool::new(false);

    // The model file is not discoverable from a test: there is no `AppHandle`
    // and therefore no app data directory. Named explicitly rather than
    // guessed, so a run that cannot find it says which path it wanted.
    let gguf = std::env::var("STENO_TEST_GGUF").map(std::path::PathBuf::from).expect(
        "set STENO_TEST_GGUF to the formatting model file to run this test",
    );

    let loaded = steno_lib::format::model::Loaded::load(steno_lib::format::model::Params {
        path: gguf,
        n_gpu_layers: settings.llm.n_gpu_layers,
    })
    .expect("the formatting model must load for this test");

    let outcome = steno_lib::format::cleanup::transfer(&loaded, &request, RAW, &cancel, &|_| {})
        .expect("the cleanup must run for this test");

    let text = match outcome {
        steno_lib::format::cleanup::Outcome::Complete(complete) => complete.text,
        steno_lib::format::cleanup::Outcome::Cancelled(_) => {
            panic!("nothing cancelled this cleanup")
        }
    };

    println!("--- model output ---\n{text}\n--------------------");

    let vocabulary: Vec<&str> = settings
        .whisper
        .vocabulary
        .iter()
        .map(String::as_str)
        .collect();
    let added = additions(RAW, &text, &vocabulary);

    assert!(
        added.is_empty(),
        "the model added words the dictation never contained: {added:?}"
    );
    assert!(
        framing_survives(&text),
        "the model dropped the framing saying this is a prompt for Claude Code"
    );
    assert!(
        repeated_phrases(&text).is_empty(),
        "the model restated an idea in two places: {:?}",
        repeated_phrases(&text)
    );
}
