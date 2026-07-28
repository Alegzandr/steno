//! The Whisper context: loading it, running one clip through it, and applying
//! the two guards that need the decoder's own output.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::filter;
use super::prompt;
use crate::audio::lock;
use crate::config::WhisperSettings;

/// Upper bound on decoder threads. Past this, contention costs more than the
/// extra cores return, and on a CUDA build the CPU is barely on the path.
const MAX_THREADS: u16 = 8;

/// A loaded model, ready to transcribe.
///
/// Held by a `Resident`, so it exists only between a window show and the
/// matching hide. Dropping it is what gives the video memory back.
pub struct Engine {
    context: WhisperContext,
    pub model_id: String,
    pub backend: &'static str,
    /// What loading this cost. Reported at the checkpoint, and the number that
    /// decides whether warming on window show is early enough.
    pub load_ms: u64,
    /// One clip at a time. Two concurrent runs would double the working-set
    /// memory on the GPU for no throughput gain: the model is already using
    /// every unit it can.
    running: Mutex<()>,
}

/// What one clip produced, with enough detail to say which guard fired.
#[derive(Clone, Debug, Default)]
pub struct Transcript {
    pub text: String,
    pub segment_count: usize,
    pub dropped_no_speech: usize,
    pub dropped_denylist: usize,
    /// Highest no-speech probability any segment reported. This is the margin
    /// on `no_speech_thold`: it says how close the second guard came to firing,
    /// which is the only way to tell a threshold that is well chosen from one
    /// that simply never gets tested.
    pub peak_no_speech: f32,
}

impl Transcript {
    pub fn dropped(&self) -> usize {
        self.dropped_no_speech + self.dropped_denylist
    }

    /// The guard that emptied the transcript, when one did. Only meaningful
    /// once `text` is known to be empty.
    pub fn emptied_by(&self) -> Option<filter::Guard> {
        match (self.dropped_denylist, self.dropped_no_speech) {
            (0, 0) => None,
            // Denylist wins the attribution: it is the specific diagnosis,
            // no-speech is the general one.
            (d, _) if d > 0 => Some(filter::Guard::Denylist),
            _ => Some(filter::Guard::NoSpeech),
        }
    }
}

impl Engine {
    /// Loads the model. Blocking and expensive — seconds, and gigabytes of
    /// video memory — so it belongs on a warm-up thread, never on the event
    /// loop.
    pub fn load(path: &Path, model_id: &str) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("the model file is missing: {}", path.display()));
        }

        let started = Instant::now();

        // `use_gpu` defaults to whether a GPU backend was compiled in, which is
        // exactly the right answer; naming it here would only let the two
        // drift apart.
        let context = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|error| format!("could not load {}: {error}", path.display()))?;

        let load_ms = started.elapsed().as_millis() as u64;
        let backend = crate::model::backend_name();
        eprintln!("whisper: loaded {model_id} on the {backend} backend in {load_ms} ms");

        Ok(Self {
            context,
            model_id: model_id.to_owned(),
            backend,
            load_ms,
            running: Mutex::new(()),
        })
    }

    /// Transcribes 16 kHz mono samples.
    ///
    /// The RMS floor is *not* applied here: it is checked before the engine is
    /// even acquired, so a silent clip never pays for a model load.
    pub fn run(&self, samples: &[f32], settings: &WhisperSettings) -> Result<Transcript, String> {
        let _running = lock(&self.running);

        let mut state = self
            .context
            .create_state()
            .map_err(|error| format!("could not create a Whisper state ({error})"))?;

        let vocabulary = prompt::build(&settings.vocabulary);
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Forced, never detected. Detection on a three-second clip guesses, and
        // guessing wrong turns a French dictation into phonetic English.
        params.set_language(Some(&settings.language));
        params.set_detect_language(false);
        params.set_translate(false);

        // Every window starts clean. Push-to-talk bursts are independent
        // thoughts, and carrying decoded context between them lets one bad
        // transcript poison the next.
        params.set_no_context(true);

        params.set_n_threads(i32::from(threads(settings)));
        params.set_temperature(0.0);
        params.set_no_speech_thold(settings.no_speech_thold);
        params.set_suppress_blank(true);
        // Drops `(musique)`, `[bruit de fond]` and similar non-speech
        // annotations, which are a hallucination class of their own.
        params.set_suppress_nst(true);

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        if vocabulary.dropped > 0 {
            eprintln!(
                "whisper: biasing on {} of {} vocabulary term(s); the rest did not fit \
                 in the prompt budget (see transcribe::prompt::RECOMMENDED_MAX_TERMS = {})",
                vocabulary.used,
                vocabulary.used + vocabulary.dropped,
                prompt::RECOMMENDED_MAX_TERMS
            );
        }

        if !vocabulary.is_empty() {
            params.set_initial_prompt(&vocabulary.text);
            // Without this the prompt biases the first 30-second window and no
            // other, so on a 90-second brainstorm two thirds of the dictation
            // decodes with no vocabulary at all. `set_carry_initial_prompt`
            // does not exist in whisper-rs 0.16 as published: it comes from the
            // vendored patch, and this call is what makes its absence a
            // compile error rather than a quiet regression. See CLAUDE.md.
            params.set_carry_initial_prompt(settings.carry_initial_prompt);
        }

        state
            .full(params, samples)
            .map_err(|error| format!("Whisper failed on this clip ({error})"))?;

        Ok(collect(&state, settings))
    }
}

/// Walks the segments, applying the two output-side guards.
fn collect(state: &whisper_rs::WhisperState, settings: &WhisperSettings) -> Transcript {
    let mut transcript = Transcript::default();
    let mut pieces: Vec<String> = Vec::new();

    for segment in state.as_iter() {
        transcript.segment_count += 1;

        let no_speech = segment.no_speech_probability();
        transcript.peak_no_speech = transcript.peak_no_speech.max(no_speech);

        let Ok(text) = segment.to_str_lossy() else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        if no_speech > settings.no_speech_thold {
            transcript.dropped_no_speech += 1;
            eprintln!("whisper: dropped a segment at p(no speech)={no_speech:.2} — {text:?}");
            continue;
        }

        pieces.push(text.to_owned());
    }

    // The denylist is applied to the transcription as a whole, once, after
    // every segment is in hand — never segment by segment inside the loop.
    // Dropping a matching segment out of a real dictation would silently delete
    // a word the user actually said: ending a burst on "merci" is normal
    // speech, and losing it is worse than the hallucination the guard exists
    // for. The guard may empty a clip; it may not edit one.
    if filter::is_wholly_denied(&pieces, &settings.hallucinations) {
        eprintln!("whisper: dropped a transcription that was only boilerplate — {pieces:?}");
        transcript.dropped_denylist = pieces.len();
        pieces.clear();
    }

    transcript.text = pieces.join(" ");
    transcript
}

/// Decoder threads: the configured value, or a sensible share of the machine.
fn threads(settings: &WhisperSettings) -> u16 {
    settings
        .threads
        .filter(|n| *n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| (n.get() as u16).min(MAX_THREADS))
                .unwrap_or(4)
        })
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_count_respects_the_override_and_the_cap() {
        let mut settings = WhisperSettings::default();

        settings.threads = Some(3);
        assert_eq!(threads(&settings), 3);

        // Zero in the file means "you decide", not "no threads".
        settings.threads = Some(0);
        assert!(threads(&settings) >= 1);

        settings.threads = None;
        assert!((1..=MAX_THREADS).contains(&threads(&settings)));
    }

    #[test]
    fn attribution_prefers_the_specific_guard() {
        let mut transcript = Transcript::default();
        assert_eq!(transcript.emptied_by(), None);

        transcript.dropped_no_speech = 2;
        assert_eq!(transcript.emptied_by(), Some(filter::Guard::NoSpeech));

        transcript.dropped_denylist = 1;
        assert_eq!(transcript.emptied_by(), Some(filter::Guard::Denylist));
        assert_eq!(transcript.dropped(), 3);
    }
}
