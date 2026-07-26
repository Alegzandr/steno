use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("no input device is available")]
    NoInputDevice,

    #[error("input device error: {0}")]
    Device(#[from] cpal::Error),

    #[error("the input device uses an unsupported sample format ({0:?})")]
    SampleFormat(cpal::SampleFormat),

    #[error("could not set up the resampler: {0}")]
    ResamplerSetup(#[from] rubato::ResamplerConstructionError),

    #[error("resampling failed: {0}")]
    Resample(#[from] rubato::ResampleError),

    #[error("could not write the WAV file: {0}")]
    Wav(#[from] hound::Error),
}
