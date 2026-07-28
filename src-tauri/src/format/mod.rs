//! The local formatting model: making sure a server is there, streaming a
//! cleanup through it, and keeping the model itself off the GPU whenever Steno
//! is not using it.

pub mod cleanup;
pub mod model;
pub mod server;

use std::sync::Arc;

use crate::resident::Resident;

/// The managed handle on the resident formatting model.
///
/// The value is a marker, not the model: the weights live in the Ollama
/// process, not in ours. What `Resident` actually owns is the *claim* on them,
/// and dropping it is what sends the unload. Same state machine as Whisper,
/// same leases, same guarantee that a cleanup in flight cannot have the model
/// pulled out from under it.
pub type Formatter = Arc<Resident<model::Loaded>>;
