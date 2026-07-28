//! How much video memory this machine has, used once to pick a default model.
//!
//! DXGI rather than shelling out to `nvidia-smi`: it needs no CUDA toolkit, no
//! vendor tooling, works for AMD and Intel adapters, and answers in
//! microseconds. `windows` is held at 0.61 to match Tauri's pin — see the note
//! in Cargo.toml.

/// Dedicated video memory on the largest hardware adapter, in bytes.
///
/// `None` means the question could not be answered, not that there is no GPU.
/// Callers treat it as "assume the small model".
#[cfg(windows)]
pub fn dedicated_video_memory() -> Option<u64> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };

    // SAFETY: every call below is a plain COM call on an interface we just
    // created, and the loop stops on the first error, which is how DXGI
    // reports "no more adapters".
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best = None;

        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let adapter: IDXGIAdapter1 = adapter;

            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };

            // The Microsoft Basic Render Driver reports memory it cannot use
            // for compute. Counting it would pick the big model on a machine
            // with no real GPU at all.
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }

            let bytes = desc.DedicatedVideoMemory as u64;
            if bytes > best.unwrap_or(0) {
                best = Some(bytes);
            }
        }

        best
    }
}

/// Nothing to probe with off Windows, and nothing on the MVP path needs it:
/// macOS is unified memory and Linux is out of scope until after the MVP.
#[cfg(not(windows))]
pub fn dedicated_video_memory() -> Option<u64> {
    None
}
