mod audio;
mod commands;
mod lifecycle;
mod shortcut;
mod window;

// Public so the measurement harness in `examples/` drives exactly the code the
// app drives — the download, the model choice, the Whisper parameters — rather
// than a second copy of it that can quietly drift.
pub mod config;
pub mod format;
pub mod model;
pub mod resident;
pub mod transcribe;

use std::sync::Arc;

use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .invoke_handler(tauri::generate_handler![
            commands::audio::start_recording,
            commands::audio::stop_recording,
            commands::audio::cancel_recording,
            commands::audio::recording_state,
            commands::audio::enumerate_input_devices,
            commands::audio::input_device,
            commands::audio::set_input_device,
            commands::model::model_status,
            commands::model::download_model,
            commands::model::cancel_model_download,
            commands::model::model_download_running,
            commands::transcribe::transcribe_file,
            commands::transcribe::residency,
            commands::format::ollama_availability,
            commands::format::clean_up,
            commands::format::cancel_cleanup,
            commands::format::cleanup_running,
            commands::window::hide_window,
            commands::window::quit_app,
        ])
        .setup(|app| {
            // Steno lives in the background: no Dock icon on macOS, no taskbar
            // entry elsewhere (`skipTaskbar` in tauri.conf.json).
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let config = config::Config::load(app.handle());
            let preferred = config.input_device();
            app.manage(config);
            app.manage(audio::Recorder::new());

            // Managed before the shortcut is armed: a recording started in the
            // next millisecond looks this up, and an unmanaged state panics.
            let warm_up = Arc::new(audio::capture::WarmUp::new());
            app.manage(warm_up.clone());

            // Both start cold and stay that way until the window is first
            // shown. Launching Steno must cost zero video memory.
            app.manage::<transcribe::Whisper>(Arc::new(resident::Resident::new("whisper")));
            app.manage::<format::Formatter>(Arc::new(resident::Resident::new("llm")));
            app.manage(Arc::new(lifecycle::Ollama::default()));
            app.manage(Arc::new(lifecycle::Activity::default()));
            app.manage(Arc::new(format::cleanup::Cleanup::default()));
            app.manage(model::download::Downloads::default());

            // Probe the hardware and record the model choice now, not on first
            // use. "Detect once at first launch" has to mean launch: leaving it
            // to whoever asks first makes the answer depend on whether the
            // hidden webview has finished mounting, and leaves `model.id` null
            // in a file the user is invited to edit.
            let spec = model::resolve(app.handle());
            eprintln!(
                "model: {} on the {} backend",
                spec.id,
                model::backend_name()
            );

            shortcut::register(app.handle())?;

            // Pay the input device's first-open cost now, off the event loop,
            // so the user's first recording starts without the driver warm-up.
            // A recording that lands during it waits on the gate rather than
            // opening the same endpoint a second time.
            std::thread::spawn(move || audio::capture::warm_up(preferred, warm_up));

            // Steno is meant to be left open on a second screen. Hiding the
            // window is not the only moment it stops working, so idleness
            // releases the models too.
            lifecycle::watch_idle(app.handle().clone());

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Tauri application");

    // `build` then `run` rather than `Builder::run`, purely so `Exit` can be
    // observed: it is the last chance to unload the formatting model and stop
    // an Ollama server we started.
    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit) {
            lifecycle::on_exit(handle);
        }
    });
}
