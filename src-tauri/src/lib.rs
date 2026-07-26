mod audio;
mod commands;
mod config;
mod shortcut;
mod window;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            commands::audio::start_recording,
            commands::audio::stop_recording,
            commands::audio::cancel_recording,
            commands::audio::recording_state,
            commands::audio::enumerate_input_devices,
            commands::audio::input_device,
            commands::audio::set_input_device,
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
            let warm_up = std::sync::Arc::new(audio::capture::WarmUp::new());
            app.manage(warm_up.clone());

            shortcut::register(app.handle())?;

            // Pay the input device's first-open cost now, off the event loop,
            // so the user's first recording starts without the driver warm-up.
            // A recording that lands during it waits on the gate rather than
            // opening the same endpoint a second time.
            std::thread::spawn(move || audio::capture::warm_up(preferred, warm_up));

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
