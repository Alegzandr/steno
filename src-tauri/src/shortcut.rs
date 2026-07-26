use tauri::{AppHandle, Runtime};
use tauri_plugin_global_shortcut::{
    Code, Error, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
};

use crate::window;

/// Cmd+Shift+D on macOS, Ctrl+Shift+D everywhere else.
fn toggle_shortcut() -> Shortcut {
    #[cfg(target_os = "macos")]
    let primary = Modifiers::SUPER;
    #[cfg(not(target_os = "macos"))]
    let primary = Modifiers::CONTROL;

    Shortcut::new(Some(primary | Modifiers::SHIFT), Code::KeyD)
}

pub fn register_toggle<R: Runtime>(app: &AppHandle<R>) -> Result<(), Error> {
    app.global_shortcut()
        .on_shortcut(toggle_shortcut(), |app, _shortcut, event| {
            // Phase 2 turns this into push-to-talk by also handling Released.
            if event.state == ShortcutState::Pressed {
                window::toggle(app);
            }
        })
}
