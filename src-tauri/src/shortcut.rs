use tauri::{AppHandle, Manager, Runtime};
use tauri_plugin_global_shortcut::{
    Code, Error, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
};

use crate::audio::Recorder;
use crate::window;

/// Cmd+Shift+D on macOS, Ctrl+Shift+D everywhere else.
fn primary() -> Modifiers {
    #[cfg(target_os = "macos")]
    return Modifiers::SUPER;
    #[cfg(not(target_os = "macos"))]
    return Modifiers::CONTROL;
}

/// Push-to-talk: hold to record, release to transcribe.
fn talk_shortcut() -> Shortcut {
    Shortcut::new(Some(primary() | Modifiers::SHIFT), Code::KeyD)
}

/// Hides without copying. Cmd/Ctrl+Enter in the editor does both, but that one
/// is a webview key: it only fires when Steno has focus, and the whole point of
/// this window is that it usually does not.
fn hide_shortcut() -> Shortcut {
    Shortcut::new(Some(primary() | Modifiers::SHIFT), Code::KeyH)
}

pub fn register<R: Runtime>(app: &AppHandle<R>) -> Result<(), Error> {
    app.global_shortcut()
        .on_shortcut(talk_shortcut(), |app, _shortcut, event| match event.state {
            ShortcutState::Pressed => {
                // Both are cheap and non-blocking: this runs on the event loop
                // thread. Opening the device happens on the session thread.
                if let Some(window) = app.get_webview_window(window::MAIN) {
                    window::show_without_stealing_focus(&window);
                }
                app.state::<Recorder>().start(app);
            }
            ShortcutState::Released => app.state::<Recorder>().stop(),
        })?;

    app.global_shortcut()
        .on_shortcut(hide_shortcut(), |app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                window::hide(app);
            }
        })
}
