use tauri::{AppHandle, Manager, Runtime, WebviewWindow};

use crate::{lifecycle, tray};

pub const MAIN: &str = "main";

/// Take the mini editor off screen, and give the video memory back.
pub fn hide<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window(MAIN) {
        let _ = window.hide();
    }

    lifecycle::on_hide(app);

    // The tray menu offers Show or Hide depending on which one is possible, so
    // it has to hear about this even though the residency change that follows
    // will refresh it again a moment later.
    tray::refresh(app);
}

/// Bring the mini editor on screen without taking the keyboard away from the
/// app the user was typing in.
///
/// Also where the two expensive resources start loading. Tauri has no "window
/// became visible" event, and this is the only place the window is shown, so
/// hooking the call rather than an event is both simpler and exact.
pub fn show_without_stealing_focus<R: Runtime>(window: &WebviewWindow<R>) {
    // On Windows, tao honours `focus: false` only for the very first show;
    // every later ShowWindow(SW_SHOW) activates the window. `set_focusable`
    // toggles WS_EX_NOACTIVATE, which suppresses that activation. Restoring it
    // right after keeps the window clickable, so the user can still put the
    // caret in the editor and fix the transcript.
    //
    // macOS runs Steno as an accessory app and X11/Wayland compositors apply
    // focus-stealing prevention, so `show` on its own is enough there.
    #[cfg(windows)]
    let _ = window.set_focusable(false);

    let _ = window.show();

    #[cfg(windows)]
    let _ = window.set_focusable(true);

    lifecycle::on_show(window.app_handle());
    tray::refresh(window.app_handle());
}
