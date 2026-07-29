//! The notification-area icon: the only place Steno is visible when its window
//! is not.
//!
//! Steno is a background app by design — `skipTaskbar`, no Dock icon, a window
//! that hides rather than closes and takes the video memory with it. The cost of
//! that design is that a hidden Steno is indistinguishable from a Steno that was
//! never launched or has crashed, and the only way out of the process was the
//! Task Manager, which kills it without `RunEvent::Exit` and so without the
//! unload that hands nine gigabytes back. This module is what closes both gaps,
//! and it does it without a taskbar button: presence in the notification area,
//! not in the window list.
//!
//! Two things it shows, and they answer different questions:
//!
//! - **The icon is there** ⇒ Steno is running. That is the whole answer to "is
//!   it alive", and it needs no badge.
//! - **The badge** ⇒ Steno is *using something*. Red while the microphone is
//!   live, blue while a model is working, amber when nothing is happening but
//!   the models are still resident, nothing at all when the card is free. The
//!   amber state is the one worth a colour of its own on this machine: "idle"
//!   and "holding the GPU" are not the same thing, and the VRAM discipline is
//!   the reason.
//!
//! **What Steno is doing is derived, never stored.** `activity` asks the
//! recorder, the transcriber and the cleanup flag; nothing here keeps a copy that
//! could disagree with them. Residency is the one exception, and `Residency`
//! says why. `refresh` is called at each transition rather than polled, so the
//! badge changes within a frame of the thing it describes, and calling it too
//! often is free — the last rendered state is remembered and an unchanged one is
//! not repainted.
//!
//! Every Tauri tray and menu setter marshals to the main thread and blocks until
//! it has run, so `refresh` is safe from any thread (`send_user_message` runs the
//! task inline when it is already on the main thread, so the event-loop callers —
//! the shortcut handler, the menu handler — do not deadlock).

use std::sync::Arc;
use std::sync::Mutex;

use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Runtime};

use crate::audio::{lock, Recorder, RecordingState};
use crate::format::cleanup::Cleanup;
use crate::resident::ResidentState;
use crate::transcribe::InFlight;
use crate::window;

/// The tray icon's id, so `refresh` can find it again without managed state.
pub const ID: &str = "steno-tray";

const SHOW: &str = "tray-show";
const HIDE: &str = "tray-hide";
const QUIT: &str = "tray-quit";

/// What Steno is doing, in one word.
///
/// Ordered by precedence, not by chronology: `activity` returns the first one
/// that applies, so a cleanup started while a clip is still being transcribed
/// reports as transcribing. Both are "working" and share a badge colour, so the
/// only visible difference is one word of tooltip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activity {
    /// The GPU runtime is missing: nothing Steno does can happen at all.
    Blocked,
    Recording,
    /// The microphone is closed; the clip is still being resampled and written.
    Finalizing,
    Transcribing,
    CleaningUp,
    /// Nothing in flight. The models may still be resident — see `Snapshot`.
    Idle,
}

impl Activity {
    fn word(self) -> &'static str {
        match self {
            Activity::Blocked => "GPU unavailable",
            Activity::Recording => "Recording",
            Activity::Finalizing => "Saving the clip",
            Activity::Transcribing => "Transcribing",
            Activity::CleaningUp => "Cleaning up",
            Activity::Idle => "Idle",
        }
    }
}

/// What the rest of the app has been *told* is resident.
///
/// The one piece of state this module keeps rather than derives, and not by
/// preference: `lifecycle::emit` announces `Ready` from inside the load closure,
/// a moment before `Resident` itself leaves `Loading`, so a tray that asked the
/// slot would draw the wrong answer at exactly that instant and then never hear a
/// correction. Consuming the same announcement the status bar consumes also makes
/// the two agree by construction.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Residency {
    whisper: ResidentState,
    llm: ResidentState,
}

impl Residency {
    /// Whether video memory is held. The distinction the amber badge exists for.
    fn any(&self) -> bool {
        [self.whisper, self.llm]
            .iter()
            .any(|state| matches!(state, ResidentState::Loading | ResidentState::Ready))
    }
}

/// Everything the tray draws, in one comparable value. What `refresh` diffs
/// against so an unchanged state costs no main-thread round trip.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Snapshot {
    activity: Activity,
    residency: Residency,
    shown: bool,
}

impl Snapshot {
    fn resident(&self) -> bool {
        self.residency.any()
    }

    /// Under Windows' 128-character tooltip limit by construction.
    fn tooltip(&self) -> String {
        let detail = match self.activity {
            Activity::Idle if self.resident() => " · models resident",
            Activity::Idle => " · nothing loaded",
            _ => "",
        };

        format!("Steno — {}{detail}", self.activity.word())
    }

    fn residency_line(&self) -> String {
        format!(
            "whisper: {} · llm: {}",
            self.residency.whisper.as_str(),
            self.residency.llm.as_str()
        )
    }
}

/// The handles `refresh` needs to keep, and the state it last drew.
///
/// Managed rather than passed around because the callers are transition points
/// scattered across the recorder, the transcriber and the cleanup, none of which
/// has any other reason to know the tray exists.
struct Handles<R: Runtime> {
    header: MenuItem<R>,
    models: MenuItem<R>,
    show: MenuItem<R>,
    hide: MenuItem<R>,
    residency: Mutex<Residency>,
    last: Mutex<Option<Snapshot>>,
}

/// Builds the icon and its menu. Called once, from `setup`.
pub fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    // Both models are cold at startup and nothing has been announced yet, so the
    // default is the truth rather than an assumption.
    let snapshot = Snapshot {
        activity: activity(app),
        residency: Residency::default(),
        shown: false,
    };

    // Disabled on purpose: the first two items are a readout, not commands. A
    // menu is the one place a background app can put a sentence.
    let header = MenuItem::with_id(app, "tray-state", snapshot.tooltip(), false, None::<&str>)?;
    let models =
        MenuItem::with_id(app, "tray-models", snapshot.residency_line(), false, None::<&str>)?;
    let show = MenuItem::with_id(app, SHOW, "Show", !snapshot.shown, None::<&str>)?;
    let hide = MenuItem::with_id(app, HIDE, "Hide", snapshot.shown, None::<&str>)?;
    let quit = MenuItem::with_id(app, QUIT, "Quit Steno", true, None::<&str>)?;

    let menu = Menu::with_items(
        app,
        &[
            &header,
            &models,
            &PredefinedMenuItem::separator(app)?,
            &show,
            &hide,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;

    TrayIconBuilder::with_id(ID)
        .icon(icon(&snapshot))
        .tooltip(snapshot.tooltip())
        .menu(&menu)
        // Left click shows the window; without this it would open the menu
        // instead, which is tray-icon's default and one click too many for the
        // thing the user wants nine times out of ten.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            SHOW => reveal(app),
            HIDE => window::hide(app),
            // The same path as the titlebar's quit button: `exit` runs the Tauri
            // exit sequence, so `lifecycle::on_exit` still gets to release both
            // models. Killing the process from the Task Manager is what this
            // menu entry exists to make unnecessary.
            QUIT => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                reveal(tray.app_handle());
            }
        })
        .build(app)?;

    app.manage(Arc::new(Handles {
        header,
        models,
        show,
        hide,
        residency: Mutex::new(snapshot.residency),
        last: Mutex::new(Some(snapshot)),
    }));

    Ok(())
}

/// Records what `lifecycle` has just announced about one of the two models, and
/// redraws.
///
/// The tray's only writer. Everything else about the badge is asked for on the
/// spot; see `Residency` for why this one is remembered.
pub fn report<R: Runtime>(app: &AppHandle<R>, resource: &str, state: ResidentState) {
    if let Some(handles) = app.try_state::<Arc<Handles<R>>>() {
        let mut residency = lock(&handles.residency);
        match resource {
            "whisper" => residency.whisper = state,
            "llm" => residency.llm = state,
            other => eprintln!("tray: ignoring a residency report for an unknown \"{other}\""),
        }
    }

    refresh(app);
}

/// Redraws the icon, its tooltip and its menu header from the live state.
///
/// Cheap and idempotent: call it from anywhere that changes what Steno is doing.
/// A state equal to the one already on screen returns without touching the tray.
pub fn refresh<R: Runtime>(app: &AppHandle<R>) {
    // Absent when the tray could not be built — which must not turn every
    // transition into a panic — and during `setup`, before it exists.
    let Some(handles) = app.try_state::<Arc<Handles<R>>>() else {
        return;
    };

    let snapshot = Snapshot {
        activity: activity(app),
        residency: *lock(&handles.residency),
        shown: app
            .get_webview_window(window::MAIN)
            .and_then(|main| main.is_visible().ok())
            .unwrap_or(false),
    };

    {
        let mut last = lock(&handles.last);
        if *last == Some(snapshot) {
            return;
        }
        *last = Some(snapshot);
    }

    if let Some(tray) = app.tray_by_id(ID) {
        let _ = tray.set_icon(Some(icon(&snapshot)));
        let _ = tray.set_tooltip(Some(snapshot.tooltip()));
    }

    let _ = handles.header.set_text(snapshot.tooltip());
    let _ = handles.models.set_text(snapshot.residency_line());
    let _ = handles.show.set_enabled(!snapshot.shown);
    let _ = handles.hide.set_enabled(snapshot.shown);
}

/// Brings the window back, from the icon or from the menu.
///
/// Deliberately the same non-stealing show the shortcut uses. Clicking the tray
/// is a request to see the editor, not to move the caret out of the app the user
/// was typing in — and it is the one path that shows the window without also
/// starting a recording.
fn reveal<R: Runtime>(app: &AppHandle<R>) {
    if let Some(main) = app.get_webview_window(window::MAIN) {
        window::show_without_stealing_focus(&main);
    }
}

/// Asks the three owners of the answer, in precedence order.
fn activity<R: Runtime>(app: &AppHandle<R>) -> Activity {
    if crate::gpu::blocker().is_some() {
        return Activity::Blocked;
    }

    match app.state::<Recorder>().state() {
        RecordingState::Recording => return Activity::Recording,
        RecordingState::Finalizing => return Activity::Finalizing,
        RecordingState::Idle => {}
    }

    if app.state::<Arc<InFlight>>().any() {
        return Activity::Transcribing;
    }

    if app.state::<Arc<Cleanup>>().is_running() {
        return Activity::CleaningUp;
    }

    Activity::Idle
}

/// The app icon, with a status badge in the corner.
///
/// Composited here rather than shipped as five PNGs: the states are a property
/// of this module, and a badge that is drawn from the same code that decides its
/// colour cannot fall out of step with one that was exported by hand.
fn icon(snapshot: &Snapshot) -> Image<'static> {
    const RING_COLOUR: [u8; 3] = [0x1A, 0x1C, 0x1F];

    let base = base_icon();
    let (width, height) = (base.width(), base.height());
    let mut rgba = base.rgba().to_vec();

    let Some(colour) = badge_colour(snapshot) else {
        return Image::new_owned(rgba, width, height);
    };

    let centre = badge_centre(width, height);

    for y in 0..height {
        for x in 0..width {
            let distance = distance_from(centre, x, y);

            // Two coverages, one pass: the ring first so the badge lands on top
            // of it rather than being cut into it.
            let offset = ((y * width + x) * 4) as usize;
            let pixel = &mut rgba[offset..offset + 4];
            over(pixel, RING_COLOUR, BADGE_RADIUS + BADGE_RING - distance);
            over(pixel, colour, BADGE_RADIUS - distance);
        }
    }

    Image::new_owned(rgba, width, height)
}

/// Bottom-right, in the corner Windows itself badges.
const BADGE_RADIUS: f32 = 7.0;

/// Dark ring under the badge, so it reads on a light taskbar as well as a dark
/// one.
const BADGE_RING: f32 = 1.5;

/// One pixel of clearance from both edges, so nothing is clipped by the icon's
/// own bounds.
fn badge_centre(width: u32, height: u32) -> (f32, f32) {
    (
        width as f32 - BADGE_RADIUS - BADGE_RING - 1.0,
        height as f32 - BADGE_RADIUS - BADGE_RING - 1.0,
    )
}

fn distance_from(centre: (f32, f32), x: u32, y: u32) -> f32 {
    let dx = x as f32 + 0.5 - centre.0;
    let dy = y as f32 + 0.5 - centre.1;
    (dx * dx + dy * dy).sqrt()
}

/// `None` when Steno is idle with nothing loaded: the plain icon means "running,
/// costing you nothing", and it is the state the app is in most of the time.
fn badge_colour(snapshot: &Snapshot) -> Option<[u8; 3]> {
    match snapshot.activity {
        Activity::Blocked => Some([0x8A, 0x8E, 0x94]),
        Activity::Recording => Some([0xE5, 0x3E, 0x3E]),
        Activity::Finalizing | Activity::Transcribing | Activity::CleaningUp => {
            Some([0x3E, 0x9B, 0xE5])
        }
        Activity::Idle if snapshot.resident() => Some([0xE5, 0xA5, 0x3E]),
        Activity::Idle => None,
    }
}

/// Composites `colour` over one straight-alpha RGBA pixel.
///
/// `edge` is the signed distance inside the shape, so the half-pixel band around
/// the boundary is what antialiases it. Straight alpha, not premultiplied: the
/// app icon is transparent at the corners and a naive blend there would leave a
/// dark fringe around the badge.
fn over(pixel: &mut [u8], colour: [u8; 3], edge: f32) {
    let alpha = (edge + 0.5).clamp(0.0, 1.0);
    if alpha == 0.0 {
        return;
    }

    let under = f32::from(pixel[3]) / 255.0;
    let out = alpha + under * (1.0 - alpha);
    if out <= 0.0 {
        return;
    }

    for channel in 0..3 {
        let src = f32::from(colour[channel]) * alpha;
        let dst = f32::from(pixel[channel]) * under * (1.0 - alpha);
        pixel[channel] = ((src + dst) / out).round().clamp(0.0, 255.0) as u8;
    }
    pixel[3] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// The app icon, decoded once.
///
/// 32×32 rather than the 128×128: Windows asks for 16 or 20 physical pixels at
/// the usual scale factors, and downscaling from four times that loses the badge
/// into a smudge.
fn base_icon() -> &'static Image<'static> {
    use std::sync::OnceLock;

    static BASE: OnceLock<Image<'static>> = OnceLock::new();

    BASE.get_or_init(|| {
        const PNG: &[u8] = include_bytes!("../icons/32x32.png");

        Image::from_bytes(PNG)
            .expect("the bundled 32x32.png is a valid PNG")
            .to_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(activity: Activity, resident: bool) -> Snapshot {
        let state = if resident {
            ResidentState::Ready
        } else {
            ResidentState::Cold
        };

        Snapshot {
            activity,
            residency: Residency {
                whisper: state,
                llm: ResidentState::Cold,
            },
            shown: false,
        }
    }

    /// The plain icon is the idle-and-cold state and nothing else. Everything a
    /// user might want to interrupt has to be visible without hovering.
    #[test]
    fn only_an_unloaded_idle_steno_has_no_badge() {
        assert_eq!(badge_colour(&snapshot(Activity::Idle, false)), None);

        for activity in [
            Activity::Blocked,
            Activity::Recording,
            Activity::Finalizing,
            Activity::Transcribing,
            Activity::CleaningUp,
        ] {
            assert!(
                badge_colour(&snapshot(activity, false)).is_some(),
                "{activity:?} must be visible on the icon"
            );
        }

        assert!(
            badge_colour(&snapshot(Activity::Idle, true)).is_some(),
            "holding video memory must not look the same as holding none"
        );
    }

    /// Recording is the one state with a consequence — a live microphone — so it
    /// may not share a colour with anything else.
    #[test]
    fn recording_has_a_colour_of_its_own() {
        let recording = badge_colour(&snapshot(Activity::Recording, false));

        for activity in [
            Activity::Blocked,
            Activity::Finalizing,
            Activity::Transcribing,
            Activity::CleaningUp,
            Activity::Idle,
        ] {
            assert_ne!(recording, badge_colour(&snapshot(activity, true)));
        }
    }

    /// Windows truncates `szTip` at 128 characters, and the truncation would fall
    /// in the middle of the part that says what Steno is doing.
    #[test]
    fn every_tooltip_fits_the_windows_limit() {
        for activity in [
            Activity::Blocked,
            Activity::Recording,
            Activity::Finalizing,
            Activity::Transcribing,
            Activity::CleaningUp,
            Activity::Idle,
        ] {
            for resident in [false, true] {
                let tooltip = snapshot(activity, resident).tooltip();
                assert!(tooltip.starts_with("Steno — "), "{tooltip}");
                assert!(tooltip.chars().count() < 128, "{tooltip}");
            }
        }
    }

    /// The idle tooltip is the one that answers "is it holding my GPU", so the
    /// two idle states must not read the same.
    #[test]
    fn idle_says_whether_anything_is_loaded() {
        assert_ne!(
            snapshot(Activity::Idle, false).tooltip(),
            snapshot(Activity::Idle, true).tooltip()
        );
    }

    /// What the badge is actually for: the composite has to differ from the app
    /// icon inside the badge and be bit-identical to it everywhere else.
    ///
    /// Also the only check that the disc is not drawn off the edge of the image,
    /// which is what a wrong centre looks like: it would still change pixels, and
    /// they would still be in the corner.
    #[test]
    fn the_badge_lands_on_the_disc_and_nowhere_else() {
        let base = base_icon();
        let plain = icon(&snapshot(Activity::Idle, false));
        assert_eq!(plain.rgba(), base.rgba(), "no badge means the icon untouched");

        let badged = icon(&snapshot(Activity::Recording, false));
        let (width, height) = (base.width(), base.height());
        assert_eq!(badged.rgba().len() as u32, width * height * 4);

        let centre = badge_centre(width, height);
        // The outer edge, plus the half pixel the antialiasing band reaches into.
        let outer = BADGE_RADIUS + BADGE_RING + 0.5;

        let mut changed = 0;
        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 4) as usize;
                if badged.rgba()[offset..offset + 4] == base.rgba()[offset..offset + 4] {
                    continue;
                }

                changed += 1;
                let distance = distance_from(centre, x, y);
                assert!(
                    distance <= outer,
                    "pixel ({x}, {y}) is {distance:.1} from the badge centre, outside {outer}"
                );
            }
        }

        // A disc of radius 8.5 covers ~227 pixels. Well under that would mean the
        // badge is being clipped by the edge of the icon.
        assert!(changed > 200, "only {changed} pixels changed");
    }

    /// The blend has to leave every pixel it touches fully opaque inside the
    /// badge, including the ones the app icon left transparent — the corner it is
    /// drawn in is transparent in the source PNG.
    #[test]
    fn the_badge_is_opaque_over_transparent_pixels() {
        let mut pixel = [0u8, 0, 0, 0];
        over(&mut pixel, [0xE5, 0x3E, 0x3E], 4.0);

        assert_eq!(pixel, [0xE5, 0x3E, 0x3E, 0xFF]);
    }

    /// And a pixel the shape does not reach must come out bit-identical, which is
    /// what the previous test's "nowhere else" assertion rests on.
    #[test]
    fn a_pixel_outside_the_shape_is_untouched() {
        let mut pixel = [0x11, 0x22, 0x33, 0x44];
        over(&mut pixel, [0xE5, 0x3E, 0x3E], -1.0);

        assert_eq!(pixel, [0x11, 0x22, 0x33, 0x44]);
    }
}
