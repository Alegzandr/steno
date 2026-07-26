import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

import { LevelMeter } from "./LevelMeter";
import { useRecording } from "./useRecording";

const isMac = navigator.userAgent.includes("Mac");
const modifier = isMac ? "⌘⇧" : "Ctrl+Shift+";
const talkShortcut = `${modifier}D`;
const hideShortcut = `${modifier}H`;

function clock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const seconds = total % 60;
  return `${Math.floor(total / 60)}:${String(seconds).padStart(2, "0")}`;
}

function App() {
  const { status, device, level, elapsedMs, clipped, clip, notice } =
    useRecording();
  const recording = status === "recording";

  return (
    <div className={recording ? "app app-recording" : "app"}>
      <header className="titlebar" data-tauri-drag-region>
        <span className="titlebar-name" data-tauri-drag-region>
          {recording ? (
            <>
              <span className="rec-dot" />
              REC {clock(elapsedMs)}
            </>
          ) : (
            "Steno"
          )}
        </span>

        <span className="titlebar-actions">
          {recording ? (
            // Esc cannot reach an unfocused webview, so cancelling needs
            // something clickable until the editor takes focus.
            <button
              className="action action-cancel"
              onClick={() => invoke("cancel_recording")}
              title="Discard this recording"
            >
              Cancel
            </button>
          ) : (
            <span className="titlebar-shortcut">{talkShortcut}</span>
          )}
          <button
            className="action action-close"
            onClick={() => getCurrentWindow().hide()}
            title={`Hide (${hideShortcut})`}
          >
            ×
          </button>
        </span>
      </header>

      <main className="editor-slot">
        <div id="editor" className="editor-placeholder">
          {notice && <p className="notice">{notice}</p>}

          {clip && !notice && (
            // Stands in for the transcript until Whisper arrives in phase 3.
            <dl className="clip">
              <dt>File</dt>
              <dd className="clip-path">{clip.path}</dd>
              <dt>Duration</dt>
              <dd>{(clip.durationMs / 1000).toFixed(2)} s</dd>
              <dt>Format</dt>
              <dd>
                {clip.sampleRate} Hz, {clip.channels} ch, 16-bit
                {clip.clipped && " · clipped"}
                {clip.reason === "max-duration" && " · hit the 120 s cap"}
              </dd>
            </dl>
          )}

          {!clip && !notice && (
            <p className="hint">Hold {talkShortcut} to record.</p>
          )}
        </div>
      </main>

      <footer className="statusbar">
        <LevelMeter level={level} clipped={clipped} active={recording} />
        <span className="device">
          {recording && device
            ? `${device.deviceName} · ${device.sampleRate} Hz · ${device.channels} ch · live in ${device.onsetMs} ms`
            : `Idle · ${hideShortcut} hides`}
        </span>
      </footer>
    </div>
  );
}

export default App;
