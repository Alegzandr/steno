import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";

import { DevicePicker } from "./DevicePicker";
import { Editor, type EditorHandle } from "./Editor";
import { LevelMeter } from "./LevelMeter";
import { ModelDownload } from "./ModelDownload";
import { useCleanup } from "./useCleanup";
import { useModel } from "./useModel";
import { useRecording } from "./useRecording";
import { useResidency, type ResidentState } from "./useResidency";
import { useTranscription } from "./useTranscription";

const isMac = navigator.userAgent.includes("Mac");
const modifier = isMac ? "⌘⇧" : "Ctrl+Shift+";
const talkShortcut = `${modifier}D`;
const hideShortcut = `${modifier}H`;
const cleanShortcut = `${modifier}K`;
const copyShortcut = isMac ? "⌘↵" : "Ctrl+Enter";

/** Long enough to read "Copied", short enough not to feel like a delay. */
const CONFIRMATION_MS = 450;

function clock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const seconds = total % 60;
  return `${Math.floor(total / 60)}:${String(seconds).padStart(2, "0")}`;
}

/** Compact residency readout: the GPU is shared, so what is loaded is worth
    keeping visible rather than hiding in a settings pane. */
function Resident({ label, state }: { label: string; state: ResidentState }) {
  return (
    <span className={`resident resident-${state}`} title={`${label}: ${state}`}>
      {label}
    </span>
  );
}

function App() {
  const { status, device, level, elapsedMs, clipped, notice } = useRecording();
  const transcription = useTranscription();
  const residency = useResidency();
  const model = useModel();

  const editor = useRef<EditorHandle>(null);
  const cleanup = useCleanup(editor);

  const [empty, setEmpty] = useState(true);
  const [copied, setCopied] = useState(false);

  const recording = status === "recording";
  const needsModel = model.status !== null && !model.status.installed;

  // Each completed transcription is a fresh object, so a new one is exactly
  // what this should react to. Appending here rather than inside the hook
  // keeps the editor the only thing that knows how a burst joins the buffer.
  useEffect(() => {
    if (transcription.transcript) {
      editor.current?.append(transcription.transcript.text);
    }
  }, [transcription.transcript]);

  const runCleanUp = useCallback(() => {
    if (cleanup.running) {
      cleanup.cancel();
      return;
    }
    const text = editor.current?.text() ?? "";
    if (text.trim()) cleanup.start(text);
  }, [cleanup]);

  const copyAndHide = useCallback(async () => {
    const text = editor.current?.text() ?? "";
    if (!text.trim()) return;

    try {
      await writeText(text);
    } catch (error) {
      console.error("clipboard write failed", error);
      return;
    }

    // Confirm first, then hide. Hiding immediately would make the copy
    // indistinguishable from a misfire.
    setCopied(true);
    window.setTimeout(() => {
      setCopied(false);
      void invoke("hide_window");
    }, CONFIRMATION_MS);
  }, []);

  const escape = useCallback(() => {
    if (cleanup.running) {
      cleanup.cancel();
    } else if (recording) {
      void invoke("cancel_recording");
    }
  }, [cleanup, recording]);

  // The editor's own keymap only fires when it has focus, and focus may well be
  // on a button. These are the same chords at window level.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const mod = isMac ? event.metaKey : event.ctrlKey;

      if (mod && event.key === "Enter") {
        event.preventDefault();
        void copyAndHide();
      } else if (mod && event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        runCleanUp();
      } else if (event.key === "Escape") {
        event.preventDefault();
        escape();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [copyAndHide, runCleanUp, escape]);

  return (
    <div
      className={[
        "app",
        recording ? "app-recording" : "",
        cleanup.running ? "app-cleaning" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
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
            // Not `getCurrentWindow().hide()`: hiding is what releases the
            // GPU, and that lives on the Rust side of the wall.
            onClick={() => invoke("hide_window")}
            title={`Hide (${hideShortcut})`}
          >
            ×
          </button>
          {/* Hiding is the everyday action, so it keeps the familiar corner.
              Quitting still needs to exist and be reachable: without it the
              only way out is the Task Manager, which skips the unload path. */}
          <button
            className="action action-quit"
            onClick={() => invoke("quit_app")}
            title="Quit Steno and release the GPU"
          >
            ⏻
          </button>
        </span>
      </header>

      {needsModel ? (
        <main className="editor-slot">
          <ModelDownload model={model} />
        </main>
      ) : (
        <>
          <main className="editor-slot">
            <Editor
              ref={editor}
              readOnly={cleanup.running}
              onChange={setEmpty}
              onCopyAndHide={copyAndHide}
              onCleanUp={runCleanUp}
              onEscape={escape}
            />

            {empty && !cleanup.running && (
              <p className="watermark">
                Hold {talkShortcut} to dictate. Bursts stack up here.
              </p>
            )}

            {/* The cleanup streams here, over the buffer, and lands in it only
                when it is finished. Nothing provisional gets into the document
                or into its undo history. */}
            {cleanup.running && (
              <div className="stream">
                <pre className="stream-text">
                  {cleanup.preview}
                  <span className="caret" />
                </pre>
              </div>
            )}

            {/* Transient state sits over the buffer rather than in it: the
                buffer is the user's text and nothing else belongs in it. */}
            <div className="overlays">
              {!recording && !transcription.running && empty && (
                <DevicePicker disabled={recording} />
              )}

              {notice && <p className="notice">{notice}</p>}

              {transcription.running && (
                <p className="working">
                  <span className="spinner" />
                  {transcription.modelCold
                    ? "Loading the model, then transcribing"
                    : "Transcribing"}
                  {" · "}
                  {(transcription.elapsedMs / 1000).toFixed(1)} s
                </p>
              )}

              {transcription.failure && (
                <div className="failure">
                  <p className="notice">{transcription.failure.message}</p>
                  <p className="hint">
                    The clip was kept at{" "}
                    <span className="clip-path">
                      {transcription.failure.wavPath}
                    </span>
                  </p>
                  <button
                    className="action"
                    onClick={() =>
                      invoke("transcribe_file", {
                        path: transcription.failure?.wavPath,
                      })
                    }
                  >
                    Try again
                  </button>
                </div>
              )}

              {transcription.notice && (
                <p className="notice">{transcription.notice}</p>
              )}

              {cleanup.failure && (
                <div className="failure">
                  <p className="notice">{cleanup.failure.message}</p>
                  {cleanup.failure.remedy && (
                    <p className="hint">
                      Run <code className="clip-path">{cleanup.failure.remedy}</code>
                    </p>
                  )}
                  <button className="action" onClick={cleanup.dismiss}>
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          </main>

          <div className="toolbar">
            <button
              className={cleanup.running ? "primary primary-busy" : "primary"}
              onClick={runCleanUp}
              disabled={empty && !cleanup.running}
              title={`${cleanup.running ? "Stop the cleanup" : "Restructure the whole buffer"} (${cleanShortcut})`}
            >
              {cleanup.running ? (
                <>
                  <span className="spinner" />
                  {cleanup.modelCold && cleanup.stats === null
                    ? "Loading the model"
                    : "Cleaning up"}
                  {" · "}
                  {(cleanup.elapsedMs / 1000).toFixed(1)} s · Cancel
                </>
              ) : (
                "Clean up"
              )}
            </button>

            <button
              className={copied ? "secondary secondary-done" : "secondary"}
              onClick={copyAndHide}
              disabled={empty || cleanup.running}
              title={`Copy as Markdown and hide (${copyShortcut})`}
            >
              {copied ? "Copied" : "Copy & hide"}
            </button>
          </div>
        </>
      )}

      <footer className="statusbar">
        <LevelMeter level={level} clipped={clipped} active={recording} />
        <span className="status-row">
          <span className="device">
            {cleanup.stats
              ? `${cleanup.stats.outputTokens} tokens · first at ${(cleanup.stats.ttftMs / 1000).toFixed(1)} s · ${cleanup.stats.tokensPerSecond.toFixed(0)} tok/s`
              : recording && device
                ? `${device.deviceName} · ${device.sampleRate} Hz · ${device.channels} ch`
                : `Idle · ${hideShortcut} hides`}
          </span>
          <span className="residency" title={residency.message ?? undefined}>
            <Resident label="whisper" state={residency.whisper} />
            <Resident label="llm" state={residency.llm} />
          </span>
        </span>
      </footer>
    </div>
  );
}

export default App;
