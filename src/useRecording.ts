import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type Status = "idle" | "recording" | "finalizing";

type RecordingStarted = {
  deviceName: string;
  sampleRate: number;
  channels: number;
  onsetMs: number;
};

type AudioLevel = {
  peak: number;
  elapsedMs: number;
  clipped: boolean;
};

export type Clip = {
  path: string;
  durationMs: number;
  sampleRate: number;
  channels: number;
  clipped: boolean;
  reason: "released" | "max-duration" | "device-lost";
};

type Discarded = {
  durationMs: number;
  reason: "too-short" | "cancelled";
};

export type Recording = {
  status: Status;
  device: RecordingStarted | null;
  // Smoothed peak, 0 to 1. The raw peak arrives every 50 ms, which reads as a
  // flicker; decaying towards it keeps the meter legible.
  level: number;
  elapsedMs: number;
  clipped: boolean;
  clip: Clip | null;
  notice: string | null;
};

const IDLE: Recording = {
  status: "idle",
  device: null,
  level: 0,
  elapsedMs: 0,
  clipped: false,
  clip: null,
  notice: null,
};

// Decay applied to the meter on every update that is quieter than the last.
const DECAY = 0.7;

export function useRecording(): Recording {
  const [state, setState] = useState<Recording>(IDLE);

  useEffect(() => {
    // A dev-server reload can land mid-recording, so ask where we stand
    // rather than assuming idle.
    invoke<Status>("recording_state")
      .then((status) => setState((prev) => ({ ...prev, status })))
      .catch(() => undefined);

    const subscriptions = [
      listen<RecordingStarted>("recording-started", ({ payload }) =>
        setState({
          ...IDLE,
          status: "recording",
          device: payload,
        }),
      ),

      listen<AudioLevel>("audio-level", ({ payload }) =>
        setState((prev) => ({
          ...prev,
          level: Math.max(payload.peak, prev.level * DECAY),
          elapsedMs: payload.elapsedMs,
          clipped: payload.clipped,
        })),
      ),

      listen<Clip>("recording-complete", ({ payload }) =>
        setState((prev) => ({
          ...prev,
          status: "idle",
          level: 0,
          clip: payload,
          notice: null,
        })),
      ),

      listen<Discarded>("recording-discarded", ({ payload }) =>
        setState((prev) => ({
          ...prev,
          status: "idle",
          level: 0,
          notice:
            payload.reason === "cancelled"
              ? "Recording cancelled."
              : `Too short (${payload.durationMs} ms), discarded.`,
        })),
      ),

      listen<{ message: string }>("recording-error", ({ payload }) =>
        setState((prev) => ({
          ...prev,
          status: "idle",
          level: 0,
          notice: payload.message,
          // A device-lost error is paired with a salvaged clip that arrived
          // just before it; keep that clip so the file stays visible. Any
          // other clip on screen is stale and the error replaces it. Order of
          // the two events does not matter: if the clip arrives after, its own
          // handler sets it.
          clip: prev.clip?.reason === "device-lost" ? prev.clip : null,
        })),
      ),
    ];

    return () => {
      subscriptions.forEach((pending) => pending.then((unlisten) => unlisten()));
    };
  }, []);

  return state;
}
