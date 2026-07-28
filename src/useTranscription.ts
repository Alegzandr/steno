import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";

export type Transcript = {
  text: string;
  durationMs: number;
  clipDurationMs: number;
  /** Below 1.0 is faster than real time. */
  realtimeFactor: number;
  segmentCount: number;
  droppedCount: number;
  modelId: string;
  backend: string;
};

type Started = {
  clipDurationMs: number;
  modelCold: boolean;
};

type Empty = {
  reason: "rms-floor" | "no-speech" | "denylist" | "empty";
  rmsDbfs: number;
  clipDurationMs: number;
};

type Failure = {
  message: string;
  wavPath: string;
};

export type Transcription = {
  running: boolean;
  /** The model still has to load, so the wait is longer than usual. */
  modelCold: boolean;
  /** Wall clock since the spinner appeared, for the elapsed-time readout. */
  elapsedMs: number;
  transcript: Transcript | null;
  notice: string | null;
  failure: Failure | null;
};

const IDLE: Transcription = {
  running: false,
  modelCold: false,
  elapsedMs: 0,
  transcript: null,
  notice: null,
  failure: null,
};

/** Why nothing was inserted, in words rather than in enum names. */
function explain(payload: Empty): string {
  switch (payload.reason) {
    case "rms-floor":
      return `Too quiet to transcribe (${payload.rmsDbfs.toFixed(0)} dBFS). Nothing was sent to Whisper.`;
    case "no-speech":
      return "Whisper judged the whole clip to be silence.";
    case "denylist":
      return "Whisper returned only a known hallucination, which was dropped.";
    default:
      return "Whisper returned nothing for this clip.";
  }
}

export function useTranscription(): Transcription {
  const [state, setState] = useState<Transcription>(IDLE);
  const startedAt = useRef<number | null>(null);

  // The spinner has to show elapsed time, and the backend emits nothing
  // between start and finish: a local tick is the only source for it.
  useEffect(() => {
    if (!state.running) {
      startedAt.current = null;
      return;
    }

    startedAt.current ??= Date.now();
    const timer = window.setInterval(() => {
      setState((prev) =>
        prev.running
          ? { ...prev, elapsedMs: Date.now() - (startedAt.current ?? Date.now()) }
          : prev,
      );
    }, 100);

    return () => window.clearInterval(timer);
  }, [state.running]);

  useEffect(() => {
    const subscriptions = [
      listen<Started>("transcription-started", ({ payload }) =>
        setState({
          ...IDLE,
          running: true,
          modelCold: payload.modelCold,
        }),
      ),

      listen<Transcript>("transcription-complete", ({ payload }) =>
        setState({ ...IDLE, transcript: payload }),
      ),

      listen<Empty>("transcription-empty", ({ payload }) =>
        setState({ ...IDLE, notice: explain(payload) }),
      ),

      listen<Failure>("transcription-error", ({ payload }) =>
        setState({ ...IDLE, failure: payload }),
      ),
    ];

    return () => {
      subscriptions.forEach((pending) => pending.then((unlisten) => unlisten()));
    };
  }, []);

  return state;
}
