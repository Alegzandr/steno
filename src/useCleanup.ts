import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type { EditorHandle } from "./Editor";

type Started = {
  model: string;
  inputChars: number;
  modelCold: boolean;
};

export type CleanupStats = {
  text: string;
  ttftMs: number;
  totalMs: number;
  promptTokens: number;
  outputTokens: number;
  tokensPerSecond: number;
};

type Failure = {
  message: string;
  /** A command to type, when one would fix it. */
  remedy: string | null;
};

export type Cleanup = {
  running: boolean;
  /** The model still has to load, so first token will be a while. */
  modelCold: boolean;
  elapsedMs: number;
  /** What has arrived so far. Rendered outside the editor: nothing provisional
      is allowed into the buffer or its undo history. */
  preview: string;
  stats: CleanupStats | null;
  failure: Failure | null;
  start(text: string): void;
  cancel(): void;
  dismiss(): void;
};

const IDLE = {
  running: false,
  modelCold: false,
  elapsedMs: 0,
  preview: "",
  stats: null as CleanupStats | null,
  failure: null as Failure | null,
};

/**
 * Drives a cleanup and feeds it into the editor.
 *
 * The editor is manipulated through its imperative handle rather than by
 * rendering its content from state: CodeMirror owns the document, and a
 * cleanup has to interleave with whatever the user is doing without React
 * rebuilding the buffer underneath them.
 */
export function useCleanup(editor: React.RefObject<EditorHandle | null>): Cleanup {
  const [state, setState] = useState(IDLE);
  const startedAt = useRef<number | null>(null);

  // The backend emits nothing between the first token and the last, so the
  // elapsed readout has to come from here.
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
      listen<Started>("cleanup-started", ({ payload }) => {
        setState({ ...IDLE, running: true, modelCold: payload.modelCold });
      }),

      listen<{ text: string }>("cleanup-delta", ({ payload }) => {
        setState((prev) =>
          prev.running ? { ...prev, preview: prev.preview + payload.text } : prev,
        );
      }),

      listen<CleanupStats>("cleanup-complete", ({ payload }) => {
        // The only moment the buffer changes, and the only undo step a cleanup
        // ever costs.
        editor.current?.replaceAll(payload.text);
        setState({ ...IDLE, stats: payload });
      }),

      // A cancelled or failed cleanup leaves the buffer untouched, because it
      // was never touched in the first place. There is nothing to roll back.
      listen("cleanup-cancelled", () => setState({ ...IDLE })),

      listen<Failure>("cleanup-error", ({ payload }) =>
        setState({ ...IDLE, failure: payload }),
      ),
    ];

    return () => {
      subscriptions.forEach((pending) => pending.then((unlisten) => unlisten()));
    };
  }, [editor]);

  return {
    ...state,

    start(text: string) {
      setState({ ...IDLE, running: true });
      invoke("clean_up", { text }).catch((error: unknown) => {
        // `clean_up` only rejects before anything started, so there is no
        // stream to unwind here — but the spinner has to come down.
        setState({
          ...IDLE,
          failure: { message: String(error), remedy: null },
        });
      });
    },

    cancel() {
      invoke("cancel_cleanup").catch(() => undefined);
    },

    dismiss() {
      setState((prev) => ({ ...prev, failure: null, stats: null }));
    },
  };
}
