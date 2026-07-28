import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type ResidentState = "cold" | "loading" | "ready" | "failed";

export type Residency = {
  whisper: ResidentState;
  llm: ResidentState;
  /** Last thing either resource had to say. Usually a load time or a failure. */
  message: string | null;
};

type Change = {
  resource: "whisper" | "llm";
  state: ResidentState;
  message: string | null;
};

/** What is currently on the GPU. Both are released when the window hides. */
export function useResidency(): Residency {
  const [state, setState] = useState<Residency>({
    whisper: "cold",
    llm: "cold",
    message: null,
  });

  useEffect(() => {
    invoke<{ whisper: ResidentState; llm: ResidentState }>("residency")
      .then((current) => setState((prev) => ({ ...prev, ...current })))
      .catch(() => undefined);

    const pending = listen<Change>("resource-state", ({ payload }) =>
      setState((prev) => ({
        ...prev,
        [payload.resource]: payload.state,
        message: payload.message,
      })),
    );

    return () => {
      pending.then((unlisten) => unlisten());
    };
  }, []);

  return state;
}
