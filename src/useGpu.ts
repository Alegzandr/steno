import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type { Progress } from "./useModel";

/** What stops this build from working at all. See `crate::gpu`. */
export type GpuBlocker = {
  missing: string;
  message: string;
  remedy: string;
};

/** What the installed NVIDIA driver says about itself. See `gpu::driver`. */
export type Driver = {
  version: string;
  cudaMajor: number;
  cudaMinor: number;
};

/** See `gpu::runtime::Status`. */
export type CublasStatus = {
  needed: boolean;
  missing: string | null;
  directory: string;
  archiveId: string;
  archiveBytes: number;
  installBytes: number;
  requiredBytes: number;
  freeBytes: number | null;
  partialBytes: number;
  archivePresent: boolean;
  driver: Driver | null;
  /** Why the download must not be offered. `null` means the button is safe. */
  obstacle: string | null;
};

export type Gpu = {
  blocker: GpuBlocker | null;
  runtime: CublasStatus | null;
  installing: boolean;
  /** `downloading`, `extracting`, `checking` — the phases with no byte count. */
  stage: string | null;
  progress: Progress | null;
  error: string | null;
  install: () => void;
  cancel: () => void;
};

/**
 * Asked once, on mount.
 *
 * The backend caches the answer on purpose, so polling would only ever return
 * the same value. One event changes it — the cuBLAS install finishing — and
 * `install_cublas` answers with the new state rather than leaving this hook to
 * notice. `blocker === null` is both "not asked yet" and "nothing wrong", which
 * is what keeps the panel from flashing on a healthy launch: the normal case
 * never renders it.
 */
export function useGpu(): Gpu {
  const [blocker, setBlocker] = useState<GpuBlocker | null>(null);
  const [runtime, setRuntime] = useState<CublasStatus | null>(null);
  const [installing, setInstalling] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The archive's own id, so the shared `model-download-*` events can be told
  // apart from a Whisper or formatter download using the same single slot.
  const archiveId = useRef<string | null>(null);

  useEffect(() => {
    let live = true;

    void invoke<GpuBlocker | null>("gpu_blocker")
      .then((found) => {
        if (!live) return;
        setBlocker(found);
        if (!found) return;
        // Only worth asking when something is actually missing: it touches the
        // disk, the driver and the loader.
        return invoke<CublasStatus>("cublas_status").then((status) => {
          if (!live) return;
          archiveId.current = status.archiveId;
          setRuntime(status);
        });
      })
      .catch((reason) => console.error("gpu status failed", reason));

    const subscriptions = [
      listen<Progress>("model-download-progress", ({ payload }) => {
        if (payload.modelId !== archiveId.current) return;
        setInstalling(true);
        setError(null);
        setProgress(payload);
      }),

      listen<string>("cublas-install-stage", ({ payload }) => {
        setInstalling(true);
        setStage(payload);
      }),
    ];

    return () => {
      live = false;
      subscriptions.forEach((pending) => pending.then((unlisten) => unlisten()));
    };
  }, []);

  const install = useCallback(() => {
    setError(null);
    setInstalling(true);
    setStage("downloading");
    invoke<CublasStatus>("install_cublas")
      .then((status) => {
        // Authoritative: the backend re-asked the loader, it did not infer this
        // from the download having finished.
        setRuntime(status);
        setBlocker(status.needed ? blocker : null);
      })
      .catch((reason) => {
        setError(String(reason));
        void invoke<CublasStatus>("cublas_status").then(setRuntime).catch(() => undefined);
      })
      .finally(() => {
        setInstalling(false);
        setStage(null);
        setProgress(null);
      });
  }, [blocker]);

  const cancel = useCallback(() => {
    invoke("cancel_model_download").catch(() => undefined);
  }, []);

  return { blocker, runtime, installing, stage, progress, error, install, cancel };
}
