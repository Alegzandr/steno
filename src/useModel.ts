import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export type ModelSpec = {
  id: string;
  url: string;
  bytes: number;
  sha256: string;
  label: string;
};

export type ModelStatus = {
  spec: ModelSpec;
  path: string;
  installed: boolean;
  partialBytes: number;
  videoMemoryBytes: number | null;
  gpuBackend: string;
};

export type Progress = {
  modelId: string;
  receivedBytes: number;
  totalBytes: number;
  bytesPerSecond: number;
  etaMs: number;
  resumed: boolean;
};

export type Model = {
  status: ModelStatus | null;
  downloading: boolean;
  progress: Progress | null;
  error: string | null;
  /** A partial file is on disk, so the button continues rather than restarts. */
  resumable: boolean;
  download: () => void;
  cancel: () => void;
};

export function useModel(): Model {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resumable, setResumable] = useState(false);

  const refresh = useCallback(() => {
    invoke<ModelStatus>("model_status")
      .then(setStatus)
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    refresh();

    // A dev-server reload can land in the middle of a multi-gigabyte transfer.
    invoke<boolean>("model_download_running")
      .then(setDownloading)
      .catch(() => undefined);

    const subscriptions = [
      listen<Progress>("model-download-progress", ({ payload }) => {
        setDownloading(true);
        setError(null);
        setProgress(payload);
      }),

      listen("model-download-complete", () => {
        setDownloading(false);
        setProgress(null);
        setResumable(false);
        refresh();
      }),

      listen<{ message: string; resumable: boolean }>(
        "model-download-error",
        ({ payload }) => {
          setDownloading(false);
          setError(payload.message);
          setResumable(payload.resumable);
          refresh();
        },
      ),
    ];

    return () => {
      subscriptions.forEach((pending) => pending.then((unlisten) => unlisten()));
    };
  }, [refresh]);

  const download = useCallback(() => {
    setError(null);
    setDownloading(true);
    // Rejection also arrives as a `model-download-error` event; this catch only
    // covers the case where the command itself could not start.
    invoke("download_model").catch((reason) => {
      setDownloading(false);
      setError(String(reason));
    });
  }, []);

  const cancel = useCallback(() => {
    invoke("cancel_model_download").catch(() => undefined);
  }, []);

  return { status, downloading, progress, error, resumable, download, cancel };
}
