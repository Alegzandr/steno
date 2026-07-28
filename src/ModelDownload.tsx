import type { Model } from "./useModel";

function gigabytes(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}

function rate(bytesPerSecond: number): string {
  if (bytesPerSecond <= 0) return "—";
  return `${(bytesPerSecond / 1_000_000).toFixed(1)} MB/s`;
}

function eta(ms: number): string {
  if (ms <= 0) return "—";
  const total = Math.round(ms / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return minutes > 0 ? `${minutes} min ${seconds} s` : `${seconds} s`;
}

/**
 * First launch, and the only screen a new user sees before they can dictate.
 * The transfer is between half a gigabyte and three, so it reports throughput
 * and a remaining time rather than only a bar: a stalled download has to be
 * distinguishable from a slow one.
 */
export function ModelDownload({ model }: { model: Model }) {
  const { status, downloading, progress, error, resumable } = model;
  if (!status) return null;

  const received = progress?.receivedBytes ?? status.partialBytes;
  const total = progress?.totalBytes ?? status.spec.bytes;
  const percent = total > 0 ? Math.min(100, (received / total) * 100) : 0;
  const verifying = downloading && received >= total && total > 0;

  return (
    <section className="download">
      <h1 className="download-title">{status.spec.label}</h1>
      <p className="download-why">
        The model is not bundled. It downloads once, to{" "}
        <span className="download-path">{status.path}</span>, and is verified
        against its published SHA-256 before use.
      </p>

      <div className="download-bar" role="progressbar" aria-valuenow={percent}>
        <div className="download-fill" style={{ width: `${percent}%` }} />
      </div>

      <dl className="download-stats">
        <dt>Progress</dt>
        <dd>
          {gigabytes(received)} of {gigabytes(total)} · {percent.toFixed(1)}%
        </dd>
        {downloading && !verifying && (
          <>
            <dt>Speed</dt>
            <dd>{rate(progress?.bytesPerSecond ?? 0)}</dd>
            <dt>Remaining</dt>
            <dd>{eta(progress?.etaMs ?? 0)}</dd>
          </>
        )}
        <dt>Chosen for</dt>
        <dd>
          {status.gpuBackend} backend
          {status.videoMemoryBytes !== null &&
            ` · ${Math.round(status.videoMemoryBytes / (1024 * 1024))} MiB VRAM`}
        </dd>
      </dl>

      {verifying && <p className="hint">Checking the SHA-256…</p>}
      {error && <p className="notice">{error}</p>}

      <div className="download-actions">
        {downloading ? (
          <button className="action" onClick={model.cancel}>
            Cancel
          </button>
        ) : (
          <button className="action action-primary" onClick={model.download}>
            {received > 0 || resumable ? "Resume download" : "Download"}
          </button>
        )}
      </div>
    </section>
  );
}
