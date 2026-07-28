import type { Gpu } from "./useGpu";

function megabytes(bytes: number): string {
  return `${Math.round(bytes / 1_000_000)} MB`;
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

const STAGES: Record<string, string> = {
  downloading: "Downloading from NVIDIA…",
  extracting: "Unpacking — this reads half a gigabyte and takes a moment…",
  checking: "Asking the loader whether it can find it…",
};

/**
 * The screen a new user without the CUDA toolkit meets first.
 *
 * It says what is missing, why Steno needs it, where it comes from and where it
 * goes, before it offers a bar: 391 MB from a host the user did not choose is a
 * thing to explain, not a thing to start. When the driver is too old or the disk
 * too full, the button is replaced by the reason — offering a download that
 * cannot end well is worse than refusing it.
 */
export function CublasDownload({ gpu }: { gpu: Gpu }) {
  const { blocker, runtime, installing, stage, progress, error } = gpu;
  if (!blocker) return null;

  const received = progress?.receivedBytes ?? runtime?.partialBytes ?? 0;
  const total = progress?.totalBytes ?? runtime?.archiveBytes ?? 0;
  const percent = total > 0 ? Math.min(100, (received / total) * 100) : 0;
  const resuming = (runtime?.partialBytes ?? 0) > 0 || runtime?.archivePresent;

  return (
    <section className="download">
      <h1 className="download-title">Steno needs an NVIDIA runtime component</h1>

      <p className="download-why">
        This build uses your GPU for both dictation and cleanup, and that needs{" "}
        <span className="download-path">{blocker.missing}</span> — part of the
        NVIDIA CUDA runtime, which is not bundled with Steno. Neither
        transcription nor formatting can run without it.
      </p>

      {runtime && (
        <p className="download-why">
          Steno can fetch it from NVIDIA: {megabytes(runtime.archiveBytes)},
          downloaded once, verified against its published SHA-256 before
          anything is unpacked, and installed to{" "}
          <span className="download-path">{runtime.directory}</span>. Installing
          the CUDA Toolkit yourself works just as well.
        </p>
      )}

      {runtime?.obstacle ? (
        <p className="notice">{runtime.obstacle}</p>
      ) : (
        <>
          {installing && (
            <div
              className="download-bar"
              role="progressbar"
              aria-valuenow={percent}
            >
              <div className="download-fill" style={{ width: `${percent}%` }} />
            </div>
          )}

          <dl className="download-stats">
            {installing && (
              <>
                <dt>Progress</dt>
                <dd>
                  {megabytes(received)} of {megabytes(total)} ·{" "}
                  {percent.toFixed(1)}%
                </dd>
                <dt>Speed</dt>
                <dd>{rate(progress?.bytesPerSecond ?? 0)}</dd>
                <dt>Remaining</dt>
                <dd>{eta(progress?.etaMs ?? 0)}</dd>
              </>
            )}
            {runtime && (
              <>
                <dt>On disk after</dt>
                <dd>{megabytes(runtime.installBytes)}</dd>
                <dt>Free space needed</dt>
                <dd>
                  {megabytes(runtime.requiredBytes)}
                  {runtime.freeBytes !== null &&
                    ` · ${megabytes(runtime.freeBytes)} available`}
                </dd>
                {runtime.driver && (
                  <>
                    <dt>Driver</dt>
                    <dd>
                      {runtime.driver.version} · CUDA{" "}
                      {runtime.driver.cudaMajor}.{runtime.driver.cudaMinor}
                    </dd>
                  </>
                )}
              </>
            )}
          </dl>

          {installing && stage && <p className="hint">{STAGES[stage] ?? stage}</p>}
          {error && <p className="notice">{error}</p>}

          <div className="download-actions">
            {installing ? (
              <button
                className="action"
                onClick={gpu.cancel}
                disabled={stage !== "downloading"}
                title={
                  stage === "downloading"
                    ? "Stop; the partial file is kept"
                    : "Unpacking cannot be interrupted safely"
                }
              >
                Cancel
              </button>
            ) : (
              <button className="action action-primary" onClick={gpu.install}>
                {resuming ? "Resume download" : "Download from NVIDIA"}
              </button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
