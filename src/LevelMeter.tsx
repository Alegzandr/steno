type Props = {
  level: number;
  clipped: boolean;
  active: boolean;
};

// Peak amplitude is linear, and speech spends most of its time near the bottom
// of that scale. Plotting dBFS instead spreads the useful range across the bar.
const FLOOR_DB = -60;

function toBar(level: number): number {
  if (level <= 0) {
    return 0;
  }

  const db = 20 * Math.log10(Math.min(level, 1));
  return Math.max(0, 1 + db / -FLOOR_DB);
}

export function LevelMeter({ level, clipped, active }: Props) {
  const filled = active ? toBar(level) : 0;

  return (
    <div className="meter-row">
      <div
        className="meter"
        role="meter"
        aria-label="Input level"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(filled * 100)}
      >
        <div className="meter-fill" style={{ width: `${filled * 100}%` }} />
      </div>
      {/* Latched for the whole clip: a peak that flashed by is exactly the one
          worth knowing about. */}
      <span className={clipped ? "clip-flag clip-flag-on" : "clip-flag"}>
        CLIP
      </span>
    </div>
  );
}
