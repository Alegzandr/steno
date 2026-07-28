import type { Format } from "./Editor";

/**
 * The row above the editor.
 *
 * Seven controls and no eighth. The window is 400 px wide, and a Markdown
 * toolbar that tries to cover Markdown ends up a scrolling strip of glyphs
 * nobody reads. These are the four marks a dictated brainstorm actually needs,
 * plus the two that make the buffer safe to experiment in.
 *
 * Undo and redo are here mainly to be *visible*. Ctrl+Z is invisible, and a
 * user who does not know the cleanup can be backed out will not risk running
 * it. They are disabled when the history has nothing to give, so their state
 * is information rather than decoration.
 */

type Props = {
  canUndo: boolean;
  canRedo: boolean;
  /** A cleanup is streaming: the buffer is read-only until it lands. */
  disabled: boolean;
  onUndo(): void;
  onRedo(): void;
  onFormat(kind: Format): void;
};

const isMac = navigator.userAgent.includes("Mac");
const mod = isMac ? "⌘" : "Ctrl+";

const MARKS: { kind: Format; glyph: string; title: string }[] = [
  { kind: "heading", glyph: "H2", title: "Heading (##)" },
  { kind: "bullet", glyph: "•—", title: "Bullet list (-)" },
  { kind: "code", glyph: "{ }", title: "Code block (```)" },
  { kind: "bold", glyph: "B", title: "Bold (**)" },
];

export function FormatBar({
  canUndo,
  canRedo,
  disabled,
  onUndo,
  onRedo,
  onFormat,
}: Props) {
  return (
    <div className="formatbar" role="toolbar" aria-label="Formatting">
      <button
        className="mark"
        onClick={onUndo}
        disabled={disabled || !canUndo}
        title={`Undo (${mod}Z)`}
      >
        ↶
      </button>
      <button
        className="mark"
        onClick={onRedo}
        disabled={disabled || !canRedo}
        title={`Redo (${mod}${isMac ? "⇧Z" : "Y"})`}
      >
        ↷
      </button>

      <span className="mark-separator" aria-hidden="true" />

      {MARKS.map(({ kind, glyph, title }) => (
        <button
          key={kind}
          className={`mark mark-${kind}`}
          onClick={() => onFormat(kind)}
          disabled={disabled}
          title={title}
        >
          {glyph}
        </button>
      ))}
    </div>
  );
}
