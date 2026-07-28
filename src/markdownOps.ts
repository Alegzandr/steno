import { isolateHistory } from "@codemirror/commands";
import {
  type ChangeSpec,
  EditorState,
  Transaction,
  type TransactionSpec,
} from "@codemirror/state";

/**
 * The toolbar's Markdown commands, as transaction specs.
 *
 * Separated from the component for the same reason as `bufferOps`: these are
 * pure functions of an `EditorState`, and testing them against a real state is
 * both cheaper and stricter than testing them through a rendered editor.
 *
 * Every command is one undo step. A toolbar button that takes two Ctrl+Z to
 * back out is worse than no button, because the second Ctrl+Z eats something
 * the user did mean to keep.
 */

function spec(changes: ChangeSpec[], event: string): TransactionSpec {
  return {
    changes,
    annotations: [
      Transaction.userEvent.of(event),
      isolateHistory.of("full"),
    ],
  };
}

/** The lines the selection touches, whole. */
function selectedLines(state: EditorState): { number: number }[] {
  const { from, to } = state.selection.main;
  const first = state.doc.lineAt(from).number;
  const last = state.doc.lineAt(to).number;
  const lines = [];
  for (let n = first; n <= last; n++) lines.push({ number: n });
  return lines;
}

function mapLines(
  state: EditorState,
  event: string,
  transform: (text: string) => string,
): TransactionSpec {
  const changes: ChangeSpec[] = [];
  for (const { number } of selectedLines(state)) {
    const line = state.doc.line(number);
    const next = transform(line.text);
    if (next !== line.text) {
      changes.push({ from: line.from, to: line.to, insert: next });
    }
  }
  return spec(changes, event);
}

const HEADING = /^(#{1,6})[ \t]+/;
const BULLET = /^([ \t]*)([-*+])[ \t]+/;

/**
 * `##` on every touched line, or off every touched line.
 *
 * The decision is taken once for the whole selection rather than per line: a
 * button that turns half the lines into headings and strips the other half is
 * unusable on a multi-line selection.
 */
export function toggleHeadingSpec(state: EditorState): TransactionSpec {
  const lines = selectedLines(state).map(({ number }) => state.doc.line(number));
  const allHeadings = lines.every((line) => line.text.startsWith("## "));

  return mapLines(state, "input.format.heading", (text) => {
    if (allHeadings) return text.slice(3);
    // An `###` becomes a `##` rather than a `#####`.
    if (HEADING.test(text)) return text.replace(HEADING, "## ");
    return `## ${text}`;
  });
}

/** `- ` on every touched line, or off every touched line. Indentation is kept. */
export function toggleBulletSpec(state: EditorState): TransactionSpec {
  const lines = selectedLines(state).map(({ number }) => state.doc.line(number));
  const allBullets = lines.every((line) => BULLET.test(line.text));

  return mapLines(state, "input.format.bullet", (text) => {
    if (allBullets) return text.replace(BULLET, "$1");
    const indent = /^[ \t]*/.exec(text)?.[0] ?? "";
    return `${indent}- ${text.slice(indent.length)}`;
  });
}

/**
 * Wraps the selected lines in a fenced block, or opens an empty one.
 *
 * It does not unwrap. Detecting "the selection is exactly the inside of a
 * fence" is guesswork on a partial selection, and guessing wrong deletes the
 * user's fences; removing two lines by hand is cheap.
 */
export function codeBlockSpec(state: EditorState): TransactionSpec | null {
  const { from, to } = state.selection.main;

  if (from === to) {
    const line = state.doc.lineAt(from);
    const insert = line.text.length === 0 ? "```\n\n```" : "\n```\n\n```";
    const at = line.text.length === 0 ? line.from : line.to;
    return {
      ...spec([{ from: at, to: at, insert }], "input.format.code"),
      // The blank line between the fences, which is where the next keystroke
      // belongs. `indexOf("\n\n")` is the newline that *ends* the opening
      // fence, so the empty line starts one character later.
      selection: { anchor: at + insert.indexOf("\n\n") + 1 },
    };
  }

  const first = state.doc.lineAt(from);
  const last = state.doc.lineAt(to);
  const body = state.doc.sliceString(first.from, last.to);
  return spec(
    [{ from: first.from, to: last.to, insert: `\`\`\`\n${body}\n\`\`\`` }],
    "input.format.code",
  );
}

/** `**` around the selection, or off it. An empty selection opens the pair. */
export function toggleBoldSpec(state: EditorState): TransactionSpec {
  const { from, to } = state.selection.main;

  if (from === to) {
    return {
      ...spec([{ from, to, insert: "****" }], "input.format.bold"),
      selection: { anchor: from + 2 },
    };
  }

  const text = state.doc.sliceString(from, to);
  if (text.length >= 4 && text.startsWith("**") && text.endsWith("**")) {
    return {
      ...spec([{ from, to, insert: text.slice(2, -2) }], "input.format.bold"),
      selection: { anchor: from, head: to - 4 },
    };
  }

  // The markers may sit just outside the selection, which is what a
  // double-click on a bold word gives you.
  const before = state.doc.sliceString(Math.max(0, from - 2), from);
  const after = state.doc.sliceString(to, Math.min(state.doc.length, to + 2));
  if (before === "**" && after === "**") {
    return {
      ...spec(
        [
          { from: from - 2, to: from, insert: "" },
          { from: to, to: to + 2, insert: "" },
        ],
        "input.format.bold",
      ),
      selection: { anchor: from - 2, head: to - 2 },
    };
  }

  return {
    ...spec([{ from, to, insert: `**${text}**` }], "input.format.bold"),
    selection: { anchor: from + 2, head: to + 2 },
  };
}
