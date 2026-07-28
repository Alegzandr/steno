import { isolateHistory } from "@codemirror/commands";
import {
  EditorState,
  Transaction,
  type TransactionSpec,
} from "@codemirror/state";

/**
 * Every change Steno makes to the buffer, as transaction specs.
 *
 * Separated from the editor component so the undo contract can be tested
 * against a real `EditorState` and the real `history()` extension, with no DOM
 * and no React. The rule it enforces — one Ctrl+Z restores the exact
 * pre-cleanup text — is a property of *which* transactions enter the history,
 * which is decided here and nowhere else.
 *
 * There are only two, and that is the point. An earlier version streamed the
 * cleanup into the document itself, using transactions marked out of the
 * history and putting the original back before committing. It satisfied the
 * single-undo rule and still destroyed the history: clearing the document maps
 * every earlier undo entry onto an empty range, so after one cleanup the bursts
 * dictated before it could no longer be undone. Streaming now happens outside
 * the editor entirely, and the document changes exactly once per cleanup.
 */

/**
 * A dictated burst, inserted at the cursor.
 *
 * At the cursor rather than at the end: the buffer accumulates while the user
 * types between dictations, and a burst belongs where they are looking. Returns
 * null for an empty burst so the caller does not dispatch a no-op.
 */
export function appendSpec(
  state: EditorState,
  text: string,
): TransactionSpec | null {
  if (!text.trim()) return null;

  const at = state.selection.main.head;
  const before = state.doc.sliceString(Math.max(0, at - 2), at);
  const after = state.doc.sliceString(at, Math.min(state.doc.length, at + 2));

  // One blank line on each side, and never two: Markdown needs the break to
  // keep bursts as separate paragraphs, and a doubled one is visible in the
  // output the user pastes. Both sides matter — the cursor can perfectly well
  // be at the top of the buffer, and a burst inserted there must not run into
  // the text it lands above.
  let prefix = "";
  if (at > 0 && !before.endsWith("\n\n")) {
    prefix = before.endsWith("\n") ? "\n" : "\n\n";
  }

  let suffix = "\n";
  if (at < state.doc.length && !after.startsWith("\n")) {
    suffix = "\n\n";
  }

  const insert = `${prefix}${text.trim()}${suffix}`;
  return {
    changes: { from: at, insert },
    selection: { anchor: at + insert.length },
    // A burst is its own undo step. Without this, CodeMirror merges changes
    // that land within half a second of each other, so a transcription
    // arriving while the user is typing would be undone together with their
    // sentence — and two bursts dictated back to back would collapse into one.
    annotations: [
      Transaction.userEvent.of("input.dictation"),
      isolateHistory.of("full"),
    ],
    scrollIntoView: true,
  };
}

/**
 * The finished cleanup: the one transaction it contributes to the history.
 *
 * The document is untouched until this runs, so history stores the inverse of
 * exactly one change — whole buffer before, whole buffer after — and a single
 * undo lands on the pre-cleanup text with everything before it still reachable.
 *
 * `isolateHistory` stops CodeMirror merging this with adjacent typing, so one
 * Ctrl+Z can never swallow both the cleanup and the sentence typed after it.
 */
export function replaceAllSpec(
  state: EditorState,
  final: string,
): TransactionSpec {
  return {
    changes: { from: 0, to: state.doc.length, insert: final },
    selection: { anchor: final.length },
    annotations: [
      Transaction.userEvent.of("input.cleanup"),
      isolateHistory.of("full"),
    ],
    scrollIntoView: true,
  };
}
