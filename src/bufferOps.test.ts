import { describe, expect, it } from "vitest";
import { history, undo } from "@codemirror/commands";
import { EditorState, type TransactionSpec } from "@codemirror/state";

import { appendSpec, replaceAllSpec } from "./bufferOps";

/**
 * The undo contract, tested against the real `history()` extension.
 *
 * No DOM and no React: `EditorState` and the `undo` command are the parts that
 * decide this, and they run headless. What this does NOT cover is that the
 * editor component dispatches these specs at the right moments — that is
 * `Editor.tsx`, and it is covered by reading it.
 */

/** A tiny harness that plays the role of the EditorView. */
function buffer(doc = "", cursor = doc.length) {
  let state = EditorState.create({
    doc,
    selection: { anchor: cursor },
    extensions: [history()],
  });

  return {
    get text() {
      return state.doc.toString();
    },
    get state() {
      return state;
    },
    apply(spec: TransactionSpec | null) {
      if (spec) state = state.update(spec).state;
    },
    /** What the user types, which enters the history normally. */
    type(text: string) {
      const at = state.selection.main.head;
      state = state.update({
        changes: { from: at, insert: text },
        selection: { anchor: at + text.length },
        userEvent: "input.type",
      }).state;
    },
    undo() {
      return undo({
        state,
        dispatch: (transaction) => {
          state = transaction.state;
        },
      });
    },
  };
}

describe("the accumulation buffer", () => {
  it("appends a burst at the cursor instead of replacing the buffer", () => {
    const b = buffer();
    b.apply(appendSpec(b.state, "première idée"));
    b.apply(appendSpec(b.state, "deuxième idée"));

    expect(b.text).toBe("première idée\n\ndeuxième idée\n");
  });

  it("never doubles the blank line between bursts", () => {
    const b = buffer("déjà là\n\n");
    b.apply(appendSpec(b.state, "la suite"));

    expect(b.text).toBe("déjà là\n\nla suite\n");
  });

  it("separates a burst from the text it lands above", () => {
    // The cursor is at the very top, which is where it sits after a cleanup
    // scrolls the user back up. The burst must not run into what follows.
    const b = buffer("le paragraphe existant\n", 0);
    b.apply(appendSpec(b.state, "une idée qui arrive avant"));

    expect(b.text).toBe("une idée qui arrive avant\n\nle paragraphe existant\n");
  });

  it("ignores an empty burst", () => {
    const b = buffer("intact");
    b.apply(appendSpec(b.state, "   \n "));

    expect(b.text).toBe("intact");
  });
});

describe("the cleanup undo contract", () => {
  it("restores the exact pre-cleanup text with one undo", () => {
    const b = buffer();
    b.apply(appendSpec(b.state, "alors je veux refactorer le middleware"));
    b.apply(appendSpec(b.state, "et brancher le endpoint sur Ollama"));

    const original = b.text;
    b.apply(replaceAllSpec(b.state, "## Refactor\n\n- Le `middleware`\n"));
    expect(b.text).not.toBe(original);

    b.undo();
    expect(b.text).toBe(original);
  });

  it("leaves the bursts dictated before the cleanup still undoable", () => {
    // The regression that killed the stream-into-the-document design: a
    // cleanup must cost one undo step, not the whole history behind it.
    const b = buffer();
    b.apply(appendSpec(b.state, "première dictée"));
    const afterFirst = b.text;
    b.apply(appendSpec(b.state, "deuxième dictée"));
    const afterSecond = b.text;

    b.apply(replaceAllSpec(b.state, "## Propre\n"));

    b.undo();
    expect(b.text).toBe(afterSecond);
    b.undo();
    expect(b.text).toBe(afterFirst);
    b.undo();
    expect(b.text).toBe("");
  });

  it("does not let a single undo swallow typing done after the cleanup", () => {
    const b = buffer();
    b.apply(appendSpec(b.state, "la dictée"));
    const original = b.text;

    b.apply(replaceAllSpec(b.state, "## Propre\n"));
    b.type("une note ajoutée après");

    // First undo takes back the typing only.
    b.undo();
    expect(b.text).toBe("## Propre\n");

    // Second undo takes back the cleanup, and lands exactly on the dictation.
    b.undo();
    expect(b.text).toBe(original);
  });

  it("costs nothing at all when a cleanup never commits", () => {
    // Cancelled and failed cleanups never touch the document, so there is
    // simply no transaction to undo.
    const b = buffer();
    b.apply(appendSpec(b.state, "la dictée"));
    const original = b.text;

    expect(b.text).toBe(original);
    b.undo();
    expect(b.text).toBe("");
  });
});
