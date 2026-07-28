import { describe, expect, it } from "vitest";
import { history, undo, undoDepth } from "@codemirror/commands";
import { EditorState, type TransactionSpec } from "@codemirror/state";

import {
  codeBlockSpec,
  toggleBoldSpec,
  toggleBulletSpec,
  toggleHeadingSpec,
} from "./markdownOps";

/**
 * The toolbar's commands, against a real `EditorState` and a real `history()`.
 *
 * The two things worth asserting are the text they produce and the fact that
 * each is exactly one undo step. What this does NOT cover is which button is
 * wired to which function, or whether it is disabled at the right moment —
 * that lives in `Editor.tsx` and `App.tsx`.
 */

function buffer(doc = "", from = doc.length, to = from) {
  let state = EditorState.create({
    doc,
    selection: { anchor: from, head: to },
    extensions: [history()],
  });

  return {
    get text() {
      return state.doc.toString();
    },
    get selected() {
      const { from, to } = state.selection.main;
      return state.doc.sliceString(from, to);
    },
    get state() {
      return state;
    },
    get depth() {
      return undoDepth(state);
    },
    apply(spec: TransactionSpec | null) {
      if (spec) state = state.update(spec).state;
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

describe("the heading button", () => {
  it("adds and removes `##` on the current line", () => {
    const b = buffer("une idée");
    b.apply(toggleHeadingSpec(b.state));
    expect(b.text).toBe("## une idée");

    b.apply(toggleHeadingSpec(b.state));
    expect(b.text).toBe("une idée");
  });

  it("demotes a heading of another level rather than stacking hashes", () => {
    const b = buffer("#### trop profond", 0);
    b.apply(toggleHeadingSpec(b.state));
    expect(b.text).toBe("## trop profond");
  });

  it("decides once for the whole selection", () => {
    // One line is already a heading, the other is not: the selection becomes
    // headings, it does not half invert.
    const b = buffer("## déjà\nautre", 0, 13);
    b.apply(toggleHeadingSpec(b.state));
    expect(b.text).toBe("## déjà\n## autre");
  });
});

describe("the bullet button", () => {
  it("adds and removes `- ` across the selection", () => {
    const b = buffer("un\ndeux", 0, 7);
    b.apply(toggleBulletSpec(b.state));
    expect(b.text).toBe("- un\n- deux");

    b.apply(toggleBulletSpec(b.state));
    expect(b.text).toBe("un\ndeux");
  });

  it("keeps the indentation of a nested item", () => {
    const b = buffer("  imbriqué", 0);
    b.apply(toggleBulletSpec(b.state));
    expect(b.text).toBe("  - imbriqué");
  });
});

describe("the code block button", () => {
  it("fences the selected lines", () => {
    const b = buffer("cargo build\nnpm run build", 0, 25);
    b.apply(codeBlockSpec(b.state));
    expect(b.text).toBe("```\ncargo build\nnpm run build\n```");
  });

  it("opens an empty block with the cursor inside it", () => {
    const b = buffer("");
    b.apply(codeBlockSpec(b.state));
    expect(b.text).toBe("```\n\n```");
    expect(b.state.selection.main.head).toBe(4);
  });

  it("starts the block on its own line when the cursor is mid-text", () => {
    const b = buffer("une phrase");
    b.apply(codeBlockSpec(b.state));
    expect(b.text).toBe("une phrase\n```\n\n```");
  });
});

describe("the bold button", () => {
  it("wraps and unwraps the selection", () => {
    const b = buffer("un mot", 3, 6);
    b.apply(toggleBoldSpec(b.state));
    expect(b.text).toBe("un **mot**");
    // The selection still covers the word, not the markers.
    expect(b.selected).toBe("mot");

    b.apply(toggleBoldSpec(b.state));
    expect(b.text).toBe("un mot");
  });

  it("unwraps when the markers sit just outside the selection", () => {
    // What a double-click on a bold word gives you.
    const b = buffer("un **mot**", 5, 8);
    b.apply(toggleBoldSpec(b.state));
    expect(b.text).toBe("un mot");
  });

  it("opens an empty pair with the cursor between the markers", () => {
    const b = buffer("un ");
    b.apply(toggleBoldSpec(b.state));
    expect(b.text).toBe("un ****");
    expect(b.state.selection.main.head).toBe(5);
  });
});

describe("the undo contract", () => {
  it("makes each command exactly one undo step", () => {
    const b = buffer("une idée");
    b.apply(toggleHeadingSpec(b.state));
    b.apply(toggleBulletSpec(b.state));
    expect(b.depth).toBe(2);

    b.undo();
    expect(b.text).toBe("## une idée");
    b.undo();
    expect(b.text).toBe("une idée");
  });

  it("never merges two commands into one step", () => {
    // CodeMirror groups changes made within half a second of each other, and
    // these arrive as fast as the user can click.
    const b = buffer("un mot", 3, 6);
    b.apply(toggleBoldSpec(b.state));
    b.apply(toggleHeadingSpec(b.state));
    expect(b.depth).toBe(2);
  });
});
