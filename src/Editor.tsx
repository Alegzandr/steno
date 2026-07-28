import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import {
  defaultKeymap,
  history,
  historyKeymap,
  redo,
  redoDepth,
  undo,
  undoDepth,
} from "@codemirror/commands";
import { markdown } from "@codemirror/lang-markdown";
import { defaultHighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { Compartment, EditorState } from "@codemirror/state";
import { EditorView, drawSelection, keymap } from "@codemirror/view";

import { appendSpec, replaceAllSpec } from "./bufferOps";
import {
  codeBlockSpec,
  toggleBoldSpec,
  toggleBulletSpec,
  toggleHeadingSpec,
} from "./markdownOps";

/**
 * The accumulation buffer.
 *
 * Every dictated burst lands here at the cursor; nothing ever replaces the
 * buffer except a cleanup, and a cleanup is one undo step. Those two sentences
 * are the whole contract, and both are harder than they look:
 *
 * - **Append, never replace.** A transcription arriving while the user is
 *   typing between bursts must not move their cursor to the end or clobber the
 *   line they are on.
 * - **One Ctrl+Z.** A cleanup takes tens of seconds and arrives token by token,
 *   but the document changes exactly once, at the end. The tokens are rendered
 *   outside the editor while they stream, so nothing provisional ever enters
 *   the buffer or its history. See `bufferOps` for why streaming *into* the
 *   document was abandoned.
 */

/** What the toolbar's four formatting buttons map to. */
export type Format = "heading" | "bullet" | "code" | "bold";

/** Whether undo and redo have anything to act on. */
export type HistoryState = { canUndo: boolean; canRedo: boolean };

export type EditorHandle = {
  /** The whole buffer, which is what a cleanup operates on. */
  text(): string;
  /** Appends a transcription at the cursor, after a blank line. */
  append(text: string): void;
  /** Applies a finished cleanup as one undoable transaction. */
  replaceAll(final: string): void;
  /** Exactly what Ctrl+Z does, which is what the Revert affordance promises. */
  undo(): void;
  redo(): void;
  format(kind: Format): void;
  focus(): void;
};

type Props = {
  /** Locks the editor while a cleanup is streaming into it. */
  readOnly: boolean;
  onChange?: (empty: boolean) => void;
  /** Fires whenever the undo or redo depth changes, so the toolbar can grey
      the buttons out. A button that is always enabled says nothing. */
  onHistory?: (state: HistoryState) => void;
  /** Chords that must work whether or not the editor has focus. */
  onCopyAndHide?: () => void;
  onCleanUp?: () => void;
  onEscape?: () => void;
};

const theme = EditorView.theme({
  "&": { height: "100%", fontSize: "13px" },
  ".cm-scroller": {
    fontFamily:
      "ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace",
    lineHeight: "1.55",
    overflow: "auto",
  },
  ".cm-content": { padding: "10px 12px" },
  "&.cm-focused": { outline: "none" },
});

export const Editor = forwardRef<EditorHandle, Props>(function Editor(
  { readOnly, onChange, onHistory, onCopyAndHide, onCleanUp, onEscape },
  ref,
) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const editable = useRef(new Compartment());
  // Last reported depths, so an update that changed neither does not re-render
  // the toolbar. Every keystroke produces an update.
  const depths = useRef({ undo: -1, redo: -1 });

  // Handlers change on every render; the editor is built once. Reading them
  // through a ref keeps the keymap pointing at the current ones without
  // tearing down CodeMirror on each keystroke.
  const handlers = useRef({
    onCopyAndHide,
    onCleanUp,
    onEscape,
    onChange,
    onHistory,
  });
  handlers.current = {
    onCopyAndHide,
    onCleanUp,
    onEscape,
    onChange,
    onHistory,
  };

  useEffect(() => {
    if (!host.current) return;

    const shortcuts = keymap.of([
      {
        key: "Mod-Enter",
        preventDefault: true,
        run: () => (handlers.current.onCopyAndHide?.(), true),
      },
      {
        key: "Mod-Shift-k",
        preventDefault: true,
        run: () => (handlers.current.onCleanUp?.(), true),
      },
      {
        key: "Escape",
        preventDefault: true,
        run: () => (handlers.current.onEscape?.(), true),
      },
    ]);

    const instance = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: "",
        extensions: [
          history(),
          drawSelection(),
          markdown(),
          syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
          EditorView.lineWrapping,
          // Before the default keymap so Mod-Enter and Escape are ours.
          shortcuts,
          keymap.of([...defaultKeymap, ...historyKeymap]),
          editable.current.of([]),
          theme,
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              handlers.current.onChange?.(
                update.state.doc.toString().trim().length === 0,
              );
            }

            const nextUndo = undoDepth(update.state);
            const nextRedo = redoDepth(update.state);
            if (
              nextUndo !== depths.current.undo ||
              nextRedo !== depths.current.redo
            ) {
              depths.current = { undo: nextUndo, redo: nextRedo };
              handlers.current.onHistory?.({
                canUndo: nextUndo > 0,
                canRedo: nextRedo > 0,
              });
            }
          }),
        ],
      }),
    });

    view.current = instance;
    return () => {
      instance.destroy();
      view.current = null;
    };
  }, []);

  useEffect(() => {
    const instance = view.current;
    if (!instance) return;
    instance.dispatch({
      effects: editable.current.reconfigure(
        readOnly
          ? [EditorState.readOnly.of(true), EditorView.editable.of(false)]
          : [],
      ),
    });
  }, [readOnly]);

  useImperativeHandle(ref, () => ({
    text: () => view.current?.state.doc.toString() ?? "",

    append(text) {
      const instance = view.current;
      if (!instance) return;

      const spec = appendSpec(instance.state, text);
      if (spec) instance.dispatch(spec);
    },

    replaceAll(final) {
      const instance = view.current;
      if (!instance || !final.trim()) return;
      instance.dispatch(replaceAllSpec(instance.state, final));
    },

    undo() {
      const instance = view.current;
      if (instance) undo(instance);
    },

    redo() {
      const instance = view.current;
      if (instance) redo(instance);
    },

    format(kind) {
      const instance = view.current;
      if (!instance) return;

      const build = {
        heading: toggleHeadingSpec,
        bullet: toggleBulletSpec,
        code: codeBlockSpec,
        bold: toggleBoldSpec,
      }[kind];

      const spec = build(instance.state);
      if (spec) instance.dispatch(spec);
      // The click moved focus to the button; formatting is a mid-typing
      // action and the caret has to come back.
      instance.focus();
    },

    focus: () => view.current?.focus(),
  }));

  return <div className="editor" ref={host} />;
});
