const isMac = navigator.userAgent.includes("Mac");
const toggleShortcut = isMac ? "⌘⇧D" : "Ctrl+Shift+D";

function App() {
  return (
    <div className="app">
      <header className="titlebar" data-tauri-drag-region>
        <span className="titlebar-name" data-tauri-drag-region>
          Steno
        </span>
        <span className="titlebar-shortcut" data-tauri-drag-region>
          {toggleShortcut}
        </span>
      </header>

      <main className="editor-slot">
        <div id="editor" className="editor-placeholder" />
      </main>
    </div>
  );
}

export default App;
