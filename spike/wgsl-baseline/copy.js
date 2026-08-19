// copy.js — a copy button for a harness page's output.
//
// Every page here ends by asking for its output to be pasted back, so the pages
// grew a manual select-and-copy step that loses the structure people actually
// need: the table alignment, which is monospace-dependent, and the verdict lines.
//
// Copies the TEXT rather than the HTML, so the colour spans become nothing and
// the alignment survives. `innerText` is deliberate over `textContent`: it
// respects the rendered line breaks, which is what makes a pasted table still
// look like a table.

/**
 * Attach a copy button next to `runButton` that copies `logEl`'s rendered text.
 * Disabled while the log is empty, so it cannot copy nothing and look broken.
 */
export function attachCopyButton(runButton, logEl) {
  const btn = document.createElement("button");
  btn.textContent = "copy output";
  btn.disabled = true;
  btn.style.marginLeft = ".5rem";
  runButton.insertAdjacentElement("afterend", btn);

  const say = (msg) => {
    const was = btn.textContent;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = was === msg ? "copy output" : was; }, 1200);
  };

  btn.addEventListener("click", async () => {
    const text = logEl.innerText;
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      say("copied");
    } catch {
      // Clipboard access can be refused — over plain http on a non-localhost
      // origin, or without a user gesture. Selecting the text is still useful,
      // and saying so beats a button that silently does nothing.
      const r = document.createRange();
      r.selectNodeContents(logEl);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      say("selected — press ⌘C");
    }
  });

  // Enable as soon as there is something to copy, and keep it in step after that.
  new MutationObserver(() => { btn.disabled = !logEl.innerText.trim(); })
    .observe(logEl, { childList: true, subtree: true, characterData: true });

  return btn;
}
