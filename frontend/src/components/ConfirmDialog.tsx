import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  title: string;
  body: string;
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/** In-page modal (no window.confirm): focus moves to Cancel, Escape closes. */
export function ConfirmDialog(p: Props) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (p.open) cancel.current?.focus();
  }, [p.open]);
  if (!p.open) return null;
  return (
    <div className="dialog-backdrop">
      <div
        className="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="dlg-title"
        aria-describedby="dlg-body"
        onKeyDown={(e) => e.key === "Escape" && p.onCancel()}
      >
        <h2 id="dlg-title">{p.title}</h2>
        <p id="dlg-body">{p.body}</p>
        <div className="row">
          <button type="button" ref={cancel} onClick={p.onCancel}>
            {p.cancelLabel}
          </button>
          <button type="button" className="danger" onClick={p.onConfirm}>
            {p.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
