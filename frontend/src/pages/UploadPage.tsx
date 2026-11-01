import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { errorMessage } from "../api/errors";
import {
  fileFormat,
  MAX_FILE_BYTES,
  MIN_NII_BYTES,
  type UploadHandle,
  type UploadRole,
  uploadCase,
} from "../api/upload";
import { Layout } from "../components/Layout";
import { ko } from "../i18n/ko";

interface Picked {
  file: File;
  format: string;
}

const MiB = 1024 * 1024;

export function UploadPage() {
  const navigate = useNavigate();
  const [picked, setPicked] = useState<Partial<Record<UploadRole, Picked>>>({});
  const [fieldError, setFieldError] = useState<Partial<Record<UploadRole, string>>>({});
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const handle = useRef<UploadHandle | null>(null);
  const inputs = useRef<Partial<Record<UploadRole, HTMLInputElement | null>>>({});
  const uploading = progress !== null;

  useEffect(() => {
    if (!uploading) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [uploading]);
  useEffect(() => () => handle.current?.abort(), []);

  function choose(role: UploadRole, file: File | undefined) {
    setError(null);
    if (!file) return;
    const format = fileFormat(file);
    const problem = !format
      ? ko.upload.badFormat
      : file.size === 0
        ? ko.upload.emptyFile
        : format === ".nii" && file.size < MIN_NII_BYTES
          ? ko.upload.tooSmall
          : file.size > MAX_FILE_BYTES
            ? ko.upload.tooLarge
            : null;
    setFieldError((f) => ({ ...f, [role]: problem ?? undefined }));
    setPicked((p) => {
      const next = { ...p };
      if (problem || !format) delete next[role];
      else next[role] = { file, format };
      return next;
    });
  }

  async function submit() {
    const dwi = picked.dwi;
    const adc = picked.adc;
    if (!dwi || !adc || uploading) return;
    setError(null);
    setProgress(0);
    handle.current = uploadCase({ dwi: dwi.file, adc: adc.file }, setProgress);
    try {
      const created = await handle.current.promise;
      navigate(`/cases/${created.case_id}`);
    } catch (e) {
      setError(
        e instanceof DOMException && e.name === "AbortError"
          ? ko.upload.cancelled
          : errorMessage(e),
      );
    } finally {
      handle.current = null;
      setProgress(null);
    }
  }

  return (
    <Layout>
      <section className="panel narrow">
        <h1>{ko.upload.title}</h1>
        <p>{ko.upload.help}</p>
        <p className="note">{ko.upload.privacy}</p>
        {(["dwi", "adc"] as const).map((role) => {
          const p = picked[role];
          const id = `file-${role}`;
          const roleLabel = role === "dwi" ? ko.upload.roleDwi : ko.upload.roleAdc;
          return (
            <div className="file-field" key={role}>
              <label htmlFor={id}>{roleLabel}</label>
              {/* The native input is hidden: browsers render the chosen file name next to it. */}
              <input
                id={id}
                ref={(el) => {
                  inputs.current[role] = el;
                }}
                className="sr-only"
                type="file"
                tabIndex={-1}
                aria-hidden="true"
                accept=".nii,.gz,application/gzip"
                disabled={uploading}
                onChange={(e) => {
                  choose(role, e.target.files?.[0]);
                  e.target.value = ""; // do not keep the name in the DOM; allows re-picking
                }}
              />
              <button
                type="button"
                className="file-choose"
                disabled={uploading}
                aria-describedby={`status-${role}`}
                onClick={() => inputs.current[role]?.click()}
              >
                {ko.upload.chooseFor(roleLabel)}
              </button>
              <span className="file-status" id={`status-${role}`} data-testid={`status-${role}`}>
                {p
                  ? `${ko.upload.selected} · ${p.format} · ${(p.file.size / MiB).toFixed(2)} MiB`
                  : ko.upload.notSelected}
              </span>
              {fieldError[role] && (
                <span className="field-error" role="alert">
                  {fieldError[role]}
                </span>
              )}
            </div>
          );
        })}
        {uploading ? (
          <div>
            <div
              className="progress"
              role="progressbar"
              aria-label={ko.upload.uploading}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress * 100)}
            >
              <div style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <p aria-live="polite">
              {ko.upload.uploading} {Math.round(progress * 100)}%
            </p>
            <button type="button" onClick={() => handle.current?.abort()}>
              {ko.upload.cancel}
            </button>
          </div>
        ) : (
          <button
            type="button"
            className="primary"
            disabled={!picked.dwi || !picked.adc}
            onClick={submit}
          >
            {ko.upload.submit}
          </button>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>
    </Layout>
  );
}
