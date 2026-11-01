/**
 * Multipart upload of the DWI + ADC pair with progress (XMLHttpRequest; fetch has no upload
 * progress). The browser sets Content-Length for FormData, which the API requires. Nothing about
 * the files (names, sizes, contents) is logged.
 */
import { ApiUnavailableError, ContractError, toApiError } from "./client";
import { type CaseInfo, caseSchema } from "./schemas";

export const MAX_FILE_BYTES = 64 * 1024 * 1024;
/** NIfTI-1 header (348 bytes) + 4-byte extension flag: the smallest valid uncompressed .nii. */
export const MIN_NII_BYTES = 352;
export type UploadRole = "dwi" | "adc";

export type FileFormat = ".nii.gz" | ".nii";

/** Extension check on the client (the server checks again). The name itself is never shown. */
export function fileFormat(file: File): FileFormat | null {
  const n = file.name.toLowerCase();
  if (n.endsWith(".nii.gz")) return ".nii.gz";
  if (n.endsWith(".nii")) return ".nii";
  return null;
}

export interface UploadHandle {
  promise: Promise<CaseInfo>;
  abort: () => void;
}

export function uploadCase(
  files: Record<UploadRole, File>,
  onProgress: (fraction: number) => void,
): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<CaseInfo>((resolve, reject) => {
    const form = new FormData();
    form.append("dwi", files.dwi);
    form.append("adc", files.adc);
    xhr.open("POST", "/api/v1/cases");
    xhr.withCredentials = false;
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiUnavailableError());
    xhr.onabort = () => reject(new DOMException("upload aborted", "AbortError"));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status === 201) {
        const parsed = caseSchema.safeParse(body);
        if (parsed.success) resolve(parsed.data);
        else reject(new ContractError());
      } else if (xhr.status === 502 || xhr.status === 504 || xhr.status === 0) {
        reject(new ApiUnavailableError());
      } else {
        reject(toApiError(xhr.status, body));
      }
    };
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}
