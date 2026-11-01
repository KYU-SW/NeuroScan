import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CASE, caseInfo } from "./fixtures";
import { HEALTH_OK, json, mockFetch, renderApp } from "./render";

/** Minimal XMLHttpRequest stand-in: captures the form, lets the test resolve the upload. */
class FakeXhr {
  static last: FakeXhr | null = null;
  upload: {
    onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null;
  } = {
    onprogress: null,
  };
  status = 0;
  responseText = "";
  responseType = "";
  withCredentials = true;
  method = "";
  url = "";
  form: FormData | null = null;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  aborted = false;
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  send(form: FormData) {
    this.form = form;
    FakeXhr.last = this;
  }
  abort() {
    this.aborted = true;
    this.onabort?.();
  }
  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

const PHI_NAME = "Hong_Gildong_19700101_DWI.nii.gz";

function file(name: string, size = 1024) {
  return new File([new Uint8Array(size)], name, { type: "application/gzip" });
}

describe("UploadPage", () => {
  let logs: unknown[][];
  beforeEach(() => {
    FakeXhr.last = null;
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    logs = [];
    for (const k of ["log", "info", "warn", "error", "debug"] as const)
      vi.spyOn(console, k).mockImplementation((...a: unknown[]) => void logs.push(a));
    mockFetch({
      "GET /api/v1/health": () => json(HEALTH_OK),
      "GET /api/v1/cases/": () => json(caseInfo()),
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows role, format and size but never the file name; uploads and navigates", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file(PHI_NAME, 2 * 1024 * 1024));
    await user.upload(screen.getByLabelText("ADC"), file("Hong_ADC.nii.gz"));
    expect(screen.getByTestId("status-dwi")).toHaveTextContent("선택됨 · .nii.gz · 2.00 MiB");
    expect(document.body.textContent).not.toContain("Hong");
    await user.click(screen.getByRole("button", { name: "업로드" }));
    const xhr = FakeXhr.last;
    if (!xhr) throw new Error("no upload");
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe("/api/v1/cases");
    expect(xhr.withCredentials).toBe(false);
    expect([...(xhr.form?.keys() ?? [])]).toEqual(["dwi", "adc"]);
    act(() => xhr.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 }));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
    expect(screen.getByRole("button", { name: "업로드 취소" })).toBeInTheDocument();
  });

  it("navigates to the case page after 201", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("a.nii.gz"));
    await user.upload(screen.getByLabelText("ADC"), file("b.nii"));
    await user.click(screen.getByRole("button", { name: "업로드" }));
    act(() => FakeXhr.last?.respond(201, caseInfo()));
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent(`/cases/${CASE}`));
    expect(logs.flat().join(" ")).not.toMatch(/a\.nii|b\.nii|Hong/);
  });

  it("rejects bad extensions and oversized files on the client", async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("scan.dcm"));
    expect(screen.getByText(".nii 또는 .nii.gz 파일만 선택할 수 있습니다.")).toBeInTheDocument();
    const big = file("big.nii.gz");
    Object.defineProperty(big, "size", { value: 65 * 1024 * 1024 });
    await user.upload(screen.getByLabelText("ADC"), big);
    expect(screen.getByText("파일이 64 MiB를 초과합니다.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "업로드" })).toBeDisabled();
  });

  it.each([
    ["0 byte .nii.gz", "empty.nii.gz", 0, "빈 파일입니다"],
    ["0 byte .nii", "empty.nii", 0, "빈 파일입니다"],
    ["100 byte .nii", "tiny.nii", 100, "NIfTI header"],
  ])("rejects a %s file on the client without uploading", async (_label, name, size, message) => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file(name, size));
    await user.upload(screen.getByLabelText("ADC"), file("b.nii.gz"));
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(screen.getByTestId("status-dwi")).toHaveTextContent("선택되지 않음");
    const submit = screen.getByRole("button", { name: "업로드" });
    expect(submit).toBeDisabled();
    await user.click(submit);
    expect(FakeXhr.last).toBeNull();
  });

  it("replaces a valid selection with an invalid one and disables upload", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("a.nii.gz"));
    await user.upload(screen.getByLabelText("ADC"), file("b.nii.gz"));
    expect(screen.getByRole("button", { name: "업로드" })).toBeEnabled();
    await user.upload(screen.getByLabelText("ADC"), file("c.nii.gz", 0));
    expect(screen.getByRole("button", { name: "업로드" })).toBeDisabled();
  });

  it("accepts real-size Validation-like .nii.gz files and shows their MiB size", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("d.nii.gz", 1_839_273));
    await user.upload(screen.getByLabelText("ADC"), file("e.nii.gz", 1_642_955));
    expect(screen.getByTestId("status-dwi")).toHaveTextContent("선택됨 · .nii.gz · 1.75 MiB");
    expect(screen.getByTestId("status-adc")).toHaveTextContent("선택됨 · .nii.gz · 1.57 MiB");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "업로드" })).toBeEnabled();
  });

  it("accepts a .nii file of at least the NIfTI-1 header size", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("a.nii", 352));
    expect(screen.getByTestId("status-dwi")).toHaveTextContent("선택됨 · .nii · 0.00 MiB");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("uses a custom picker: the native input (which shows file names) is hidden", async () => {
    const user = userEvent.setup();
    renderApp("/");
    for (const role of ["DWI", "ADC"]) {
      const input = screen.getByLabelText(role);
      expect(input).toHaveClass("sr-only");
      expect(input).toHaveAttribute("aria-hidden", "true");
      expect(input).toHaveAttribute("tabindex", "-1");
      expect(screen.getByRole("button", { name: `${role} 파일 선택` })).toBeInTheDocument();
    }
    const click = vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(() => {});
    await user.click(screen.getByRole("button", { name: "DWI 파일 선택" }));
    expect(click).toHaveBeenCalledTimes(1);
    expect(click.mock.contexts[0]).toBe(screen.getByLabelText("DWI"));
  });

  it("never shows the file name or a local path after selection", async () => {
    const user = userEvent.setup();
    renderApp("/");
    const dwi = screen.getByLabelText<HTMLInputElement>("DWI");
    await user.upload(dwi, file(PHI_NAME));
    await user.upload(screen.getByLabelText("ADC"), file("C__Users_patient_ADC.nii.gz"));
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/Hong|Gildong|patient|C:\\|fakepath/);
    expect(dwi.value).toBe(""); // the DOM input does not keep the name ("C:\fakepath\…")
  });

  it.each([
    [403, "SEALED_TEST_CASE", "봉인된 평가용(Test) 데이터는 사용할 수 없습니다."],
    [400, "MISSING_FILE", "DWI 또는 ADC 파일이 없거나 비어 있습니다."],
    [413, "FILE_TOO_LARGE", "파일이 허용 크기를 초과합니다."],
    [422, "INVALID_NIFTI", "올바른 NIfTI 파일이 아닙니다."],
    [507, "INSUFFICIENT_STORAGE", "서버 저장 공간이 부족합니다."],
  ])("shows a safe message for %s %s", async (status, code, message) => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("a.nii.gz"));
    await user.upload(screen.getByLabelText("ADC"), file("b.nii.gz"));
    await user.click(screen.getByRole("button", { name: "업로드" }));
    act(() =>
      FakeXhr.last?.respond(status, {
        error: {
          code,
          message: "server text C:\\Users\\x",
          request_id: "r",
          retryable: false,
          details: {},
        },
      }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(document.body.textContent).not.toContain("C:\\Users");
  });

  it("can cancel an upload in progress", async () => {
    const user = userEvent.setup();
    renderApp("/");
    await user.upload(screen.getByLabelText("DWI"), file("a.nii.gz"));
    await user.upload(screen.getByLabelText("ADC"), file("b.nii.gz"));
    await user.click(screen.getByRole("button", { name: "업로드" }));
    await user.click(screen.getByRole("button", { name: "업로드 취소" }));
    expect(FakeXhr.last?.aborted).toBe(true);
    expect(await screen.findByText("업로드를 취소했습니다.")).toBeInTheDocument();
  });
});
