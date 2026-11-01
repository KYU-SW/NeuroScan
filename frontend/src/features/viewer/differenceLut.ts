/**
 * Phase 7-B research overlays on the difference labelmap (computed by the backend, never here):
 * 0 background, 1 = TP (AI and GT), 2 = FP (AI only), 3 = FN (GT only).
 *
 * "difference": TP / FP / FN in the approved colour-blind-safe palette, each class switchable.
 * "ground_truth": the ground-truth extent (TP + FN) in the GT-only colour, AI-only voxels hidden.
 * Colour is never the only cue: the research page always shows the text legend.
 * Pure module (no NiiVue); the adapter turns the bytes into a draw LUT.
 */
import { hexToRgb } from "./ViewerEngine";

export const DIFFERENCE_LABELS = { tp: 1, fp: 2, fn: 3 } as const;
export type DifferenceClass = keyof typeof DIFFERENCE_LABELS;
export const DIFFERENCE_CLASSES: DifferenceClass[] = ["tp", "fp", "fn"];

/** Approved palette (Phase 7 decision): Okabe-Ito colours. */
export const DIFFERENCE_PALETTE: Record<DifferenceClass, string> = {
  tp: "#009E73",
  fp: "#E69F00",
  fn: "#56B4E9",
};
export const GT_ONLY_COLOR = "#CC79A7";

export type ResearchOverlay =
  | { kind: "difference"; visible: Readonly<Record<DifferenceClass, boolean>> }
  | { kind: "ground_truth" };

export function isResearchOverlay(value: unknown): value is ResearchOverlay {
  if (typeof value !== "object" || value === null) return false;
  const kind = Reflect.get(value, "kind");
  if (kind === "ground_truth") return true;
  if (kind !== "difference") return false;
  const visible = Reflect.get(value, "visible");
  return (
    typeof visible === "object" &&
    visible !== null &&
    DIFFERENCE_CLASSES.every((c) => typeof Reflect.get(visible, c) === "boolean")
  );
}

/** 256 RGBA entries; everything other than labels 1..3 is transparent. */
export function buildDifferenceLut(overlay: ResearchOverlay): Uint8ClampedArray {
  if (!isResearchOverlay(overlay)) throw new Error("invalid research overlay");
  const lut = new Uint8ClampedArray(256 * 4);
  for (const cls of DIFFERENCE_CLASSES) {
    const label = DIFFERENCE_LABELS[cls];
    let colour: string | null = null;
    if (overlay.kind === "difference")
      colour = overlay.visible[cls] ? DIFFERENCE_PALETTE[cls] : null;
    else colour = cls === "fp" ? null : GT_ONLY_COLOR; // GT extent = TP + FN
    if (colour) lut.set([...hexToRgb(colour), 255], label * 4);
  }
  return lut;
}
