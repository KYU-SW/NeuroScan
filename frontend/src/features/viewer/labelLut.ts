/**
 * Phase 7-0: colour look-up table for a component labelmap shown through the drawing layer.
 *
 * Label 0 = background (always fully transparent), 1..254 = components, 255 = reserved "other"
 * (components beyond 254 are grouped there). The LUT only changes how labels are *displayed*:
 * selecting, dimming or hiding a label never changes the labelmap or the prediction mask.
 *
 * Colours 1..254 come from a fixed RGB lattice (levels 35 apart) without near-grey or dark
 * entries, so every pair of labels differs by at least 35 in some channel and stays visible on
 * a grey-scale MRI. Pure module: no NiiVue import (the adapter turns the bytes into a draw LUT).
 */

export type Rgb = readonly [number, number, number];

export const BACKGROUND_LABEL = 0;
export const MAX_COMPONENT_LABEL = 254;
export const OTHER_LABEL = 255;
/** Alpha of non-selected labels while one label is selected. */
export const DIM_ALPHA = 0.25;
/** Neutral grey for the reserved "other" label. */
export const OTHER_COLOR: Rgb = [153, 153, 153];

const LEVELS = [40, 75, 110, 145, 180, 215, 250] as const;
const STRIDE = 97; // coprime with the candidate count, so consecutive labels contrast

function latticeColours(): Rgb[] {
  const out: Rgb[] = [];
  for (const r of LEVELS)
    for (const g of LEVELS)
      for (const b of LEVELS) {
        const hi = Math.max(r, g, b);
        if (hi - Math.min(r, g, b) >= 70 && hi >= 145) out.push([r, g, b]);
      }
  return out;
}

const CANDIDATES = latticeColours();

/** True for an integer label that can be selected or hidden (1..255). */
export function isDisplayLabel(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= OTHER_LABEL;
}

function requireLabel(value: unknown): number {
  if (!isDisplayLabel(value)) throw new Error("invalid component label");
  return value;
}

export function componentColor(label: number): Rgb {
  const l = requireLabel(label);
  if (l === OTHER_LABEL) return OTHER_COLOR;
  const c = CANDIDATES[(l * STRIDE) % CANDIDATES.length];
  if (!c) throw new Error("colour table");
  return c;
}

export interface ComponentDisplay {
  /** Highlighted label (own colour, full brightness); the others are dimmed. null = none. */
  selected: number | null;
  /** Labels removed from the display only (alpha 0). */
  hidden: ReadonlySet<number>;
  /** How non-selected labels are de-emphasised (Phase 7-A policy: "gray" in 2D and 3D).
   * "gray": opaque, colour mixed 80 % towards DIM_GRAY. "alpha": alpha DIM_ALPHA — kept only
   * for the Phase 7-0 spike comparison: NiiVue's 3D render does not dim a thick label through
   * partial alpha (the spike measured it brighter). */
  dimStyle?: DimStyle;
  /** "component" (default): each label its own colour. "single": every label uses the given
   * single colour (the Phase 6 mask palette colour). */
  coloring?: Coloring;
}

export type DimStyle = "alpha" | "gray";
export type Coloring = "component" | "single";
export const DEFAULT_DIM_STYLE: DimStyle = "gray";
export const DIM_GRAY: Rgb = [96, 96, 96];
/** Share of DIM_GRAY in a dimmed label colour. */
export const DIM_GRAY_MIX = 0.8;

export const SHOW_ALL: ComponentDisplay = { selected: null, hidden: new Set() };

/** Validates a display state: every label must be 1..255. */
export function checkDisplay(display: ComponentDisplay): ComponentDisplay {
  if (display.selected !== null) requireLabel(display.selected);
  for (const h of display.hidden) requireLabel(h);
  if (
    display.dimStyle !== undefined &&
    display.dimStyle !== "alpha" &&
    display.dimStyle !== "gray"
  ) {
    throw new Error("invalid dim style");
  }
  if (
    display.coloring !== undefined &&
    display.coloring !== "component" &&
    display.coloring !== "single"
  ) {
    throw new Error("invalid coloring");
  }
  return display;
}

const dimStyleOf = (display: ComponentDisplay): DimStyle => display.dimStyle ?? DEFAULT_DIM_STYLE;

/** True when the label is shown but de-emphasised because another label is selected. */
export function isDimmed(label: number, display: ComponentDisplay): boolean {
  return display.selected !== null && display.selected !== label && !display.hidden.has(label);
}

/** Displayed colour of one label (grey-mixed when dimmed with the "gray" style). */
export function labelColor(
  label: number,
  display: ComponentDisplay,
  singleColor: Rgb = OTHER_COLOR,
): Rgb {
  requireLabel(label);
  const c = display.coloring === "single" ? singleColor : componentColor(label);
  if (dimStyleOf(display) !== "gray" || !isDimmed(label, display)) return c;
  const m = (i: 0 | 1 | 2) => Math.round(c[i] * (1 - DIM_GRAY_MIX) + DIM_GRAY[i] * DIM_GRAY_MIX);
  return [m(0), m(1), m(2)];
}

/** Display alpha (0..1) of one label. Hidden wins over selected. */
export function labelAlpha(label: number, display: ComponentDisplay): number {
  if (label === BACKGROUND_LABEL) return 0;
  const l = requireLabel(label);
  if (display.hidden.has(l)) return 0;
  if (!isDimmed(l, display)) return 1;
  return dimStyleOf(display) === "gray" ? 1 : DIM_ALPHA;
}

/** 256 RGBA entries (1024 bytes) for the drawing layer. */
export function buildComponentLut(
  display: ComponentDisplay,
  singleColor: Rgb = OTHER_COLOR,
): Uint8ClampedArray {
  checkDisplay(display);
  const lut = new Uint8ClampedArray(256 * 4);
  for (let label = 1; label <= OTHER_LABEL; label++) {
    const [r, g, b] = labelColor(label, display, singleColor);
    lut.set([r, g, b, Math.round(255 * labelAlpha(label, display))], label * 4);
  }
  return lut; // entry 0 stays 0,0,0,0 (transparent background)
}
