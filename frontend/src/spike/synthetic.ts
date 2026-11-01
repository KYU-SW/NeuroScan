/**
 * Synthetic, non-sensitive NIfTI-1 volumes for the NiiVue spike (no patient data).
 *
 * LAS affine like ISLES'22: world x = -sx * i + ox, so a larger array index i is further to the
 * patient's LEFT. DWI/ADC carry a voxel checkerboard inside an ellipsoid "brain" so that linear
 * vs nearest filtering is measurable; the mask is a block on the patient's left and anterior
 * (anterior offset lets the spike tell the 3D camera presets apart).
 */

export const DIMS = [64, 64, 20] as const;
export const SPACING = [2, 2, 3] as const;
export const ORIGIN = [63, -63, -27] as const;
/** inclusive voxel ranges of the lesion block (i on the patient's left, j anterior) */
export const LESION = { i: [40, 47], j: [36, 43], k: [8, 11] } as const;

export const AFFINE: number[][] = [
  [-SPACING[0], 0, 0, ORIGIN[0]],
  [0, SPACING[1], 0, ORIGIN[1]],
  [0, 0, SPACING[2], ORIGIN[2]],
  [0, 0, 0, 1],
];

export function voxToWorld(i: number, j: number, k: number): [number, number, number] {
  const row = (r: number) => {
    const a = AFFINE[r];
    if (!a) throw new Error("affine row");
    return (a[0] ?? 0) * i + (a[1] ?? 0) * j + (a[2] ?? 0) * k + (a[3] ?? 0);
  };
  return [row(0), row(1), row(2)];
}

export const LESION_CENTER_MM = voxToWorld(
  (LESION.i[0] + LESION.i[1]) / 2,
  (LESION.j[0] + LESION.j[1]) / 2,
  (LESION.k[0] + LESION.k[1]) / 2,
);

type Kind = "float32" | "uint8";

function niftiBytes(data: Float32Array | Uint8Array, kind: Kind): ArrayBuffer {
  const header = new ArrayBuffer(352);
  const v = new DataView(header);
  v.setInt32(0, 348, true);
  const dims = [3, DIMS[0], DIMS[1], DIMS[2], 1, 1, 1, 1];
  dims.forEach((d, n) => v.setInt16(40 + 2 * n, d, true));
  v.setInt16(70, kind === "float32" ? 16 : 2, true); // datatype
  v.setInt16(72, kind === "float32" ? 32 : 8, true); // bitpix
  const pixdim = [1, SPACING[0], SPACING[1], SPACING[2], 1, 1, 1, 1];
  pixdim.forEach((p, n) => v.setFloat32(76 + 4 * n, p, true));
  v.setFloat32(108, 352, true); // vox_offset
  v.setFloat32(112, 1, true); // scl_slope
  v.setUint8(123, 2); // xyzt_units: mm
  v.setInt16(252, 0, true); // qform_code
  v.setInt16(254, 1, true); // sform_code
  for (let r = 0; r < 3; r++) {
    const row = AFFINE[r] ?? [];
    for (let c = 0; c < 4; c++) v.setFloat32(280 + 16 * r + 4 * c, row[c] ?? 0, true);
  }
  [0x6e, 0x2b, 0x31, 0x00].forEach((b, n) => v.setUint8(344 + n, b)); // "n+1\0"
  const out = new Uint8Array(352 + data.byteLength);
  out.set(new Uint8Array(header), 0);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 352);
  return out.buffer;
}

async function gzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

function insideBrain(i: number, j: number, k: number): boolean {
  const [cx, cy, cz] = [(DIMS[0] - 1) / 2, (DIMS[1] - 1) / 2, (DIMS[2] - 1) / 2];
  return ((i - cx) / 26) ** 2 + ((j - cy) / 28) ** 2 + ((k - cz) / 9) ** 2 <= 1;
}

export async function syntheticVolumes(): Promise<{
  dwi: ArrayBuffer;
  adc: ArrayBuffer;
  mask: ArrayBuffer;
}> {
  const n = DIMS[0] * DIMS[1] * DIMS[2];
  const dwi = new Float32Array(n);
  const adc = new Float32Array(n);
  const mask = new Uint8Array(n);
  for (let k = 0; k < DIMS[2]; k++)
    for (let j = 0; j < DIMS[1]; j++)
      for (let i = 0; i < DIMS[0]; i++) {
        const idx = i + DIMS[0] * (j + DIMS[1] * k);
        if (!insideBrain(i, j, k)) continue;
        const checker = (i + j + k) % 2;
        dwi[idx] = 100 + 100 * checker;
        adc[idx] = 0.0006 + 0.0006 * (1 - checker);
        const inLesion =
          i >= LESION.i[0] &&
          i <= LESION.i[1] &&
          j >= LESION.j[0] &&
          j <= LESION.j[1] &&
          k >= LESION.k[0] &&
          k <= LESION.k[1];
        if (inLesion) {
          mask[idx] = 1;
          dwi[idx] = 400;
        }
      }
  return {
    dwi: await gzip(niftiBytes(dwi, "float32")),
    adc: await gzip(niftiBytes(adc, "float32")),
    mask: await gzip(niftiBytes(mask, "uint8")),
  };
}
