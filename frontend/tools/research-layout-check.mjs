// Development-only layout stability check (Phase 7-B). Not part of the app or the test suite.
//
// Drives an already running headless Chrome (--remote-debugging-port) over the DevTools
// protocol (Node >= 22 global WebSocket; no packages) against a running preview + API, and
// records DOM measurements for each viewport size and viewer state:
//   A loaded · B region selected (research: selection summary) · C 3-plane · D 3D ·
//   E after a window resize · F after waiting 10 s
// In every state the canvas CSS / pixel height and the document height are sampled 10 times
// every 0.5 s; they must not drift. Containment and side-panel checks follow the Phase 7-B
// layout contract. Exit code 0 = all checks passed.
//
//   node tools/research-layout-check.mjs --port 9222 \
//     --url http://127.0.0.1:4173/research/cases/val-001 --mode research \
//     --sizes 1920x1080,1536x864,1366x768,1280x720,1536x864@1.25
//   (--mode general for /runs/<id>/viewer pages: no selection summary, no research panels)

const args = Object.fromEntries(
    process.argv
        .slice(2)
        .reduce(
            (acc, v, i, a) => (v.startsWith("--") ? [...acc, [v.slice(2), a[i + 1]]] : acc),
            [],
        ),
);
const port = Number(args.port ?? 9222);
const url = args.url;
const mode = args.mode ?? "research";
const sizes = (args.sizes ?? "1920x1080").split(",").map((s) => {
    const [wh, scale] = s.split("@");
    const [w, h] = wh.split("x").map(Number);
    return { w, h, scale: Number(scale ?? 1) };
});
if (!url) throw new Error("--url is required");

const target = await (
    await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })
).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map();
const exceptions = [];
ws.addEventListener("message", (m) => {
    const msg = JSON.parse(m.data);
    if (msg.method === "Runtime.exceptionThrown") exceptions.push(msg.params.exceptionDetails.text);
    if (msg.id && pending.has(msg.id)) {
        pending.get(msg.id)(msg);
        pending.delete(msg.id);
    }
});
const send = (method, params = {}) =>
    new Promise((r) => {
        const n = ++id;
        pending.set(n, r);
        ws.send(JSON.stringify({ id: n, method, params }));
    });
const ev = async (expr) =>
    (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }))
        .result?.result?.value;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(expr, ms = 60000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
        if (await ev(expr)) return true;
        await sleep(250);
    }
    return false;
}
const setSize = (s) =>
    send("Emulation.setDeviceMetricsOverride", {
        width: s.w,
        height: s.h,
        deviceScaleFactor: s.scale,
        mobile: false,
    });

const MEASURE = `(() => {
  const box = (sel) => { const e = document.querySelector(sel); if (!e) return null;
    const b = e.getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, width: b.width, height: b.height }; };
  const c = document.querySelector('canvas.viewer-canvas');
  return {
    viewport: { w: innerWidth, h: innerHeight },
    doc: { h: document.documentElement.scrollHeight, w: document.documentElement.scrollWidth },
    canvas: c ? { css: c.getBoundingClientRect().height, px: c.height, pxw: c.width } : null,
    center: box('section.center'), slot: box('.viewer-slot'), host: box('.viewer-host'),
    canvasBox: box('canvas.viewer-canvas'), banner: box('.research-banner'),
    left: box('aside.left'), right: box('aside.right'), summary: box('.selection-summary'),
  };
})()`;

async function samples() {
    await sleep(1000); // let one resize settle
    const out = [];
    for (let i = 0; i < 10; i++) {
        out.push(await ev(MEASURE));
        await sleep(500);
    }
    return out;
}

function check(name, list, size, results) {
    const fail = [];
    const last = list.at(-1);
    const css = list.map((m) => m.canvas?.css ?? -1);
    const px = list.map((m) => m.canvas?.px ?? -1);
    const doc = list.map((m) => m.doc.h);
    const spread = (a) => Math.max(...a) - Math.min(...a);
    if (!last.canvas) fail.push("no canvas");
    if (spread(css) > 1)
        fail.push(`canvas css height drifts (${Math.min(...css)}..${Math.max(...css)})`);
    if (spread(px) > 1)
        fail.push(`canvas pixel height drifts (${Math.min(...px)}..${Math.max(...px)})`);
    if (spread(doc) > 1)
        fail.push(`document height drifts (${Math.min(...doc)}..${Math.max(...doc)})`);
    if (last.doc.w > last.viewport.w + 1)
        fail.push(`horizontal overflow (${last.doc.w} > ${last.viewport.w})`);
    const inside = (a, b) =>
        a &&
        b &&
        a.top >= b.top - 1 &&
        a.bottom <= b.bottom + 1 &&
        a.left >= b.left - 1 &&
        a.right <= b.right + 1;
    if (last.center && last.canvasBox && !inside(last.canvasBox, last.center))
        fail.push("canvas outside the viewer area");
    if (mode === "research") {
        if (!inside(last.host, last.slot)) fail.push("viewer-host overflows viewer-slot");
        if (!inside(last.slot, last.center)) fail.push("viewer-slot overflows the viewer area");
        if (last.slot && last.slot.height < 319)
            fail.push(`viewer-slot below 320 px (${last.slot.height})`);
        if (last.banner && last.left && last.banner.bottom > last.left.top + 0.5)
            fail.push("banner covers the left panel");
        if (last.viewport.w >= 1280) {
            if (last.right && last.right.width < 359.5)
                fail.push(`right panel ${last.right.width} px < 360`);
            if (last.center && last.center.bottom > last.viewport.h + 1)
                fail.push("viewer area below the viewport");
            if (last.doc.h > last.viewport.h + 1)
                fail.push(`document taller than the viewport (${last.doc.h})`);
        }
    }
    results.push({
        size: `${size.w}x${size.h}@${size.scale}`,
        state: name,
        ok: fail.length === 0,
        canvasCss: Math.round(last.canvas?.css ?? -1),
        canvasPx: last.canvas ? `${last.canvas.pxw}x${last.canvas.px}` : null,
        doc: last.doc.h,
        slot: last.slot ? Math.round(last.slot.height) : null,
        right: last.right ? Math.round(last.right.width) : null,
        summary: last.summary ? Math.round(last.summary.height) : null,
        fail,
    });
    return last;
}

const results = [];
for (const size of sizes) {
    await setSize(size);
    await send("Page.navigate", { url });
    const loaded = await until(
        `!!document.querySelector('canvas.viewer-canvas') && [...document.querySelectorAll('input[type=range]')].some(i => !i.disabled)`,
    );
    if (!loaded) {
        results.push({
            size: `${size.w}x${size.h}`,
            state: "load",
            ok: false,
            fail: ["viewer did not load"],
        });
        continue;
    }
    const a = check("A loaded", await samples(), size, results);
    if (mode === "research") {
        // select the smallest AI region (selection summary + small-region notes)
        await ev(
            `(() => { const b = [...document.querySelectorAll('button[aria-label$="선택 및 위치로 이동"]')].at(-1); b?.click(); return !!b; })()`,
        );
        const b = check("B selected", await samples(), size, results);
        results.at(-1).summaryShrink =
            a.canvas && b.canvas ? Math.round(a.canvas.css - b.canvas.css) : null;
    } else {
        await ev(`document.querySelector('button[aria-label$="선택 및 위치로 이동"]')?.click()`);
        check("B selected", await samples(), size, results);
    }
    const view = (label) =>
        ev(
            `(() => { const b = [...document.querySelectorAll('[role=radiogroup][aria-label="단면"] [role=radio]')].find(x => x.textContent === ${JSON.stringify(label)}); b?.click(); return !!b; })()`,
        );
    await view("3면");
    check("C 3-plane", await samples(), size, results);
    await view("3D");
    check("D 3D", await samples(), size, results);
    await setSize({ ...size, w: Math.round(size.w * 0.9), h: Math.round(size.h * 0.9) });
    await sleep(500);
    await setSize(size);
    check("E after resize", await samples(), size, results);
    await sleep(10000);
    const f = check("F after 10 s", await samples(), size, results);
    if (mode === "research") {
        // G: clearing the selection removes the summary; the canvas grows back once and settles
        await ev(
            `(() => { const b = [...document.querySelectorAll('.selection-summary button')].find(x => x.textContent === '선택 해제'); b?.click(); return !!b; })()`,
        );
        const g = check("G cleared", await samples(), size, results);
        results.at(-1).summaryGrow =
            f.canvas && g.canvas ? Math.round(g.canvas.css - f.canvas.css) : null;
        if (g.summary) {
            results.at(-1).ok = false;
            results.at(-1).fail.push("selection summary still shown after clearing");
        }
    }
}
const ok = results.every((r) => r.ok) && exceptions.length === 0;
process.stdout.write(`${JSON.stringify({ url, mode, ok, exceptions, results }, null, 1)}\n`);
ws.close();
process.exit(ok ? 0 : 1);
