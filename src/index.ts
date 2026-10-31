/**
 * assetid-recognizer: a pure-TypeScript Cloudflare Worker with the same HTTP contract as recognizer/app.py.
 *   GET  /health                               -> { ok, model }
 *   POST /extract  (multipart: file)            -> feature object
 *   POST /match    (multipart: file, candidates) -> { matched, best, alternatives, model }
 * Auth: header x-service-secret must equal RECOGNIZER_SECRET. Fails closed (503) if the secret is unset.
 *
 * Accuracy trade-off vs the OpenCV/ORB container: there are no local keypoints. Reference images are
 * described by pHash/dHash over 1 full + 18 sub-region variants plus an HSV histogram. This is robust to
 * resizing, recompression, screenshots and crops that keep roughly 50% or more of the image near those
 * regions. It is weak on strong perspective, rotation, heavy occlusion and photos of prints.
 */
import jpeg from "jpeg-js";
import UPNG from "upng-js";

export interface Env { RECOGNIZER_SECRET?: string; MAX_BYTES?: string; MAX_PIXELS?: string }
const MODEL = "ts-phash-grid-v1";

class HttpError extends Error { constructor(public status: number, msg: string) { super(msg); } }
const json = (d: unknown, s = 200) => new Response(JSON.stringify(d), { status: s, headers: { "content-type": "application/json" } });

function timingSafeEqual(a: string, b: string) {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i % (x.length || 1)] ?? 0) ^ (y[i % (y.length || 1)] ?? 0);
  return diff === 0;
}
function auth(req: Request, env: Env) {
  if (!env.RECOGNIZER_SECRET) throw new HttpError(503, "RECOGNIZER_SECRET is not configured");
  if (!timingSafeEqual(req.headers.get("x-service-secret") || "", env.RECOGNIZER_SECRET)) throw new HttpError(401, "invalid service secret");
}

type Img = { w: number; h: number; rgba: Uint8Array };
function decode(buf: Uint8Array, env: Env): Img {
  const maxPx = Number(env.MAX_PIXELS || 40_000_000);
  if (buf[0] === 0xff && buf[1] === 0xd8) {
    const r = jpeg.decode(buf, { useTArray: true, maxResolutionInMP: maxPx / 1e6, maxMemoryUsageInMB: 512, formatAsRGBA: true });
    return { w: r.width, h: r.height, rgba: r.data as Uint8Array };
  }
  if (buf[0] === 0x89 && buf[1] === 0x50) {
    const p = UPNG.decode(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
    if (p.width * p.height > maxPx) throw new HttpError(413, "image too large");
    return { w: p.width, h: p.height, rgba: new Uint8Array(UPNG.toRGBA8(p)[0]) };
  }
  throw new HttpError(400, "invalid image (JPEG or PNG required)");
}

/** Area-average a rectangle of the image into a tw x th grayscale grid. */
function grayGrid(im: Img, rx: number, ry: number, rw: number, rh: number, tw: number, th: number) {
  const out = new Float32Array(tw * th);
  for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
    const x0 = Math.floor(rx + (tx * rw) / tw), x1 = Math.max(x0 + 1, Math.floor(rx + ((tx + 1) * rw) / tw));
    const y0 = Math.floor(ry + (ty * rh) / th), y1 = Math.max(y0 + 1, Math.floor(ry + ((ty + 1) * rh) / th));
    const sx = Math.max(1, Math.floor((x1 - x0) / 8)), sy = Math.max(1, Math.floor((y1 - y0) / 8));
    let s = 0, n = 0;
    for (let y = y0; y < y1 && y < im.h; y += sy) for (let x = x0; x < x1 && x < im.w; x += sx) {
      const i = (y * im.w + x) * 4; s += 0.299 * im.rgba[i] + 0.587 * im.rgba[i + 1] + 0.114 * im.rgba[i + 2]; n++;
    }
    out[ty * tw + tx] = n ? s / n : 0;
  }
  return out;
}
const bitsToHex = (bits: boolean[]) => { let h = ""; for (let i = 0; i < bits.length; i += 4) h += ((+bits[i] << 3) | (+bits[i + 1] << 2) | (+bits[i + 2] << 1) | +bits[i + 3]).toString(16); return h; };
function dhash(im: Img, r: number[]) { const g = grayGrid(im, r[0], r[1], r[2], r[3], 9, 8); const b: boolean[] = []; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) b.push(g[y * 9 + x + 1] > g[y * 9 + x]); return bitsToHex(b); }
const COS = (() => { const c = new Float32Array(32 * 8); for (let u = 0; u < 8; u++) for (let x = 0; x < 32; x++) c[u * 32 + x] = Math.cos(((2 * x + 1) * u * Math.PI) / 64); return c; })();
function phash(im: Img, r: number[]) {
  const g = grayGrid(im, r[0], r[1], r[2], r[3], 32, 32); const d: number[] = [];
  for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) { let s = 0; for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) s += g[y * 32 + x] * COS[u * 32 + x] * COS[v * 32 + y]; d.push(s); }
  const med = [...d.slice(1)].sort((a, b) => a - b)[31];
  return bitsToHex(d.map((x) => x > med));
}
function hsvHist(im: Img) {
  const h = new Float32Array(16 * 8); const step = Math.max(1, Math.floor(Math.sqrt((im.w * im.h) / 40000)));
  for (let y = 0; y < im.h; y += step) for (let x = 0; x < im.w; x += step) {
    const i = (y * im.w + x) * 4, r = im.rgba[i] / 255, g = im.rgba[i + 1] / 255, b = im.rgba[i + 2] / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), c = mx - mn;
    let hue = 0; if (c) hue = mx === r ? ((g - b) / c + 6) % 6 : mx === g ? (b - r) / c + 2 : (r - g) / c + 4;
    const s = mx ? c / mx : 0; h[Math.min(15, Math.floor((hue / 6) * 16)) * 8 + Math.min(7, Math.floor(s * 8))]++;
  }
  const norm = Math.hypot(...h) || 1; return Array.from(h, (v) => Math.round((v / norm) * 1e6) / 1e6);
}
function regions(w: number, h: number) {
  const rs: number[][] = [[0, 0, w, h]];
  for (const f of [0.5, 0.7]) for (const oy of [0, 0.5, 1]) for (const ox of [0, 0.5, 1]) rs.push([ox * (1 - f) * w, oy * (1 - f) * h, f * w, f * h]);
  return rs;
}
type Features = { model: string; shape: [number, number]; hist: number[]; variants: { p: string; d: string }[] };
function features(im: Img, withVariants: boolean): Features {
  const rs = withVariants ? regions(im.w, im.h) : [[0, 0, im.w, im.h]];
  return { model: MODEL, shape: [im.w, im.h], hist: hsvHist(im), variants: rs.map((r) => ({ p: phash(im, r), d: dhash(im, r) })) };
}
function ham(a: string, b: string) { if (!a || !b || a.length !== b.length) return 64; let n = 0; for (let i = 0; i < a.length; i++) { let x = parseInt(a[i], 16) ^ parseInt(b[i], 16); while (x) { n += x & 1; x >>= 1; } } return n; }
function corr(a: number[], b: number[]) {
  if (a.length !== b.length || !a.length) return 0; const ma = a.reduce((s, x) => s + x, 0) / a.length, mb = b.reduce((s, x) => s + x, 0) / b.length;
  let n = 0, da = 0, db = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return da && db ? Math.max(0, n / Math.sqrt(da * db)) : 0;
}
function score(q: Features, c: Features) {
  const qv = q.variants[0]; let best = { hp: 0, hd: 0, v: 0 };
  c.variants.forEach((cv, i) => { const hp = 1 - ham(qv.p, cv.p) / 64, hd = 1 - ham(qv.d, cv.d) / 64; if (hp + 0.5 * hd > best.hp + 0.5 * best.hd) best = { hp, hd, v: i }; });
  const hs = corr(q.hist, c.hist); const total = 0.55 * best.hp + 0.25 * best.hd + 0.2 * hs;
  return { total: Math.round(total * 1e4) / 1e4, parts: { phash: +best.hp.toFixed(3), dhash: +best.hd.toFixed(3), color: +hs.toFixed(3), variant: best.v } };
}

async function readFile(req: Request, env: Env) {
  const max = Number(env.MAX_BYTES || 10_000_000);
  if (Number(req.headers.get("content-length") || 0) > max + 2_000_000) throw new HttpError(413, "file too large");
  const form = await req.formData(); const f = form.get("file") as unknown;
  if (!(f instanceof File)) throw new HttpError(400, "file is required");
  if (f.size > max) throw new HttpError(413, "file too large");
  return { form, img: decode(new Uint8Array(await f.arrayBuffer()), env) };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const u = new URL(req.url);
    try {
      if (req.method === "GET" && u.pathname === "/health") return json({ ok: true, model: MODEL, configured: Boolean(env.RECOGNIZER_SECRET) });
      if (req.method === "POST" && u.pathname === "/extract") { auth(req, env); const { img } = await readFile(req, env); return json(features(img, true)); }
      if (req.method === "POST" && u.pathname === "/match") {
        auth(req, env); const { form, img } = await readFile(req, env);
        let cs: { assetId: string; features: Features }[];
        try { cs = JSON.parse(String(form.get("candidates") || "[]")); if (!Array.isArray(cs)) throw 0; } catch { throw new HttpError(400, "invalid candidates JSON"); }
        const q = features(img, false);
        const out = cs.filter((c) => c?.features?.model === MODEL && Array.isArray(c.features.variants))
          .map((c) => { const s = score(q, c.features); return { assetId: c.assetId, score: s.total, goodKeypoints: 0, parts: s.parts }; })
          .sort((a, b) => b.score - a.score);
        const best = out[0] ?? null;
        // Conservative default; calibrate against a positive/negative corpus (see INTEGRATION.md section 5).
        const matched = Boolean(best && best.score >= 0.72 && best.parts.phash >= 0.8);
        return json({ matched, best, alternatives: out.slice(0, 5), model: MODEL });
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      console.error(e); return json({ error: "internal error" }, 500);
    }
  },
};
