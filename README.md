# AssetIDProtocolRecognizer

Visual recognizer Worker (`assetidprotocolrecognizer`) for [AssetIDProtocol](https://github.com/pjrny/AssetIDProtocol).
It is a pure-TypeScript Cloudflare Worker with the same HTTP contract as the reference OpenCV service (`recognizer/app.py` in the main repo).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | `{ ok, model, configured }` |
| POST | `/extract` | multipart `file` (JPEG/PNG) | feature object (`ts-phash-grid-v1`) |
| POST | `/match` | multipart `file`, `candidates` (JSON `[{assetId, features}]`) | `{ matched, best, alternatives, model }` |

**Auth:** every POST requires the `x-service-secret` header to equal `RECOGNIZER_SECRET`. If the secret is unset, the Worker **fails closed** with 503.
The same secret must be set on the `assetidprotocol` Worker. Its `RECOGNIZER_URL` points here.

```bash
npm install
npx wrangler secret put RECOGNIZER_SECRET
npx wrangler deploy        # Workers Builds also deploys on push to main
```

## Method and accuracy trade-off
There are no OpenCV/ORB keypoints. Each reference image is described by a 64-bit pHash (DCT) and a dHash over
**19 variants**: the full frame, plus 3x3 grids of 50% and 70% sub-regions. An HSV colour histogram is added on top.
The query's full-frame hashes are compared against every reference variant, and the best variant wins.

- **Good:** exact copies, resized or recompressed copies, screenshots, and crops that keep about 50% or more of the image.
  Local test on synthetic images: resize 0.99, 88% crop 0.83, 70% crop 0.99, centre 50% crop 0.96, all matched.
- **Weak:** perspective or oblique camera photos, rotation, heavy occlusion, small crops (under 50%), glare and prints.
  These cases need the keypoint-based (ORB) container or embeddings plus geometric verification.
- **Threshold:** `score >= 0.72 && phash >= 0.8`. Calibrate it on a real positive/negative corpus before relying on it.
- **Limits:** `MAX_BYTES` (10 MB) and `MAX_PIXELS` (40 MP). Only JPEG and PNG are decoded.
