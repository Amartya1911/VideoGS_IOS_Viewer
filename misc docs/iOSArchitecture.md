
# VideoGS_IOS_Viewer — Full Architecture

## App Entry & Navigation

- **AppDelegate.swift** — Standard UIKit `@main` entry point, no custom logic.
- **MainViewController.swift** — Presents a `SplatChoiceView` (SwiftUI grid) with 4 selectable datasets:

| Tap | Dataset | Pipeline | View |
|-----|--------------------------|----------|---------------------------------------------|
| 1 – "boxing" | ykx_boxing_long | MP4 | SplatSimpleView(index: 1) |
| 2 – "dancing low res" | 4K_Actor2_sh0_res4 | MP4 | SplatSimpleView(index: 2) |
| 3 – "dancing medium res" | 4K_Actor2_sh0_res2 | MP4 | SplatSimpleView(index: 3) |
| 4 – "actor2 (bin)" | 4K_Actor2_qp22_bin | BIN | SplatBinView(binConfig: .actor2DancingBin) |

---

## Shared Core: Splat Struct & SplatCloud

- **ShaderTypes.h** defines the shared GPU/CPU data types:
  - `Splat` — 4×float4: center, color, scale, quat (64 bytes per Gaussian)
  - `Uniforms` — MVP matrices, camera params, viewport, focal lengths, FoV tangents, drag alpha, time
- **SplatCloud.swift** (`SplatCloud : Object, Renderable`) is the core render object. It holds:
  - `splats` / `temp_splats` — MetalBuffer<Splat> (double-buffered for sorting)
  - `splat_indices` — MetalBuffer<Int64> (packed depth + index for sort)
  - Two initializers — one per pipeline (detailed below)
  - `render()` — Calls `sortSplats()` (every 4th frame) then draws instanced quads
  - `sortSplats()` → GPU compute `splat_set_depths` kernel packs depth into upper 32 bits of int64, then C++ `sort_splats()` does `std::sort` on the packed array and reorders splats by depth

---

## Pipeline 1: MP4 (Video-Compressed Gaussians)

**Data on disk:** Per-group folders like `ykx_boxing_long_qp15_380/group0/` containing **17 .mp4 files** (`0.mp4–5.mp4, 9.mp4–19.mp4`). Each video encodes one quantized Gaussian attribute channel as grayscale frames. Each frame = one timestep of the dynamic scene.

**Configuration:** `DatasetConfig` in `Models.swift` maps `dataIndex` → `minmaxPath`, `videoFolder`, `groupInfoPath`, `stopNum`, `totalFrames`, `initPos`.

**Flow (in SplatSimpleView.swift):**

1. `SplatSimpleView` (SwiftUI) creates `CameraControllerRenderer` with a `dataIndex`.
2. `setupMtkView()` in `CameraControllerRenderer`:
	- Parses `group_info_*.json` → list of `(groupIndex, startFrame, endFrame)` tuples
	- Loads `viewer_min_max_*.json` → per-frame min/max arrays for dequantization
	- **First group loaded synchronously** on the main thread:
		- `VideoProcessor.processVideos(groupIndex:dataIndex:)` calls `OpencvTest.processVideo()` (Objective-C++ via `OpencvTest.mm`) which uses OpenCV `VideoCapture` to decode each `.mp4` from the app bundle into an array of grayscale `UIImage`s
		- For each frame index, constructs `SplatCloud` via the MP4 init concurrently using `DispatchQueue.concurrentPerform`
	- **Remaining groups queued** via `OperationQueue` (serial, max 1 concurrent) for background loading with backpressure (`stopNum` controls pause/resume)
3. `SplatCloud` MP4 init (`SplatCloud.swift`):
	- Takes `groupFrame: [[UIImage]]` (17 video channels × N frames) + `minmax` + `frameIndex`
	- Converts each video channel's frame image into a Metal texture via `createDFTexture()`
	- Dispatches GPU compute kernel `generateSplats` (in `SplatShaders.metal`) which:
		- Reads 17 textures: `x0,x1` (16-bit position X), `y0,y1` (Y), `z0,z1` (Z) → assembled from two 8-bit channels into 16-bit, dequantized via `minmax` ÷ 65535
		- `fdc0,fdc1,fdc2` (SH DC color), `opacity`, `scale0–2`, `rot0–3` → dequantized via `minmax` ÷ 255 (single 8-bit channel)
		- Applies SH_C0 color conversion, writes `Splat` struct
	- Calls `copySplats()` (blit copy to `temp_splats`)
4. **Playback** in `draw()`:
	- `progress.progressValue` increments; every `stepInd=2` draw calls, swaps the visible `SplatCloud` in the Satin scene graph (removes old, adds new)
	- Old `SplatCloud`s are `nil`-ed to free memory; `splatFinishNum` tracks buffered frames
	- On loop end, calls `selectFrame(0)` to restart

---

## Pipeline 2: BIN (Raw Float32 Gaussians)

**Data on disk:** `4K_Actor2_Dancing_qp22_bin/` contains numbered subdirectories (`0/`, `1/`, …`199/`), each with 5 raw float32 `.bin` files:

- `means3d.bin` (N×3), `scales3d.bin` (N×3), `quats3d.bin` (N×4), `colors3d.bin` (N×3), `opacities3d.bin` (N×1)

**Configuration:** `BinDatasetConfig` in `Models.swift` — `binFolder`, `totalFrames`, `initPos`.

**Flow (in SplatBinView.swift):**

1. `SplatBinView` (SwiftUI) creates `BinCameraControllerRenderer` with a `BinDatasetConfig`.
2. `setupMtkView()`:
	- `BinDataLoader.loadAllFrames()` reads all `.bin` files into `[BinFrameData]` (pure `Data` blobs in memory)
	- For each frame, creates Metal buffers from the raw `Data` and calls the `SplatCloud` Bin init
3. `SplatCloud` Bin init (`SplatCloud.swift`):
	- Takes `binBuffers` (5 `MTLBuffer`s) + `numPoints`
	- Dispatches GPU compute kernel `generateSplatsFromBin` which:
		- Reads raw float32 means → `center`
		- Applies `sigmoid()` to raw logit opacities
		- Applies `exp()` to log-space scales
		- Normalizes quaternions
		- Applies SH_C0 to color DC coefficients
		- Writes `Splat` struct
	- Calls `copySplats()`
	- **All frames loaded at startup** (no streaming/backpressure needed — data is smaller)
4. **Playback** in `draw()`:
	- Same `stepInd=2` cadence; swaps `SplatCloud` in scene graph with circular wrapping
	- Includes `BinPerfLogger` — records per-draw timing (CPU stage, GPU queue wait, GPU+callback, end-to-end) and exports CSV to Documents on dismiss/deinit

---

## Key Differences Between the Two Pipelines

| Aspect | MP4 Pipeline | BIN Pipeline |
|--------|-------------|--------------|
| Storage format | 17 H.264-compressed .mp4 video files per group (quantized 8/16-bit grayscale) | 5 raw float32 .bin files per frame |
| Decode | OpenCV VideoCapture → grayscale UIImage → Metal textures | Data(contentsOf:) → MTLBuffer directly |
| Dequantization | GPU kernel reads textures, reconstructs 16-bit positions from two 8-bit channels, applies per-frame minmax JSON | GPU kernel applies sigmoid, exp, normalize to raw floats — no external min/max |
| GPU kernel | generateSplats (17 textures + minmax buffer) | generateSplatsFromBin (5 float buffers) |
| Loading strategy | Streaming: first group sync, rest via OperationQueue with backpressure (stopNum) | Eager: all frames loaded at startup |
| Memory management | Old splat clouds nil-ed after playback passes them | All frames kept in memory |
| Performance logging | None | Full BinPerfLogger with CSV export |

---

## Rendering (shared)

Both pipelines produce the same `SplatCloud` objects. Rendering uses the Satin/Forge framework:

- **Vertex shader** `splat_vertex` — projects each Gaussian center, computes 2D covariance (3D→2D via EWA), sizes a screen-space quad
- **Fragment shader** `splat_fragment` — evaluates Gaussian falloff from conic, applies alpha blending
- **Sorting** — GPU compute `splat_set_depths` packs camera-space depth + index into int64, C++ `std::sort` reorders back-to-front, `memcpy` into sorted order
- **Camera** — `PerspectiveCameraController` (touch gestures for orbit/pan/zoom) or AR-tracked via `ARPerspectiveCamera` in `ARSplatView.swift`