# Timer Audit for `SplatBinView.swift`

This note maps each CSV timer column to the exact code path in `BinCameraControllerRenderer.draw(_:_: )`, and checks whether `end_to_end_ms` / `full_frame_ms` cover the full pipeline. It also documents `display_fps`, which is sampled from the on-screen FPS counter and written per row.

## Where each timer starts/ends

- `frameStart = CACurrentMediaTime()` is taken at the very start of `draw()`.

- `drawStart = CACurrentMediaTime()` is taken **after**:
  - `guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }`
  - `guard currentFrameNum > 0 else { return }`

- `drawEnd = CACurrentMediaTime()` is taken after:
  - visible frame math
  - clear color assignment
  - `renderer.draw(...)`
  - progress update / frame wrap / `scene.remove(...)` + `scene.add(...)` swap logic

- `callbackEntry = CACurrentMediaTime()` is taken inside `commandBuffer.addCompletedHandler`.

## Column-by-column inclusion

### 1) `pre_draw_ms`

Defined as:

```swift
preDrawMs = (drawStart - frameStart) * 1000
```

Includes CPU/runtime time between `draw()` entry and `drawStart`, primarily:
- `view.currentRenderPassDescriptor` acquisition/wait
- early guard path overhead

---

### 2) `cpu_stage_ms`

Defined as:

```swift
cpuStageMs = (drawEnd - drawStart) * 1000
```

Includes CPU work inside `draw()` between `drawStart` and `drawEnd`:
- frame index computation
- `renderPassDescriptor` color clear assignment
- `renderer.draw(renderPassDescriptor:commandBuffer:scene:camera:)` CPU-side encoding work
- playback bookkeeping (`progress.progressValue` updates)
- optional scene swap (`scene.remove`, `scene.add`)

Does **not** include:
- time before `drawStart` (notably any wait in `view.currentRenderPassDescriptor`)
- CPU work after `drawEnd` (draw-call counter increment, `beginInFlightSample`, adding completion handler)
- GPU execution time

---

### 3) `gpu_queue_wait_ms`

Defined as:

```swift
gpuQueueWaitMs = (cb.gpuStartTime - drawEnd) * 1000
```

Includes wall time from end of measured CPU stage to GPU start timestamp.

Important nuance:
- because the baseline is `drawEnd` (not command-buffer commit), this bucket also absorbs any CPU/runtime time between `drawEnd` and actual commit/scheduling done outside this method.
- if `cb.gpuStartTime <= 0`, code stores `-1.0` (invalid/unavailable).

---

### 4) `gpu_and_callback_ms`

Defined as:

```swift
gpuAndCallbackMs = (callbackEntry - cb.gpuStartTime) * 1000
```

Includes:
- GPU execution interval beginning at `gpuStartTime`
- plus completion-handler dispatch latency until `callbackEntry`

So this is **not pure GPU time**. It is GPU time + callback wakeup/scheduling overhead.

---

### 5) `end_to_end_ms`

Defined as:

```swift
endToEndMs = (callbackEntry - drawStart) * 1000
```

By construction (when `gpuStartTime` is valid):

`end_to_end_ms = cpu_stage_ms + gpu_queue_wait_ms + gpu_and_callback_ms`

That is exactly why your rows appear to sum perfectly.

---

### 6) `display_fps` (non-time column)

Defined as:

```swift
displayFPS = fpsCounter.snapshotFPS()
```

Includes:
- the most recent FPS value produced by `CADisplayLink` (`FPSCounter`), sampled when the draw call's command buffer completes.

Alignment behavior:
- written on the same CSV row as `draw_call` and `frame_index`, so it is aligned to those row identifiers.
- because `FPSCounter` updates roughly once per second, many consecutive rows can share the same `display_fps` value.
- early rows can be `-1`/`0` depending on startup state before the first FPS update.

---

### 7) `full_frame_ms`

Defined as:

```swift
fullFrameMs = (callbackEntry - frameStart) * 1000
```

Includes everything from function entry of `draw()` up to completion callback entry, so it adds pre-`drawStart` time such as:
- waiting/overhead in `view.currentRenderPassDescriptor`
- the `currentFrameNum` guard check path

Relationship to existing columns:

`full_frame_ms = pre_draw_ms + end_to_end_ms`

and equivalently:

`full_frame_ms = pre_draw_ms + cpu_stage_ms + gpu_queue_wait_ms + gpu_and_callback_ms`

So `full_frame_ms` is expected to be >= `end_to_end_ms`.

Current CSV order:

`draw_call,frame_index,pre_draw_ms,cpu_stage_ms,gpu_queue_wait_ms,gpu_and_callback_ms,end_to_end_ms,display_fps,full_frame_ms`

## Is full pipeline timing correct?

### What is correct

- Internal decomposition is self-consistent.
- `end_to_end_ms` is a valid wall-clock span from measured draw start to completion callback entry.

### What is still missing from a true “entire pipeline” frame cost

Even with `full_frame_ms`, some pipeline pieces remain outside these per-row metrics:

1. Work outside this `draw()` scope
   - `update()` (`cameraController.update()`)
   - framework/Forge/MTKView scheduling around command-buffer commit.

2. Display presentation/scanout latency
   - completion handler timing is not the same as photons-on-screen at next VSync.

3. One-time startup cost per session
   - bin loading and SplatCloud build are captured only in CSV header fields (`initLoadMs`, `initBuildMs`), not per-row `end_to_end_ms`.

## Practical interpretation

- Treat `end_to_end_ms` as **post-descriptor draw-scope wall time**.
- Treat `pre_draw_ms` as the explicit descriptor/early-overhead bucket.
- Treat `full_frame_ms` as **draw()-entry wall time** (pre-draw + draw-scope timeline).
- For true whole-frame/app timing, also instrument around `update()` and presentation/VSync markers.
