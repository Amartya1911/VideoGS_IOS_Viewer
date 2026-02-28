# VisionOS Port Plan (BIN Pipeline Only)

## Scope
Port the current iOS viewer into a visionOS-compatible app using only the `.bin` Gaussian loading/render pipeline (`SplatBinView` + `BinCameraControllerRenderer` + `SplatCloud(binBuffers:...)`).

---

## 1) Define Target Architecture

1. Keep the existing Metal Gaussian render core:
   - `SplatCloud.swift`
   - `SplatShaders.metal` (`generateSplatsFromBin`, `splat_vertex`, `splat_fragment`, `splat_set_depths`)
2. Remove MP4/video pipeline from the product target:
   - Exclude `SplatSimpleView.swift` and OpenCV decode path from visionOS target.
3. Build a visionOS-first app shell:
   - `WindowGroup` for control UI.
   - `ImmersiveSpace` (recommended) for 3D viewing.

Deliverable: design doc confirming **BIN-only + immersive rendering path**.

---

## 2) Create visionOS App Target

1. In Xcode, add a new **visionOS App** target (or a clean visionOS project and migrate source files).
2. Set deployment target (visionOS 1.x/2.x based on team requirement).
3. Add required source files to visionOS target membership:
   - Include: `SplatCloud.swift`, `SplatBinView.swift`, `ShaderTypes.h`, `SplatShaders.metal`, required `Utils/*`.
   - Exclude: `SplatSimpleView.swift`, `OpencvTest.*`, MP4 dataset JSON/files not needed for BIN-only runtime.
4. Verify package/framework availability on visionOS:
   - `Forge`, `Satin`, `SatinCore` compatibility.

Deliverable: target builds with core sources added (even if runtime not complete yet).

---

## 3) Remove/Isolate Non-visionOS APIs

1. Audit UIKit dependencies in shared files:
   - Replace iOS-only symbols (`UIImage`, `UIColor`, `UIViewController`, presentation APIs) where needed.
2. Replace `MainViewController` flow with SwiftUI app entry for visionOS.
3. Keep `SplatBinView` as main UI entry and avoid ARKit phone-style camera APIs.
4. Ensure conditional compilation guards:
   - `#if os(visionOS)` for vision-specific paths.
   - `#if os(iOS)` for legacy code retained in repo.

Deliverable: no compile-time usage of unsupported iOS-only APIs in visionOS target.

---

## 4) Build visionOS UI Shell (BIN-only)

1. Create `App` entry with:
   - `WindowGroup` containing controls (play/pause, scrubber, FPS, dataset picker).
   - Optional button to enter `ImmersiveSpace`.
2. Host renderer view in visionOS:
   - Start with existing `ForgeView(renderer:)` if compatible.
   - If needed, create a `MetalViewRepresentable` for visionOS wrapping `MTKView`.
3. Wire app state:
   - `RendererProgress`, pause/play, selected frame, current dataset.

Deliverable: a visionOS window shows BIN renderer and controls.

---

## 5) Keep Only BIN Data Pipeline

1. Retain these runtime components:
   - `BinDatasetConfig` in `Models.swift`
   - `BinDataLoader`
   - `BinCameraControllerRenderer`
   - `SplatCloud(binBuffers:numPoints:renderDestination:)`
2. Remove/disable MP4 pipeline codepaths from visionOS target:
   - `VideoProcessor`, `loadAndExtractData`, group-info/minmax dependencies, OpenCV.
3. Validate bundle resource layout on visionOS:
   - `4K_Actor2_Dancing_qp22_bin/<frame>/{means3d,scales3d,quats3d,colors3d,opacities3d}.bin`
4. Add startup checks:
   - Confirm frame count, per-frame point count consistency, memory footprint logs.

Deliverable: successful load of all BIN frames and creation of all `SplatCloud` instances.

---

## 6) Camera/Input Strategy for visionOS

1. Replace touch-centric camera controller interactions with visionOS-friendly input:
   - Drag gesture for orbit.
   - Pinch/scale for zoom.
   - Optional reset camera action.
2. Keep deterministic default camera transform from current BIN renderer.
3. If using immersive space, define world anchoring/scaling rules.

Deliverable: stable and usable 3D navigation in visionOS.

---

## 7) Rendering & Performance Tuning

1. Validate render loop on visionOS:
   - `preferredFramesPerSecond`, drawable size/downsample strategy.
2. Preserve existing sort path:
   - GPU depth pass + CPU `sort_splats` reorder.
3. Profile memory and startup:
   - Loading all BIN frames at startup may be heavy; add optional lazy paging if needed.
4. Keep `BinPerfLogger` CSV export, adapted for visionOS filesystem behavior.

Deliverable: smooth playback target (e.g., 60 FPS where feasible) and measured startup timings.

---

## 8) Packaging & Build Settings

1. Confirm bridging/header setup for mixed Swift/ObjC++ files:
   - `BridgingHeader.h`, `splat_utils.mm`, C++ stdlib settings.
2. Ensure Metal shader compilation in visionOS target.
3. Remove unused dependencies from visionOS target to reduce binary size.
4. Verify code signing and entitlement requirements for visionOS deployment.

Deliverable: clean archive/build for visionOS device/simulator.

---

## 9) Validation Matrix

1. Functional tests:
   - Launch app, load BIN sequence, play/pause, scrub timeline, loop playback.
2. Data integrity tests:
   - Spot-check frame-to-frame continuity and no NaNs/invalid splats.
3. Stability tests:
   - Long playback session, repeated enter/exit view, memory growth checks.
4. Device tests:
   - visionOS simulator + real device.

Deliverable: test checklist signed off.

---

## 10) Migration Cleanup

1. Mark iOS MP4 path as legacy or move to separate target.
2. Keep shared renderer core platform-agnostic where possible.
3. Document final architecture and dataset onboarding steps for BIN datasets.

Deliverable: maintainable BIN-only visionOS codebase.

---

## Suggested Execution Order (Practical)

1. Create visionOS target + compile baseline.
2. Run BIN renderer in windowed mode.
3. Fix platform API mismatches.
4. Add immersive space + gestures.
5. Tune performance/memory.
6. Final cleanup and testing.

---

## Risks & Mitigations

- **Framework compatibility risk (`Forge`/`Satin` on visionOS):**
  - Mitigation: early spike test with minimal `MTKView` renderer.
- **Memory pressure from eager loading all frames:**
  - Mitigation: fallback to chunked/lazy frame loading.
- **Input UX mismatch from iOS controls:**
  - Mitigation: redesign camera controls around visionOS gestures/spatial UX.

---

## Definition of Done

- App runs natively on visionOS.
- Only BIN pipeline is used in production target.
- Sequence playback, scrubbing, and pause/resume work.
- Rendering is stable with acceptable FPS and memory.
- Build/test checklist completed on simulator and device.
