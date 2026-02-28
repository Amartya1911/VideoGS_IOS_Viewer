# VisionOS BIN Port — Actionable TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

## Core migration tasks

1. [x] Align BIN dataset metadata with packaged data
   - Updated BIN frame count to 200 in `BinDatasetConfig.actor2DancingBin`.

2. [x] Add BIN startup integrity checks
   - Added logs for expected vs discovered frame count.
   - Added missing-frame index warnings.
   - Added point-count consistency checks.
   - Added total in-memory raw BIN size logging.

3. [x] Remove unnecessary UIKit dependency from BIN renderer path
   - Removed `UIKit` import from BIN renderer file.
   - Removed `UIColor`-specific setup line in BIN renderer setup.

4. [x] Add visionOS SwiftUI app entry shell
   - Added a visionOS app entry (`WindowGroup`) that launches BIN viewer.
   - Added simple dataset picker scaffold (currently one BIN dataset).

5. [x] Gate legacy iOS entry/navigation path
   - Added platform guards around iOS app entry and UIKit navigation flow.
   - Next: confirm final source membership in Xcode target settings.

6. [x] Isolate MP4-only codepaths from BIN visionOS target
   - Added platform availability guards around UIImage/MP4-specific pieces in `SplatCloud`.
   - Next: confirm `SplatSimpleView.swift` and OpenCV bridge are excluded from visionOS target membership.

7. [x] Build a native visionOS control shell
   - Added play/pause state at shell level.
   - Added immersive-space enter/exit controls.
   - Renderer is no longer hosted in a window; it is hosted in ImmersiveSpace.

8. [~] Verify resource packaging in visionOS target
   - Ensure `4K_Actor2_Dancing_qp22_bin/**` is copied into bundle for visionOS target.

9. [~] Performance/stability pass
   - Measure startup time + playback FPS.
   - Evaluate memory pressure from eager load; add lazy paging if needed.

10. [ ] Validation checklist run
   - Simulator run
   - Device run
   - Long-session memory and stability check

11. [x] Lock visionOS minimum target to 2.0
   - Added `XROS_DEPLOYMENT_TARGET = 2.0` in project/target build settings.

12. [x] Separate iOS MTKView and visionOS compositor sources
   - Added platform-filtered source membership (`ios` vs `xros`) in project file.
   - Added visionOS-only compositor source files.
