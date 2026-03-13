# visionOS Gaussian Splatting Port - Engineering Notes

This document tracks the technical fixes implemented during the port of the Gaussian Splatting BIN viewer from iOS to visionOS.

---

## 🚀 Key Fixes for Immersive Rendering

### 1. Correcting the Projection Matrix (Black Screen Fix #1)
Apple's `view.tangents` in `CompositorServices` are provided as **positive magnitudes** (e.g., 1.0, 1.0, 0.75, 0.75).
- **Issue:** The original code treated them as signed, resulting in `width = right - left = 1.0 - 1.0 = 0`, causing a division by zero.
- **Solution:** Manually negate the left and bottom tangents: `left = -tangents.x`, `bottom = -tangents.z`.
- **Result:** Correct frustum calculation with proper focal length (approx. 1775).

### 2. Stereo Rendering & Layered Layout (Black Screen Fix #2)
On physical Vision Pro hardware, the compositor uses a `.layered` texture layout (2D Array).
- **Issue:** The vertex shader lacks `[[render_target_array_index]]`. Rendering once with `renderTargetArrayLength = 2` only filled the left eye (layer 0). The compositor silently drops frames if layer 1 is empty.
- **Solution:** 
    - Create a `MTLTexture` view for each slice (eye) of the array.
    - Execute a **separate render pass per eye** within the same command buffer.
    - This ensures both eyes are drawn correctly without changing the core shader.

### 3. Solving Double Vision (Per-Eye Parallax)
- **Issue:** Using a single `viewMatrix` based only on the `deviceAnchor` caused "ghosting" or double images when looking closely at the actor.
- **Solution:** Incorporate the `view.transform` (the offset from the bridge of the nose to each eye).
    - `eyeWorldTransform = anchorTransform * view.transform`
    - `viewMatrix = inverse(eyeWorldTransform)`
- **Result:** Proper stereo parallax and IPD (Inter-Pupillary Distance) support.

---

## 🤖 ARKit & Stability on Device

### 4. ARKit Initialization Race Condition
- **Issue:** Attempting to query the `deviceAnchor` before the ARKit session is fully active caused crash-like behavior and black screens on device.
- **Solution:**
    - **Priority Startup:** Moved `arSession.run()` before the heavy dataset loading.
    - **Spin-Wait:** Implemented a 5-second max wait loop for `worldTracking.state == .running` before starting the render loop.
    - **Anchor Caching:** Added `lastDeviceAnchor` to provide the last valid pose if a specific frame query returns nil, preventing dropped frames.

---

## 📐 Model Transformation & Positioning

### 5. Coordinate Space & Orientation
- **Dataset Assumption:** The BIN data is Y-up.
- **Initial Fix:** Removed the iOS -90° X rotation which made the model appear to be lying down.
- **Final Config:**
    - **Rotation:** 180° Y-axis rotation to face the user.
    - **Scale:** 0.35 (normalized for visionOS space).
    - **Position:** `SIMD3<Float>(0, 1.2, -1.5)`.
        - `Y = 1.2` places the model at chest/face height (ARKit origin is the floor).
        - `Z = -1.5` places the model at a comfortable viewing distance in front of the user.

---

## 🖥️ Simulator vs. Physical Device
- **Double Vision:** Does not appear in the Simulator because it renders a single monoscopic view to a 2D window, ignoring IPD disparity.
- **Anchor Requirement:** The Simulator is more forgiving of `nil` anchors; the physical device requires a non-nil anchor for every single drawable presentation.

---

## 🛠️ Performance & Maintenance
- **Depth Format:** Explicitly set `configuration.depthFormat = .depth32Float` in the `LayerRenderer` configuration to match the Metal pipeline.
- **Animation Sync:** Frames advance once per composite frame (not per eye) to keep the animation speed consistent across stereo passes.