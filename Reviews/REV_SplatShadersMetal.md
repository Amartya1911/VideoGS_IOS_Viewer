# Code Review: SplatShaders.metal

## Overview
`SplatShaders.metal` contains the GPU-side logic for the Gaussian Splatting pipeline. This file tells the graphics card how to:
1.  **Dequantize** video data back into 3D points.
2.  **Calculate depths** for sorting.
3.  **Project** 3D Gaussians into 2D screen space (Vertex Shader).
4.  **Draw** the actual fuzzy blobs pixel-by-pixel (Fragment Shader).

---

## 1. Struct Definitions (The Data Contracts)
These structures define exactly how data is organized when moving between the CPU and the GPU. They must match the C structs defined in `ShaderTypes.h`.

-   **`VertexIn`** (Lines 102-104): Represents the raw geometry of a splat instance. It is just a simple 2D point because each splat starts as a simple quad (two triangles).
-   **`VertexOut`** (Lines 106-118): The data passed from the Vertex Shader to the Fragment Shader.
    -   `center_screen_pos`: Where the center of the blob is on the screen.
    -   `conic`: A mathematically computed 2x2 inverse covariance matrix that describes the shape and rotation of the ellipse in 2D.
    -   `color`: The base color of the splat.
-   **`CameraParameters`** (Lines 120-125): Technical camera details (focal length, field of view) needed for the projection math.

---

## 2. Math Helper Functions (The 3D to 2D Magic)
These functions implement the core Gaussian Splatting mathematics (EWA Splatting).

-   **`computeCov3D`** (Lines 131-157):
    -   Takes a splat's **scale** (how wide/tall/deep) and **rotation** (quaternion).
    -   Combines them into a **3D Covariance Matrix**. This matrix mathematically describes the 3D ellipsoid shape.
-   **`computeCov2D`** (Lines 169-216):
    -   This is the heavy lifter. It takes the 3D ellipsoid (from `computeCov3D`) and the camera view.
    -   It projects the 3D shape onto the 2D camera sensor plane.
    -   **Result**: A 2D covariance matrix that describes the flat ellipse we see on screen.
    -   *Tech Note*: It includes a "low-pass filter" (lines 212-213) to prevent aliasing (jagged edges) when splats are very small.

---

## 3. The Vertex Shader: `splat_vertex`
**Location**: Lines 220-310

This function runs once for every corner of every splat quad.

1.  **Input**: Receives the quad geometry (`vertices`) and the specific Gaussian instance data (`instances`).
2.  **Projection**:
    -   Calculates where the center of the 3D Gaussian is on the screen (`center_clip_pos`).
3.  **Shape Calculation**:
    -   Calls `computeCov3D` and `computeCov2D` to figure out the 2D ellipse shape.
    -   Inverts the covariance to get the `conic` (needed for easy pixel math later).
4.  **Bounding Box Expansion**:
    -   Calculates the "radius" (Line 287) of the splat on screen.
    -   Stretches the simple input quad (`quad_pos`) to be large enough to cover the entire fuzzy blob.
5.  **Color Setup**: Prepares the color but notably adds a "drag color" effect (Line 304) that highlights splats if they are being manipulated by the user.

---

## 4. The Fragment Shader: `splat_fragment`
**Location**: Lines 338-362

This function runs for every single pixel covered by the expanded quad.

1.  **Distance Check**:
    -   Calculates `d` (Line 346), the distance from the current pixel to the center of the Gaussian.
2.  **Gaussian Evaluation**:
    -   Uses `CalcPowerFromConic` to compute the Gaussian falloff value ($e^{-x}$).
    -   Pixels at the center get full alpha (1.0).
    -   Pixels at the edge get near-zero alpha.
3.  **Alpha Modulation**:
    -   Multiplies the splat's base alpha by this calculated falloff.
    -   This creates the smooth, fuzzy appearance of the blob.
4.  **Hardware Discard**:
    -   If the pixel is too transparent (`alpha < 1/255`), it discards it entirely to save detailed processing.

---

## 5. Compute Kernels (Data Processing)
These run distinct tasks separate from the drawing pipeline.

### A. `splat_set_depths`
**Location**: Lines 370-390
-   **Purpose**: Prepares the splats for sorting.
-   **Logic**:
    -   Takes the 3D center of each splat.
    -   Projects it along the camera's Z-axis to find its "depth".
    -   **Packing**: It combines the depth (32 bits) and the splat's index ID (32 bits) into a single 64-bit integer (`packed`).
    -   This packed list is sent to the CPU to be sorted, ensuring fast depth comparisons.

### B. `generateSplats` (The Dequantizer)
**Location**: Lines 465-529
-   **Purpose**: The "translator" that turns video pixels back into 3D data.
-   **Inputs**: 17 different textures (MP4 video frames) representing x, y, z, rotation, scale, color, etc.
-   **Logic**:
    -   **Reconstruction**: Reads high/low bytes for X, Y, Z coordinates (Lines 482-484) to rebuild precise 16-bit positions.
    -   **Dequantization**: Uses the `minmax` buffer to verify the value range.
        -   *Formula*: `real_value = pixel_val * range + min_val`.
    -   **Color Conversion**: Converts raw spherical harmonic coefficients into RGB colors (Lines 519-521).
    -   **Writing**: Saves the final `Splat` struct into the GPU buffer, ready for the rendering pipeline.
