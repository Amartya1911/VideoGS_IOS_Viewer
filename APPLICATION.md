# Simplified Gaussian Splatting iOS Application

## Direct Splat Rendering from Pre-Computed Binary Data

---

## Overview

This document describes how to build a **simplified iOS Gaussian Splatting renderer** that takes **pre-dequantized Splat structs** as input. Since the input data is already in the final `Splat` format (center, color, scale, quaternion), we bypass:

- ❌ Video decoding (OpenCV)
- ❌ Texture creation
- ❌ GPU compute dequantization (`generateSplats` kernel)

We only need to implement:

- ✅ **Depth Sorting** (GPU depth computation + CPU sorting)
- ✅ **GPU Rendering** (vertex shader, fragment shader, alpha blending)

---

## Simplified Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: .bin FILE                            │
├─────────────────────────────────────────────────────────────────┤
│  Array of Splat structs (pre-dequantized float values)          │
│  Each Splat: 64 bytes (center, color, scale, quat)              │
│  File size = numSplats × 64 bytes                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LOAD INTO METAL BUFFER                          │
├─────────────────────────────────────────────────────────────────┤
│  Direct memory copy from .bin → MTLBuffer                       │
│  No dequantization needed                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DEPTH SORTING (Hybrid)                        │
├─────────────────────────────────────────────────────────────────┤
│  GPU: splat_set_depths() → Compute camera-space depth           │
│  CPU: std::sort() in splat_utils.mm → Sort by packed depth|idx  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GPU RENDERING (Metal)                        │
├─────────────────────────────────────────────────────────────────┤
│  Vertex Shader: splat_vertex() → 2D screen-space ellipse        │
│  Fragment Shader: splat_fragment() → EWA alpha blending         │
│  Instanced drawing: 1 quad × numSplats instances                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Data Structures](#1-data-structures)
2. [Binary File Loading](#2-binary-file-loading)
3. [Metal Buffer Management](#3-metal-buffer-management)
4. [Depth Sorting Pipeline](#4-depth-sorting-pipeline)
5. [Rendering Pipeline Setup](#5-rendering-pipeline-setup)
6. [Vertex Shader](#6-vertex-shader)
7. [Fragment Shader](#7-fragment-shader)
8. [Alpha Blending Configuration](#8-alpha-blending-configuration)
9. [Uniforms Management](#9-uniforms-management)
10. [Render Loop](#10-render-loop)
11. [Complete Implementation Checklist](#11-complete-implementation-checklist)

---

## 1. Data Structures

### 1.1 Splat Structure (INPUT FORMAT)

This is your input data format. Each splat is exactly **64 bytes**.

**Source:** [ShaderTypes.h](MetalSplat/ShaderTypes.h), lines 47-59

```c
// ShaderTypes.h
typedef struct
{
    simd_float4 center; // xyz position + padding (w=1.0)
    simd_float4 color;  // rgba (RGB color + opacity)
    simd_float4 scale;  // xyz scale + padding (w=1.0)
    simd_float4 quat;   // xyzw quaternion (rotation)
} Splat;
```

**Memory Layout:**
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 16 bytes | center | float4: (x, y, z, 1.0) |
| 16 | 16 bytes | color | float4: (r, g, b, opacity) |
| 32 | 16 bytes | scale | float4: (sx, sy, sz, 1.0) |
| 48 | 16 bytes | quat | float4: (qx, qy, qz, qw) |
| **Total** | **64 bytes** | | |

### 1.2 Uniforms Structure

Camera and viewport parameters passed to shaders every frame.

**Source:** [ShaderTypes.h](MetalSplat/ShaderTypes.h), lines 24-44

```c
// ShaderTypes.h
typedef struct
{
    matrix_float4x4 projection_matrix;
    matrix_float4x4 model_matrix;
    matrix_float4x4 model_view_matrix;
    matrix_float4x4 inv_model_view_matrix;
    
    simd_float4 camera_pos;
    simd_float4 camera_pos_orig;
    
    float viewport_width;
    float viewport_height;
        
    float focal_x;
    float focal_y;
    float tan_fovx;
    float tan_fovy;
    
    float drag_alpha;
    float time;
} Uniforms;
```

---

## 2. Binary File Loading

### 2.1 Loading Pre-Computed Splats from .bin File

**New code** (not in original project - you'll implement this):

```swift
// SplatLoader.swift (NEW FILE)
import Foundation
import Metal

class SplatLoader {
    
    /// Load splats directly from a binary file into a Metal buffer
    /// - Parameters:
    ///   - path: Path to .bin file containing array of Splat structs
    ///   - device: Metal device
    /// - Returns: Tuple of (MTLBuffer, splatCount)
    static func loadSplatsFromBinary(path: String, device: MTLDevice) throws -> (MTLBuffer, Int) {
        
        // Read binary file
        guard let data = FileManager.default.contents(atPath: path) else {
            throw NSError(domain: "SplatLoader", code: 1, 
                          userInfo: [NSLocalizedDescriptionKey: "Failed to read file at \(path)"])
        }
        
        // Calculate number of splats (each Splat is 64 bytes)
        let splatSize = MemoryLayout<Splat>.stride  // Should be 64
        let numSplats = data.count / splatSize
        
        guard numSplats > 0 else {
            throw NSError(domain: "SplatLoader", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "File contains no splats"])
        }
        
        // Create Metal buffer directly from data
        guard let buffer = data.withUnsafeBytes({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, 
                              length: data.count, 
                              options: .storageModeShared)
        }) else {
            throw NSError(domain: "SplatLoader", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal buffer"])
        }
        
        print("Loaded \(numSplats) splats from \(path)")
        return (buffer, numSplats)
    }
    
    /// Load splats from bundle resource
    static func loadSplatsFromBundle(name: String, device: MTLDevice) throws -> (MTLBuffer, Int) {
        guard let path = Bundle.main.path(forResource: name, ofType: "bin") else {
            throw NSError(domain: "SplatLoader", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Resource \(name).bin not found in bundle"])
        }
        return try loadSplatsFromBinary(path: path, device: device)
    }
}
```

---

## 3. Metal Buffer Management

### 3.1 MetalBuffer Wrapper Class

**Source:** [MetalBuffer.swift](MetalSplat/Utils/MetalBuffer.swift), lines 1-50 (approximate)

```swift
// MetalBuffer.swift
import Metal

class MetalBuffer<T> {
    let buffer: MTLBuffer
    let count: Int
    let index: UInt32
    
    init(device: MTLDevice, count: Int, index: UInt32, label: String, 
         options: MTLResourceOptions = .storageModeShared) {
        
        let size = count * MemoryLayout<T>.stride
        self.buffer = device.makeBuffer(length: size, options: options)!
        self.buffer.label = label
        self.count = count
        self.index = index
    }
    
    init(device: MTLDevice, array: [T], index: UInt32, 
         options: MTLResourceOptions = .storageModeShared) {
        
        let size = array.count * MemoryLayout<T>.stride
        self.buffer = device.makeBuffer(bytes: array, length: size, options: options)!
        self.count = array.count
        self.index = index
    }
    
    // Direct buffer initialization (for loading from binary)
    init(buffer: MTLBuffer, count: Int, index: UInt32 = 0) {
        self.buffer = buffer
        self.count = count
        self.index = index
    }
}
```

### 3.2 Required Buffers

You need these buffers for rendering:

```swift
// In your SplatCloud or renderer class:

var splats: MetalBuffer<Splat>           // Main splat buffer (from .bin file)
var temp_splats: MetalBuffer<Splat>      // Temporary buffer for sorting
var splat_indices: MetalBuffer<Int64>    // Index buffer for depth sorting
var quadBuffer: MetalBuffer<packed_float2>  // Fixed quad vertices
```

**Quad buffer initialization:**

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 407-413

```swift
// SplatCloud.swift, lines 407-413
let _quads: [packed_float2] = [
    [1, -1],   // Bottom-right
    [1, 1],    // Top-right
    [-1, -1],  // Bottom-left
    [-1, 1]    // Top-left
]

self.quadBuffer = MetalBuffer(device: device,
                              array: _quads,
                              index: 0,
                              options: .storageModePrivate)
```

---

## 4. Depth Sorting Pipeline

Gaussian splatting requires **back-to-front rendering** for correct alpha blending. This is a two-step process:

### 4.1 GPU Depth Computation Kernel

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 370-390

```metal
// SplatShaders.metal, lines 370-390
kernel void splat_set_depths(
    device int64_t * splat_indices [[buffer(0)]],
    const device Splat * splats [[buffer(1)]],
    constant Uniforms & uniforms [[ buffer(2) ]],
    uint index [[thread_position_in_grid]])
{
    Splat splat = splats[index];
    
    float x = splat.center.x;
    float y = splat.center.y;
    float z = splat.center.z;

    float depth = 0;
    
    // Compute camera-space depth using inverse model-view matrix
    float4x4 mat = uniforms.inv_model_view_matrix;
    depth = mat.columns[2].x * x + mat.columns[2].y * y + mat.columns[2].z * z;
    
    // Scale for integer precision
    depth = depth * 1000.0f;
    
    // Pack depth (32 bits) | index (32 bits) into 64-bit integer
    // This allows sorting by depth while preserving original index
    int32_t depthInt = static_cast<int32_t>(depth);
    int64_t packed = static_cast<int64_t>(depthInt) << 32 | index;
    
    splat_indices[index] = packed;
}
```

**Swift dispatch code:**

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 640-660

```swift
// SplatCloud.swift, lines 640-660
private func setSplatDepthsComputeShader() {
    
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        return
    }
    
    computeEncoder.setComputePipelineState(computePipelineState)
    
    computeEncoder.setBuffer(self.splat_indices.buffer, offset: 0, index: 0)
    computeEncoder.setBuffer(self.splats.buffer, offset: 0, index: 1)
    
    var uni: Uniforms = self.uniforms
    computeEncoder.setBytes(&uni, length: MemoryLayout<Uniforms>.stride, index: 2)
                                    
    let threadPerGrid = MTLSize(width: numPoints, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}
```

### 4.2 CPU Sorting

**Source:** [splat_utils.mm](MetalSplat/Utils/splat_utils.mm), lines 23-48

```cpp
// splat_utils.mm, lines 23-48
#include <algorithm>
#include <cstdint>
#include <cstring>
#import "ShaderTypes.h"

extern "C" void sort_splats(void * splat_buffer,
                            void * temp_splat_buffer,
                            void * splat_index_buffer,
                            Uniforms uniforms,
                            int num_splats)
{
    Splat * splats = (Splat*)splat_buffer;
    Splat * temp_splats = (Splat*)temp_splat_buffer;
    int64_t * splat_index = (int64_t*)splat_index_buffer;
    
    // Sort by packed depth|index (sorts by upper 32 bits = depth)
    std::sort(splat_index, splat_index + num_splats);
    
    // Reorder splats according to sorted indices
    for (int i = 0; i < num_splats; ++i) {
        // Extract original index from lower 32 bits
        int index = static_cast<int>(splat_index[i] & 0xFFFFFFFF);
        temp_splats[i] = splats[index];
    }
    
    // Copy sorted splats back to main buffer
    memcpy(splats, temp_splats, num_splats * sizeof(Splat));
}
```

**Swift wrapper:**

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 494-502

```swift
// SplatCloud.swift, lines 494-502
private func _sortSplatsCpp() {
    sort_splats(splats.buffer.contents(),
                temp_splats.buffer.contents(),
                splat_indices.buffer.contents(),
                uniforms,
                Int32(numPoints))
}
```

### 4.3 Bridging Header for C++ Function

**Source:** [SplatCloudC.h](MetalSplat/Utils/SplatCloudC.h)

```c
// SplatCloudC.h
#ifndef SplatCloudC_h
#define SplatCloudC_h

#include "ShaderTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

void sort_splats(void * splat_buffer,
                 void * temp_splat_buffer,
                 void * splat_index_buffer,
                 Uniforms uniforms,
                 int num_splats);

#ifdef __cplusplus
}
#endif

#endif /* SplatCloudC_h */
```

### 4.4 Complete Sorting Function

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 458-488

```swift
// SplatCloud.swift, lines 458-488
func sortSplats() {
    // Only sort every 4 frames for performance
    if frame_index % 4 == 0 && !isSorting {
        
        isSorting = true
        
        // Step 1: GPU - compute depths and pack into indices
        self.setSplatDepthsComputeShader()
        
        // Step 2: CPU - sort indices, reorder splats
        self._sortSplatsCpp()
        
        self.isSorting = false
    }
}
```

---

## 5. Rendering Pipeline Setup

### 5.1 Compute Pipeline (for depth sorting)

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 618-632

```swift
// SplatCloud.swift, lines 618-632
private func setupCompute(_ renderDestination: RenderDestinationProvider) {
    
    guard let commandQueue = device.makeCommandQueue() else {
        fatalError("Failed to create command queue.")
    }
    self.commandQueue = commandQueue
    
    let defaultLibrary = device.makeDefaultLibrary()
    
    // Create compute pipeline for depth calculation
    let computeFunction = defaultLibrary?.makeFunction(name: "splat_set_depths")
    do {
        computePipelineState = try device.makeComputePipelineState(function: computeFunction!)
    } catch {
        fatalError("Failed to create compute pipeline state.")
    }
}
```

### 5.2 Render Pipeline

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 548-606

```swift
// SplatCloud.swift, lines 548-606
private func makePipelineState(_ renderDestination: RenderDestinationProvider) -> MTLRenderPipelineState? {
    
    guard let vertexFunction = library.makeFunction(name: "splat_vertex"),
          let fragmentFunction = library.makeFunction(name: "splat_fragment") else {
        return nil
    }
    
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertexFunction
    descriptor.fragmentFunction = fragmentFunction
    
    descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
    descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
    descriptor.stencilAttachmentPixelFormat = .invalid
    descriptor.rasterSampleCount = renderDestination.sampleCount
    
    // =========== Alpha Blending Configuration ============= //
    descriptor.colorAttachments[0].isBlendingEnabled = true
    
    // Premultiplied alpha blending for back-to-front compositing
    descriptor.colorAttachments[0].sourceRGBBlendFactor = .one
    descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    descriptor.colorAttachments[0].sourceAlphaBlendFactor = .oneMinusSourceAlpha
    descriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
    
    descriptor.colorAttachments[0].rgbBlendOperation = .add
    descriptor.colorAttachments[0].alphaBlendOperation = .add
    
    return try? device.makeRenderPipelineState(descriptor: descriptor)
}
```

### 5.3 Depth Stencil State

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 540-546

```swift
// SplatCloud.swift, lines 540-546
private func setupShaders(_ renderDestination: RenderDestinationProvider) {
    
    let depthStateDescriptor = MTLDepthStencilDescriptor()
    depthStateDescriptor.depthCompareFunction = .always  // No depth testing
    depthStateDescriptor.isDepthWriteEnabled = false     // Don't write depth
    
    self.depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    self.pipelineState = makePipelineState(renderDestination)!
}
```

---

## 6. Vertex Shader

The vertex shader projects each Gaussian to screen space and computes the 2D bounding quad.

### 6.1 Helper Functions

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 118-200

```metal
// SplatShaders.metal, lines 118-152
// Compute 3D covariance matrix from scale and rotation quaternion
float3x3 computeCov3D(float3 scale, float mod, float4 rot) {
    
    // Create scaling matrix
    float3x3 S = float3x3(1.0);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Quaternion to rotation matrix
    float4 q = rot;
    float r = q.x;  // w component
    float x = q.y;
    float y = q.z;
    float z = q.w;

    float3x3 R = float3x3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    float3x3 M = S * R;

    // Covariance: Sigma = M^T * M
    float3x3 Sigma = transpose(M) * M;

    return Sigma;
}

// SplatShaders.metal, lines 154-157
float3 transformPoint4x3(thread const float3& p, constant const float4x4& matrix) {
    float3 transformed = {
        matrix[0][0] * p.x + matrix[1][0] * p.y + matrix[2][0] * p.z + matrix[3][0],
        matrix[0][1] * p.x + matrix[1][1] * p.y + matrix[2][1] * p.z + matrix[3][1],
        matrix[0][2] * p.x + matrix[1][2] * p.y + matrix[2][2] * p.z + matrix[3][2]
    };
    return transformed;
}

// SplatShaders.metal, lines 159-200
// Project 3D covariance to 2D screen-space covariance (EWA splatting)
float3 computeCov2D(thread const float3& mean,
                    thread const CameraParameters& params,
                    thread const float3x3& cov3D,
                    constant const float4x4& viewmatrix)
{
    // Transform point to camera space
    float3 t = transformPoint4x3(mean, viewmatrix);

    // Clamp to frustum limits
    const float limx = 1.3f * params.tan_fovx;
    const float limy = 1.3f * params.tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // Jacobian of perspective projection
    float3x3 J = float3x3(
        params.focal_x / t.z, 0.0f, -(params.focal_x * t.x) / (t.z * t.z),
        0.0f, params.focal_y / t.z, -(params.focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );

    // View rotation matrix (3x3)
    float3x3 W = float3x3(
        viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0],
        viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1],
        viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]
    );

    // Transform: T = W * J
    float3x3 T = W * J;

    // Project covariance to 2D
    float3x3 cov = transpose(T) * transpose(cov3D) * T;

    // Low-pass filter (anti-aliasing)
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    // Return upper triangle (symmetric matrix): (σxx, σxy, σyy)
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}
```

### 6.2 Vertex Shader Input/Output Structures

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 87-109

```metal
// SplatShaders.metal, lines 87-109
struct VertexIn {
    float2 position;  // Quad corner: (-1,-1), (-1,1), (1,-1), or (1,1)
};

struct VertexOut {
    float4 position [[position]];  // Clip-space position
    float4 color;                  // RGBA color
    float3 conic;                  // Inverse 2D covariance (for fragment shader)
    float2 center_screen_pos;      // Screen-space center
    float is_valid;                // Validity flag
};

struct CameraParameters {
    float focal_x;
    float focal_y;
    float tan_fovx;
    float tan_fovy;
};
```

### 6.3 Main Vertex Shader

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 208-310

```metal
// SplatShaders.metal, lines 208-310
vertex VertexOut splat_vertex(
    device VertexIn const *vertices [[buffer(0)]],     // Quad corners
    const device Splat *instances [[buffer(1)]],       // Per-Gaussian data
    constant Uniforms & uniforms [[ buffer(2) ]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]])
{
    VertexIn vertexIn = vertices[vertexID];
    Splat instance = instances[instanceID];
    
    VertexOut out;
    out.position = float4(0, 0, 0, 1);
    out.is_valid = 0.0;
    
    float2 quad_pos = vertexIn.position;  // [-1,1] × [-1,1]
    float2 viewport(uniforms.viewport_width, uniforms.viewport_height);
    
    // Project Gaussian center to clip space
    float3 p_orig = instance.center.xyz;
    float4 p_world = uniforms.model_matrix * float4(p_orig, 1);
    
    const float4 center_clip_pos = uniforms.projection_matrix * 
                                   uniforms.model_view_matrix * 
                                   float4(p_orig, 1);
    
    // Compute screen-space center for fragment shader
    const float proj_x = -1.0;
    out.center_screen_pos = (center_clip_pos.xy / center_clip_pos.w * 
                             float2(0.5, 0.5 * proj_x) + 0.5) * viewport;
    
    // Compute 3D covariance from scale and rotation
    const float scale_modifier = 1.0;
    const float3x3 cov3D = computeCov3D(instance.scale.xyz, scale_modifier, instance.quat);
    
    // Setup camera parameters
    CameraParameters cam_params;
    cam_params.focal_x = uniforms.focal_x;
    cam_params.focal_y = uniforms.focal_y;
    cam_params.tan_fovx = uniforms.tan_fovx;
    cam_params.tan_fovy = uniforms.tan_fovy;
    
    // Compute 2D screen-space covariance
    float3 cov = computeCov2D(p_orig, cam_params, cov3D, uniforms.model_view_matrix);
    
    // Compute conic (inverse covariance) for EWA evaluation
    float det = (cov.x * cov.z - cov.y * cov.y);
    
    if (det == 0.0f)
        return out;  // Degenerate Gaussian
    
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    out.conic = conic;
    
    // Compute bounding radius from eigenvalues
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    
    // Expand quad to cover bounding circle
    float2 deltaScreenPos = quad_pos * radius * 2 / viewport;
    
    out.position = center_clip_pos;
    out.position.xy += deltaScreenPos * center_clip_pos.w;
    
    out.is_valid = true;
    out.color = instance.color;
    
    return out;
}
```

---

## 7. Fragment Shader

The fragment shader evaluates the Gaussian function at each pixel.

### 7.1 Helper Functions

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 312-328

```metal
// SplatShaders.metal, lines 312-328
inline float CalcPowerFromConic(float3 conic, float2 d) {
    // Evaluate: -0.5 * d^T * Sigma^{-1} * d
    // conic = (σyy/det, -σxy/det, σxx/det)
    return -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
}

inline float2 CalcScreenSpaceDelta(float2 svPositionXY, float2 centerXY, float proj_x) {
    float2 d = svPositionXY - centerXY;
    d.y *= proj_x;
    return d;
}
```

### 7.2 Main Fragment Shader

**Source:** [SplatShaders.metal](MetalSplat/SplatShaders.metal), lines 330-360

```metal
// SplatShaders.metal, lines 330-360
fragment float4 splat_fragment(
    VertexOut in [[stage_in]],
    constant Uniforms & uni [[ buffer(2) ]])
{
    // Discard invalid splats
    if (in.is_valid < 0.5) {
        discard_fragment();
    }
    
    // Compute distance from pixel to Gaussian center
    const float proj_x = 1.0;
    const float2 d = CalcScreenSpaceDelta(in.position.xy, in.center_screen_pos, proj_x);
    
    // Evaluate Gaussian: exp(-0.5 * d^T * Sigma^{-1} * d)
    float power = CalcPowerFromConic(in.conic, d);
    
    // Modulate alpha by Gaussian falloff
    in.color.a *= saturate(exp(power));
    
    // Discard nearly transparent fragments
    if (in.color.a < 1.0 / 255.0) {
        discard_fragment();
    }
    
    // Output premultiplied alpha
    const float alpha = in.color.a;
    return float4(in.color.rgb * alpha, alpha);
}
```

---

## 8. Alpha Blending Configuration

Alpha blending is critical for correct Gaussian compositing.

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 583-604

```swift
// SplatCloud.swift, lines 583-604
// Enable blending
descriptor.colorAttachments[0].isBlendingEnabled = true

// Premultiplied alpha blending equation:
// C_out = C_src * 1 + C_dst * (1 - alpha_src)
// A_out = A_src * (1 - A_dst) + A_dst * 1

descriptor.colorAttachments[0].sourceRGBBlendFactor = .one
descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
descriptor.colorAttachments[0].sourceAlphaBlendFactor = .oneMinusSourceAlpha
descriptor.colorAttachments[0].destinationAlphaBlendFactor = .one

descriptor.colorAttachments[0].rgbBlendOperation = .add
descriptor.colorAttachments[0].alphaBlendOperation = .add
```

**Mathematical explanation:**

For back-to-front rendering with premultiplied alpha:

$$C_{out} = C_{src} + C_{dst} \cdot (1 - \alpha_{src})$$

Where $C_{src} = \text{RGB} \cdot \alpha$ (premultiplied in fragment shader).

---

## 9. Uniforms Management

### 9.1 Updating Uniforms Each Frame

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 787-830

```swift
// SplatCloud.swift, lines 787-830
override func update(camera: Satin.Camera, viewport: simd_float4) {
    
    let modelMatrix = self.worldMatrix
    let modelViewMatrix = simd_mul(camera.viewMatrix, modelMatrix)
    
    let width = viewport.z
    let height = viewport.w

    // Extract tangent of half-angles of the FoVs
    let tan_fovx = 1.0 / camera.projectionMatrix[0][0]
    let tan_fovy = 1.0 / camera.projectionMatrix[1][1]
    
    let focal_y = height / (2.0 * tan_fovy)
    let focal_x = width / (2.0 * tan_fovx)
    
    let time: Double = CACurrentMediaTime()
    
    let cameraPos = simd_float4(camera.worldPosition, 1.0)
    let cameraPosOrig = simd_mul(simd_inverse(modelMatrix), cameraPos)
    
    let uni = Uniforms(
        projection_matrix: camera.projectionMatrix,
        model_matrix: modelMatrix,
        model_view_matrix: modelViewMatrix,
        inv_model_view_matrix: simd_inverse(modelViewMatrix),
        camera_pos: cameraPos,
        camera_pos_orig: cameraPosOrig,
        viewport_width: viewport.z,
        viewport_height: viewport.w,
        focal_x: focal_x,
        focal_y: focal_y,
        tan_fovx: tan_fovx,
        tan_fovy: tan_fovy,
        drag_alpha: 0.0,
        time: Float(time)
    )
    
    self.updateUniforms(uniforms: uni)
}

func updateUniforms(uniforms: Uniforms) {
    self.uniforms = uniforms
}
```

---

## 10. Render Loop

### 10.1 Main Render Function

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 506-536

```swift
// SplatCloud.swift, lines 506-536
func render(renderEncoder: MTLRenderCommandEncoder) {
    
    // Sort splats by depth (every 4 frames)
    self.sortSplats()
    
    // Set depth stencil state (no depth testing)
    renderEncoder.setDepthStencilState(depthState)
    
    // Set render pipeline state
    renderEncoder.setRenderPipelineState(pipelineState)
    
    // Bind vertex buffers
    renderEncoder.setVertexBuffer(self.quadBuffer.buffer, offset: 0, index: 0)  // Quad vertices
    renderEncoder.setVertexBuffer(self.splats.buffer, offset: 0, index: 1)      // Splat instances
    
    // Bind uniforms
    var uni: Uniforms = self.uniforms
    renderEncoder.setVertexBytes(&uni, length: MemoryLayout<Uniforms>.stride, index: 2)
    renderEncoder.setFragmentBytes(&uni, length: MemoryLayout<Uniforms>.stride, index: 2)
    
    // Draw instanced quads (one quad per Gaussian)
    renderEncoder.drawPrimitives(
        type: .triangleStrip,
        vertexStart: 0,
        vertexCount: self.quadBuffer.count,      // 4 vertices per quad
        instanceCount: self.splats.count         // Number of Gaussians
    )
    
    frame_index += 1
}
```

### 10.2 Satin Renderable Protocol

**Source:** [SplatCloud.swift](MetalSplat/SplatCloud.swift), lines 832-840

```swift
// SplatCloud.swift, lines 832-840
func draw(renderEncoder: MTLRenderCommandEncoder, shadow: Bool) {
    self.render(renderEncoder: renderEncoder)
}
```

---

## 11. Complete Implementation Checklist

### Files to Create/Modify

| File | Purpose | Action |
|------|---------|--------|
| `ShaderTypes.h` | Splat & Uniforms structs | ✅ Copy from original |
| `SplatShaders.metal` | GPU shaders | ⚠️ Keep only: `splat_set_depths`, `splat_vertex`, `splat_fragment`, helper functions |
| `splat_utils.mm` | CPU sorting | ✅ Copy from original |
| `SplatCloudC.h` | C bridging header | ✅ Copy from original |
| `MetalBuffer.swift` | Buffer wrapper | ✅ Copy from original |
| `SplatLoader.swift` | Binary loading | 🆕 Create new (see Section 2) |
| `SimpleSplatRenderer.swift` | Main renderer | 🆕 Create new (combine from SplatCloud.swift) |

### Minimal Metal Shader File

You can create a stripped-down shader file with only the required functions:

```metal
// SimpleSplatShaders.metal

#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

// ============= DATA STRUCTURES =============
struct VertexIn {
    float2 position;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float3 conic;
    float2 center_screen_pos;
    float is_valid;
};

struct CameraParameters {
    float focal_x;
    float focal_y;
    float tan_fovx;
    float tan_fovy;
};

// ============= HELPER FUNCTIONS =============
// [Copy computeCov3D from SplatShaders.metal lines 118-152]
// [Copy transformPoint4x3 from SplatShaders.metal lines 154-157]
// [Copy computeCov2D from SplatShaders.metal lines 159-200]

// ============= DEPTH SORTING KERNEL =============
// [Copy splat_set_depths from SplatShaders.metal lines 370-390]

// ============= VERTEX SHADER =============
// [Copy splat_vertex from SplatShaders.metal lines 208-310]

// ============= FRAGMENT SHADER =============
// [Copy CalcPowerFromConic from SplatShaders.metal lines 312-318]
// [Copy CalcScreenSpaceDelta from SplatShaders.metal lines 320-328]
// [Copy splat_fragment from SplatShaders.metal lines 330-360]
```

### Dependencies

| Dependency | Required? | Notes |
|------------|-----------|-------|
| OpenCV | ❌ No | Not needed (no video decoding) |
| Satin | ⚠️ Optional | Only for camera/scene management. Can use custom implementation. |
| Forge | ⚠️ Optional | Only for MTKView wrapper. Can use standard MetalKit. |
| Metal | ✅ Yes | Core rendering |
| MetalKit | ✅ Yes | View and render loop |

### Simplified Class Structure

```swift
// SimpleSplatRenderer.swift
class SimpleSplatRenderer {
    // Metal resources
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // Pipeline states
    private var renderPipelineState: MTLRenderPipelineState!
    private var computePipelineState: MTLComputePipelineState!
    private var depthState: MTLDepthStencilState!
    
    // Buffers
    private var splats: MTLBuffer!
    private var tempSplats: MTLBuffer!
    private var splatIndices: MTLBuffer!
    private var quadBuffer: MTLBuffer!
    
    // State
    private var numSplats: Int = 0
    private var uniforms: Uniforms = Uniforms()
    private var frameIndex: Int = 0
    private var isSorting: Bool = false
    
    // Initialize with .bin file path
    init(binFilePath: String) throws { ... }
    
    // Update camera each frame
    func updateCamera(viewMatrix: simd_float4x4, 
                      projectionMatrix: simd_float4x4,
                      viewport: simd_float4) { ... }
    
    // Render to encoder
    func render(encoder: MTLRenderCommandEncoder) { ... }
}
```

---

## Summary

To build the simplified application:

1. **Load** pre-computed Splat structs from `.bin` file directly into Metal buffer
2. **Sort** splats by depth every few frames (GPU depth computation + CPU std::sort)
3. **Render** using instanced quad drawing with EWA splatting shaders
4. **Blend** using premultiplied alpha for correct compositing

The entire video decoding and dequantization pipeline is bypassed. Your input file format is simply:

```
[Splat_0][Splat_1][Splat_2]...[Splat_N-1]
```

Where each `Splat` is 64 bytes containing `(center, color, scale, quat)` as `float4` vectors.

---

*Application design document for simplified Gaussian Splatting iOS renderer*
