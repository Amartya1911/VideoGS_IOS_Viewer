# VideoGS iOS Viewer: Technical Report

## Real-Time Volumetric Video Streaming Using 2D Dynamic Gaussians

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Data Encoding Format](#3-data-encoding-format)
4. [Video Decoding Pipeline](#4-video-decoding-pipeline)
5. [GPU-Based Splat Generation](#5-gpu-based-splat-generation)
6. [Gaussian Splatting Mathematics](#6-gaussian-splatting-mathematics)
7. [Depth Sorting Pipeline](#7-depth-sorting-pipeline)
8. [Real-Time Rendering Pipeline](#8-real-time-rendering-pipeline)
9. [Alpha Blending and Compositing](#9-alpha-blending-and-compositing)
10. [Performance Optimizations](#10-performance-optimizations)
11. [Key Data Structures](#11-key-data-structures)
12. [Complete Rendering Workflow](#12-complete-rendering-workflow)
13. [Reproduction Guide](#13-reproduction-guide)
14. [Connection to Python Compression Pipeline](#14-connection-to-python-compression-pipeline)
15. [AR Mode (ARSplatView)](#15-ar-mode-arsplatview)
16. [Legacy PLY Loader (SplatCloudC)](#16-legacy-ply-loader-splatcloudc)
17. [Utility Components](#17-utility-components)
18. [Dataset Configurations](#18-dataset-configurations)

---

## 1. Executive Summary

This iOS application implements **real-time volumetric video playback** using **3D Gaussian Splatting (3DGS)**, a novel neural rendering technique. The key innovation is encoding Gaussian splat parameters into video streams, allowing standard video codecs (H.264/HEVC) to efficiently compress temporal coherence in dynamic scenes.

### Core Technique

Instead of storing raw floating-point Gaussian parameters per frame, the system:
1. **Quantizes** each parameter channel to 8-bit or 16-bit integers
2. **Encodes** quantized values as grayscale video frames
3. **Decodes** videos on-device using OpenCV
4. **Dequantizes** pixel values back to floats using per-frame min/max bounds
5. **Renders** using a Metal GPU pipeline with EWA splatting

This approach achieves ~30+ FPS playback of dynamic volumetric content on iPhone hardware.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA STRUCTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  64 Group Folders × 17-20 MP4 Videos × N Frames per Video       │
│  + group_info_ykx_380.json (frame index mappings)               │
│  + viewer_min_max_ykx_380.json (dequantization parameters)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO DECODING (OpenCV)                      │
├─────────────────────────────────────────────────────────────────┤
│  OpencvTest.mm: processVideo() → Array of grayscale UIImage     │
│  17 videos × N frames = 17N grayscale images per group          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GPU TEXTURE CREATION (Metal)                    │
├─────────────────────────────────────────────────────────────────┤
│  SplatCloud.swift: createDFTexture() → 17 Metal textures        │
│  Each texture: R8Unorm format (single channel, 8-bit)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              GPU COMPUTE: SPLAT GENERATION                      │
├─────────────────────────────────────────────────────────────────┤
│  SplatShaders.metal: generateSplats() kernel                    │
│  Input: 17 textures + minmax buffer + init_pos buffer           │
│  Output: Array of Splat structs (center, color, scale, quat)    │
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

## 3. Data Encoding Format

### 3.1 Directory Structure

```
ykx_boxing_long_qp15_380/
├── group0/
│   ├── 0.mp4   (x position, low byte)
│   ├── 1.mp4   (x position, high byte)
│   ├── 2.mp4   (y position, low byte)
│   ├── 3.mp4   (y position, high byte)
│   ├── 4.mp4   (z position, low byte)
│   ├── 5.mp4   (z position, high byte)
│   ├── 9.mp4   (f_dc_0: SH coefficient R)
│   ├── 10.mp4  (f_dc_1: SH coefficient G)
│   ├── 11.mp4  (f_dc_2: SH coefficient B)
│   ├── 12.mp4  (opacity)
│   ├── 13.mp4  (scale_0)
│   ├── 14.mp4  (scale_1)
│   ├── 15.mp4  (scale_2)
│   ├── 16.mp4  (rot_0: quaternion w)
│   ├── 17.mp4  (rot_1: quaternion x)
│   ├── 18.mp4  (rot_2: quaternion y)
│   └── 19.mp4  (rot_3: quaternion z)
├── group1/
│   └── ...
└── group63/
```

### 3.2 Video File Mapping (17 Channels)

The video files map to the `urls` array in `VideoProcessor`:

```swift
// SplatSimpleView.swift, line 237
var urls = ["0.mp4", "1.mp4", "2.mp4", "3.mp4", "4.mp4", "5.mp4", 
            "9.mp4", "10.mp4", "11.mp4", "12.mp4", "13.mp4", 
            "14.mp4", "15.mp4", "16.mp4", "17.mp4", "18.mp4", "19.mp4"]
```

**Index Mapping:**
| Index | Filename | Parameter | Bit Depth | Notes |
|-------|----------|-----------|-----------|-------|
| 0 | 0.mp4 | x_low | 8-bit | Low byte of 16-bit x position |
| 1 | 1.mp4 | x_high | 8-bit | High byte of 16-bit x position |
| 2 | 2.mp4 | y_low | 8-bit | Low byte of 16-bit y position |
| 3 | 3.mp4 | y_high | 8-bit | High byte of 16-bit y position |
| 4 | 4.mp4 | z_low | 8-bit | Low byte of 16-bit z position |
| 5 | 5.mp4 | z_high | 8-bit | High byte of 16-bit z position |
| 6 | 9.mp4 | f_dc_0 | 8-bit | Spherical harmonic (red) |
| 7 | 10.mp4 | f_dc_1 | 8-bit | Spherical harmonic (green) |
| 8 | 11.mp4 | f_dc_2 | 8-bit | Spherical harmonic (blue) |
| 9 | 12.mp4 | opacity | 8-bit | Alpha/transparency |
| 10 | 13.mp4 | scale_0 | 8-bit | Scale X |
| 11 | 14.mp4 | scale_1 | 8-bit | Scale Y |
| 12 | 15.mp4 | scale_2 | 8-bit | Scale Z |
| 13 | 16.mp4 | rot_0 | 8-bit | Quaternion w |
| 14 | 17.mp4 | rot_1 | 8-bit | Quaternion x |
| 15 | 18.mp4 | rot_2 | 8-bit | Quaternion y |
| 16 | 19.mp4 | rot_3 | 8-bit | Quaternion z |

### 3.3 Quantization Scheme

**Position channels (x, y, z):** 16-bit quantization (2 videos per axis)
- Combines low byte + high byte: `uint16 = low | (high << 8)`
- Dequantization: `value = uint16 * (max - min) / 65535.0 + min`

**All other channels:** 8-bit quantization (1 video per parameter)
- Dequantization: `value = uint8 * (max - min) / 255.0 + min`

### 3.4 Min/Max JSON Structure

```json
// viewer_min_max_ykx_380.json
{
  "0": {
    "num": 0,
    "info": [
      // First 6 values: position min/max (x_min, x_max, y_min, y_max, z_min, z_max)
      // Values 12-17: color min/max (fdc0_min, fdc0_max, fdc1_min, fdc1_max, fdc2_min, fdc2_max)
      // Last 16 values: opacity, scale (3), rotation (4) min/max pairs
    ]
  },
  "1": { ... }
}
```

The `extractNeededValues()` function extracts specific indices:

```swift
// SplatSimpleView.swift, lines 51-60
func extractNeededValues(info: [Double]) -> [Double] {
    var extractedValues = [Double]()
    extractedValues.append(contentsOf: info[0...5])      // Position min/max
    extractedValues.append(contentsOf: info[12...17])    // Color min/max
    extractedValues.append(contentsOf: info.suffix(16))  // Opacity, scale, rot min/max
    return extractedValues
}
```

---

## 4. Video Decoding Pipeline

### 4.1 OpenCV Video Processing

The `OpencvTest` class (Objective-C++) handles video decoding:

```objectivec
// OpencvTest.mm, lines 32-74
+ (NSArray<UIImage *> *)processVideo:(NSString *)url frameRate:(NSInteger)frameRate {
    NSString *filePath = [[NSBundle mainBundle] pathForResource:url ofType:nil];
    
    std::string urllink = [filePath UTF8String];
    cv::VideoCapture videoCapture(urllink);
    
    if (!videoCapture.isOpened()) {
        return @[];
    }
    
    videoCapture.set(cv::CAP_PROP_FPS, frameRate);
    
    cv::Mat frame;
    NSMutableArray<UIImage *> *images = [NSMutableArray array];
    
    while (videoCapture.read(frame)) {
        // Convert BGR to grayscale
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        
        // Create CGImage from raw bytes
        NSData *data = [NSData dataWithBytes:frame.data 
                                      length:frame.total() * frame.elemSize()];
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
        
        CGImageRef imageRef = CGImageCreate(
            frame.cols, frame.rows,
            8,                              // bits per component
            8 * frame.elemSize(),           // bits per pixel
            frame.step[0],                  // bytes per row
            colorSpace,
            kCGBitmapByteOrderDefault,
            provider, NULL, false,
            kCGRenderingIntentDefault
        );
        
        UIImage *image = [UIImage imageWithCGImage:imageRef];
        [images addObject:image];
        
        CGImageRelease(imageRef);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
    }
    
    return [images copy];
}
```

### 4.2 Video Processing Orchestration

```swift
// SplatSimpleView.swift, lines 285-305
func processVideos(groupIndex ind: Int, dataIndex: Int) -> [[UIImage]] {
    var allImages: [[UIImage]] = []
    var urlpath = "ykx_boxing_long_qp15_380"

    for i in 0..<17 {
        let frameRate = 25
        let videoURL = "\(urlpath)/group\(ind)/\(urls[i])"
        let images = OpencvTest.processVideo(videoURL, frameRate: frameRate) ?? []
        allImages.append(images)
    }
    return allImages
}
```

**Output:** `[[UIImage]]` - 17 arrays (one per channel), each containing N grayscale frames.

---

## 5. GPU-Based Splat Generation

### 5.1 Texture Creation

Each video frame is converted to a Metal texture:

```swift
// SplatCloud.swift, lines 349-378
func createDFTexture(device: MTLDevice, image: UIImage) throws -> MTLTexture? {
    guard let cgImage = image.cgImage else { return nil }

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm,           // Single channel, 8-bit normalized [0,1]
        width: cgImage.width,
        height: cgImage.height,
        mipmapped: false
    )
    textureDescriptor.usage = [.shaderRead]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
        throw NSError(domain: "TextureCreationError", code: 1, userInfo: nil)
    }

    guard let pixelData = cgImage.dataProvider?.data else {
        throw NSError(domain: "TextureCreationError", code: 2, userInfo: nil)
    }

    let bytesPerRow = cgImage.bytesPerRow
    texture.replace(
        region: MTLRegionMake2D(0, 0, cgImage.width, cgImage.height),
        mipmapLevel: 0,
        slice: 0,
        withBytes: CFDataGetBytePtr(pixelData),
        bytesPerRow: bytesPerRow,
        bytesPerImage: bytesPerRow * cgImage.height
    )

    return texture
}
```

### 5.2 Compute Shader Dispatch

```swift
// SplatCloud.swift, lines 674-717
private func setGenerateSplatComputeShader(width: Int, height: Int, 
                                            textures: [MTLTexture], 
                                            minmaxBuffer: MTLBuffer, 
                                            initPosBuffer: MTLBuffer) {
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        return
    }
    
    computeEncoder.setComputePipelineState(generateSplatPipelineState)
    
    // Bind all 17 textures
    for i in 0..<17 {
        computeEncoder.setTexture(textures[i], index: i)
    }
    
    // Bind output buffer and dequantization parameters
    computeEncoder.setBuffer(self.splats.buffer, offset: 0, index: 0)
    computeEncoder.setBuffer(minmaxBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(initPosBuffer, offset: 0, index: 2)
    
    // Dispatch threads: one per pixel (= one per splat)
    let threadsPerGrid = MTLSize(width: width, height: height, depth: 1)
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}
```

### 5.3 The generateSplats Kernel

This is the **core dequantization kernel** that converts video frames to Gaussian splats:

```metal
// SplatShaders.metal, lines 465-529
kernel void generateSplats(
    texture2d<float, access::read> x0 [[ texture(0) ]],   // x low byte
    texture2d<float, access::read> x1 [[ texture(1) ]],   // x high byte
    texture2d<float, access::read> y0 [[ texture(2) ]],   // y low byte
    texture2d<float, access::read> y1 [[ texture(3) ]],   // y high byte
    texture2d<float, access::read> z0 [[ texture(4) ]],   // z low byte
    texture2d<float, access::read> z1 [[ texture(5) ]],   // z high byte
    texture2d<float, access::read> fdc0 [[ texture(6) ]], // SH red
    texture2d<float, access::read> fdc1 [[ texture(7) ]], // SH green
    texture2d<float, access::read> fdc2 [[ texture(8) ]], // SH blue
    texture2d<float, access::read> opacity [[ texture(9) ]],
    texture2d<float, access::read> scale0 [[ texture(10) ]],
    texture2d<float, access::read> scale1 [[ texture(11) ]],
    texture2d<float, access::read> scale2 [[ texture(12) ]],
    texture2d<float, access::read> rot0 [[ texture(13) ]],
    texture2d<float, access::read> rot1 [[ texture(14) ]],
    texture2d<float, access::read> rot2 [[ texture(15) ]],
    texture2d<float, access::read> rot3 [[ texture(16) ]],
    device Splat* splats [[ buffer(0) ]],
    device const float* minmax [[ buffer(1) ]],
    device const float* init_pos [[ buffer(2) ]],
    uint2 gid [[thread_position_in_grid]])
{
    uint index = gid.y * x0.get_width() + gid.x;
    
    // ============================================
    // STEP 1: Reconstruct 16-bit position values
    // ============================================
    // Read normalized [0,1] values, convert to uint8, combine into uint16
    float xVal = float((uint(x1.read(gid).r * 255.0) << 8) + uint(x0.read(gid).r * 255.0));
    float yVal = float((uint(y1.read(gid).r * 255.0) << 8) + uint(y0.read(gid).r * 255.0));
    float zVal = float((uint(z1.read(gid).r * 255.0) << 8) + uint(z0.read(gid).r * 255.0));

    // ============================================
    // STEP 2: Dequantize positions using minmax
    // ============================================
    // Formula: value = quantized * (max - min) / 65535.0 + min + offset
    xVal = xVal * (minmax[1] - minmax[0]) / 65535.0 + minmax[0] + init_pos[0];
    yVal = yVal * (minmax[3] - minmax[2]) / 65535.0 + minmax[2] + init_pos[1];
    zVal = zVal * (minmax[5] - minmax[4]) / 65535.0 + minmax[4] + init_pos[2];
    float4 center = float4(xVal, yVal, zVal, 1.0);
    
    // ============================================
    // STEP 3: Dequantize 8-bit channels
    // ============================================
    // Color (SH degree 0 coefficients)
    float f_dc0Val = fdc0.read(gid).r * (minmax[7] - minmax[6]) + minmax[6];
    float f_dc1Val = fdc1.read(gid).r * (minmax[9] - minmax[8]) + minmax[8];
    float f_dc2Val = fdc2.read(gid).r * (minmax[11] - minmax[10]) + minmax[10];
    
    // Opacity
    float opacityVal = opacity.read(gid).r * (minmax[13] - minmax[12]) + minmax[12];
    
    // Scale (log space)
    float scale0Val = scale0.read(gid).r * (minmax[15] - minmax[14]) + minmax[14];
    float scale1Val = scale1.read(gid).r * (minmax[17] - minmax[16]) + minmax[16];
    float scale2Val = scale2.read(gid).r * (minmax[19] - minmax[18]) + minmax[18];
    
    // Rotation quaternion
    float rot0Val = rot0.read(gid).r * (minmax[21] - minmax[20]) + minmax[20];
    float rot1Val = rot1.read(gid).r * (minmax[23] - minmax[22]) + minmax[22];
    float rot2Val = rot2.read(gid).r * (minmax[25] - minmax[24]) + minmax[24];
    float rot3Val = rot3.read(gid).r * (minmax[27] - minmax[26]) + minmax[26];

    float4 quat = float4(rot0Val, rot1Val, rot2Val, rot3Val);
    float4 scale = float4(scale0Val, scale1Val, scale2Val, 1.0);

    // ============================================
    // STEP 4: Convert SH to RGB color
    // ============================================
    // SH_C0 = 0.28209479177387814 (0th order spherical harmonic coefficient)
    float SH_C0 = 0.28209479177387814;
    float4 color = float4(
        0.5 + SH_C0 * f_dc0Val,  // Red
        0.5 + SH_C0 * f_dc1Val,  // Green
        0.5 + SH_C0 * f_dc2Val,  // Blue
        opacityVal               // Alpha (sigmoid already applied during training)
    );

    // ============================================
    // STEP 5: Store final Splat struct
    // ============================================
    Splat splat = {center, color, scale, quat};
    splats[index] = splat;
}
```

---

## 6. Gaussian Splatting Mathematics

### 6.1 3D Covariance Matrix from Scale and Rotation

Each Gaussian is parameterized by:
- **Scale**: $\mathbf{s} = (s_x, s_y, s_z)$ - anisotropic scaling
- **Rotation**: $\mathbf{q} = (w, x, y, z)$ - unit quaternion

The 3D covariance matrix $\Sigma$ is computed as:

$$\Sigma = R S S^T R^T = (SR)^T (SR)$$

Where $S$ is a diagonal scaling matrix and $R$ is the rotation matrix from the quaternion.

```metal
// SplatShaders.metal, lines 118-152
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
```

### 6.2 2D Covariance Projection (EWA Splatting)

The 3D covariance is projected to screen space using the Jacobian of the perspective projection:

$$\Sigma' = J W \Sigma W^T J^T$$

Where:
- $W$ is the view rotation matrix (3×3 upper-left of view matrix)
- $J$ is the Jacobian of the perspective projection

```metal
// SplatShaders.metal, lines 158-200
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
    t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
    t.y = min(limy, max(-limy, t.y / t.z)) * t.z;

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

    // Return upper triangle (symmetric matrix)
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}
```

### 6.3 Conic (Inverse Covariance)

For efficient fragment evaluation, we compute the **conic** (inverse of 2D covariance):

$$\text{conic} = \Sigma'^{-1} = \frac{1}{\det(\Sigma')} \begin{pmatrix} \sigma_{yy} & -\sigma_{xy} \\ -\sigma_{xy} & \sigma_{xx} \end{pmatrix}$$

```metal
// SplatShaders.metal, lines 268-278
float det = (cov.x * cov.z - cov.y * cov.y);

if (det == 0.0f)
    return out;

float det_inv = 1.f / det;
float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
out.conic = conic;
```

### 6.4 Gaussian Bounding Radius

The screen-space extent is computed from the eigenvalues of the 2D covariance:

$$\lambda_{1,2} = \frac{\sigma_{xx} + \sigma_{yy}}{2} \pm \sqrt{\left(\frac{\sigma_{xx} + \sigma_{yy}}{2}\right)^2 - \det(\Sigma')}$$

$$\text{radius} = \lceil 3 \sqrt{\max(\lambda_1, \lambda_2)} \rceil$$

```metal
// SplatShaders.metal, lines 280-287
float mid = 0.5f * (cov.x + cov.z);
float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
```

---

## 7. Depth Sorting Pipeline

Gaussian splatting requires back-to-front rendering for correct alpha blending.

### 7.1 GPU Depth Computation

```metal
// SplatShaders.metal, lines 370-390
kernel void splat_set_depths(
    device int64_t * splat_indices [[buffer(0)]],
    const device Splat * splats [[buffer(1)]],
    constant Uniforms & uniforms [[ buffer(2) ]],
    uint index [[thread_position_in_grid]] )
{
    Splat splat = splats[index];
    
    float x = splat.center.x;
    float y = splat.center.y;
    float z = splat.center.z;

    // Compute camera-space depth using inverse model-view matrix
    float4x4 mat = uniforms.inv_model_view_matrix;
    float depth = mat.columns[2].x * x + mat.columns[2].y * y + mat.columns[2].z * z;
    
    // Scale for integer precision
    depth = depth * 1000.0f;
    
    // Pack depth (32 bits) | index (32 bits) into 64-bit integer
    int32_t depthInt = static_cast<int32_t>(depth);
    int64_t packed = static_cast<int64_t>(depthInt) << 32 | index;
    
    splat_indices[index] = packed;
}
```

### 7.2 CPU Sorting

```cpp
// splat_utils.mm, lines 23-48
extern "C" void sort_splats(void * splat_buffer,
                            void * temp_splat_buffer,
                            void * splat_index_buffer,
                            Uniforms uniforms,
                            int num_splats)
{
    Splat * splats = (Splat*)splat_buffer;
    Splat * temp_splats = (Splat*)temp_splat_buffer;
    int64_t * splat_index = (int64_t*)splat_index_buffer;
    
    // std::sort on packed depth|index values
    // This sorts by depth (upper 32 bits) automatically
    std::sort(splat_index, splat_index + num_splats);
    
    // Reorder splats according to sorted indices
    for (int i = 0; i < num_splats; ++i) {
        int index = static_cast<int>(splat_index[i] & 0xFFFFFFFF);  // Extract original index
        temp_splats[i] = splats[index];
    }
    
    // Copy sorted splats back
    memcpy(splats, temp_splats, num_splats * sizeof(Splat));
}
```

### 7.3 Sorting Frequency

Sorting is performed every 4 frames to balance quality and performance:

```swift
// SplatCloud.swift, lines 458-488
func sortSplats() {
    if frame_index % 4 == 0 && !isSorting {
        isSorting = true
        
        // GPU: Compute depths
        self.setSplatDepthsComputeShader()
        
        // CPU: Sort by depth
        self._sortSplatsCpp()
        
        self.isSorting = false
    }
}
```

---

## 8. Real-Time Rendering Pipeline

### 8.1 Vertex Shader: Quad Expansion

Each Gaussian is rendered as an instanced quad that covers its screen-space extent:

```metal
// SplatShaders.metal, lines 208-310
vertex VertexOut splat_vertex(
    device VertexIn const *vertices [[buffer(0)]],     // Quad corners: (-1,-1), (-1,1), (1,-1), (1,1)
    const device Splat *instances [[buffer(1)]],       // Per-Gaussian data
    constant Uniforms & uniforms [[ buffer(2) ]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]])
{
    VertexIn vertexIn = vertices[vertexID];
    Splat instance = instances[instanceID];
    
    VertexOut out;
    out.position = float4(0,0,0,1);
    out.is_valid = 0.0;
    
    float2 quad_pos = vertexIn.position;  // [-1,1] × [-1,1]
    float2 viewport(uniforms.viewport_width, uniforms.viewport_height);
    
    // Project Gaussian center to clip space
    float3 p_orig = instance.center.xyz;
    const float4 center_clip_pos = uniforms.projection_matrix * 
                                   uniforms.model_view_matrix * 
                                   float4(p_orig, 1);
    
    // Compute screen-space center for fragment shader
    const float proj_x = -1.0;
    out.center_screen_pos = (center_clip_pos.xy / center_clip_pos.w * 
                             float2(0.5, 0.5*proj_x) + 0.5) * viewport;
    
    // Compute 3D covariance
    const float scale_modifier = 1.0;
    const float3x3 cov3D = computeCov3D(instance.scale.xyz, scale_modifier, instance.quat);
    
    // Project to 2D covariance
    CameraParameters cam_params;
    cam_params.focal_x = uniforms.focal_x;
    cam_params.focal_y = uniforms.focal_y;
    cam_params.tan_fovx = uniforms.tan_fovx;
    cam_params.tan_fovy = uniforms.tan_fovy;
    
    float3 cov = computeCov2D(p_orig, cam_params, cov3D, uniforms.model_view_matrix);
    
    // Compute conic (inverse covariance)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f) return out;
    
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

### 8.2 Fragment Shader: Gaussian Evaluation

```metal
// SplatShaders.metal, lines 330-360
inline float CalcPowerFromConic(float3 conic, float2 d) {
    // Evaluate: -0.5 * d^T * Sigma^{-1} * d
    return -0.5 * (conic.x * d.x*d.x + conic.z * d.y*d.y) + conic.y * d.x*d.y;
}

inline float2 CalcScreenSpaceDelta(float2 svPositionXY, float2 centerXY, float proj_x) {
    float2 d = svPositionXY - centerXY;
    d.y *= proj_x;
    return d;
}

fragment float4 splat_fragment(VertexOut in [[stage_in]],
                               constant Uniforms & uni [[ buffer(2) ]])
{
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
    if (in.color.a < 1.0/255.0) {
        discard_fragment();
    }
    
    // Premultiplied alpha output
    const float alpha = in.color.a;
    return float4(in.color.rgb * alpha, alpha);
}
```

---

## 9. Alpha Blending and Compositing

### 9.1 Blend State Configuration

```swift
// SplatCloud.swift, lines 583-604
descriptor.colorAttachments[0].isBlendingEnabled = true

// Source: ONE (premultiplied alpha)
// Dest: ONE_MINUS_SOURCE_ALPHA
descriptor.colorAttachments[0].sourceRGBBlendFactor = .one
descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
descriptor.colorAttachments[0].sourceAlphaBlendFactor = .oneMinusSourceAlpha
descriptor.colorAttachments[0].destinationAlphaBlendFactor = .one

descriptor.colorAttachments[0].rgbBlendOperation = .add
descriptor.colorAttachments[0].alphaBlendOperation = .add
```

### 9.2 Blending Mathematics

For premultiplied alpha, the blend equation is:

$$C_{out} = C_{src} + C_{dst} \cdot (1 - \alpha_{src})$$

Where $C_{src} = \text{RGB} \cdot \alpha$ (premultiplied in fragment shader).

This achieves correct back-to-front compositing when splats are depth-sorted.

---

## 10. Performance Optimizations

### 10.1 Streaming Group Loading

Groups are loaded asynchronously to avoid blocking:

```swift
// SplatSimpleView.swift, lines 368-380
func setupProcessingQueue() {
    for group in self.groupIndexList {
        if [0].contains(group.0) {
            continue  // Skip first group (already loaded)
        }
        operationQueue.addOperation {
            self.processGroup(group)
        }
    }
}
```

### 10.2 Adaptive Loading Suspension

```swift
// SplatSimpleView.swift, lines 590-600
// Pause loading when buffer is full
if self.splatFinishNum > stopNum && self.suspend == false {
    self.suspend = true
    self.operationQueue.isSuspended = true
}
// Resume loading when buffer has space
else if self.splatFinishNum < stopNum && self.suspend == true {
    self.suspend = false
    self.operationQueue.isSuspended = false
}
```

### 10.3 GPU Buffer Management

```swift
// SplatCloud.swift, lines 718-728
private func copySplats() {
    let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()!
    let blitEncoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()!
    blitEncoder.copy(from: self.splats.buffer,
                     sourceOffset: 0,
                     to: self.temp_splats.buffer,
                     destinationOffset: 0,
                     size: self.splats.buffer.length)
    blitEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}
```

---

## 11. Key Data Structures

### 11.1 Splat Structure

```c
// ShaderTypes.h, lines 47-59
typedef struct {
    simd_float4 center; // xyz position + padding
    simd_float4 color;  // rgba (premultiplied alpha)
    simd_float4 scale;  // xyz scale + padding
    simd_float4 quat;   // xyzw quaternion
} Splat;
```

**Size:** 64 bytes per Gaussian

### 11.2 Uniforms Structure

```c
// ShaderTypes.h, lines 24-44
typedef struct {
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

## 12. Complete Rendering Workflow

### Frame-by-Frame Pipeline:

```
1. VIDEO LOADING (Background Thread)
   ├── VideoProcessor.processVideos()
   ├── OpencvTest.processVideo() × 17 channels
   └── Returns: [[UIImage]] - 17 × N frames

2. SPLATCLOUD INITIALIZATION (Per Frame)
   ├── Create 17 Metal textures from UIImages
   ├── Create Splat buffer (width × height entries)
   ├── Dispatch generateSplats compute kernel
   │   ├── Read 17 textures at (gid.x, gid.y)
   │   ├── Reconstruct 16-bit position from two 8-bit channels
   │   ├── Dequantize all values using minmax buffer
   │   ├── Convert SH coefficients to RGB
   │   └── Store Splat struct
   └── Copy to temp_splats buffer

3. RENDER LOOP (60 FPS)
   ├── Update uniforms (camera, projection, etc.)
   ├── Sort splats (every 4 frames):
   │   ├── GPU: splat_set_depths kernel
   │   └── CPU: std::sort + buffer reorder
   ├── Set pipeline state
   ├── Bind buffers (quads, splats, uniforms)
   └── drawPrimitives(triangleStrip, vertexCount: 4, instanceCount: numSplats)
       ├── Vertex shader: Project Gaussian, compute 2D covariance, expand quad
       └── Fragment shader: Evaluate Gaussian, output premultiplied alpha

4. FRAME ADVANCEMENT
   ├── Remove previous SplatCloud from scene
   ├── Add new SplatCloud to scene
   └── Update progress slider
```

---

## 13. Reproduction Guide

To reproduce this pipeline from scratch:

### 13.1 Data Preparation (Training Side)

1. Train a 3D Gaussian Splatting model on your video sequence
2. For each frame, extract per-Gaussian parameters:
   - Position (x, y, z)
   - Spherical harmonic coefficients (f_dc_0, f_dc_1, f_dc_2)
   - Opacity
   - Scale (s0, s1, s2)
   - Rotation quaternion (r0, r1, r2, r3)

3. Quantize parameters:
   ```python
   # Position: 16-bit quantization
   def quantize_16bit(values):
       min_val, max_val = values.min(), values.max()
       normalized = (values - min_val) / (max_val - min_val)
       quantized = (normalized * 65535).astype(np.uint16)
       return quantized, min_val, max_val
   
   # Other params: 8-bit quantization
   def quantize_8bit(values):
       min_val, max_val = values.min(), values.max()
       normalized = (values - min_val) / (max_val - min_val)
       quantized = (normalized * 255).astype(np.uint8)
       return quantized, min_val, max_val
   ```

4. Reshape quantized values to 2D images (e.g., reshape N splats to W×H)

5. Encode as grayscale video using FFmpeg:
   ```bash
   ffmpeg -framerate 25 -i frame_%04d.png -c:v libx264 -pix_fmt gray -qp 15 output.mp4
   ```

6. Save min/max values to JSON for dequantization

### 13.2 iOS Application Structure

1. **OpenCV Integration:**
   - Add opencv2.framework to project
   - Create bridging header for Objective-C++
   - Implement video decoding to UIImage array

2. **Metal Rendering:**
   - Create compute shader for dequantization
   - Implement vertex shader with covariance projection
   - Implement fragment shader with Gaussian evaluation
   - Configure alpha blending for back-to-front compositing

3. **Frame Management:**
   - Load video groups in background
   - Create SplatCloud per frame
   - Swap SplatClouds in scene graph at playback rate

### 13.3 Key Dependencies

- **OpenCV 4.x** (iOS framework)
- **Satin** (v13.0.0) - 3D rendering framework
- **Forge** (drunknbass fork) - Metal utilities
- **Metal** - GPU compute and rendering
- **MetalKit** - View and rendering loop

---

## Appendix (Original): File Reference (Superseded by Appendix A)

*See Appendix A below for the complete, updated file reference.*

---

## 14. Connection to Python Compression Pipeline

This section maps the **exact** relationship between the Python-side compression scripts (in the `VideoGS/compress/` directory) and the iOS viewer's data ingestion.

### 14.1 End-to-End Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PYTHON COMPRESSION SIDE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  train_sequence.py                                                   │
│    └── Produces PLY files (one per frame, grouped by 20)             │
│         └── Attributes: [x,y,z, nx,ny,nz, f_dc_0..2,               │
│              f_rest_0..N, opacity, scale_0..2, rot_0..3]             │
│                                                                      │
│  compress_to_png_full_sh.py                                          │
│    └── Reads PLYs, quantizes, reshapes to 2D images                  │
│    └── Output: PNGs + viewer_min_max.json + group_info.json          │
│         ├── Position: uint16 → split into 2×uint8 channels           │
│         ├── Others: uint8 channels                                   │
│         └── 20 channels for SH0 (6 pos + 3 normal + 3 dc            │
│              + 1 opacity + 3 scale + 4 rotation)                     │
│                                                                      │
│  compress_png_2_video.py                                             │
│    └── ffmpeg H.264 encoding per channel                             │
│    └── Output: 20 MP4 files per group                                │
│         ├── Position high bytes (1,3,5): QP=0 (lossless)             │
│         ├── DC/scale/rotation: QP capped at ≤22                      │
│         └── Others: user-specified QP                                │
│                                                                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            │  Bundle into iOS app:
                            │    • 17 of 20 MP4 files per group (skip 6,7,8)
                            │    • viewer_min_max.json
                            │    • group_info.json
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      iOS VIEWER SIDE                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  VideoProcessor.processVideos()                                      │
│    └── Loads 17 MP4 files (skips normals 6.mp4, 7.mp4, 8.mp4)       │
│    └── OpencvTest.processVideo() → [[UIImage]] (grayscale frames)    │
│                                                                      │
│  SplatCloud.__init__()                                               │
│    └── Creates 17 Metal R8Unorm textures from UIImages               │
│    └── Dispatches generateSplats compute kernel                      │
│                                                                      │
│  generateSplats (Metal kernel)                                       │
│    └── Reconstructs uint16 position from two uint8 channels          │
│    └── Dequantizes using minmax buffer                               │
│    └── Outputs Splat structs → rendering pipeline                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 14.2 Channel Mapping: Python ↔ iOS

The Python compression creates **20 video channels** (for SH degree 0). The iOS viewer loads only **17**, skipping channels 6, 7, 8 (normals, always zero):

| Python Channel | MP4 File | PLY Attribute | iOS `urls[]` Index | Metal Texture Index |
|---------------|----------|---------------|--------------------|--------------------|
| 0 | 0.mp4 | x low byte | 0 | `x0` (texture 0) |
| 1 | 1.mp4 | x high byte | 1 | `x1` (texture 1) |
| 2 | 2.mp4 | y low byte | 2 | `y0` (texture 2) |
| 3 | 3.mp4 | y high byte | 3 | `y1` (texture 3) |
| 4 | 4.mp4 | z low byte | 4 | `z0` (texture 4) |
| 5 | 5.mp4 | z high byte | 5 | `z1` (texture 5) |
| 6 | 6.mp4 | normal_x | **SKIPPED** | — |
| 7 | 7.mp4 | normal_y | **SKIPPED** | — |
| 8 | 8.mp4 | normal_z | **SKIPPED** | — |
| 9 | 9.mp4 | f_dc_0 | 6 | `fdc0` (texture 6) |
| 10 | 10.mp4 | f_dc_1 | 7 | `fdc1` (texture 7) |
| 11 | 11.mp4 | f_dc_2 | 8 | `fdc2` (texture 8) |
| 12 | 12.mp4 | opacity | 9 | `opacity` (texture 9) |
| 13 | 13.mp4 | scale_0 | 10 | `scale0` (texture 10) |
| 14 | 14.mp4 | scale_1 | 11 | `scale1` (texture 11) |
| 15 | 15.mp4 | scale_2 | 12 | `scale2` (texture 12) |
| 16 | 16.mp4 | rot_0 | 13 | `rot0` (texture 13) |
| 17 | 17.mp4 | rot_1 | 14 | `rot1` (texture 14) |
| 18 | 18.mp4 | rot_2 | 15 | `rot2` (texture 15) |
| 19 | 19.mp4 | rot_3 | 16 | `rot3` (texture 16) |

### 14.3 Min/Max Metadata Mapping

**Python** (`compress_to_png_full_sh.py`) writes `viewer_min_max.json` with **34 values** per frame (17 attributes × min,max):

```
Index: [0  1   2  3   4  5   6  7   8  9   10 11  12 13  14 15  16 17  18 19  20 21  22 23  24 25  26 27  28 29  30 31  32 33]
Attr:  [x_min x_max y_min y_max z_min z_max nx_min nx_max ny_min ny_max nz_min nz_max fdc0_min fdc0_max fdc1_min fdc1_max fdc2_min fdc2_max op_min op_max s0_min s0_max s1_min s1_max s2_min s2_max r0_min r0_max r1_min r1_max r2_min r2_max r3_min r3_max]
```

**iOS** (`extractNeededValues()` in `SplatSimpleView.swift`) extracts **28 of 34** values, **skipping normal indices 6–11**:

```swift
func extractNeededValues(info: [Double]) -> [Double] {
    var extractedValues = [Double]()
    extractedValues.append(contentsOf: info[0...5])     // Position: indices 0-5 → minmax[0-5]
    extractedValues.append(contentsOf: info[12...17])   // DC color: indices 12-17 → minmax[6-11]
    extractedValues.append(contentsOf: info.suffix(16)) // Op+Scale+Rot: indices 18-33 → minmax[12-27]
    return extractedValues
}
```

The resulting **28-value minmax array** maps to the Metal shader's `minmax` buffer indices as:

| minmax Index | Attribute | Used By |
|-------------|-----------|---------|
| 0, 1 | x_min, x_max | Position dequantization |
| 2, 3 | y_min, y_max | Position dequantization |
| 4, 5 | z_min, z_max | Position dequantization |
| 6, 7 | fdc0_min, fdc0_max | Color dequantization |
| 8, 9 | fdc1_min, fdc1_max | Color dequantization |
| 10, 11 | fdc2_min, fdc2_max | Color dequantization |
| 12, 13 | opacity_min, opacity_max | Opacity dequantization |
| 14, 15 | scale0_min, scale0_max | Scale dequantization |
| 16, 17 | scale1_min, scale1_max | Scale dequantization |
| 18, 19 | scale2_min, scale2_max | Scale dequantization |
| 20, 21 | rot0_min, rot0_max | Quaternion dequantization |
| 22, 23 | rot1_min, rot1_max | Quaternion dequantization |
| 24, 25 | rot2_min, rot2_max | Quaternion dequantization |
| 26, 27 | rot3_min, rot3_max | Quaternion dequantization |

### 14.4 Dequantization Formula Comparison

**Python** (`compress_to_png_full_sh.py` quantization / `decompress_from_png_full_sh.py` dequantization):

```python
# Quantization (compress side)
position: uint16 = ((value - min) / (max - min) * 65535).astype(np.uint16)
others:   uint8  = ((value - min) / (max - min) * 255).astype(np.uint8)

# Dequantization (decompress side)
position: value = (uint16 / 65535.0) * (max - min) + min
others:   value = (uint8  / 255.0)  * (max - min) + min
```

**Metal** (`generateSplats` kernel in `SplatShaders.metal`):

```metal
// Position: reconstruct uint16 from two R8Unorm textures, then dequantize
float xVal = float((uint(x1.read(gid).r * 255.0) << 8) + uint(x0.read(gid).r * 255.0));
xVal = xVal * (minmax[1] - minmax[0]) / 65535.0 + minmax[0] + init_pos[0];

// Others: R8Unorm texture auto-normalizes [0,255]→[0,1], so .r is already value/255
float f_dc0Val = fdc0.read(gid).r * (minmax[7] - minmax[6]) + minmax[6];
// Equivalent to: (uint8 / 255.0) * (max - min) + min
```

**Key difference**: The Metal shader adds `init_pos[i]` to position — a viewer-specific scene placement offset defined per dataset in `Models.swift` (e.g., `[0.0, 0.0, 0.0]` for actor2).

### 14.5 Activation Functions: Training vs. Viewer

During training (`train.py`), opacity and scale are stored in **raw/log space** in the PLY:
- Opacity: stored as inverse-sigmoid (logit) → compressed as-is → dequantized as-is
- Scale: stored as log-scale → compressed as-is → dequantized as-is

The **Metal vertex shader** (`splat_vertex`) applies the activations at render time:
- Scale: `exp()` is applied via `computeCov3D()` → `S[i][i] = mod * scale.xyz` (scale is already exponentiated in `SplatCloudC.mm` for PLY path, but for the video path, the raw log-scale values are used directly and exponentiation happens implicitly in the covariance computation since scale enters as `M = S * R` and `Sigma = M^T * M`)
- Opacity: the `generateSplats` kernel passes `opacityVal` directly to `color.a` — sigmoid is assumed to be baked in during the Python compression step (applied before quantization)

**Important note about the compression pipeline's treatment of activations:**

In `compress_to_png_full_sh.py`, the PLY attributes are read **as-is** (raw values from training output). The training code (`train.py`) stores:
- `opacity`: inverse-sigmoid (logit) space
- `scale`: log space
- `rotation`: raw quaternion components

These raw values get quantized and encoded. The iOS viewer's `generateSplats` kernel dequantizes them back to the same raw space. The vertex shader then interprets them assuming the scale values are in log space (the `computeCov3D` function uses them directly as scaling factors).

### 14.6 Group Structure Connection

**Python** (`compress_to_png_full_sh.py`) groups frames:

```python
# Groups of 20 frames each
group_size = 20
num_groups = ceil(total_frames / group_size)
# group_info.json: {"0": {"start": 0, "end": 19}, "1": {"start": 20, "end": 39}, ...}
```

**iOS** (`CameraControllerRenderer` in `SplatSimpleView.swift`) reads `group_info.json`:

```swift
// Build groupIndexList: [(groupId, startFrame, endFrame)]
let groupInfo = loadGroupInfo()  // e.g., [(0, 0, 19), (1, 20, 39), ...]

// Load group 0 synchronously on startup
processGroup(groupIndexList[0])

// Queue remaining groups in background
for group in groupIndexList[1...] {
    operationQueue.addOperation { self.processGroup(group) }
}
```

Each `processGroup()` call:
1. Calls `VideoProcessor.processVideos(groupIndex:)` → decodes 17 MP4 files for that group
2. For each frame in the group: creates a `SplatCloud` with the decoded textures + per-frame minmax
3. Appends to the `splatClouds` array for playback

### 14.7 Spatial Ordering (Morton Code)

The Python compression (`compress_to_png_full_sh.py`) applies **Morton (Z-order) sorting** to Gaussians before writing to images. This ensures spatially adjacent splats map to nearby pixels, improving H.264's spatial prediction efficiency.

The legacy PLY loader (`SplatCloudC.mm`) also implements Morton sorting:

```cpp
// SplatCloudC.mm, lines 183-200
simd_float3 rel = (p - minn) / (maxx - minn);
simd_float3 scaled = simd_make_float3(float((1 << 21) - 1) * rel.x, ...);
int xyz[3] = { (int)round(scaled.x), (int)round(scaled.y), (int)round(scaled.z) };

uint64_t code = 0;
for (int j = 0; j < 21; j++) {
    code |= ((uint64_t(xyz[0] & (1 << j))) << (2 * j + 0));
    code |= ((uint64_t(xyz[1] & (1 << j))) << (2 * j + 1));
    code |= ((uint64_t(xyz[2] & (1 << j))) << (2 * j + 2));
}
```

This 63-bit Morton code interleaves 21 bits per axis, creating a space-filling curve that preserves 3D locality in the 1D ordering.

---

## 15. AR Mode (ARSplatView)

The `ARSplatView` provides an **augmented reality** mode where the volumetric video is rendered on top of the live camera feed using ARKit.

### 15.1 Architecture

```
┌─────────────────────────────────────────────────┐
│                   ARSplatRenderer               │
├─────────────────────────────────────────────────┤
│  1. ARSession (ARWorldTrackingConfiguration)     │
│  2. ARPerspectiveCamera (pose from ARKit)        │
│  3. ARBackgroundRenderer (camera feed texture)   │
│  4. Satin.Renderer (splat rendering offscreen)   │
│  5. PostProcessor (blend splats + camera feed)   │
└─────────────────────────────────────────────────┘
```

### 15.2 Rendering Pipeline

Each draw call in `ARSplatRenderer` performs three passes:

1. **Background Pass**: `ARBackgroundRenderer` captures the AR camera feed (YCbCr pixel buffer from `ARFrame`) and renders it into a `backgroundTexture`

2. **Content Pass**: `Satin.Renderer` renders the current `SplatCloud` into a `contentTexture` (offscreen, potentially downsampled by `renderDownsample`)

3. **Compositing Pass**: `PostProcessor` with `BlendMaterial` blends the two textures together using a custom pipeline (alpha-blending splats over the camera feed)

### 15.3 Camera Integration

`ARPerspectiveCamera` extends Satin's `PerspectiveCamera` to pull view and projection matrices from `ARFrame.camera`:

```swift
// ARPerspectiveCamera.swift
override func update(_ commandBuffer: MTLCommandBuffer) {
    viewMatrix = frame.camera.viewMatrix(for: orientation)
    let arkitProjectionMatrix = frame.camera.projectionMatrix(
        for: orientation, viewportSize: mtkView.drawableSize, 
        zNear: CGFloat(near), zFar: CGFloat(far))
    setProjectionMatrixFromARKit(arkitProjectionMatrix)
}
```

### 15.4 Gesture Controls (DragHelper)

- **1-finger pan**: Rotates the splat cloud around Y-axis
- **2-finger pan**: Translates in the XZ plane
- **Pinch**: Uniform scale adjustment
- **Tap**: Triggers a "glow" effect (sets `isDragging` flag for 3 seconds)

---

## 16. Legacy PLY Loader (SplatCloudC)

The `SplatCloudC` class (`SplatCloudC.mm`) provides a direct PLY file loading path, used for static demo models (Mic, Lego) defined in `Models.swift`.

### 16.1 PLY Parsing

Reads binary PLY files with the `RichPoint<D>` struct layout matching the 3DGS training output format:

```cpp
template<int D>
struct RichPoint {
    Pos pos;            // MTLPackedFloat3 (12 bytes)
    float n[3];         // normals (unused, always 0 for trained models)
    SHs<D> shs;         // (D+1)^2 * 3 floats (e.g., 48 floats for SH3)
    float opacity;      // inverse-sigmoid space
    Scale scale;        // log space (3 floats)
    Rot rot;            // quaternion (4 floats)
};
```

### 16.2 Activation Application

Unlike the video path (where activations may already be baked), the PLY loader applies activations on the CPU:

```cpp
// Quaternion normalization
for (int j = 0; j < 4; j++)
    rot[k].rot[j] = points[i].rot.rot[j] / length;

// Scale exponentiation (log → linear)
for (int j = 0; j < 3; j++)
    scales[k].scale[j] = std::exp(points[i].scale.scale[j]);

// Opacity sigmoid activation (logit → probability)
opacities[k] = sigmoid(points[i].opacity);
```

### 16.3 SH Coefficient Reordering

The PLY stores SH coefficients in a specific interleaved order. `SplatCloudC` reorders them:

```cpp
// DC coefficients (band 0): direct copy
shs[k].shs[0] = points[i].shs.shs[0];  // R
shs[k].shs[1] = points[i].shs.shs[1];  // G
shs[k].shs[2] = points[i].shs.shs[2];  // B

// Higher bands: deinterleave RGB
for (int j = 1; j < SH_N; j++) {
    shs[k].shs[j*3 + 0] = points[i].shs.shs[(j-1) + 3];
    shs[k].shs[j*3 + 1] = points[i].shs.shs[(j-1) + SH_N + 2];
    shs[k].shs[j*3 + 2] = points[i].shs.shs[(j-1) + 2*SH_N + 1];
}
```

---

## 17. Utility Components

### 17.1 MetalBuffer

Type-safe wrapper around `MTLBuffer` providing:
- Typed subscript access (`buffer[i]`)
- Array initialization (`MetalBuffer(device:array:index:)`)
- Element count tracking

### 17.2 Metal+Extensions

Extension on `MTLPixelFormat` providing a computed property `srgb: Bool` that returns whether a given pixel format is an sRGB format. Used by `ARBackgroundRenderer` to configure color space handling.

### 17.3 DragHelper

Encapsulates gesture recognition for interactive 3D manipulation:
- `UIRotationGestureRecognizer` → roll rotation (unused)
- `UIPanGestureRecognizer` (1 touch) → Y-axis rotation
- `UIPanGestureRecognizer` (2 touches) → XZ translation
- `UIPinchGestureRecognizer` → uniform scaling

### 17.4 ARBackgroundRenderer

Extends Satin's `PostProcessor` to render the ARKit camera feed into a Metal texture. Handles:
- `CVPixelBuffer` → `CVMetalTexture` conversion (Y plane as `.r8Unorm`, CbCr plane as `.rg8Unorm`)
- Display orientation handling via `displayTransform(for:viewportSize:)`
- Texture coordinate adjustment for aspect-fill rendering

### 17.5 ARPerspectiveCamera

Subclass of Satin's `PerspectiveCamera` that derives view/projection matrices from the ARKit `ARFrame.camera` pose, with proper handling of device orientation rotations (portrait, landscape, etc.).

---

## 18. Dataset Configurations

Three datasets are pre-configured in `Models.swift`:

| Config Name | Dataset | Frames | Groups | Video Folder | Splat Count | Init Position |
|------------|---------|--------|--------|-------------|------------|---------------|
| `.ykxBoxing` | YKX Boxing | 630 | 63 | `ykx_boxing_long_qp15_380/` | ~380K | `[0, 0, 0]` |
| `.actor2Dancing4K` | 4K Actor2 | 40 | 2 | `4K_Actor2_Dancing_sh0_res4_qp15/` | ~80K | `[0, 0, 0]` |
| `.actor2DancingMedRes` | 4K Actor2 (med) | 40 | 2 | `4K_Actor2_Dancing_sh0_res2_qp15/` | ~80K | `[0, 0, 0]` |

**Entry point**: `MainViewController.swift` presents a `SplatChoiceView` with three buttons. Each navigates to `SplatSimpleView` with the corresponding `dataIndex`:

```
dataIndex 1 → .ykxBoxing       (630 frames, 63 groups × 10 frames each)
dataIndex 2 → .actor2Dancing4K (40 frames, 2 groups × 20 frames each)
dataIndex 3 → .actor2DancingMedRes (40 frames, 2 groups × 20 frames each)
```

---

## Appendix A: Complete File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `AppDelegate.swift` | 47 | Standard iOS app lifecycle (no custom logic) |
| `MainViewController.swift` | ~300 | SwiftUI entry view with dataset selection buttons |
| `SplatSimpleView.swift` | 742 | Core view: video loading, frame management, rendering loop, UI controls |
| `SplatCloud.swift` | 897 | Splat buffer management, GPU compute dispatch, sorting, instanced rendering |
| `SplatShaders.metal` | 529 | All GPU shaders: `generateSplats`, `splat_vertex`, `splat_fragment`, `splat_set_depths` |
| `ShaderTypes.h` | ~60 | Shared C structs: `Splat` (64 bytes), `Uniforms` |
| `BridgingHeader.h` | 17 | Exposes `ShaderTypes.h`, `OpencvTest.h`, `sort_splats()` to Swift |
| `ARSplatView.swift` | 345 | AR mode: ARKit session + splat rendering + camera feed compositing |
| `Models/Models.swift` | 187 | `SplatModelInfo`, `DatasetConfig`, dataset configurations |
| `OpencvTest.h` / `.mm` | 142 | OpenCV H.264 video decoding to grayscale `UIImage` arrays |
| `Utils/SplatCloudC.h` / `.mm` | 376 | Legacy PLY file parser with Morton-code sorting |
| `Utils/splat_utils.mm` | 51 | CPU-side `std::sort` + splat buffer reordering |
| `Utils/MetalBuffer.swift` | ~100 | Type-safe `MTLBuffer` wrapper |
| `Utils/DragHelper.swift` | ~130 | Gesture recognition for 3D object manipulation |
| `Utils/ARBackgroundRenderer.swift` | ~180 | ARKit camera feed → Metal texture rendering |
| `Utils/ARPerspectiveCamera.swift` | ~90 | ARKit-driven perspective camera |
| `Utils/Metal+Extensions.swift` | 294 | `MTLPixelFormat.srgb` computed property |

## Appendix B: Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                      External Frameworks                     │
├─────────────────────────────────────────────────────────────┤
│  opencv2.framework    │ H.264 video decoding (VideoCapture)  │
│  Satin (v13)          │ 3D rendering (Object, Renderer)      │
│  SatinCore            │ Core math types, geometry             │
│  Forge                │ Metal app lifecycle (ForgeView)       │
│  ARKit                │ World tracking, camera pose           │
│  Metal / MetalKit     │ GPU compute + rendering               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      App Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  AppDelegate                                                  │
│    └── MainViewController (SwiftUI)                           │
│         ├── SplatSimpleView (non-AR mode)                     │
│         │    ├── VideoProcessor (decodes H.264 → UIImages)    │
│         │    ├── CameraControllerRenderer (Forge.Renderer)    │
│         │    │    ├── SplatCloud × N (one per frame)           │
│         │    │    │    ├── Metal textures (17 R8Unorm)         │
│         │    │    │    ├── generateSplats compute kernel       │
│         │    │    │    ├── Splat buffer (64 bytes/splat)       │
│         │    │    │    ├── sort_splats (CPU std::sort)         │
│         │    │    │    └── splat_vertex/fragment shaders       │
│         │    │    └── PerspectiveCamera (Satin)                │
│         │    └── UI: Slider + FPS counter                     │
│         │                                                      │
│         └── ARSplatView (AR mode)                             │
│              ├── ARSplatRenderer (Forge.Renderer)             │
│              │    ├── ARSession (ARWorldTrackingConfiguration) │
│              │    ├── ARPerspectiveCamera                     │
│              │    ├── ARBackgroundRenderer                    │
│              │    ├── SplatCloud × N                          │
│              │    └── PostProcessor (blend splats + camera)   │
│              └── DragHelper (gestures)                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

*Report generated for VideoGS iOS Viewer - Real-Time Volumetric Video Streaming Using 2D Dynamic Gaussians*
