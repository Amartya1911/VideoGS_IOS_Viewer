#if os(visionOS)
import Metal
import CompositorServices
import ARKit
import _CompositorServices_SwiftUI
import QuartzCore
import Satin
import SatinCore
import simd

// MARK: - CompositorLayer Configuration

/// Configures the CompositorLayer — color format, depth mode, and layout.
struct ContentStageConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities,
                           configuration: inout LayerRenderer.Configuration) {
        print("[ContentStageConfiguration] makeConfiguration called")
        configuration.colorFormat = .bgra8Unorm_srgb
        configuration.depthFormat = .depth32Float
        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled

        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions =
            foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)
        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated

        print("[ContentStageConfiguration] layout=\(supportedLayouts.contains(.layered) ? "layered" : "dedicated"), foveation=\(foveationEnabled)")
    }
}

// MARK: - RenderDestinationProvider shim for visionOS

/// Supplies pixel-format info that `SplatCloud.init(binBuffers:…)` needs.
/// On iOS this role is filled by `MTKView`.
struct CompositorRenderDestination: RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { nil }
    var currentDrawable: CAMetalDrawable? { nil }
    var colorPixelFormat: MTLPixelFormat = .bgra8Unorm_srgb
    var depthStencilPixelFormat: MTLPixelFormat = .depth32Float
    var sampleCount: Int = 1
}

// MARK: - Compositor Renderer

/// Metal renderer that drives a visionOS CompositorLayer.
///
/// Call ``run(layerRenderer:dataset:isPlaying:)`` from the
/// `CompositorLayer` content closure — it **blocks** the compositor thread
/// for the layer's lifetime (required by CompositorServices).
final class VisionCompositorRenderer: @unchecked Sendable {

    // MARK: State (compositor thread only after run() entry)

    private var layerRenderer: LayerRenderer?
    private var isRunning = false
    private var isPlaying = true
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?

    // Splat data
    private var splatClouds: [SplatCloud] = []
    private var currentFrameIndex: Int = 0
    private var progressCounter: Int = 0
    private let stepInd: Int = 2  // draw calls per frame advance (matches iOS pipeline)

    // ARKit — provides the device anchor so the compositor knows
    // where to place rendered content in the user's physical space.
    private let arSession = ARKitSession()
    private let worldTracking = WorldTrackingProvider()

    // Cached anchor — on device, every drawable MUST have a device anchor
    // or the system silently drops the frame.  We cache the last valid one
    // so intermittent nil returns from queryDeviceAnchor don't cause
    // "Presenting a drawable without a device anchor" black-screen issues.
    private var lastDeviceAnchor: DeviceAnchor?

    // MARK: Public API

    /// Blocking entry point — call from the CompositorLayer closure.
    /// Does **not** return until ``stopRenderLoop()`` is called or the
    /// immersive space is dismissed.
    func run(layerRenderer: LayerRenderer,
             dataset: BinDatasetConfig,
             isPlaying: Bool) {

        print("[VisionCompositorRenderer] run() blocking entry point called")

        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device ?? MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.makeCommandQueue()
        self.isPlaying = isPlaying

        // Start world tracking concurrently.
        // We do this BEFORE loading the dataset to give ARKit time to initialize
        // while we process the BIN files.
        Task {
            do {
                print("[Compositor] Starting ARKit session...")
                try await arSession.run([worldTracking])
                print("[Compositor] ARKit world tracking active")
            } catch {
                print("[Compositor] ARKit error: \(error)")
            }
        }

        // Load BIN frames and build SplatClouds
        loadDataset(dataset)

        // Wait for ARKit world tracking to be running before entering
        // the render loop.  On a real device, presenting a drawable
        // without a device anchor causes the frame to be silently dropped
        // ("This drawable won't be presented").  Dataset loading above
        // already gave ARKit ~300ms of head start; spin-wait here if
        // it still needs more time.
        do {
            let waitStart = CACurrentMediaTime()
            while worldTracking.state != .running {
                Thread.sleep(forTimeInterval: 0.01) // 10ms poll
                let elapsed = CACurrentMediaTime() - waitStart
                if elapsed > 5.0 {
                    print("[Compositor] ARKit still not running after 5s — proceeding anyway")
                    break
                }
            }
            let waited = (CACurrentMediaTime() - waitStart) * 1000.0
            print(String(format: "[Compositor] ARKit wait: %.0fms (state: %@)",
                         waited, String(describing: worldTracking.state)))
        }

        // Blocking render loop.
        isRunning = true
        while isRunning {
            autoreleasepool {
                renderFrame()
            }
        }
    }

    func setPlaying(_ playing: Bool) { isPlaying = playing }

    func reloadDataset(_ dataset: BinDatasetConfig) {
        splatClouds.removeAll()
        currentFrameIndex = 0
        progressCounter = 0
        loadDataset(dataset)
    }

    func stopRenderLoop() { isRunning = false }

    // MARK: - Dataset Loading

    private func loadDataset(_ dataset: BinDatasetConfig) {
        guard let device else {
            print("[Compositor] No Metal device — cannot load dataset")
            return
        }

        let loadStart = CACurrentMediaTime()
        let loader = BinDataLoader(binFolder: dataset.binFolder, totalFrames: dataset.totalFrames)
        let allFrames = loader.loadAllFrames()
        let loadElapsed = (CACurrentMediaTime() - loadStart) * 1000.0

        guard !allFrames.isEmpty else {
            print("[Compositor] No BIN frames loaded!")
            return
        }

        print(String(format: "[Compositor] Loaded %d frames in %.1fms", allFrames.count, loadElapsed))

        var renderDest = CompositorRenderDestination()
        let buildStart = CACurrentMediaTime()

        for (index, frameData) in allFrames.enumerated() {
            do {
                let numPoints = frameData.numPoints

                let meansBuffer = frameData.means.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: frameData.means.count, options: .storageModeShared)
                }
                let scalesBuffer = frameData.scales.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: frameData.scales.count, options: .storageModeShared)
                }
                let quatsBuffer = frameData.quats.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: frameData.quats.count, options: .storageModeShared)
                }
                let colorsBuffer = frameData.colors.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: frameData.colors.count, options: .storageModeShared)
                }
                let opacitiesBuffer = frameData.opacities.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: frameData.opacities.count, options: .storageModeShared)
                }

                guard let m = meansBuffer, let s = scalesBuffer, let q = quatsBuffer,
                      let c = colorsBuffer, let o = opacitiesBuffer else {
                    print("[Compositor] Failed to create Metal buffers for frame \(index)")
                    continue
                }

                let binBuffers = (means: m, scales: s, quats: q, colors: c, opacities: o)

                guard let splatCloud = try SplatCloud(binBuffers: binBuffers,
                                                     numPoints: numPoints,
                                                     renderDestination: renderDest) else {
                    print("[Compositor] Failed to create SplatCloud for frame \(index)")
                    continue
                }

                // Match the iOS viewer's model transform:
                // Note: User reported -90° X rotation resulted in "lying down" / "backside" view.
                // Assuming native data is Y-up or similar. We rotate 180° Y to face the user.
                // We also lower it (y=-1.0) so it stands on the "floor" relative to eye level (y=0).
                let rotate = simd_quatf(angle: Float.pi, axis: .init(x: 1, y: 0, z: 0))
                splatCloud.orientation = rotate
                splatCloud.scale = .init(repeating: 0.35)
                splatCloud.position = SIMD3<Float>(0, 1.8, -1.75)

                splatClouds.append(splatCloud)
            } catch {
                print("[Compositor] Error creating SplatCloud for frame \(index): \(error)")
            }
        }

        let buildElapsed = (CACurrentMediaTime() - buildStart) * 1000.0
        print(String(format: "[Compositor] Built %d SplatClouds in %.1fms", splatClouds.count, buildElapsed))
    }

    // MARK: - Frame Rendering

    private var debugFrameCounter: Int = 0

    private func renderFrame() {
        guard let layerRenderer else { return }

        // queryNextFrame() blocks until the compositor needs a new frame.
        guard let frame    = layerRenderer.queryNextFrame(),
              let timing   = frame.predictTiming(),
              let drawable = frame.queryDrawable() else { return }

        // --- Update phase: supply device anchor ---
        frame.startUpdate()

        // Use CACurrentMediaTime() for the anchor query.
        // (timing.presentationTime is a Clock.Instant and not directly convertible to TimeInterval here)
        if worldTracking.state == .running {
            if let freshAnchor = worldTracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime()) {
                lastDeviceAnchor = freshAnchor
            }
        }

        drawable.deviceAnchor = lastDeviceAnchor
        frame.endUpdate()

        // --- Submission phase ---
        frame.startSubmission()

        guard let commandQueue,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            frame.endSubmission()
            return
        }

        // ------------------------------------------------------------------
        // Prepare per-view textures.
        //
        // On the real device the compositor uses a `.layered` layout where
        // both eyes share a single 2D-array texture.  The vertex shader does
        // NOT output [[render_target_array_index]], so when we used to set
        // renderTargetArrayLength = 2 and render once, ALL primitives went
        // to layer 0 (left eye) only.  The compositor then rejected the
        // frame because layer 1 was empty → permanent black screen.
        //
        // Fix: render each view in its OWN render pass that targets a 2D
        // texture view of the correct array slice (layered) or the correct
        // dedicated texture.
        // ------------------------------------------------------------------

        let viewCount = drawable.views.count
        let isLayered = drawable.colorTextures[0].textureType == .type2DArray

        // Build per-view color & depth texture references
        var colorTexPerView: [MTLTexture] = []
        var depthTexPerView: [MTLTexture?] = []

        if isLayered {
            let colorArray = drawable.colorTextures[0]
            let depthArray = drawable.depthTextures.first
            for i in 0..<viewCount {
                if let cSlice = colorArray.makeTextureView(
                    pixelFormat: colorArray.pixelFormat,
                    textureType: .type2D,
                    levels: 0..<1, slices: i..<(i + 1)
                ) {
                    colorTexPerView.append(cSlice)
                }
                if let dArr = depthArray,
                   let dSlice = dArr.makeTextureView(
                    pixelFormat: dArr.pixelFormat,
                    textureType: .type2D,
                    levels: 0..<1, slices: i..<(i + 1)
                ) {
                    depthTexPerView.append(dSlice)
                } else {
                    depthTexPerView.append(nil)
                }
            }
        } else {
            // Dedicated — one texture per view
            for i in 0..<viewCount {
                if i < drawable.colorTextures.count {
                    colorTexPerView.append(drawable.colorTextures[i])
                }
                if i < drawable.depthTextures.count {
                    depthTexPerView.append(drawable.depthTextures[i])
                } else {
                    depthTexPerView.append(nil)
                }
            }
        }

        // Fallback: if texture-view creation failed, use the raw textures
        // with the old renderTargetArrayLength approach (at least one eye).
        if colorTexPerView.isEmpty {
            colorTexPerView.append(drawable.colorTextures[0])
            depthTexPerView.append(drawable.depthTextures.first)
        }

        // ------------------------------------------------------------------
        // Render
        // ------------------------------------------------------------------
        if !splatClouds.isEmpty {
            let frameIdx = currentFrameIndex % splatClouds.count
            let splatCloud = splatClouds[frameIdx]

            let modelMatrix = splatCloud.worldMatrix

            // ---------- Render each view (eye) ----------
            for viewIndex in 0..<min(viewCount, colorTexPerView.count) {
                let view = drawable.views[viewIndex]
                let projectionMatrix = Self.computeProjectionFromTangents(
                    view.tangents, nearZ: 0.01, farZ: 100.0)

                // -------- Per-eye view matrix --------
                let viewTransform: simd_float4x4
                let cameraWorldPos: SIMD3<Float>

                if let anchor = lastDeviceAnchor {
                    let anchorTransform = anchor.originFromAnchorTransform
                    // Multiply anchor transform by the eye's offset (view.transform)
                    let eyeWorldTransform = simd_mul(anchorTransform, view.transform)
                    viewTransform = simd_inverse(eyeWorldTransform)
                    cameraWorldPos = SIMD3<Float>(eyeWorldTransform.columns.3.x,
                                                  eyeWorldTransform.columns.3.y,
                                                  eyeWorldTransform.columns.3.z)
                } else {
                    viewTransform = matrix_identity_float4x4
                    cameraWorldPos = SIMD3<Float>(0, 0, 0)
                }

                let modelViewMatrix = simd_mul(viewTransform, modelMatrix)
                let cameraPos       = simd_float4(cameraWorldPos, 1.0)
                let cameraPosOrig   = simd_mul(simd_inverse(modelMatrix), cameraPos)

                let viewWidth  = Float(view.textureMap.viewport.width)
                let viewHeight = Float(view.textureMap.viewport.height)

                let tan_fovx = 1.0 / projectionMatrix[0][0]
                let tan_fovy = 1.0 / projectionMatrix[1][1]
                let focal_x  = viewWidth  / (2.0 * tan_fovx)
                let focal_y  = viewHeight / (2.0 * tan_fovy)

                let uni = Uniforms(
                    projection_matrix: projectionMatrix,
                    model_matrix: modelMatrix,
                    model_view_matrix: modelViewMatrix,
                    inv_model_view_matrix: simd_inverse(modelViewMatrix),
                    camera_pos: cameraPos,
                    camera_pos_orig: cameraPosOrig,
                    viewport_width: viewWidth,
                    viewport_height: viewHeight,
                    focal_x: focal_x,
                    focal_y: focal_y,
                    tan_fovx: tan_fovx,
                    tan_fovy: tan_fovy,
                    drag_alpha: 0.0,
                    time: Float(CACurrentMediaTime())
                )

                splatCloud.updateUniforms(uniforms: uni)

                // -------- diagnostic logging (first 3 actual frames) --------
                if debugFrameCounter < 3 {
                    let mcClip = simd_mul(projectionMatrix,
                                         simd_mul(modelViewMatrix,
                                                  simd_float4(0, 0, 0, 1)))
                    let ndc = mcClip.w != 0
                        ? SIMD3<Float>(mcClip.x / mcClip.w,
                                       mcClip.y / mcClip.w,
                                       mcClip.z / mcClip.w)
                        : SIMD3<Float>(0, 0, 0)

                    print("[Compositor] Frame \(debugFrameCounter+1) eye\(viewIndex): " +
                          "anchor=\(lastDeviceAnchor != nil), " +
                          "views=\(viewCount), layered=\(isLayered), " +
                          "vp=\(viewWidth)x\(viewHeight), " +
                          "focal=(\(focal_x),\(focal_y)), " +
                          "tan_fov=(\(tan_fovx),\(tan_fovy)), " +
                          "cam=\(cameraWorldPos), " +
                          "modelPos=\(splatCloud.position), " +
                          "NDC=\(ndc), " +
                          "tangents=\(view.tangents), " +
                          "numSplats=\(splatCloud.numPoints), " +
                          "colTexType=\(drawable.colorTextures[0].textureType.rawValue), " +
                          "depFmt=\(drawable.depthTextures.first?.pixelFormat.rawValue ?? 0)")
                }

                // -------- build per-view render pass --------
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture    = colorTexPerView[viewIndex]
                rpd.colorAttachments[0].loadAction  = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 0)

                if let dTex = depthTexPerView[viewIndex] {
                    rpd.depthAttachment.texture    = dTex
                    rpd.depthAttachment.loadAction  = .clear
                    rpd.depthAttachment.storeAction = .store
                    rpd.depthAttachment.clearDepth  = 1.0
                }

                rpd.renderTargetArrayLength = 0  // single 2D texture per pass

                if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
                    let vp = view.textureMap.viewport
                    encoder.setViewport(MTLViewport(
                        originX: vp.originX,
                        originY: vp.originY,
                        width:   vp.width,
                        height:  vp.height,
                        znear: 0.0, zfar: 1.0))

                    splatCloud.render(renderEncoder: encoder)
                    encoder.endEncoding()
                }
            }

            // Advance animation once per composite frame (not per eye)
            if isPlaying {
                progressCounter += 1
                if progressCounter % stepInd == 0 {
                    currentFrameIndex = (currentFrameIndex + 1) % splatClouds.count
                }
            }

            if debugFrameCounter < 3 { debugFrameCounter += 1 }

        } else {
            // No splat data — clear all views to dark blue
            for viewIndex in 0..<colorTexPerView.count {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture    = colorTexPerView[viewIndex]
                rpd.colorAttachments[0].loadAction  = .clear
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0.05, 0.05, 0.25, 1.0)
                rpd.renderTargetArrayLength = 0

                if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
                    encoder.endEncoding()
                }
            }
        }

        drawable.encodePresent(commandBuffer: commandBuffer)
        commandBuffer.commit()
        frame.endSubmission()
    }
}

// MARK: - Projection Matrix from view tangents

/// Builds a reverse-Z perspective projection matrix from tangent half-angles.
/// Apple's `view.tangents` components are **positive magnitudes**:
///   x = left extent, y = right extent, z = bottom extent, w = top extent.
/// We negate left and bottom to get signed frustum boundaries.
private extension VisionCompositorRenderer {
    static func computeProjectionFromTangents(_ tangents: SIMD4<Float>, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let left   = -tangents.x  // negate: left is negative
        let right  =  tangents.y  // positive
        let bottom = -tangents.z  // negate: bottom is negative
        let top    =  tangents.w  // positive

        let width  = right - left    // e.g. 1.0 - (-1.0) = 2.0
        let height = top - bottom    // e.g. 0.75 - (-0.75) = 1.5

        // Reverse-Z projection (matches iOS / Metal convention, depth 1→0)
        let col0 = SIMD4<Float>(2.0 / width, 0.0, 0.0, 0.0)
        let col1 = SIMD4<Float>(0.0, 2.0 / height, 0.0, 0.0)
        let col2 = SIMD4<Float>((right + left) / width,
                                (top + bottom) / height,
                                nearZ / (farZ - nearZ),
                                -1.0)
        let col3 = SIMD4<Float>(0.0,
                                0.0,
                                (farZ * nearZ) / (farZ - nearZ),
                                0.0)

        return simd_float4x4(col0, col1, col2, col3)
    }
}
#endif
