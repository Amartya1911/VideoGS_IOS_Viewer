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
        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled

        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions =
            foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)
        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
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

        // Load BIN frames and build SplatClouds
        loadDataset(dataset)

        // Start world tracking concurrently.
        Task {
            do {
                try await arSession.run([worldTracking])
                print("[Compositor] ARKit world tracking active")
            } catch {
                print("[Compositor] ARKit error: \(error)")
            }
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
                // -90° rotation around X (converts Z-up Gaussian data to Y-up),
                // scale 0.35 to fit the scene, and position 2m in front of the user.
                let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
                splatCloud.orientation = rotate
                splatCloud.scale = .init(repeating: 0.35)
                splatCloud.position = SIMD3<Float>(0, 0, -2)

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
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime())
        drawable.deviceAnchor = deviceAnchor
        frame.endUpdate()

        // --- Submission phase ---
        frame.startSubmission()

        guard let commandQueue,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            frame.endSubmission()
            return
        }

        // If we have splat clouds, render the current frame's splats
        if !splatClouds.isEmpty {
            let frameIdx = currentFrameIndex % splatClouds.count
            let splatCloud = splatClouds[frameIdx]

            // Always build projection from compositor view tangents
            let view = drawable.views[0]
            let projectionMatrix = Self.computeProjectionFromTangents(view.tangents, nearZ: 0.01, farZ: 100.0)

            // View matrix: from device anchor, or identity for simulator
            let viewTransform: simd_float4x4
            let cameraWorldPos: SIMD3<Float>

            if let anchor = deviceAnchor {
                let anchorTransform = anchor.originFromAnchorTransform
                viewTransform = simd_inverse(anchorTransform)
                cameraWorldPos = SIMD3<Float>(anchorTransform.columns.3.x,
                                              anchorTransform.columns.3.y,
                                              anchorTransform.columns.3.z)
            } else {
                // Before ARKit starts: identity (camera at origin, looking -Z)
                // SplatCloud already positioned at (0,0,-2), so it's in view.
                viewTransform = matrix_identity_float4x4
                cameraWorldPos = SIMD3<Float>(0, 0, 0)
            }

            // Build viewport dimensions from first drawable view
            let viewWidth = Float(view.textureMap.viewport.width)
            let viewHeight = Float(view.textureMap.viewport.height)

            let modelMatrix = splatCloud.worldMatrix

            let modelViewMatrix = simd_mul(viewTransform, modelMatrix)

            let tan_fovx = 1.0 / projectionMatrix[0][0]
            let tan_fovy = 1.0 / projectionMatrix[1][1]
            let focal_x = viewWidth / (2.0 * tan_fovx)
            let focal_y = viewHeight / (2.0 * tan_fovy)

            let cameraPos = simd_float4(cameraWorldPos, 1.0)
            let cameraPosOrig = simd_mul(simd_inverse(modelMatrix), cameraPos)

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

            // Debug print first few frames
            if debugFrameCounter < 3 {
                debugFrameCounter += 1
                print("[Compositor] Frame \(debugFrameCounter): anchor=\(deviceAnchor != nil), " +
                      "viewport=\(viewWidth)x\(viewHeight), " +
                      "focal=(\(focal_x),\(focal_y)), " +
                      "tan_fov=(\(tan_fovx),\(tan_fovy)), " +
                      "cameraPos=\(cameraWorldPos), " +
                      "numSplats=\(splatCloud.numPoints), " +
                      "tangents=\(view.tangents)")
            }

            // Build render pass descriptor
            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = drawable.colorTextures[0]
            rpd.colorAttachments[0].loadAction = .clear
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0)

            if let depthTex = drawable.depthTextures.first {
                rpd.depthAttachment.texture = depthTex
                rpd.depthAttachment.loadAction = .clear
                rpd.depthAttachment.storeAction = .store
                rpd.depthAttachment.clearDepth = 1.0
            }

            rpd.renderTargetArrayLength = drawable.views.count

            if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
                let vp = view.textureMap.viewport
                encoder.setViewport(MTLViewport(
                    originX: vp.originX,
                    originY: vp.originY,
                    width: vp.width,
                    height: vp.height,
                    znear: 0.0, zfar: 1.0))

                splatCloud.render(renderEncoder: encoder)
                encoder.endEncoding()
            }

            // Advance animation if playing
            if isPlaying {
                progressCounter += 1
                if progressCounter % stepInd == 0 {
                    currentFrameIndex = (currentFrameIndex + 1) % splatClouds.count
                }
            }

        } else {
            // No splat data — clear to dark blue so user knows renderer is alive
            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = drawable.colorTextures[0]
            rpd.colorAttachments[0].loadAction = .clear
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.25, 1.0)
            rpd.renderTargetArrayLength = drawable.views.count

            if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
                encoder.endEncoding()
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
