#if os(visionOS)
import Metal
import CompositorServices

@MainActor
final class VisionCompositorRenderer {
    private var layerRenderer: LayerRenderer?
    private var isRunning = false
    private var isPlaying = true
    private var device: MTLDevice?
    private var renderTask: Task<Void, Never>?

    func attach(layerRenderer: LayerRenderer, dataset: BinDatasetConfig, isPlaying: Bool) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        if device == nil {
            self.device = MTLCreateSystemDefaultDevice()
        }
        self.isPlaying = isPlaying

        // Dataset hook point for BIN loading + SplatCloud creation on visionOS.
        // This keeps iOS MTKView renderer fully separate.
        _ = dataset

        if renderTask == nil {
            renderTask = Task {
                await startRenderLoop(isPlaying: isPlaying)
            }
        }
    }

    func setPlaying(_ isPlaying: Bool) {
        self.isPlaying = isPlaying
    }

    func reloadDataset(_ dataset: BinDatasetConfig) {
        // TODO: wire BIN reload for visionOS compositor path.
        _ = dataset
    }

    func startRenderLoop(isPlaying: Bool) async {
        self.isPlaying = isPlaying
        guard !isRunning else { return }
        isRunning = true

        while isRunning {
            autoreleasepool {
                renderFrameIfPossible()
            }
            await Task.yield()
        }
    }

    func stopRenderLoop() {
        isRunning = false
        renderTask?.cancel()
        renderTask = nil
    }

    private func renderFrameIfPossible() {
        guard let layerRenderer else { return }
        guard isPlaying else { return }

        guard let frame = layerRenderer.queryNextFrame(),
              let timing = frame.predictTiming(),
              let drawable = frame.queryDrawable() else {
            return
        }

        frame.startUpdate()
        frame.endUpdate()

        frame.startSubmission()

        guard let commandQueue = device?.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPass = drawable.colorTextures.first else {
            frame.endSubmission()
            return
        }

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = renderPass
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)

        if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) {
            encoder.endEncoding()
        }

        commandBuffer.commit()

        drawable.encodePresent(commandBuffer: commandBuffer)
        frame.endSubmission()

        _ = timing
    }
}
#endif
