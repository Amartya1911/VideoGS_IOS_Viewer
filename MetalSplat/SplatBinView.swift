//
//  SplatBinView.swift
//  MetalSplat
//
//  Bin-based viewer: loads raw float32 .bin files per frame,
//  constructs Splat structs on GPU, reuses existing sort + render pipeline.
//

import SwiftUI
import UIKit
import Metal
import MetalKit
import Forge
import Satin
import SatinCore


// MARK: - Performance Logger

/// Stores one row of per-draw-call timing data.
struct BinPerfSample {
    let frameIndex: Int         // which splat frame is showing
    let drawCallIndex: Int      // monotonic draw counter
    let preDrawMs: Double       // draw() entry -> drawStart
    let cpuStageMs: Double      // drawStart -> drawEnd (all CPU work inside draw())
    let gpuQueueWaitMs: Double  // drawEnd -> gpuStartTime (CPU/GPU queue wait)
    let gpuAndCallbackMs: Double // gpuStartTime -> completion callback entry
    let endToEndMs: Double      // CPU start of draw() -> completion handler entry
    let displayFPS: Int         // snapshot of on-screen CADisplayLink FPS counter
    let fullFrameMs: Double     // draw() entry -> completion handler entry
}

/// Accumulates timing samples in memory, dumps to CSV on demand.
/// - Debug builds: also prints every Nth sample to console.
/// - Release builds: silent until exportToCSV().
class BinPerfLogger {
    private(set) var samples: [BinPerfSample] = []
    private var debugPrintInterval: Int = 60  // print every N samples in debug
    private var hasExported = false
    private let lock = NSLock()
    private var inFlightSamples: Int = 0
    
    func beginInFlightSample() {
        lock.lock()
        inFlightSamples += 1
        lock.unlock()
    }

    /// Record one completed sample and mark one in-flight draw as finished.
    /// In DEBUG prints to console periodically.
    func recordCompleted(_ sample: BinPerfSample) {
        lock.lock()
        samples.append(sample)
        inFlightSamples = max(0, inFlightSamples - 1)
        let sampleCount = samples.count
        lock.unlock()
        
        #if DEBUG
        if sampleCount % debugPrintInterval == 1 {
            NSLog("[BinPerf] draw#%d  frame=%d  preDraw=%.2fms  cpuStage=%.2fms  gpuQueueWait=%.2fms  gpuAndCallback=%.2fms  endToEnd=%.2fms  displayFPS=%d  fullFrame=%.2fms",
                  sample.drawCallIndex, sample.frameIndex,
                sample.preDrawMs,
                  sample.cpuStageMs, sample.gpuQueueWaitMs,
                  sample.gpuAndCallbackMs,
                  sample.endToEndMs,
                  sample.displayFPS,
                  sample.fullFrameMs)
        }
        #endif
    }

    /// Wait briefly for command-buffer completion handlers so final frames aren't dropped on export.
    private func waitForPendingSamples(maxWaitMs: Int = 500) {
        let start = CACurrentMediaTime()
        while true {
            lock.lock()
            let pending = inFlightSamples
            lock.unlock()

            if pending == 0 { return }

            let elapsedMs = (CACurrentMediaTime() - start) * 1000.0
            if elapsedMs >= Double(maxWaitMs) {
                NSLog("[BinPerf] Export timeout: %d samples still in-flight", pending)
                return
            }

            usleep(2_000) // 2ms poll
        }
    }
    
    /// Also store a one-time init record (bin loading + SplatCloud creation).
    var initLoadMs: Double = 0
    var initSplatCloudBuildMs: Double = 0
    var numFrames: Int = 0
    var numSplatsPerFrame: Int = 0
    
    /// Export all samples to a CSV file in the app's Documents directory.
    /// Returns the file URL on success.
    @discardableResult
    func exportToCSV() -> URL? {
        waitForPendingSamples()

        guard !hasExported else {
            NSLog("[BinPerf] CSV already exported, skipping duplicate")
            return nil
        }

        lock.lock()
        let snapshot = samples
        lock.unlock()

        guard !snapshot.isEmpty else {
            NSLog("[BinPerf] No samples to export")
            return nil
        }
        hasExported = true
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            NSLog("[BinPerf] ERROR: cannot find Documents directory")
            return nil
        }
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let stamp = dateFormatter.string(from: Date())
        let filename = "bin_perf_\(stamp).csv"
        let fileURL = docs.appendingPathComponent(filename)
        
        var csv = "# BinPerfLog  frames=\(numFrames)  splats/frame=\(numSplatsPerFrame)  initLoadMs=\(String(format: "%.2f", initLoadMs))  initBuildMs=\(String(format: "%.2f", initSplatCloudBuildMs))\n"
        csv += "draw_call,frame_index,pre_draw_ms,cpu_stage_ms,gpu_queue_wait_ms,gpu_and_callback_ms,end_to_end_ms,display_fps,full_frame_ms\n"
        
        for s in snapshot {
            csv += String(format: "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f\n",
                          s.drawCallIndex, s.frameIndex,
                          s.preDrawMs,
                          s.cpuStageMs, s.gpuQueueWaitMs,
                          s.gpuAndCallbackMs,
                          s.endToEndMs,
                          s.displayFPS,
                          s.fullFrameMs)
        }
        
        do {
            try csv.write(to: fileURL, atomically: true, encoding: .utf8)
            NSLog("[BinPerf] CSV exported: %@ (%d samples)  →  Files > On My iPhone > MetalSplat", fileURL.lastPathComponent, snapshot.count)
            return fileURL
        } catch {
            NSLog("[BinPerf] ERROR writing CSV: %@", error.localizedDescription)
            return nil
        }
    }
}


// MARK: - Bin Data Loader

/// Holds raw float32 data for a single frame's Gaussian attributes.
struct BinFrameData {
    let means: Data       // N×3 float32
    let scales: Data      // N×3 float32
    let quats: Data       // N×4 float32
    let colors: Data      // N×3 float32
    let opacities: Data   // N×1 float32
    
    var numPoints: Int {
        return means.count / (3 * MemoryLayout<Float>.size)
    }
}

/// Loads all bin frame data from the app bundle at startup.
class BinDataLoader {
    
    let binFolder: String
    let totalFrames: Int
    
    init(binFolder: String, totalFrames: Int) {
        self.binFolder = binFolder
        self.totalFrames = totalFrames
    }
    
    /// Discover available frame directories in the bundle.
    func enumerateFrames() -> [Int] {
        var frames = [Int]()
        for i in 0..<totalFrames {
            let testPath = "\(binFolder)/\(i)/means3d"
            if Bundle.main.url(forResource: testPath, withExtension: "bin") != nil {
                frames.append(i)
            }
        }
        frames.sort()
        return frames
    }
    
    /// Load a single frame's bin files from the bundle.
    func loadFrame(_ frameIndex: Int) -> BinFrameData? {
        let base = "\(binFolder)/\(frameIndex)"
        
        guard let meansURL = Bundle.main.url(forResource: "\(base)/means3d", withExtension: "bin"),
              let scalesURL = Bundle.main.url(forResource: "\(base)/scales3d", withExtension: "bin"),
              let quatsURL = Bundle.main.url(forResource: "\(base)/quats3d", withExtension: "bin"),
              let colorsURL = Bundle.main.url(forResource: "\(base)/colors3d", withExtension: "bin"),
              let opacitiesURL = Bundle.main.url(forResource: "\(base)/opacities3d", withExtension: "bin") else {
            print("BinDataLoader: Missing bin files for frame \(frameIndex)")
            return nil
        }
        
        do {
            let means = try Data(contentsOf: meansURL)
            let scales = try Data(contentsOf: scalesURL)
            let quats = try Data(contentsOf: quatsURL)
            let colors = try Data(contentsOf: colorsURL)
            let opacities = try Data(contentsOf: opacitiesURL)
            return BinFrameData(means: means, scales: scales, quats: quats,
                                colors: colors, opacities: opacities)
        } catch {
            print("BinDataLoader: Error reading bin files for frame \(frameIndex): \(error)")
            return nil
        }
    }
    
    /// Load all frames into memory.
    func loadAllFrames() -> [BinFrameData] {
        let frameIndices = enumerateFrames()
        print("BinDataLoader: Found \(frameIndices.count) frames in \(binFolder)")
        
        var allFrames = [BinFrameData]()
        for i in frameIndices {
            if let data = loadFrame(i) {
                allFrames.append(data)
                print("  Frame \(i): \(data.numPoints) splats, \(data.means.count + data.scales.count + data.quats.count + data.colors.count + data.opacities.count) bytes")
            }
        }
        return allFrames
    }
}


// MARK: - Bin Renderer

class BinCameraControllerRenderer: Forge.Renderer {
    
    let model: SplatModelInfo
    var progress: RendererProgress
    
    var splatClouds: [SplatCloud] = []
    var currentFrameNum: Int = 0
    
    var renderTime: Int = 0
    var isPaused: Bool = false
    
    let stepInd: Int = 2  // draw calls per frame (same as video pipeline)
    
    let binConfig: BinDatasetConfig
    weak var fpsCounter: FPSCounter?
    
    let perfLogger = BinPerfLogger()
    var drawCallCounter: Int = 0
    
    func togglePause(_ isPaused: Bool) {
        self.isPaused = isPaused
    }
    
    init(model: SplatModelInfo, progress: RendererProgress, binConfig: BinDatasetConfig, fpsCounter: FPSCounter?) {
        self.model = model
        self.progress = progress
        self.binConfig = binConfig
        self.fpsCounter = fpsCounter
        super.init()
    }
    
    lazy var scene = Object("Scene", [])
    
    lazy var context: Context = .init(device, sampleCount, colorPixelFormat, depthPixelFormat, stencilPixelFormat)
    
    lazy var camera: PerspectiveCamera = {
        let pos = SIMD3<Float>(10.0, 10.0, 10.0)
        let camera = PerspectiveCamera(position: pos, near: 0.01, far: 100.0)
        camera.aspect = 0.9429056
        camera.orientation = simd_quatf(real: 1.0, imag: SIMD3<Float>(0.0, 0.0, 0.0))
        camera.fov = 45.0
        camera.viewMatrix = simd_float4x4([[-0.3292983, -0.17486514, 0.9278904, 0.0], [0.92797965, -0.24143358, 0.28383055, 0.0], [0.17439224, 0.9545303, 0.24177541, -0.0], [-0.18122181, 0.18167843, -0.97597855, 1.0000001]])
        camera.worldPosition = SIMD3<Float>(0.8776981, 0.48904794, 0.09415415)
        camera.projectionMatrix = simd_float4x4([[2.5603979, 0.0, 0.0, 0.0], [0.0, 2.4142134, 0.0, 0.0], [0.0, 0.0, 0.000100016594, -1.0], [0.0, 0.0, 0.010001, 0.0]])
        camera.scale = SIMD3<Float>(1.0000015, 1.0000017, 1.0000018)
        camera.localMatrix = simd_float4x4([[1.0000015, 0.0, 0.0, 0.0], [0.0, 1.0000017, 0.0, 0.0], [0.0, 0.0, 1.0000018, 0.0], [0.0, 0.0, 0.9759802, 1.0]])
        camera.worldMatrix = simd_float4x4([[-0.32929954, 0.92798316, 0.1743929, 0.0], [-0.17486589, -0.24143456, 0.9545341, 0.0], [0.9278944, 0.28383178, 0.24177642, 0.0], [0.8776981, 0.48904794, 0.09415415, 1.0]])
        camera.worldOrientation = simd_quatf(real: 0.409586, imag: SIMD3<Float>(0.40937737, 0.45991546, 0.67314726))
        return camera
    }()
    
    lazy var cameraController: PerspectiveCameraController = .init(camera: camera, view: mtkView)
    lazy var renderer: Satin.Renderer = .init(context: context)
    
    override func setupMtkView(_ metalKitView: MTKView) {
        
        metalKitView.depthStencilPixelFormat = .invalid
        metalKitView.backgroundColor = UIColor.white
        metalKitView.autoResizeDrawable = true
        metalKitView.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0)
        metalKitView.preferredFramesPerSecond = 60
        
        metalKitView.drawableSize = mtkView.drawableSize.applying(
            CGAffineTransform(scaleX: 1.0 / CGFloat(model.rendererDownsample),
                              y: 1.0 / CGFloat(model.rendererDownsample))
        )
        print("Bin viewer drawable size: ", mtkView.drawableSize)
        
        renderer.clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0)
        
        // Load all bin data at startup
        let loadStart = CACurrentMediaTime()
        let loader = BinDataLoader(binFolder: binConfig.binFolder, totalFrames: binConfig.totalFrames)
        let allFrames = loader.loadAllFrames()
        let loadElapsed = (CACurrentMediaTime() - loadStart) * 1000.0
        
        guard !allFrames.isEmpty else {
            print("BinRenderer: No frames loaded!")
            return
        }
        
        self.currentFrameNum = allFrames.count
        print("BinRenderer: Loaded \(currentFrameNum) frames in \(String(format: "%.1f", loadElapsed))ms")
        print("BinRenderer: Building \(currentFrameNum) SplatClouds on GPU...")
        
        let buildStart = CACurrentMediaTime()
        
        // Build SplatCloud for each frame using GPU kernel
        for (index, frameData) in allFrames.enumerated() {
            do {
                let numPoints = frameData.numPoints
                
                // Create Metal buffers from raw Data
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
                    print("BinRenderer: Failed to create Metal buffers for frame \(index)")
                    continue
                }
                
                let binBuffers = (means: m, scales: s, quats: q, colors: c, opacities: o)
                
                guard let splatCloud = try SplatCloud(binBuffers: binBuffers,
                                                     numPoints: numPoints,
                                                     renderDestination: metalKitView) else {
                    print("BinRenderer: Failed to create SplatCloud for frame \(index)")
                    continue
                }
                
                splatCloud.orientation = model.initialOrientation
                splatCloud.scale = .init(repeating: model.initialScale)
                
                self.splatClouds.append(splatCloud)
                
                // Metal buffers for this frame's raw data are released here
                // (meansBuffer, scalesBuffer, etc. go out of scope)
                
            } catch {
                print("BinRenderer: Error creating SplatCloud for frame \(index): \(error)")
            }
        }
        
        let buildElapsed = (CACurrentMediaTime() - buildStart) * 1000.0
        print("BinRenderer: Built \(splatClouds.count) SplatClouds in \(String(format: "%.1f", buildElapsed))ms")
        self.currentFrameNum = splatClouds.count
        
        // Record init stats
        perfLogger.initLoadMs = loadElapsed
        perfLogger.initSplatCloudBuildMs = buildElapsed
        perfLogger.numFrames = currentFrameNum
        perfLogger.numSplatsPerFrame = splatClouds.first?.numPoints ?? 0
    }
    
    override func setup() {
        print("BinRenderer: setup")
        scene.attach(cameraController.target)
        
        if !splatClouds.isEmpty {
            scene.add(splatClouds[0])
        }
    }
    
    deinit {
        perfLogger.exportToCSV()
        cameraController.disable()
    }
    
    override func update() {
        cameraController.update()
    }
    
    override func draw(_ view: MTKView, _ commandBuffer: MTLCommandBuffer) {
        let frameStart = CACurrentMediaTime()
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
        guard currentFrameNum > 0 else { return }
        
        let drawStart = CACurrentMediaTime()

        // Frame index shown for this draw (before optional swap for next draw)
        let visibleFrame = (progress.progressValue / stepInd) % currentFrameNum
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0)
        
        renderer.draw(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            scene: scene,
            camera: camera
        )
        
        if !isPaused {
            progress.progressValue += 1
            
            // Circular playback: wrap around
            if progress.progressValue >= stepInd * currentFrameNum {
                progress.progressValue = 0
            }
            
            if progress.progressValue % stepInd == 0 {
                let newFrameIndex = progress.progressValue / stepInd
                let oldFrameIndex = (newFrameIndex - 1 + currentFrameNum) % currentFrameNum
                
                // Swap SplatCloud in scene
                scene.remove(splatClouds[oldFrameIndex])
                scene.add(splatClouds[newFrameIndex])
            }
        }

        let drawEnd = CACurrentMediaTime()
        let cpuStageMs = (drawEnd - drawStart) * 1000.0
        
        // Record sample on command-buffer completion so endToEndMs captures CPU+GPU wall time.
        let drawCallIndex = drawCallCounter
        drawCallCounter += 1
        perfLogger.beginInFlightSample()
        commandBuffer.addCompletedHandler { [weak self] cb in
            guard let self else { return }

            let callbackEntry = CACurrentMediaTime()
            let preDrawMs = (drawStart - frameStart) * 1000.0
            let endToEndMs = (callbackEntry - drawStart) * 1000.0
            let displayFPS = self.fpsCounter?.snapshotFPS() ?? -1
            let fullFrameMs = (callbackEntry - frameStart) * 1000.0

            var gpuQueueWaitMs = -1.0
            var gpuAndCallbackMs = -1.0
            if cb.gpuStartTime > 0 {
                gpuQueueWaitMs = (cb.gpuStartTime - drawEnd) * 1000.0
                gpuAndCallbackMs = (callbackEntry - cb.gpuStartTime) * 1000.0
            }

            let sample = BinPerfSample(
                frameIndex: visibleFrame,
                drawCallIndex: drawCallIndex,
                preDrawMs: preDrawMs,
                cpuStageMs: cpuStageMs,
                gpuQueueWaitMs: gpuQueueWaitMs,
                gpuAndCallbackMs: gpuAndCallbackMs,
                endToEndMs: endToEndMs,
                displayFPS: displayFPS,
                fullFrameMs: fullFrameMs
            )

            self.perfLogger.recordCompleted(sample)
        }
    }
    
    override func resize(_ size: (width: Float, height: Float)) {
        camera.aspect = size.width / size.height
        renderer.resize(size)
    }
    
    /// Jump to a specific frame (slider scrubbing).
    func selectFrame(chosenFrame: Double) {
        let frameInd = Int(chosenFrame) / stepInd
        let clampedFrame = min(max(frameInd, 0), currentFrameNum - 1)
        
        // Remove current splat from scene
        let currentVisibleFrame = (progress.progressValue / stepInd) % currentFrameNum
        scene.remove(splatClouds[currentVisibleFrame])
        
        // Set to new frame
        progress.progressValue = clampedFrame * stepInd
        scene.add(splatClouds[clampedFrame])
    }
}


// MARK: - SwiftUI View

struct SplatBinView: View {
    @SwiftUI.Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    let model: SplatModelInfo
    let binConfig: BinDatasetConfig
    
    @StateObject var progress: RendererProgress = RendererProgress()
    @State private var renderer: BinCameraControllerRenderer?
    @State private var isPaused = false
    @State private var sliderValue: Double = 0
    @StateObject private var fpsCounter = FPSCounter()
    
    var body: some View {
        VStack {
            HStack {
                Button(action: {
                    renderer?.perfLogger.exportToCSV()
                    self.presentationMode.wrappedValue.dismiss()
                }) {
                    Image(systemName: "arrow.left")
                        .foregroundColor(.black)
                        .bold()
                        .padding()
                        .background(Color.gray.opacity(0.2))
                        .clipShape(Circle())
                }
                Spacer()
                Text("FPS: \(fpsCounter.fps)")
                    .foregroundColor(.blue)
                    .padding(.trailing)
                    .bold()
                    .font(.system(size: 32))
            }
            .padding()
            
            if let renderer = renderer {
                ForgeView(renderer: renderer)
                    .ignoresSafeArea()
            } else {
                Text("Loading bin data...")
            }
            
            Slider(value: $sliderValue,
                   in: 0...Double(binConfig.totalFrames * 2),
                   onEditingChanged: sliderEditingChanged)
                .accentColor(.black)
                .padding()
                .background(Color.white)
                .onChange(of: progress.progressValue) { newValue in
                    sliderValue = Double(newValue)
                }
            
            Button(action: {
                isPaused.toggle()
                renderer?.togglePause(isPaused)
            }) {
                Image(systemName: isPaused ? "play.fill" : "pause.fill")
                    .font(.largeTitle)
                    .foregroundColor(.black)
            }
            .buttonStyle(PlainButtonStyle())
            .padding()
            .background(Color.white)
        }
        .background(Color.white)
        .onAppear {
            if renderer == nil {
                renderer = BinCameraControllerRenderer(model: model, progress: progress, binConfig: binConfig, fpsCounter: fpsCounter)
            }
        }
        .onDisappear {
            renderer?.perfLogger.exportToCSV()
        }
    }
    
    private func sliderEditingChanged(_ isEditing: Bool) {
        if !isEditing {
            renderer?.selectFrame(chosenFrame: sliderValue)
        }
    }
}
