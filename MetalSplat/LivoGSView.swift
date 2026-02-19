import SwiftUI
import UIKit
import Metal
import MetalKit
import Forge
import Satin
import SatinCore

class LivoGSBufferRenderer: Forge.Renderer {
    let model: SplatModelInfo
    var progress: RendererProgress
    var isPaused: Bool = false

    private var splatCloud: SplatCloud?
    private var sceneReady: Bool = false
    private var cloudAddedToScene: Bool = false
    private var didSortOnce: Bool = false

    init(model: SplatModelInfo, progress: RendererProgress) {
        self.model = model
        self.progress = progress
        super.init()
    }

    func togglePause(_ paused: Bool) {
        self.isPaused = paused
    }

    private func loadData(url: URL) throws -> Data {
        return try Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private func loadSingleFrameCloud() throws -> SplatCloud {
        guard let device = self.device else {
            throw SplatError.deviceCreationFailed
        }

        guard let library = device.makeDefaultLibrary(),
              let commandQueue = device.makeCommandQueue(),
              let livoFolder = Bundle.main.url(forResource: "LivoGS_data1", withExtension: nil)?
                .appendingPathComponent("LivoGS_Frame1") else {
            throw SplatError.deviceCreationFailed
        }

        let meansData = try loadData(url: livoFolder.appendingPathComponent("means3d.bin"))
        let colorsData = try loadData(url: livoFolder.appendingPathComponent("colors3d.bin"))
        let scalesData = try loadData(url: livoFolder.appendingPathComponent("scales3d.bin"))
        let quatsData = try loadData(url: livoFolder.appendingPathComponent("quats3d.bin"))
        let opacitiesData = try loadData(url: livoFolder.appendingPathComponent("opacities3d.bin"))

        let meansCount = meansData.count / MemoryLayout<Float>.stride
        let colorsCount = colorsData.count / MemoryLayout<Float>.stride
        let scalesCount = scalesData.count / MemoryLayout<Float>.stride
        let quatsCount = quatsData.count / MemoryLayout<Float>.stride
        let opacitiesCount = opacitiesData.count / MemoryLayout<Float>.stride

        let count = meansCount / 3
        guard count > 0,
              colorsCount == count * 3,
              scalesCount == count * 3,
              quatsCount == count * 4,
              opacitiesCount == count else {
            throw SplatError.plyParsingFailed
        }

        guard let meansBuffer = device.makeBuffer(bytes: (meansData as NSData).bytes,
                                                  length: meansData.count,
                                                  options: .storageModeShared),
              let colorsBuffer = device.makeBuffer(bytes: (colorsData as NSData).bytes,
                                                   length: colorsData.count,
                                                   options: .storageModeShared),
              let scalesBuffer = device.makeBuffer(bytes: (scalesData as NSData).bytes,
                                                   length: scalesData.count,
                                                   options: .storageModeShared),
              let quatsBuffer = device.makeBuffer(bytes: (quatsData as NSData).bytes,
                                                  length: quatsData.count,
                                                  options: .storageModeShared),
              let opacitiesBuffer = device.makeBuffer(bytes: (opacitiesData as NSData).bytes,
                                                      length: opacitiesData.count,
                                                      options: .storageModeShared),
              let packedSplatsBuffer = device.makeBuffer(length: count * MemoryLayout<Splat>.stride,
                                                         options: .storageModeShared) else {
            throw SplatError.deviceCreationFailed
        }

        guard let packFunction = library.makeFunction(name: "packSplatsFromBuffers") else {
            throw SplatError.deviceCreationFailed
        }

        let packPipeline = try device.makeComputePipelineState(function: packFunction)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatError.deviceCreationFailed
        }

        computeEncoder.setComputePipelineState(packPipeline)
        computeEncoder.setBuffer(meansBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(colorsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(quatsBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(opacitiesBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(packedSplatsBuffer, offset: 0, index: 5)

        var splatCount = UInt32(count)
        computeEncoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(packPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard let cloud = try SplatCloud(model: model,
                                         renderDestination: mtkView,
                                         splatBuffer: packedSplatsBuffer,
                                         count: count) else {
            throw SplatError.deviceCreationFailed
        }

        cloud.shouldSortSplats = false
        cloud.orientation = model.initialOrientation
        cloud.scale = SIMD3<Float>(repeating: model.initialScale)
        return cloud
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
        metalKitView.clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0)
        metalKitView.preferredFramesPerSecond = 60

        metalKitView.drawableSize = mtkView.drawableSize.applying(
            CGAffineTransform(scaleX: 1.0 / CGFloat(model.rendererDownsample),
                              y: 1.0 / CGFloat(model.rendererDownsample))
        )

        renderer.clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            do {
                let cloud = try self.loadSingleFrameCloud()
                DispatchQueue.main.async {
                    self.splatCloud = cloud
                    self.didSortOnce = false
                    if self.sceneReady && !self.cloudAddedToScene {
                        self.scene.add(cloud)
                        self.cloudAddedToScene = true
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    print("[LivoGS] Failed to load single frame: \(error)")
                }
            }
        }
    }

    override func setup() {
        scene.attach(cameraController.target)
        sceneReady = true
        if let cloud = splatCloud, !cloudAddedToScene {
            scene.add(cloud)
            cloudAddedToScene = true
        }
    }

    deinit {
        cameraController.disable()
    }

    override func update() {
        cameraController.update()
    }

    override func draw(_ view: MTKView, _ commandBuffer: MTLCommandBuffer) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        renderer.draw(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            scene: scene,
            camera: camera
        )

        if !didSortOnce, let cloud = splatCloud {
            cloud.sortSplatsOnce()
            didSortOnce = true
        }

        if !isPaused {
            progress.progressValue += 1
        }
    }

    override func resize(_ size: (width: Float, height: Float)) {
        camera.aspect = size.width / size.height
        renderer.resize(size)
    }
}

struct LivoGSSplatView: View {
    @SwiftUI.Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    let model: SplatModelInfo
    @StateObject var progress: RendererProgress = RendererProgress()
    @State private var renderer: LivoGSBufferRenderer?

    var body: some View {
        VStack {
            HStack {
                Button(action: {
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
                Text("LivoGS Single Frame")
                    .foregroundColor(.black)
                    .bold()
            }
            .padding()

            if let renderer = renderer {
                ForgeView(renderer: renderer)
                    .ignoresSafeArea()
            } else {
                Text("Loading LivoGS frame...")
            }
        }
        .background(Color.white)
        .onAppear {
            if renderer == nil {
                renderer = LivoGSBufferRenderer(model: model, progress: progress)
            }
        }
    }
}
