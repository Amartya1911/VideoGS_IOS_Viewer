#if os(visionOS)
import SwiftUI
import CompositorServices

@main
struct MetalSplatVisionApp: App {
    @State private var appModel = VisionAppModel()

    var body: some Scene {
        WindowGroup("VideoGS Controls") {
            VisionControlPanelView()
                .environment(appModel)
        }

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            CompositorLayer { layerRenderer in
                appModel.compositorRenderer.attach(
                    layerRenderer: layerRenderer,
                    dataset: appModel.selectedDataset,
                    isPlaying: appModel.isPlaying
                )
            }
        }
        .immersionStyle(selection: .constant(.full), in: .full)
    }
}

@Observable
@MainActor
final class VisionAppModel {
    let immersiveSpaceID = "VideoGSImmersive"
    var selectedDatasetKey: Int = 1
    var isImmersiveActive: Bool = false
    var isPlaying: Bool = true
    let compositorRenderer = VisionCompositorRenderer()

    var selectedDataset: BinDatasetConfig {
        BinDatasetConfig.config(for: selectedDatasetKey)
    }
}

struct VisionControlPanelView: View {
    @Environment(VisionAppModel.self) private var appModel
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace

    @State private var openTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("VideoGS BIN Viewer")
                .font(.title2)
                .bold()

            Text("Renderer runs only inside ImmersiveSpace via CompositorServices.")
                .foregroundStyle(.secondary)

            Picker("Dataset", selection: Bindable(appModel).selectedDatasetKey) {
                Text("Actor2 Dancing (BIN)")
                    .tag(1)
            }
            .pickerStyle(.segmented)

            Toggle("Play", isOn: Bindable(appModel).isPlaying)

            HStack(spacing: 12) {
                Button(appModel.isImmersiveActive ? "Exit Immersive Renderer" : "Enter Immersive Renderer") {
                    openTask?.cancel()
                    openTask = Task {
                        if appModel.isImmersiveActive {
                            await dismissImmersiveSpace()
                            appModel.isImmersiveActive = false
                        } else {
                            let result = await openImmersiveSpace(id: appModel.immersiveSpaceID)
                            if case .opened = result {
                                appModel.isImmersiveActive = true
                            }
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
    }
}

#endif
