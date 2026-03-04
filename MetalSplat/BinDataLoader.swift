//
//  BinDataLoader.swift
//  MetalSplat
//
//  Cross-platform BIN frame data loader (iOS + visionOS).
//  Extracted from SplatBinView.swift so the visionOS compositor can reuse it.
//

import Foundation

// MARK: - Bin Frame Data

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

// MARK: - Bin Data Loader

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
        print("BinDataLoader: Found \(frameIndices.count)/\(totalFrames) frames in \(binFolder)")

        if frameIndices.count != totalFrames {
            print("BinDataLoader: WARNING expected \(totalFrames) frames, found \(frameIndices.count)")
        }

        if !frameIndices.isEmpty {
            var gaps = [Int]()
            for expected in 0..<totalFrames {
                if !frameIndices.contains(expected) {
                    gaps.append(expected)
                }
            }
            if !gaps.isEmpty {
                print("BinDataLoader: WARNING missing frame indices: \(gaps.prefix(10))\(gaps.count > 10 ? " ..." : "")")
            }
        }

        var allFrames = [BinFrameData]()
        var totalBytes = 0
        var expectedPointsPerFrame: Int?
        var inconsistentPointFrames: [Int] = []

        for i in frameIndices {
            if let data = loadFrame(i) {
                allFrames.append(data)
                let frameBytes = data.means.count + data.scales.count + data.quats.count + data.colors.count + data.opacities.count
                totalBytes += frameBytes

                if let expected = expectedPointsPerFrame {
                    if data.numPoints != expected {
                        inconsistentPointFrames.append(i)
                    }
                } else {
                    expectedPointsPerFrame = data.numPoints
                }

                print("  Frame \(i): \(data.numPoints) splats, \(frameBytes) bytes")
            }
        }

        let totalMiB = Double(totalBytes) / (1024.0 * 1024.0)
        print(String(format: "BinDataLoader: Total in-memory raw bin size: %.2f MiB", totalMiB))

        if let expected = expectedPointsPerFrame {
            if inconsistentPointFrames.isEmpty {
                print("BinDataLoader: Point-count check OK (\(expected) splats/frame)")
            } else {
                print("BinDataLoader: WARNING point-count mismatch for frames: \(inconsistentPointFrames.prefix(10))\(inconsistentPointFrames.count > 10 ? " ..." : "")")
            }
        }

        return allFrames
    }
}
