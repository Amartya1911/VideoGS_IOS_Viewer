//
//  Models.swift
//  MetalSplat
//
//  Created by CC Laan on 9/18/23.
//

import Foundation
import simd

extension simd_quatf {
    
    static let identity : simd_quatf = .init(ix: 1, iy: 0, iz: 0, r: 0)
    
}

struct SplatModelInfo {
    
    
    let plyUrl : URL
    let centroid : simd_float3
    
    let initialOrientation : simd_quatf
    let initialScale : Float
    
    // first subtract centroid, then possibly clip
    let clipOutsideRadius : Float
    let randomDownsample : Float
    
    let rendererDownsample : Float
    
    
    init(_ name : String,
         centroid: simd_float3 = .zero,
         initialOrientation: simd_quatf = simd_quatf.identity,
         initialScale: Float = 1.0,
         clipOutsideRadius: Float = -1,
         randomDownsample : Float = -1,
         rendererDownsample : Float = 2 ) {
        
        let plyUrl = Bundle.main.url(forResource: name, withExtension: "ply")!
        assert( FileManager.default.fileExists(atPath: plyUrl.path ))
        
        self.plyUrl = plyUrl
        self.centroid = centroid
        self.initialOrientation = initialOrientation
        self.initialScale = initialScale
        self.clipOutsideRadius = clipOutsideRadius
        self.randomDownsample = randomDownsample
        self.rendererDownsample = rendererDownsample
        
    }
    
}

// MARK: - Dataset Configuration
// To add a new dataset: 1) Add a static property below  2) Add a case in config(for:)
struct DatasetConfig {
    let minmaxPath: String      // JSON filename (without extension)
    let videoFolder: String     // Video dataset folder name in bundle
    let groupInfoPath: String   // Group info JSON filename (without extension)
    let stopNum: Int            // Max buffered frames before pausing load queue
    let totalFrames: Int        // Total number of frames in dataset
    let initPos: [Float]        // Initial position [x, y, z]

    static let ykxBoxing = DatasetConfig(
        minmaxPath: "viewer_min_max_ykx_380",
        videoFolder: "ykx_boxing_long_qp15_380",
        groupInfoPath: "group_info_ykx_380",
        stopNum: 100,
        totalFrames: 630,
        initPos: [0.0, -0.6, 0.5]
    )

    static let actor2Dancing4K = DatasetConfig(
        minmaxPath: "viewer_min_max_4k_actor2_lowres",
        videoFolder: "4K_Actor2_Dancing_sh0_res4_qp15",
        groupInfoPath: "group_info_4k_actor2_lowres",
        stopNum: 40,
        totalFrames: 40,
        initPos: [0.0, -0.6, 0.5]
    )

    /// Look up dataset config by dataIndex.
    /// To add a new dataset, add a new case here.
    static func config(for dataIndex: Int) -> DatasetConfig {
        switch dataIndex {
        case 1:  return .ykxBoxing
        case 2:  return .actor2Dancing4K
        default: return .ykxBoxing
        }
    }
}

struct Models {
    
    
    
    static var Mic : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("mic_60k",
                              centroid: [0, 0, -0.7],
                              initialOrientation: rotate,
                              initialScale: 0.35,
                              rendererDownsample: 2)
        
    }()
    
    static var MicLowRes : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("mic_60k",
                              centroid: [0, 0, -0.7],
                              initialOrientation: rotate,
                              initialScale: 0.35,
                              rendererDownsample: 4)
        
    }()
    
    
    static var Lego : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("lego_60k",
                              centroid: [0, 0, -0.35],
                              initialOrientation: rotate,
                              initialScale: 0.35,
                              rendererDownsample: 2)
        
    }()
    
    /*
    static var Plush : SplatModelInfo = {
        let rotate = simd_quatf(angle: Float.pi * 0.8, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("plush_cloud", centroid: [0.28, 1.97, 1.41], 
                              initialOrientation: rotate, rendererDownsample: 2)
    }()
    
    static var Nike : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * 0.8, axis: .init(x: 1, y: 0, z: 0))
        
        return SplatModelInfo("nike_cloud", 
                              //centroid: [-0.37, 2.18 + 0.25, 1.32],
                              centroid: [-0.37, 2.18 - 0.75, 1.32],
                              initialOrientation: rotate, initialScale: 0.2,
                              rendererDownsample: 2)
        
    }()
    
    static var Ship : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("ship_60k",
                              centroid: [0, 0, -0.7],
                              initialOrientation: rotate,
                              initialScale: 0.35,
                              rendererDownsample: 2)
        
    }()
    
    static var Drums : SplatModelInfo = {
        
        let rotate = simd_quatf(angle: Float.pi * -0.5, axis: .init(x: 1, y: 0, z: 0))
        return SplatModelInfo("drums_60k",
                              centroid: [0, 0, -0.7],
                              initialOrientation: rotate,
                              initialScale: 0.35,
                              rendererDownsample: 2)
        
    }()
        */
    
    
}
