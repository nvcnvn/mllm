// Time any fixed-shape fp16 Core ML model on CPU+ANE with every input and output an IOSurface-backed CVPixelBuffer
// (no CPU copies: tools/ane_weight_input.swift). Reports median predict ms, CPU ms per call, and TFLOPS counting
// 2 * S * Cout * Cin per weight input (inputs named w_*, shaped [Cout, Cin]; S = x's first dim; tools/ane_layer.py).
// With SECONDS, predicts in a loop that long (to run next to GPU decode) and reports the same.
//
//   swiftc -O tools/ane_time.swift -o /tmp/ane_time && /tmp/ane_time MODEL.mlpackage [SECONDS]
import CoreML
import CoreVideo
import Foundation

let arg = CommandLine.arguments
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndNeuralEngine
let t0 = Date()
let model = try! MLModel(contentsOf: try! MLModel.compileModel(at: URL(fileURLWithPath: arg[1])), configuration: cfg)
let loadS = Date().timeIntervalSince(t0)

let table = (0..<4096).map { Float16(sin(Double($0) * 0.7)) }
func surface(_ shape: [Int], scale: Float16) -> MLMultiArray {
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
    let w = shape.last!, h = shape.dropLast().reduce(1, *)
    precondition(CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pb)
                 == kCVReturnSuccess, "CVPixelBufferCreate \(shape)")
    let a = MLMultiArray(pixelBuffer: pb!, shape: shape as [NSNumber])
    a.withUnsafeMutableBytes { raw, st in
        let p = raw.bindMemory(to: Float16.self)
        for r in 0..<h { for c in 0..<w { p[r * st[st.count - 2] + c] = table[(r * 7919 + c) & 4095] * scale } }
    }
    return a
}
func shape(_ d: MLFeatureDescription) -> [Int] { d.multiArrayConstraint!.shape.map { $0.intValue } }

var feats = [String: MLMultiArray](), flops = 0.0
let S = shape(model.modelDescription.inputDescriptionsByName["x"]!)[0]
var wBytes = 0
for (name, d) in model.modelDescription.inputDescriptionsByName {
    let s = shape(d)
    feats[name] = surface(s, scale: name.hasPrefix("w_") ? 0.02 : 1)
    if name.hasPrefix("w_") { flops += 2.0 * Double(S * s[0] * s[1]); wBytes += s[0] * s[1] * 2 }
}
let opts = MLPredictionOptions()
opts.outputBackings = model.modelDescription.outputDescriptionsByName.mapValues { surface(shape($0), scale: 0) }
let inputs = try! MLDictionaryFeatureProvider(dictionary: feats)

func cpuMs() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3
}
for _ in 0..<3 { _ = try! model.prediction(from: inputs, options: opts) }
let seconds = arg.count > 2 ? Double(arg[2])! : 2.0
var ts = [Double](), cpu = 0.0
let end = DispatchTime.now().uptimeNanoseconds + UInt64(seconds * 1e9)
while DispatchTime.now().uptimeNanoseconds < end || ts.count < 5 {
    let c0 = cpuMs(), s0 = DispatchTime.now().uptimeNanoseconds
    _ = try! model.prediction(from: inputs, options: opts)
    ts.append(Double(DispatchTime.now().uptimeNanoseconds - s0) / 1e6)
    cpu += cpuMs() - c0
}
let med = ts.sorted()[ts.count / 2]
print(String(format: "%@: S %d, %d inputs, W %.0f MiB, load %.1f s: %.2f ms median (%d calls), %.2f ms CPU/call, %.2f TFLOPS",
             URL(fileURLWithPath: arg[1]).lastPathComponent, S, feats.count, Double(wBytes) / 1048576, loadS, med,
             ts.count, cpu / Double(ts.count), flops / med / 1e9))
