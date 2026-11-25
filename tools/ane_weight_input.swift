// Weights as ANE matmul inputs: is Core ML's ~5.6 GB/s weight intake a CPU-side copy, and do IOSurface-backed inputs
// skip it? (target-27b.md §5 "One weight copy"; tools/ane_dynamic_weights.py measured the plain-input case.)
//
// y = x @ Wᵀ, x [S, Cin], W [Cout, Cin], fp16. Four runs: the const-W model and the input-W model, each with plain
// MLMultiArrays and with IOSurface-backed CVPixelBuffers for every input and the output. Per run: median wall ms and
// this process's CPU ms per predict (getrusage: a CPU copy shows up here). The input-W runs first rewrite W rows
// between predicts and check y against the new rows, so a cached copy of W can't pass.
//
//   swiftc -O tools/ane_weight_input.swift -o /tmp/ane_weight_input
//   for s in "512 5120 5120" "512 5120 1024" "512 1024 4096" "2048 1024 4096"; do set -- $s
//     P=$(.venv/bin/python -c "import sys,tempfile;sys.path.insert(0,'tools');import ane_dynamic_weights as d;\
//   w=tempfile.mkdtemp();print(d.build($1,$2,$3,False,w)[0],d.build($1,$2,$3,True,w)[0])")
//     /tmp/ane_weight_input $P $1 $2 $3; done
import CoreML
import CoreVideo
import Foundation

let arg = CommandLine.arguments
let S = Int(arg[3])!, Cin = Int(arg[4])!, Cout = Int(arg[5])!

func load(_ path: String) -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    return try! MLModel(contentsOf: try! MLModel.compileModel(at: URL(fileURLWithPath: path)), configuration: cfg)
}
let constModel = load(arg[1]), dynModel = load(arg[2])

func array(_ rows: Int, _ cols: Int, surface: Bool) -> MLMultiArray {
    guard surface else { return try! MLMultiArray(shape: [rows, cols] as [NSNumber], dataType: .float16) }
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
    precondition(CVPixelBufferCreate(nil, cols, rows, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pb)
                 == kCVReturnSuccess)
    return MLMultiArray(pixelBuffer: pb!, shape: [rows, cols] as [NSNumber])
}
func value(_ r: Int, _ c: Int, _ seed: Int) -> Float { Float(sin(Double(r * 131 + c * 7 + seed * 1009))) }
// fill rows `rs` with value(r, c, seed) * scale
func fill(_ a: MLMultiArray, rows rs: Range<Int>, seed: Int, scale: Float) {
    a.withUnsafeMutableBytes { raw, st in
        let p = raw.bindMemory(to: Float16.self)
        for r in rs { for c in 0..<a.shape[1].intValue { p[r * st[0] + c * st[1]] = Float16(value(r, c, seed) * scale) } }
    }
}
func get(_ a: MLMultiArray, _ r: Int, _ c: Int) -> Float {
    a.withUnsafeBytes { raw in Float(raw.bindMemory(to: Float16.self)[r * a.strides[0].intValue + c * a.strides[1].intValue]) }
}

func cpuMs() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) * 1e3 + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e3
}

let flops = 2.0 * Double(S * Cin * Cout)
print(String(format: "S %d x Cin %d x Cout %d (W %.0f MiB)", S, Cin, Cout, Double(Cout * Cin * 2) / 1048576))
for dynamic in [false, true] {
    for surface in [false, true] {
        let model = dynamic ? dynModel : constModel
        let x = array(S, Cin, surface: surface), y = array(S, Cout, surface: surface)
        fill(x, rows: 0..<S, seed: 1, scale: 1)
        var feats: [String: MLMultiArray] = ["x": x]
        let w = array(Cout, Cin, surface: surface)
        if dynamic {
            fill(w, rows: 0..<Cout, seed: 2, scale: 1 / Float(Cin).squareRoot())
            feats["w"] = w
        }
        let inputs = try! MLDictionaryFeatureProvider(dictionary: feats)
        let outName = model.modelDescription.outputDescriptionsByName.keys.first!
        let opts = MLPredictionOptions()
        opts.outputBackings = [outName: y]
        let predict = { () -> MLMultiArray in try! model.prediction(from: inputs, options: opts).featureValue(for: outName)!.multiArrayValue! }

        var check = "W const"
        if dynamic {  // rewrite one W row, predict, and compare that output column against the new row
            var worst: Float = 0
            for it in 0..<4 {
                let r = (it * 7919) % Cout
                fill(w, rows: r..<r + 1, seed: 100 + it, scale: 1 / Float(Cin).squareRoot())
                let out = predict()
                for s in [0, S / 2, S - 1] {
                    var want: Float = 0
                    for c in 0..<Cin { want += Float(Float16(value(s, c, 1))) * Float(Float16(value(r, c, 100 + it) / Float(Cin).squareRoot())) }
                    worst = max(worst, abs(get(out, s, r) - want) / max(abs(want), 0.05))
                }
            }
            check = String(format: "fresh W rows: worst rel err %.1e%@", worst, worst < 2e-2 ? "" : " MISMATCH")
        }

        for _ in 0..<3 { _ = predict() }
        var ts = [Double](), cpu = 0.0
        let end = DispatchTime.now().uptimeNanoseconds + 1_500_000_000
        while DispatchTime.now().uptimeNanoseconds < end || ts.count < 5 {
            if dynamic { fill(w, rows: 0..<1, seed: ts.count, scale: 0.01) }  // W changes every call
            let c0 = cpuMs(), t0 = DispatchTime.now().uptimeNanoseconds
            _ = predict()
            ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            cpu += cpuMs() - c0
        }
        let med = ts.sorted()[ts.count / 2]
        print(String(format: "  W %@ %@: %8.2f ms wall, %7.2f ms CPU/call, %5.2f TFLOPS  (%@)", dynamic ? "input" : "const",
                     surface ? "IOSurface" : "plain    ", med, cpu / Double(ts.count), flops / med / 1e9, check))
    }
}
