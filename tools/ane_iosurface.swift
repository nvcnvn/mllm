// Can the GPU read ANE output in place? (memory-design.md, "To verify" #3)
//
// Core ML writes a prediction into an IOSurface-backed CVPixelBuffer we own (outputBackings).
// Metal then reads that same IOSurface two ways, with no copy: as an r16Float texture and as an
// MTLBuffer (bytesNoCopy). Both must match Core ML's own output bit for bit, and must still match
// after a second prediction into the same surface (same Metal objects, no re-wrap).
//
//   P=$(.venv/bin/python -c "import sys;sys.path.insert(0,'tools');import ane_bench as a,coremltools as ct,tempfile;\
//   p,_=a.build(2048,256,4,tempfile.mkdtemp());print(a.placement(ct.models.MLModel(p,compute_units=a.CU['ANE'])),file=sys.stderr);print(p)")
//   swiftc -O tools/ane_iosurface.swift -o /tmp/ane_iosurface && nice -n 10 /tmp/ane_iosurface "$P"
import CoreML
import CoreVideo
import Foundation
import IOSurface
import Metal

let C = 2048, S = 256   // must match the build() call above; smaller models get put on the CPU
let shape: [NSNumber] = [1, NSNumber(value: C), 1, NSNumber(value: S)]

let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: try MLModel.compileModel(at: URL(fileURLWithPath: CommandLine.arguments[1])), configuration: cfg)
let inName = model.modelDescription.inputDescriptionsByName.keys.first!
let outName = model.modelDescription.outputDescriptionsByName.keys.first!

func input(seed: Int) -> MLFeatureProvider {
    let a = try! MLMultiArray(shape: shape, dataType: .float16)
    let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
    let cs = a.strides[1].intValue, ss = a.strides[3].intValue
    for c in 0..<C { for s in 0..<S { p[c * cs + s * ss] = Float16(sin(Double(c * 31 + s * 7 + seed * 1000)) ) } }
    return try! MLDictionaryFeatureProvider(dictionary: [inName: a])
}

// Read any [1,C,1,S] fp16 grid into a dense array (for comparison only).
func dense(_ p: UnsafePointer<Float16>, rowStride: Int, colStride: Int = 1) -> [UInt16] {
    var o = [UInt16](repeating: 0, count: C * S)
    for c in 0..<C { for s in 0..<S { o[c * S + s] = p[c * rowStride + s * colStride].bitPattern } }
    return o
}
func reference(_ x: MLFeatureProvider) -> [UInt16] {   // Core ML's own output buffer, no backing
    let a = try! model.prediction(from: x).featureValue(for: outName)!.multiArrayValue!
    precondition(a.dataType == .float16, "output is \(a.dataType.rawValue), expected fp16")
    return dense(a.dataPointer.bindMemory(to: Float16.self, capacity: a.count), rowStride: a.strides[1].intValue, colStride: a.strides[3].intValue)
}

// The surface we own.
var pbOut: CVPixelBuffer?
let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary, kCVPixelBufferMetalCompatibilityKey: true]
precondition(CVPixelBufferCreate(nil, S, C, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pbOut) == kCVReturnSuccess)
let pb = pbOut!
let surf = CVPixelBufferGetIOSurface(pb)!.takeUnretainedValue()
let backing = MLMultiArray(pixelBuffer: pb, shape: shape)
let opts = MLPredictionOptions()
opts.outputBackings = [outName: backing]
let bpr = IOSurfaceGetBytesPerRow(surf), alloc = IOSurfaceGetAllocSize(surf)
let base = IOSurfaceGetBaseAddress(surf)
print("surface: \(S)x\(C) fp16, bytesPerRow \(bpr), allocSize \(alloc), base 0x\(String(UInt(bitPattern: base), radix: 16))")

// Metal views of the same bytes, created once.
let dev = MTLCreateSystemDefaultDevice()!
let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: S, height: C, mipmapped: false)
td.usage = .shaderRead
td.storageMode = .shared
let tex = dev.makeTexture(descriptor: td, iosurface: surf, plane: 0)!
let page = Int(getpagesize())
// allocSize carries a small tail (1 MiB + 64 B here), so wrap just the pixel rows, rounded up to a page.
let viewLen = (bpr * C + page - 1) / page * page
let bufOK = UInt(bitPattern: base) % UInt(page) == 0 && viewLen <= (alloc + page - 1) / page * page
let surfBuf = bufOK ? dev.makeBuffer(bytesNoCopy: base, length: viewLen, options: .storageModeShared, deallocator: nil) : nil
print("texture view: ok · buffer view: \(surfBuf != nil ? "ok" : "not possible (base or size not \(page)-aligned)")")

let lib = try dev.makeLibrary(source: """
#include <metal_stdlib>
using namespace metal;
kernel void fromTex(texture2d<half, access::read> t [[texture(0)]], device half *o [[buffer(0)]],
                    constant uint &W [[buffer(1)]], uint2 g [[thread_position_in_grid]]) { o[g.y * W + g.x] = t.read(g).r; }
kernel void fromBuf(device const half *b [[buffer(0)]], device half *o [[buffer(1)]], constant uint &W [[buffer(2)]],
                    constant uint &row [[buffer(3)]], uint2 g [[thread_position_in_grid]]) { o[g.y * W + g.x] = b[g.y * row + g.x]; }
""", options: nil)
let q = dev.makeCommandQueue()!
let pTex = try dev.makeComputePipelineState(function: lib.makeFunction(name: "fromTex")!)
let pBuf = try dev.makeComputePipelineState(function: lib.makeFunction(name: "fromBuf")!)
let outT = dev.makeBuffer(length: C * S * 2, options: .storageModeShared)!
let outB = dev.makeBuffer(length: C * S * 2, options: .storageModeShared)!

func gpuRead() -> (tex: [UInt16], buf: [UInt16]?) {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    var W = UInt32(S), row = UInt32(bpr / 2)
    let grid = MTLSize(width: S, height: C, depth: 1), tg = MTLSize(width: 32, height: 8, depth: 1)
    e.setComputePipelineState(pTex); e.setTexture(tex, index: 0); e.setBuffer(outT, offset: 0, index: 0)
    e.setBytes(&W, length: 4, index: 1); e.dispatchThreads(grid, threadsPerThreadgroup: tg)
    if let sb = surfBuf {
        e.setComputePipelineState(pBuf); e.setBuffer(sb, offset: 0, index: 0); e.setBuffer(outB, offset: 0, index: 1)
        e.setBytes(&W, length: 4, index: 2); e.setBytes(&row, length: 4, index: 3); e.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    precondition(cb.status == .completed, "GPU error: \(String(describing: cb.error))")
    let rd = { (b: MTLBuffer) in dense(b.contents().bindMemory(to: Float16.self, capacity: C * S), rowStride: S) }
    return (rd(outT), surfBuf != nil ? rd(outB) : nil)
}

func mismatches(_ a: [UInt16], _ b: [UInt16]) -> Int { zip(a, b).reduce(0) { $0 + ($1.0 != $1.1 ? 1 : 0) } }
var allOK = true
for seed in [1, 2] {
    let x = input(seed: seed), ref = reference(x)
    let got = try model.prediction(from: x, options: opts).featureValue(for: outName)!.multiArrayValue!
    let inPlace = got.pixelBuffer.map { $0 === pb } ?? false
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    let cpu = dense(CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: Float16.self), rowStride: bpr / 2)
    CVPixelBufferUnlockBaseAddress(pb, .readOnly)
    let g = gpuRead()
    let m = (cpu: mismatches(cpu, ref), tex: mismatches(g.tex, ref), buf: g.buf.map { mismatches($0, ref) })
    let nonzero = ref.filter { $0 & 0x7fff != 0 }.count
    print("prediction \(seed): result is our pixel buffer: \(inPlace) · nonzero outputs \(nonzero)/\(C * S) · mismatches vs Core ML's own output: CPU \(m.cpu), GPU texture \(m.tex), GPU buffer \(m.buf.map(String.init) ?? "n/a")")
    allOK = allOK && inPlace && nonzero > 0 && m.cpu == 0 && m.tex == 0 && (m.buf ?? 0) == 0
}

// Does writing into our surface cost anything vs letting Core ML allocate? (a hint about internal copies, not proof)
func median(_ f: () -> Void) -> Double {
    for _ in 0..<10 { f() }
    var t = [Double]()
    for _ in 0..<200 { let s = DispatchTime.now().uptimeNanoseconds; f(); t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6) }
    return t.sorted()[t.count / 2]
}
let x = input(seed: 3)
let plain = try MLMultiArray(shape: shape, dataType: .float16)
let optsPlain = MLPredictionOptions(); optsPlain.outputBackings = [outName: plain]
let tNone = median { _ = try! model.prediction(from: x) }
let tSurf = median { _ = try! model.prediction(from: x, options: opts) }
let tPlain = median { _ = try! model.prediction(from: x, options: optsPlain) }
print(String(format: "predict median ms (output %d KiB): Core ML's buffer %.3f · our IOSurface %.3f · our plain MLMultiArray %.3f", C * S * 2 / 1024, tNone, tSurf, tPlain))
print(allOK ? "PASS: the GPU reads ANE output in place through the IOSurface, no copy on the Metal side" : "FAIL: see mismatches above")
exit(allOK ? 0 : 1)
