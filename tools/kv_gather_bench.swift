// Do scattered arena units keep full bandwidth? Which tokens-per-page? (memory-design.md, "To verify" #1, build step 1)
//
// Reads K/V the way a decode attention kernel would, through a slot table over one 4 GiB arena laid out as in
// memory-design.md: unit = 8 layers × (K,V) × 8 heads × block, block = T tokens × 256 B (fp16) or T × 130 B
// (int8 values + fp16 scale per row). One threadgroup = (request, K/V·head, 512-token chunk); 8 layer dispatches
// per command buffer. Compares slot tables in order (contiguous) vs shuffled, T = 16/32/64/128, fp16 vs int8.
// Numbers are bytes read / GPU time; compute is a trivial xor so this is a pure read test.
//
//   swiftc -O tools/kv_gather_bench.swift -o /tmp/kvb && nice -n 10 /tmp/kvb
import Foundation
import Metal

let REQ = 16, CTX = 4096, LAYERS = 8, BLOCKS = 16 /* (K,V) × 8 heads */, CHUNK = 512, ARENA = 4 << 30
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: """
#include <metal_stdlib>
using namespace metal;
// p: unitVecs, layerVecs, blockVecs, readVecs, pagesPerReq, pagesPerTG, layer
kernel void gather(device const uint4 *arena [[buffer(0)]], device const uint *slots [[buffer(1)]],
                   device uint *out [[buffer(2)]], constant uint *p [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint3 tpg3 [[threads_per_threadgroup]]) {
    uint tpg = tpg3.x;
    uint acc = 0, p0 = tg.x * p[5];
    for (uint pg = p0; pg < p0 + p[5]; pg++) {
        ulong base = (ulong)slots[tg.z * p[4] + pg] * p[0] + (ulong)p[6] * p[1] + (ulong)tg.y * p[2];
        device const uint4 *b = arena + base;
        for (uint v = tid; v < p[3]; v += tpg) { uint4 x = b[v]; acc += x.x ^ x.y ^ x.z ^ x.w; }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;   // keeps the loads alive
}
kernel void stream(device const uint4 *a [[buffer(0)]], device uint *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    uint acc = 0;
    for (uint i = gid; i < n; i += gs) { uint4 x = a[i]; acc += x.x ^ x.y ^ x.z ^ x.w; }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
""", options: nil)
let pGather = try dev.makeComputePipelineState(function: lib.makeFunction(name: "gather")!)
let pStream = try dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)

// One arena, allocated and pre-touched once (as the design says).
let arena = dev.makeBuffer(length: ARENA, options: .storageModeShared)!
memset(arena.contents(), 0xA5, ARENA)
for i in stride(from: 0, to: ARENA, by: 4096) { arena.contents().storeBytes(of: UInt32(truncatingIfNeeded: i &* 2654435761), toByteOffset: i, as: UInt32.self) }
let sink = dev.makeBuffer(length: 16, options: .storageModeShared)!

struct Variant { let T: Int, int8: Bool, shuffled: Bool; var lone = false   // lone: read only one head's K block per page
    var rowBytes: Int { int8 ? 130 : 256 }   // int8: 128 values + fp16 scale per row
    var blockBytes: Int { T * rowBytes }
    var unitBytes: Int { (LAYERS * BLOCKS * blockBytes + 16383) / 16384 * 16384 }   // whole OS pages
    var pagesPerReq: Int { CTX / T }
    var blocksRead: Int { lone ? 1 : BLOCKS }
    var bytesPerCB: Int { REQ * pagesPerReq * blocksRead * blockBytes * LAYERS }
    var name: String { "T=\(T) \(int8 ? "int8" : "fp16") \(shuffled ? "shuffled" : "in order")\(lone ? " lone" : "")" }
}
var rng = SystemRandomNumberGenerator()
func slotTable(_ v: Variant) -> MTLBuffer {
    let n = REQ * v.pagesPerReq, slots = ARENA / v.unitBytes
    precondition(n <= slots, "arena too small for \(v.name)")
    let s: [UInt32] = v.shuffled ? Array((0..<UInt32(slots)).shuffled(using: &rng).prefix(n)) : (0..<UInt32(n)).map { $0 }
    return dev.makeBuffer(bytes: s, length: n * 4, options: .storageModeShared)!
}
func runGather(_ v: Variant, _ table: MTLBuffer) -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pGather); e.setBuffer(arena, offset: 0, index: 0); e.setBuffer(table, offset: 0, index: 1); e.setBuffer(sink, offset: 0, index: 2)
    for l in 0..<LAYERS {
        var p: [UInt32] = [UInt32(v.unitBytes / 16), UInt32(BLOCKS * v.blockBytes / 16), UInt32(v.blockBytes / 16), UInt32(v.blockBytes / 16),
                           UInt32(v.pagesPerReq), UInt32(CHUNK / v.T), UInt32(l)]
        e.setBytes(&p, length: p.count * 4, index: 3)
        e.dispatchThreadgroups(MTLSize(width: CTX / CHUNK, height: v.blocksRead, depth: REQ), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    precondition(cb.status == .completed, "GPU error: \(String(describing: cb.error))")
    return cb.gpuEndTime - cb.gpuStartTime
}
func runStream(bytes: Int) -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    var n = UInt32(bytes / 16)
    e.setComputePipelineState(pStream); e.setBuffer(arena, offset: 0, index: 0); e.setBuffer(sink, offset: 0, index: 1); e.setBytes(&n, length: 4, index: 2)
    e.dispatchThreads(MTLSize(width: 20 * 1024 * 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

var variants = [Variant]()
for T in [16, 32, 64, 128] { for int8 in [false, true] { for sh in [false, true] { variants.append(Variant(T: T, int8: int8, shuffled: sh)) } } }
for T in [16, 32, 64, 128] { for sh in [false, true] { variants.append(Variant(T: T, int8: false, shuffled: sh, lone: true)) } }   // worst case: isolated runs
let tables = variants.map(slotTable)
let streamBytes = 2 << 30
var times = Array(repeating: [Double](), count: variants.count), streamT = [Double]()
for _ in 0..<3 { _ = runStream(bytes: streamBytes) }
for round in 0..<5 {   // interleave so background load hits every variant alike
    for _ in 0..<4 { streamT.append(runStream(bytes: streamBytes)) }
    for (i, v) in variants.enumerated() {
        if round == 0 { _ = runGather(v, tables[i]) }   // warm up
        for _ in 0..<6 { times[i].append(runGather(v, tables[i])) }
    }
}
func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
let sGBs = Double(streamBytes) / med(streamT) / 1e9
print(String(format: "baseline: one contiguous 2 GiB stream  %6.1f GB/s (median)  %6.1f GB/s (best)", sGBs, Double(streamBytes) / streamT.min()! / 1e9))
print("decode reads for \(REQ) requests × \(CTX) tokens × \(LAYERS) full-attention layers, per step:")
print("variant                    run size   GB/s med  best   vs stream   ms/step")
for (i, v) in variants.enumerated() {
    let m = med(times[i]), b = times[i].min()!, g = Double(v.bytesPerCB) / m / 1e9
    print(String(format: "%-26@ %6.1f KiB  %7.1f %6.1f   %6.0f%%   %7.3f", v.name as NSString, Double(v.blockBytes) / 1024,
                 g, Double(v.bytesPerCB) / b / 1e9, 100 * g / sGBs, m * 1e3))
}
print("run size = one (layer, K/V, head) block. Normal rows read all 16 blocks of a layer, which sit side by side in each unit;\n\"lone\" rows read one block per unit, so every run is isolated (the worst case)")
