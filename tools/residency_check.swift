// Does an MTLResidencySet keep file-backed (mmap) weight pages resident? (memory-design.md, "To verify" #2)
//
// Two identical files on the weights volume, both mmap'ed and wrapped with bytesNoCopy. A goes in a residency set
// (requestResidency, queue.addResidencySet); B does not and is the control. Both are read by the GPU, then the
// script adds memory pressure (incompressible anonymous memory, 1 GiB at a time) until B starts getting evicted,
// and reports what fraction of A and B is still in RAM (mincore) and how fast the GPU reads each afterwards.
//
// Safety: stops at the first of B < 10% resident, pressure level critical, swap growth > 2 GiB, or 52 GiB allocated,
// then frees everything at once. Expect other apps to get compressed for a few seconds. Needs ~8 GiB free on DIR.
//
// Mode "anon" repeats the test with anonymous buffers (makeBuffer + fill, the "pread into anonymous memory" fallback).
// The GPU reads under pressure page memory back in, so swap can grow well past the 2 GiB stop (7 GiB seen).
//
//   swiftc -O tools/residency_check.swift -o /tmp/rc && /tmp/rc /Volumes/T5 [GiB per file, default 4] [file|anon]
import Darwin
import Foundation
import Metal

let dir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/Volumes/T5"
let GiB = 1 << 30, size = (CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 4) * GiB
let page = Int(getpagesize())
let anon = CommandLine.arguments.count > 3 && CommandLine.arguments[3] == "anon"
func sysctlInt(_ n: String) -> Int { var v = 0, s = MemoryLayout<Int>.size; return sysctlbyname(n, &v, &s, nil, 0) == 0 ? v : -1 }
func swapUsed() -> Int { var u = xsw_usage(), s = MemoryLayout<xsw_usage>.size; sysctlbyname("vm.swapusage", &u, &s, nil, 0); return Int(u.xsu_used) }
func pressure() -> Int { var v: Int32 = 0, s = 4; sysctlbyname("kern.memorystatus_vm_pressure_level", &v, &s, nil, 0); return Int(v) }

// 1. files
func makeFile(_ name: String) -> (UnsafeMutableRawPointer, String) {
    let path = "\(dir)/.residency_check_\(name).bin"
    let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o600); precondition(fd >= 0, "cannot create \(path)")
    var chunk = [UInt64](repeating: 0, count: 64 << 20 / 8), x: UInt64 = name == "A" ? 0x1234 : 0x9876
    for off in stride(from: 0, to: size, by: 64 << 20) {
        for i in chunk.indices { x ^= x << 13; x ^= x >> 7; x ^= x << 17; chunk[i] = x }
        precondition(chunk.withUnsafeBytes { pwrite(fd, $0.baseAddress, $0.count, off_t(off)) } == 64 << 20)
    }
    fsync(fd)
    let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)!; precondition(p != MAP_FAILED); close(fd)
    return (p, path)
}

func resident(_ p: UnsafeMutableRawPointer) -> Double {
    var v = [CChar](repeating: 0, count: size / page)
    precondition(mincore(p, size, &v) == 0)
    return Double(v.filter { $0 & 1 != 0 }.count) / Double(v.count)
}

// 2. Metal
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func anonBuffer(_ seed: UInt64) -> MTLBuffer {
    let b = dev.makeBuffer(length: size, options: .storageModeShared)!
    DispatchQueue.concurrentPerform(iterations: 16) { t in
        let w = b.contents().advanced(by: t * size / 16).bindMemory(to: UInt64.self, capacity: size / 128)
        var x = (seed + UInt64(t)) &* 0x9E3779B97F4A7C15
        for i in 0..<(size / 128) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x }
    }
    return b
}
let bufA: MTLBuffer, bufB: MTLBuffer
var paths = [String]()
if anon {
    print("mode: anonymous buffers, 2 × \(size / GiB) GiB")
    bufA = anonBuffer(1); bufB = anonBuffer(1000)
} else {
    print("mode: mmap'ed files, writing 2 × \(size / GiB) GiB to \(dir) …")
    let (fa, pathA) = makeFile("A"), (fb, pathB) = makeFile("B")
    paths = [pathA, pathB]
    bufA = dev.makeBuffer(bytesNoCopy: fa, length: size, options: .storageModeShared, deallocator: nil)!
    bufB = dev.makeBuffer(bytesNoCopy: fb, length: size, options: .storageModeShared, deallocator: nil)!
}
defer { paths.forEach { unlink($0) } }
let pA = bufA.contents(), pB = bufB.contents()
let rd = MTLResidencySetDescriptor(); rd.label = "weights"
let set = try dev.makeResidencySet(descriptor: rd)
set.addAllocation(bufA); set.commit(); set.requestResidency(); q.addResidencySet(set)
let lib = try dev.makeLibrary(source: """
#include <metal_stdlib>
using namespace metal;
kernel void stream(device const uint4 *a [[buffer(0)]], device uint *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    uint acc = 0;
    for (uint i = gid; i < n; i += gs) { uint4 x = a[i]; acc += x.x ^ x.y ^ x.z ^ x.w; }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
""", options: nil)
let ps = try dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)
let sink = dev.makeBuffer(length: 16, options: .storageModeShared)!
func gpuRead(_ b: MTLBuffer) -> String {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    var n = UInt32(size / 16)
    e.setComputePipelineState(ps); e.setBuffer(b, offset: 0, index: 0); e.setBuffer(sink, offset: 0, index: 1); e.setBytes(&n, length: 4, index: 2)
    e.dispatchThreads(MTLSize(width: 20 * 1024 * 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); let wall = Date().timeIntervalSince(t0)
    if cb.status != .completed { return "GPU error: \(String(describing: cb.error))" }
    return String(format: "%.1f GB/s GPU time, %.0f ms wall", Double(size) / (cb.gpuEndTime - cb.gpuStartTime) / 1e9, wall * 1e3)
}
func report(_ tag: String) {
    print(String(format: "%@: resident A (in set) %5.1f%% · B (control) %5.1f%% · pressure level %d · swap %.1f GiB · free pages %.1f GiB", tag as NSString,
                 100 * resident(pA), 100 * resident(pB), pressure(), Double(swapUsed()) / Double(GiB), Double(sysctlInt("vm.page_free_count") * page) / Double(GiB)))
}
print("GPU read A: \(gpuRead(bufA)) · B: \(gpuRead(bufB))")
print("GPU read A: \(gpuRead(bufA)) · B: \(gpuRead(bufB))")
report("before pressure")

// 3. pressure
var hogs = [UnsafeMutableRawPointer]()
let swap0 = swapUsed()
var why = "cap of 52 GiB reached"
while hogs.count < 52 {
    let p = mmap(nil, GiB, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)!; precondition(p != MAP_FAILED)
    DispatchQueue.concurrentPerform(iterations: 16) { t in   // incompressible, so the compressor can't absorb it
        let w = p.advanced(by: t * GiB / 16).bindMemory(to: UInt64.self, capacity: GiB / 128)
        var x = UInt64(hogs.count * 16 + t + 1) &* 0x9E3779B97F4A7C15
        for i in 0..<(GiB / 128) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x }
    }
    hogs.append(p)
    let rB = resident(pB), lvl = pressure()
    if hogs.count % 4 == 0 || rB < 0.9 { report("+\(hogs.count) GiB") }
    if rB < 0.10 { why = "control B mostly evicted"; break }
    if lvl >= 4 { why = "pressure level critical"; break }
    if swapUsed() - swap0 > 2 * GiB { why = "swap grew by 2 GiB"; break }
}
print("stopped: \(why)")
report("under pressure")
print("GPU read A: \(gpuRead(bufA)) · B: \(gpuRead(bufB))")
report("after GPU reads")
hogs.forEach { munmap($0, GiB) }
report("pressure released")
