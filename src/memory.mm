#include "memory.h"

#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>

#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "spec.h"

namespace fs = std::filesystem;

namespace mllm {

// Metal needs buffer offsets aligned, but safetensors only pads its JSON header if the writer chose to
// (mlx-community files don't). Such a file gets a copy with the header padded by spaces so data starts at
// 0 mod 16, kept under <cache>/weights/<model_key>/ and reused. Tensor bytes are unchanged.
static std::vector<std::string> aligned(const std::string& path) {
    std::vector<std::string> out;
    for (const auto& f : files_in(path)) {
        std::ifstream in(f, std::ios::binary);
        uint64_t n = 0;
        if (!in.read((char*)&n, 8)) throw std::runtime_error(f + ": cannot read header length");
        if ((8 + n) % 16 == 0) {
            out.push_back(f);
            continue;
        }
        fs::path dst = fs::path(spec::cache_dir()) / "weights" / spec::model_key(path) / fs::path(f).filename();
        if (!fs::exists(dst)) {
            uint64_t n16 = (8 + n + 15) / 16 * 16 - 8;
            std::string h(n16, ' ');
            if (n > n16 || !in.read(h.data(), n)) throw std::runtime_error(f + ": bad header");
            fs::create_directories(dst.parent_path());
            fs::path tmp = dst;
            tmp += ".tmp" + std::to_string(getpid());
            std::ofstream o(tmp, std::ios::binary);
            o.write((const char*)&n16, 8).write(h.data(), n16) << in.rdbuf();
            o.close();
            if (!o) throw std::runtime_error(tmp.string() + ": write failed");
            fs::rename(tmp, dst);  // atomic: a reader never sees a partial copy
        }
        out.push_back(dst.string());
    }
    return out;
}

Weights::Weights(const std::string& path, id<MTLDevice> device) : st(aligned(path)) {
    for (const auto& [name, t] : st.tensors)
        if (t.offset % 16) throw std::runtime_error(name + ": tensor offset not 16-byte aligned");
    for (const auto& m : st.files) {
        // the mapping outlives the buffer (both live in this object), so no deallocator
        id<MTLBuffer> b = [device newBufferWithBytesNoCopy:m.base
                                                    length:m.size
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (!b) throw std::runtime_error(m.path + ": newBufferWithBytesNoCopy failed");
        buffers.push_back(b);
    }
}

Weights::View Weights::operator[](const std::string& name) const {
    const Tensor& t = st.at(name);
    return {buffers[t.file], t.offset, t};
}

Arena::Arena(size_t unit_bytes, uint32_t n_units)
    : unit_bytes(unit_bytes), n_units(n_units), free((n_units + 63) / 64, ~0ull), req_refs(n_units),
      cache_refs(n_units), free_units(n_units) {
    if (n_units % 64) free.back() = (1ull << (n_units % 64)) - 1;
}

bool Arena::admit(uint32_t n) {
    if (!can_ever(n)) return false;
    // ponytail: no prefix cache yet, so cache_only_units == 0 and can_ever == fits. With the cache, the eviction
    // loop goes here: while (!fits(n)) evict the cache's next victim.
    assert(fits(n));
    reserved += n;
    return true;
}

uint32_t Arena::alloc() {
    assert(reserved > 0 && "alloc without reservation");
    size_t w = 0;
    while (w < free.size() && !free[w]) w++;
    // admit guarantees a unit with no references; if all of them are still pending, the caller waits for the
    // GPU event to pass their step and calls retire first.
    assert(w < free.size() && "no free unit: retire(signaled) first");
    uint32_t s = (uint32_t)(w * 64 + __builtin_ctzll(free[w]));
    free[w] &= free[w] - 1;
    req_refs[s] = 1;
    free_units--, reserved--;
    return s;
}

void Arena::unreserve(uint32_t n) {
    assert(n <= reserved);
    reserved -= n;
}

void Arena::release(const std::vector<uint32_t>& slots) {
    for (uint32_t s : slots) {
        assert(req_refs[s] > 0);
        if (--req_refs[s]) continue;
        if (cache_refs[s]) {
            cache_only_units++;
            continue;
        }
        assert(pending.empty() || pending.back().second <= committed);
        pending.emplace_back(s, committed);
        free_units++;
    }
}

void Arena::retire(uint64_t signaled) {
    for (; !pending.empty() && pending.front().second <= signaled; pending.pop_front())
        free[pending.front().first / 64] |= 1ull << (pending.front().first % 64);
}

id<MTLBuffer> StepTables::begin(uint64_t step, uint64_t signaled) {
    // before filling step N+2's table, step N must be done: then only step N+1 can be in flight, on the other copy
    assert(signaled + 2 >= step && "CPU more than one step ahead of the GPU");
    return copies[step % 2];
}

void StepTables::write(id<MTLBuffer> table, const std::vector<Row>& rows) const {
    const uint32_t R = (uint32_t)rows.size();
    size_t n_slots = 0, n_state = R ? rows[0].state.size() : 0;
    for (const Row& r : rows) n_slots += r.slots.size() + r.state.size(), assert(r.state.size() == n_state);
    assert(3 + 3 * R + n_slots <= words && table.length >= words * 4);
    uint32_t* t = (uint32_t*)table.contents;
    t[0] = R, t[1] = (uint32_t)n_state;
    uint32_t* off = t + 2 + 2 * R;
    off[0] = 0;
    for (uint32_t b = 0; b < R; b++) {
        assert(rows[b].pos >= 0 && rows[b].n >= 0);
        for (uint32_t s : rows[b].slots) assert(s < n_units);
        t[2 + b] = rows[b].pos, t[2 + R + b] = rows[b].n;
        off[b + 1] = off[b] + (uint32_t)rows[b].slots.size();
        std::copy(rows[b].slots.begin(), rows[b].slots.end(), off + R + 1 + off[b]);
    }
    for (uint32_t b = 0; b < R; b++) {
        for (uint32_t s : rows[b].state) assert(s < n_units);
        std::copy(rows[b].state.begin(), rows[b].state.end(), off + R + 1 + off[R] + b * n_state);
    }
}

// memory.py: a unit is one KV page (T tokens) of every full layer, rounded up to the OS page.
static size_t unit_of(const ModelSpec& s, int T) {
    size_t kv = s.full_layers().size() * 2 * s.n_kv_heads * T * s.head_dim * ELEM;
    return (kv + PAGE - 1) / PAGE * PAGE;
}

Memory::Memory(const ModelSpec& spec, id<MTLDevice> device, size_t arena_bytes, int page_tokens)
    : spec(spec), device(device), T(page_tokens),
      arena(unit_of(spec, page_tokens), (uint32_t)(arena_bytes / unit_of(spec, page_tokens))) {
    conv_bytes = (spec.conv_kernel - 1) * spec.conv_dim() * ELEM;
    delta_bytes = (size_t)spec.lin_v_heads * spec.lin_v_dim * spec.lin_k_dim * 4;
    size_t per_layer = conv_bytes + delta_bytes;
    auto lin = spec.linear_layers();
    state_units = (uint32_t)((per_layer * lin.size() + arena.unit_bytes - 1) / arena.unit_bytes);
    state_off.assign(spec.n_layers(), SIZE_MAX);
    for (size_t j = 0; j < lin.size(); j++) state_off[lin[j]] = j * per_layer;

    arena_buffer = buffer((size_t)arena.n_units * arena.unit_bytes,
                          MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked);
    memset(arena_buffer.contents, 0, arena_buffer.length);  // pre-touch: page in now, not during a step
    NSError* err = nil;
    residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&err];
    if (!residency) throw std::runtime_error(std::string("residency set: ") + err.localizedDescription.UTF8String);
    [residency addAllocation:arena_buffer];
    [residency commit];
    [residency requestResidency];
}

size_t Memory::kv_offset(int g, int kv, int head, int token) const {
    return ((((size_t)g * 2 + kv) * spec.n_kv_heads + head) * T + token) * spec.head_dim * ELEM;
}

size_t Memory::state_offset(int layer) const {
    assert(state_off[layer] != SIZE_MAX && "not a linear layer");
    return state_off[layer];
}

id<MTLBuffer> Memory::buffer(size_t bytes, MTLResourceOptions options) {
    assert(!frozen && "allocation after startup");
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:options];
    if (!b) throw std::runtime_error("newBufferWithLength failed: " + std::to_string(bytes) + " bytes");
    allocations++;
    return b;
}

Memory::Surface Memory::surface(int rows, int cols) {
    assert(!frozen && "allocation after startup");
    NSDictionary* attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}, (id)kCVPixelBufferMetalCompatibilityKey : @YES};
    CVPixelBufferRef pb = nullptr;
    if (CVPixelBufferCreate(nullptr, cols, rows, kCVPixelFormatType_OneComponent16Half, (__bridge CFDictionaryRef)attrs,
                            &pb) != kCVReturnSuccess)
        throw std::runtime_error("CVPixelBufferCreate failed: " + std::to_string(rows) + " x " + std::to_string(cols));
    id owner = (__bridge_transfer id)pb;
    IOSurfaceRef s = CVPixelBufferGetIOSurface(pb);
    // allocSize has a small tail: wrap just the rows, rounded up to a page
    const size_t page = getpagesize(), bpr = IOSurfaceGetBytesPerRow(s), len = (bpr * rows + page - 1) / page * page;
    void* base = IOSurfaceGetBaseAddress(s);
    // ponytail: no fallback to a texture view (tools/ane_iosurface.swift); add one if a surface ever comes unaligned
    if ((uintptr_t)base % page || len > (IOSurfaceGetAllocSize(s) + page - 1) / page * page || bpr % 2)
        throw std::runtime_error("IOSurface base or size not page-aligned: no MTLBuffer view");
    id<MTLBuffer> b = [device newBufferWithBytesNoCopy:base length:len options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (!b) throw std::runtime_error("newBufferWithBytesNoCopy over an IOSurface failed");
    allocations++;
    return {owner, b, (uint32_t)(bpr / 2)};
}

Workspace Memory::workspace(int chunk, int max_batch) {
    const ModelSpec& s = spec;
    size_t R = std::max(chunk, max_batch) * ELEM;  // bytes per element column, over all rows
    size_t qd = (size_t)s.n_heads * s.head_dim, kvd = (size_t)s.n_kv_heads * s.head_dim;
    size_t vd = (size_t)s.lin_v_heads * s.lin_v_dim;
    return {
        .tokens = buffer(std::max(chunk, max_batch) * sizeof(int32_t)),
        .x = buffer(R * s.hidden),
        .o = buffer(R * std::max(qd, vd)),
        .q = buffer(R * qd),
        .k = buffer(R * kvd),
        .v = buffer(R * kvd),
        .gate = buffer(R * qd),
        .qkv = buffer(R * s.conv_dim()),
        .z = buffer(R * vd),
        .b = buffer(R * s.lin_v_heads),
        .a = buffer(R * s.lin_v_heads),
        .logits = buffer((size_t)max_batch * s.vocab * ELEM),
    };
}

StepTables Memory::step_tables(int max_rows) {
    // ponytail: slot words <= n_units while no unit is shared; the prefix cache lets rows share units, then size
    // by the sum of the rows' slot counts.
    size_t words = 3 + 3 * (size_t)max_rows + arena.n_units;
    return {{buffer(words * 4), buffer(words * 4)}, words, arena.n_units};
}

}  // namespace mllm
