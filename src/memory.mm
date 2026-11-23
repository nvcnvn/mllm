#include "memory.h"

#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>

#include <unistd.h>

#include <algorithm>
#include <nlohmann/json.hpp>

#include <cassert>
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "spec.h"

namespace fs = std::filesystem;

namespace mllm {

// The weights as GpuOps reads them, rewritten once into <cache>/weights/<model_key>-v3/ and reused: (a) Metal needs
// buffer offsets aligned, but safetensors only pads its JSON header if the writer chose to (mlx-community files
// don't), so the header is padded to put data at 0 mod 16 and every tensor at 0 mod 16; (b) each MERGED chain's
// weight / scales / biases are written back to back, so a merged matmul is a view of the file, not a copy at load
// (architecture.md Deferred, "before the 27B target"); (c) only language_model.* tensors (v3: the vision tower,
// 0.9 GB at 27B, is never read). Tensor bytes and names are unchanged. A chain member in another source file moves
// to its first member's file.
static std::vector<std::string> aligned(const std::string& path) {
    const std::vector<std::string> files = files_in(path);
    const fs::path dir = fs::path(spec::cache_dir()) / "weights" / (spec::model_key(path) + "-v3");
    std::vector<std::string> out;
    for (const auto& f : files) out.push_back((dir / fs::path(f).filename()).string());
    if (std::all_of(out.begin(), out.end(), [](const std::string& o) { return fs::exists(o); })) return out;

    const Safetensors src(files);
    // name -> its chain's names (first member) or "" (a later member: written with the first)
    auto chain = [&](const std::string& name) -> std::vector<std::string> {
        const size_t dot = name.rfind('.');
        const std::string module = name.substr(0, dot), suffix = name.substr(dot);
        for (const auto& c : MERGED)
            for (size_t k = 0; k < c.size(); k++) {
                if (module.size() <= c[k].size() || module.compare(module.size() - c[k].size(), c[k].size(), c[k]) ||
                    module[module.size() - c[k].size() - 1] != '.')
                    continue;
                const std::string prefix = module.substr(0, module.size() - c[k].size());
                if (k) return {};
                std::vector<std::string> names;
                for (const auto& m : c) src.at(prefix + m + suffix), names.push_back(prefix + m + suffix);
                return names;
            }
        return {name};
    };
    fs::create_directories(dir);
    for (const auto& e : fs::directory_iterator(dir)) {  // a crashed rewrite's partial copies (<file>.tmp<pid>)
        const std::string n = e.path().filename().string();
        const size_t at = n.rfind(".tmp");
        if (at != std::string::npos && kill(std::atoi(n.c_str() + at + 4), 0) && errno == ESRCH) fs::remove(e.path());
    }
    for (size_t fi = 0; fi < files.size(); fi++) {
        std::vector<std::pair<size_t, std::string>> by_offset;  // this file's tensors in their source order
        for (const auto& [name, t] : src.tensors)
            if (t.file == (int)fi && name.starts_with("language_model.")) by_offset.emplace_back(t.offset, name);
        std::sort(by_offset.begin(), by_offset.end());
        nlohmann::ordered_json h;
        std::vector<const Tensor*> order;
        size_t off = 0;
        for (const auto& [_, first] : by_offset) {
            const std::vector<std::string> names = chain(first);
            off = (off + 15) / 16 * 16;
            for (const auto& name : names) {
                const Tensor& t = src.at(name);
                if (off % 16) throw std::runtime_error(name + ": chain member not 16-byte sized");
                h[name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, off + t.nbytes}}};
                order.push_back(&t);
                off += t.nbytes;
            }
        }
        std::string header = h.dump();
        header.resize((8 + header.size() + 15) / 16 * 16 - 8, ' ');
        const uint64_t n = header.size();
        const fs::path tmp = out[fi] + ".tmp" + std::to_string(getpid());
        std::ofstream o(tmp, std::ios::binary);
        o.write((const char*)&n, 8).write(header.data(), n);
        size_t pos = 0;
        static const char zeros[16] = {};
        for (const Tensor* t : order) {
            o.write(zeros, (16 - pos % 16) % 16), pos = (pos + 15) / 16 * 16;
            o.write((const char*)src.data(*t), t->nbytes), pos += t->nbytes;
        }
        o.close();
        if (!o) throw std::runtime_error(tmp.string() + ": write failed");
        fs::rename(tmp, out[fi]);  // atomic: a reader never sees a partial copy
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

bool Arena::admit(uint32_t n, const std::function<bool()>& evict) {
    if (!can_ever(n)) return false;
    while (!fits(n)) {
        [[maybe_unused]] const bool evicted = evict && evict();
        assert(evicted && "can_ever counted cache-only units nothing evicts");
    }
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
        if (cache_refs[s]) cache_only_units++;
        else drop(s);
    }
}

void Arena::ref(uint32_t s) {
    assert((req_refs[s] || cache_refs[s]) && "ref on a free unit");
    if (!req_refs[s]++) cache_only_units--;
}

void Arena::cache_ref(uint32_t s) {
    assert(req_refs[s] && "the cache takes units from a request");
    cache_refs[s]++;
}

void Arena::cache_release(uint32_t s) {
    assert(cache_refs[s] > 0);
    if (--cache_refs[s] || req_refs[s]) return;
    cache_only_units--;
    drop(s);
}

void Arena::drop(uint32_t s) {
    assert(pending.empty() || pending.back().second <= committed);
    pending.emplace_back(s, committed);
    free_units++;
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
size_t Memory::unit_of(const ModelSpec& s, int T) {
    size_t kv = s.full_layers().size() * 2 * s.n_kv_heads * T * s.head_dim * ELEM;
    return (kv + PAGE - 1) / PAGE * PAGE;
}

static size_t state_layer_bytes(const ModelSpec& s) {  // conv then delta, per linear layer
    return (s.conv_kernel - 1) * s.conv_dim() * ELEM + (size_t)s.lin_v_heads * s.lin_v_dim * s.lin_k_dim * 4;
}

uint32_t Memory::state_units_of(const ModelSpec& s, int T) {
    const size_t u = unit_of(s, T);
    return (uint32_t)((state_layer_bytes(s) * s.linear_layers().size() + u - 1) / u);
}

Memory::Memory(const ModelSpec& spec, id<MTLDevice> device, size_t arena_bytes, int page_tokens)
    : spec(spec), device(device), T(page_tokens),
      arena(unit_of(spec, page_tokens), (uint32_t)(arena_bytes / unit_of(spec, page_tokens))) {
    conv_bytes = (spec.conv_kernel - 1) * spec.conv_dim() * ELEM;
    delta_bytes = (size_t)spec.lin_v_heads * spec.lin_v_dim * spec.lin_k_dim * 4;
    size_t per_layer = conv_bytes + delta_bytes;
    assert(per_layer == state_layer_bytes(spec));
    auto lin = spec.linear_layers();
    state_units = state_units_of(spec, page_tokens);
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
    allocations++, allocated += b.length;
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
    allocations++, allocated += b.length;
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

StepTables Memory::step_tables(int max_rows, size_t max_ctx) {
    // Rows share units (prefix cache), so size by rows: a row lists its pages and state units, each at most once
    const size_t per_row = std::min<size_t>(arena.n_units, units_for(max_ctx) + state_units);
    size_t words = 3 + (size_t)max_rows * (3 + per_row);
    return {{buffer(words * 4), buffer(words * 4)}, words, arena.n_units};
}

}  // namespace mllm
