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

// Does module (e.g. "language_model.model.layers.3.mlp.gate_proj") end in "." + member ("mlp.gate_proj")?
static bool ends_in_member(const std::string& module, const std::string& member) {
    return module.size() > member.size() &&
           !module.compare(module.size() - member.size(), member.size(), member) &&
           module[module.size() - member.size() - 1] == '.';
}

// The tensors written together starting at tensor `name`: its whole MERGED chain if name belongs to a chain's first
// member; nothing if it belongs to a later member (written with the first); else just name.
static std::vector<std::string> written_together(const Safetensors& source, const std::string& name) {
    const size_t dot = name.rfind('.');
    const std::string module = name.substr(0, dot), suffix = name.substr(dot);
    for (const auto& chain : MERGED)
        for (size_t member = 0; member < chain.size(); member++) {
            if (!ends_in_member(module, chain[member])) continue;
            const std::string prefix = module.substr(0, module.size() - chain[member].size());
            if (member) return {};
            std::vector<std::string> names;
            for (const auto& m : chain) source.at(prefix + m + suffix), names.push_back(prefix + m + suffix);
            return names;
        }
    return {name};
}

// A crashed rewrite's partial copies (<file>.tmp<pid>, the pid no longer running) are removed.
static void remove_stale_partial_copies(const fs::path& dir) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string file_name = entry.path().filename().string();
        const size_t at = file_name.rfind(".tmp");
        if (at != std::string::npos && kill(std::atoi(file_name.c_str() + at + 4), 0) && errno == ESRCH)
            fs::remove(entry.path());
    }
}

// Writes source file file_index's language_model.* tensors, in their source order with each chain together, to
// out_path: the header padded so data starts at 0 mod 16, every tensor at 0 mod 16.
static void write_aligned_file(const Safetensors& source, size_t file_index, const std::string& out_path) {
    std::vector<std::pair<size_t, std::string>> by_offset;  // this file's tensors in their source order
    for (const auto& [name, tensor] : source.tensors)
        if (tensor.file == (int)file_index && name.starts_with("language_model."))
            by_offset.emplace_back(tensor.offset, name);
    std::sort(by_offset.begin(), by_offset.end());
    nlohmann::ordered_json header_json;
    std::vector<const Tensor*> write_order;
    size_t offset = 0;
    for (const auto& [_, first] : by_offset) {
        const std::vector<std::string> names = written_together(source, first);
        offset = (offset + 15) / 16 * 16;
        for (const auto& name : names) {
            const Tensor& tensor = source.at(name);
            if (offset % 16) throw std::runtime_error(name + ": chain member not 16-byte sized");
            header_json[name] = {{"dtype", tensor.dtype}, {"shape", tensor.shape},
                                 {"data_offsets", {offset, offset + tensor.nbytes}}};
            write_order.push_back(&tensor);
            offset += tensor.nbytes;
        }
    }
    std::string header = header_json.dump();
    header.resize((8 + header.size() + 15) / 16 * 16 - 8, ' ');
    const uint64_t header_size = header.size();
    const fs::path temporary = out_path + ".tmp" + std::to_string(getpid());
    std::ofstream file(temporary, std::ios::binary);
    file.write((const char*)&header_size, 8).write(header.data(), header_size);
    size_t written = 0;
    static const char zeros[16] = {};
    for (const Tensor* tensor : write_order) {
        file.write(zeros, (16 - written % 16) % 16), written = (written + 15) / 16 * 16;
        file.write((const char*)source.data(*tensor), tensor->nbytes), written += tensor->nbytes;
    }
    file.close();
    if (!file) throw std::runtime_error(temporary.string() + ": write failed");
    fs::rename(temporary, out_path);  // atomic: a reader never sees a partial copy
}

// The weights as GpuOps reads them, rewritten once into <cache>/weights/<model_key>-v3/ and reused: (a) Metal needs
// buffer offsets aligned, but safetensors only pads its JSON header if the writer chose to (mlx-community files
// don't), so the header is padded to put data at 0 mod 16 and every tensor at 0 mod 16; (b) each MERGED chain's
// weight / scales / biases are written back to back, so a merged multiply is a view of the file, not a copy at load
// (architecture.md Deferred, "before the 27B target"); (c) only language_model.* tensors (v3: the vision tower,
// 0.9 GB at 27B, is never read). Tensor bytes and names are unchanged. A chain member in another source file moves
// to its first member's file.
static std::vector<std::string> aligned_weight_files(const std::string& path) {
    const std::vector<std::string> files = files_in(path);
    const fs::path dir = fs::path(spec::cache_dir()) / "weights" / (spec::model_key(path) + "-v3");
    std::vector<std::string> out;
    for (const auto& file : files) out.push_back((dir / fs::path(file).filename()).string());
    if (std::all_of(out.begin(), out.end(), [](const std::string& o) { return fs::exists(o); })) return out;

    const Safetensors source(files);
    fs::create_directories(dir);
    remove_stale_partial_copies(dir);
    for (size_t file_index = 0; file_index < files.size(); file_index++)
        write_aligned_file(source, file_index, out[file_index]);
    return out;
}

Weights::Weights(const std::string& path, id<MTLDevice> device) : st(aligned_weight_files(path)) {
    for (const auto& [name, tensor] : st.tensors)
        if (tensor.offset % 16) throw std::runtime_error(name + ": tensor offset not 16-byte aligned");
    for (const auto& file : st.files) {
        // the mapping outlives the buffer (both live in this object), so no deallocator
        id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:file.base
                                                         length:file.size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!buffer) throw std::runtime_error(file.path + ": newBufferWithBytesNoCopy failed");
        buffers.push_back(buffer);
    }
}

Weights::View Weights::operator[](const std::string& name) const {
    const Tensor& tensor = st.at(name);
    return {buffers[tensor.file], tensor.offset, tensor};
}

Arena::Arena(size_t unit_bytes, uint32_t n_units)
    : unit_bytes(unit_bytes), n_units(n_units), free_bitmap((n_units + 63) / 64, ~0ull), request_refs(n_units),
      cache_refs(n_units), free_units(n_units) {
    if (n_units % 64) free_bitmap.back() = (1ull << (n_units % 64)) - 1;
}

bool Arena::admit(uint32_t count, const std::function<bool()>& evict) {
    if (!can_ever(count)) return false;
    while (!fits(count)) {
        [[maybe_unused]] const bool evicted = evict && evict();
        assert(evicted && "can_ever counted cache-only units nothing evicts");
    }
    reserved += count;
    return true;
}

uint32_t Arena::alloc() {
    assert(reserved > 0 && "alloc without reservation");
    size_t word = 0;
    while (word < free_bitmap.size() && !free_bitmap[word]) word++;
    // admit guarantees a unit with no references; if all of them are still pending, the caller waits for the
    // GPU event to pass their step and calls retire first.
    assert(word < free_bitmap.size() && "no free unit: retire(signaled) first");
    uint32_t unit = (uint32_t)(word * 64 + __builtin_ctzll(free_bitmap[word]));
    free_bitmap[word] &= free_bitmap[word] - 1;  // clear the lowest set bit: that unit
    request_refs[unit] = 1;
    free_units--, reserved--;
    return unit;
}

void Arena::unreserve(uint32_t count) {
    assert(count <= reserved);
    reserved -= count;
}

void Arena::release(const std::vector<uint32_t>& units) {
    for (uint32_t unit : units) {
        assert(request_refs[unit] > 0);
        if (--request_refs[unit]) continue;
        if (cache_refs[unit]) cache_only_units++;
        else make_pending(unit);
    }
}

void Arena::ref(uint32_t unit) {
    assert((request_refs[unit] || cache_refs[unit]) && "ref on a free unit");
    if (!request_refs[unit]++) cache_only_units--;
}

void Arena::cache_ref(uint32_t unit) {
    assert(request_refs[unit] && "the cache takes units from a request");
    cache_refs[unit]++;
}

void Arena::cache_release(uint32_t unit) {
    assert(cache_refs[unit] > 0);
    if (--cache_refs[unit] || request_refs[unit]) return;
    cache_only_units--;
    make_pending(unit);
}

void Arena::make_pending(uint32_t unit) {
    assert(pending.empty() || pending.back().second <= committed);
    pending.emplace_back(unit, committed);
    free_units++;
}

void Arena::retire(uint64_t signaled) {
    for (; !pending.empty() && pending.front().second <= signaled; pending.pop_front())
        free_bitmap[pending.front().first / 64] |= 1ull << (pending.front().first % 64);
}

id<MTLBuffer> StepTables::begin(uint64_t step, uint64_t signaled) {
    // before filling step N+2's table, step N must be done: then only step N+1 can be in flight, on the other copy
    assert(signaled + 2 >= step && "CPU more than one step ahead of the GPU");
    return copies[step % 2];
}

void StepTables::write(id<MTLBuffer> table, const std::vector<Row>& rows) const {
    const uint32_t row_count = (uint32_t)rows.size();
    size_t unit_count = 0, state_per_row = row_count ? rows[0].state.size() : 0;
    for (const Row& r : rows) unit_count += r.pages.size() + r.state.size(), assert(r.state.size() == state_per_row);
    assert(3 + 3 * row_count + unit_count <= words && table.length >= words * 4);
    uint32_t* table_words = (uint32_t*)table.contents;
    table_words[0] = row_count, table_words[1] = (uint32_t)state_per_row;
    uint32_t* offsets = table_words + 2 + 2 * row_count;  // row b's pages start at units[offsets[b]]
    uint32_t* units = offsets + row_count + 1;
    offsets[0] = 0;
    for (uint32_t b = 0; b < row_count; b++) {
        assert(rows[b].start_position >= 0 && rows[b].token_count >= 0);
        for (uint32_t unit : rows[b].pages) assert(unit < n_units);
        table_words[2 + b] = rows[b].start_position, table_words[2 + row_count + b] = rows[b].token_count;
        offsets[b + 1] = offsets[b] + (uint32_t)rows[b].pages.size();
        std::copy(rows[b].pages.begin(), rows[b].pages.end(), units + offsets[b]);
    }
    for (uint32_t b = 0; b < row_count; b++) {
        for (uint32_t unit : rows[b].state) assert(unit < n_units);
        std::copy(rows[b].state.begin(), rows[b].state.end(), units + offsets[row_count] + b * state_per_row);
    }
}

// memory.py: a unit is one KV page (page_tokens tokens) of every full layer, rounded up to the OS page.
size_t Memory::unit_of(const ModelSpec& model, int page_tokens) {
    size_t kv = model.full_layers().size() * 2 * model.key_value_heads * page_tokens * model.head_dim * ELEM;
    return (kv + PAGE - 1) / PAGE * PAGE;
}

static size_t state_layer_bytes(const ModelSpec& model) {  // conv then delta, per linear layer
    return (model.conv_kernel - 1) * model.conv_dim() * ELEM +
           (size_t)model.linear_value_heads * model.linear_value_dim * model.linear_key_dim * 4;
}

uint32_t Memory::state_units_of(const ModelSpec& model, int page_tokens) {
    const size_t unit = unit_of(model, page_tokens);
    return (uint32_t)((state_layer_bytes(model) * model.linear_layers().size() + unit - 1) / unit);
}

Memory::Memory(const ModelSpec& spec, id<MTLDevice> device, size_t arena_bytes, int page_tokens)
    : spec(spec), device(device), page_tokens(page_tokens),
      arena(unit_of(spec, page_tokens), (uint32_t)(arena_bytes / unit_of(spec, page_tokens))) {
    conv_bytes = (spec.conv_kernel - 1) * spec.conv_dim() * ELEM;
    delta_bytes = (size_t)spec.linear_value_heads * spec.linear_value_dim * spec.linear_key_dim * 4;
    size_t bytes_per_layer = conv_bytes + delta_bytes;
    assert(bytes_per_layer == state_layer_bytes(spec));
    auto linear_layers = spec.linear_layers();
    state_units = state_units_of(spec, page_tokens);
    state_offsets.assign(spec.n_layers(), SIZE_MAX);
    for (size_t j = 0; j < linear_layers.size(); j++) state_offsets[linear_layers[j]] = j * bytes_per_layer;

    arena_buffer = buffer((size_t)arena.n_units * arena.unit_bytes,
                          MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked);
    memset(arena_buffer.contents, 0, arena_buffer.length);  // pre-touch: page in now, not during a step
    NSError* error = nil;
    residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!residency) throw std::runtime_error(std::string("residency set: ") + error.localizedDescription.UTF8String);
    [residency addAllocation:arena_buffer];
    [residency commit];
    [residency requestResidency];
}

size_t Memory::kv_offset(int full_layer, int key_or_value, int head, int token) const {
    return ((((size_t)full_layer * 2 + key_or_value) * spec.key_value_heads + head) * page_tokens + token) *
           spec.head_dim * ELEM;
}

size_t Memory::state_offset(int layer) const {
    assert(state_offsets[layer] != SIZE_MAX && "not a linear layer");
    return state_offsets[layer];
}

id<MTLBuffer> Memory::buffer(size_t bytes, MTLResourceOptions options) {
    assert(!frozen && "allocation after startup");
    id<MTLBuffer> allocation = [device newBufferWithLength:bytes options:options];
    if (!allocation) throw std::runtime_error("newBufferWithLength failed: " + std::to_string(bytes) + " bytes");
    allocations++, allocated += allocation.length;
    return allocation;
}

Memory::Surface Memory::surface(int rows, int cols) {
    assert(!frozen && "allocation after startup");
    NSDictionary* attributes =
        @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}, (id)kCVPixelBufferMetalCompatibilityKey : @YES};
    CVPixelBufferRef pixel_buffer = nullptr;
    if (CVPixelBufferCreate(nullptr, cols, rows, kCVPixelFormatType_OneComponent16Half,
                            (__bridge CFDictionaryRef)attributes, &pixel_buffer) != kCVReturnSuccess)
        throw std::runtime_error("CVPixelBufferCreate failed: " + std::to_string(rows) + " x " + std::to_string(cols));
    id owner = (__bridge_transfer id)pixel_buffer;
    IOSurfaceRef io_surface = CVPixelBufferGetIOSurface(pixel_buffer);
    // allocSize has a small tail: wrap just the rows, rounded up to a page
    const size_t page_size = getpagesize(), bytes_per_row = IOSurfaceGetBytesPerRow(io_surface),
                 length = (bytes_per_row * rows + page_size - 1) / page_size * page_size;
    void* base = IOSurfaceGetBaseAddress(io_surface);
    // ponytail: no fallback to a texture view (tools/ane_iosurface.swift); add one if a surface ever comes unaligned
    if ((uintptr_t)base % page_size ||
        length > (IOSurfaceGetAllocSize(io_surface) + page_size - 1) / page_size * page_size || bytes_per_row % 2)
        throw std::runtime_error("IOSurface base or size not page-aligned: no MTLBuffer view");
    id<MTLBuffer> view = [device newBufferWithBytesNoCopy:base length:length options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!view) throw std::runtime_error("newBufferWithBytesNoCopy over an IOSurface failed");
    allocations++, allocated += view.length;
    return {owner, view, (uint32_t)(bytes_per_row / 2)};
}

Workspace Memory::workspace(int chunk, int max_batch) {
    const ModelSpec& s = spec;
    size_t column_bytes = std::max(chunk, max_batch) * ELEM;  // bytes of one element per row, over all rows
    size_t query_size = (size_t)s.query_heads * s.head_dim, key_value_size = (size_t)s.key_value_heads * s.head_dim;
    size_t linear_value_size = (size_t)s.linear_value_heads * s.linear_value_dim;
    return {
        .token_ids = buffer(std::max(chunk, max_batch) * sizeof(int32_t)),
        .residual = buffer(column_bytes * s.hidden),
        .attention_result = buffer(column_bytes * std::max(query_size, linear_value_size)),
        .query = buffer(column_bytes * query_size),
        .key = buffer(column_bytes * key_value_size),
        .value = buffer(column_bytes * key_value_size),
        .query_gate = buffer(column_bytes * query_size),
        .query_key_value = buffer(column_bytes * s.conv_dim()),
        .output_gate = buffer(column_bytes * linear_value_size),
        .update_rate_input = buffer(column_bytes * s.linear_value_heads),
        .decay_input = buffer(column_bytes * s.linear_value_heads),
        .next_token_scores = buffer((size_t)max_batch * s.vocab_size * ELEM),
    };
}

StepTables Memory::step_tables(int max_rows, size_t max_context) {
    // Rows share units (prefix cache), so size by rows: a row lists its pages and state units, each at most once
    const size_t units_per_row = std::min<size_t>(arena.n_units, units_for(max_context) + state_units);
    size_t words = 3 + (size_t)max_rows * (3 + units_per_row);
    return {{buffer(words * 4), buffer(words * 4)}, words, arena.n_units};
}

}  // namespace mllm
