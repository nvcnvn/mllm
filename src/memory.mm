#include "memory.h"

#include <unistd.h>

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

}  // namespace mllm
