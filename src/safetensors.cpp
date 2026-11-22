#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mllm {

size_t dtype_size(const std::string& d) {
    static const std::map<std::string, size_t> sizes = {
        {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1}, {"BF16", 2}, {"F16", 2}, {"I16", 2},
        {"U16", 2},  {"F32", 4}, {"I32", 4}, {"U32", 4},      {"F64", 8},      {"I64", 8},  {"U64", 8}};
    auto it = sizes.find(d);
    if (it == sizes.end()) throw std::runtime_error("unknown safetensors dtype " + d);
    return it->second;
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
}

std::vector<std::string> files_in(const std::string& path) {
    if (!fs::is_directory(path)) return {path};
    std::vector<std::string> paths;
    for (const auto& e : fs::directory_iterator(path))
        if (e.path().extension() == ".safetensors") paths.push_back(e.path().string());
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error(path + ": no *.safetensors");
    return paths;
}

Safetensors::Safetensors(const std::vector<std::string>& paths) {
    try {
        for (const auto& p : paths) {
            int fd = open(p.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error(p + ": " + strerror(errno));
            struct stat st;
            if (fstat(fd, &st)) {
                close(fd);
                throw std::runtime_error(p + ": " + strerror(errno));
            }
            size_t fsize = (size_t)st.st_size, size = (fsize + PAGE - 1) / PAGE * PAGE;
            void* base = fsize < 8 ? MAP_FAILED : mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (base == MAP_FAILED) throw std::runtime_error(p + ": cannot map");
            files.push_back({p, base, size});

            uint64_t hlen;
            memcpy(&hlen, base, 8);
            if (hlen > fsize - 8) throw std::runtime_error(p + ": bad header length");
            auto h = nlohmann::json::parse((const char*)base + 8, (const char*)base + 8 + hlen);
            size_t data0 = 8 + hlen;
            for (const auto& [name, v] : h.items()) {
                if (name == "__metadata__") continue;
                Tensor t{v.at("dtype"), v.at("shape").get<std::vector<int64_t>>(), (int)files.size() - 1, 0, 0};
                size_t a = v.at("data_offsets")[0], b = v.at("data_offsets")[1];
                if (a > b || b > fsize - data0 || b - a != t.numel() * dtype_size(t.dtype))
                    throw std::runtime_error(p + ": bad offsets for " + name);
                t.offset = data0 + a;
                t.nbytes = b - a;
                if (!tensors.emplace(name, std::move(t)).second) throw std::runtime_error(p + ": duplicate " + name);
            }
        }
    } catch (...) {  // the destructor doesn't run for a throwing constructor
        for (auto& m : files) munmap(m.base, m.size);
        throw;
    }
}

Safetensors::~Safetensors() {
    for (auto& m : files) munmap(m.base, m.size);
}

const Tensor& Safetensors::at(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("no tensor " + name);
    return it->second;
}

}  // namespace mllm
