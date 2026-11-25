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

size_t dtype_size(const std::string& dtype) {
    static const std::map<std::string, size_t> sizes = {
        {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1}, {"BF16", 2}, {"F16", 2}, {"I16", 2},
        {"U16", 2},  {"F32", 4}, {"I32", 4}, {"U32", 4},      {"F64", 8},      {"I64", 8},  {"U64", 8}};
    auto found = sizes.find(dtype);
    if (found == sizes.end()) throw std::runtime_error("unknown safetensors dtype " + dtype);
    return found->second;
}

size_t Tensor::numel() const {
    size_t count = 1;
    for (auto size : shape) count *= (size_t)size;
    return count;
}

std::vector<std::string> files_in(const std::string& path) {
    if (!fs::is_directory(path)) return {path};
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(path))
        if (entry.path().extension() == ".safetensors") paths.push_back(entry.path().string());
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error(path + ": no *.safetensors");
    return paths;
}

// Maps the whole file read-only, rounded up to PAGE; sets file_size. Throws for a file under 8 bytes (no header length).
static Mapping map_file(const std::string& path, size_t& file_size) {
    int file = open(path.c_str(), O_RDONLY);
    if (file < 0) throw std::runtime_error(path + ": " + strerror(errno));
    struct stat status;
    if (fstat(file, &status)) {
        close(file);
        throw std::runtime_error(path + ": " + strerror(errno));
    }
    file_size = (size_t)status.st_size;
    const size_t mapped_size = (file_size + PAGE - 1) / PAGE * PAGE;
    void* base = file_size < 8 ? MAP_FAILED : mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, file, 0);
    close(file);
    if (base == MAP_FAILED) throw std::runtime_error(path + ": cannot map");
    return {path, base, mapped_size};
}

// The mapped file at base: an 8-byte header length, the JSON header, then the tensors' data. Adds a view of each tensor
// to tensors.
static void read_tensors(const std::string& path, const char* base, size_t file_size, int file_index,
                         std::map<std::string, Tensor>& tensors) {
    uint64_t header_length;
    memcpy(&header_length, base, 8);
    if (header_length > file_size - 8) throw std::runtime_error(path + ": bad header length");
    auto header = nlohmann::json::parse(base + 8, base + 8 + header_length);
    size_t data_start = 8 + header_length;
    for (const auto& [name, entry] : header.items()) {
        if (name == "__metadata__") continue;
        Tensor tensor{entry.at("dtype"), entry.at("shape").get<std::vector<int64_t>>(), file_index, 0, 0};
        size_t begin = entry.at("data_offsets")[0], end = entry.at("data_offsets")[1];
        if (begin > end || end > file_size - data_start || end - begin != tensor.numel() * dtype_size(tensor.dtype))
            throw std::runtime_error(path + ": bad offsets for " + name);
        tensor.offset = data_start + begin;
        tensor.nbytes = end - begin;
        if (!tensors.emplace(name, std::move(tensor)).second) throw std::runtime_error(path + ": duplicate " + name);
    }
}

Safetensors::Safetensors(const std::vector<std::string>& paths) {
    try {
        for (const auto& path : paths) {
            size_t file_size;
            files.push_back(map_file(path, file_size));
            read_tensors(path, (const char*)files.back().base, file_size, (int)files.size() - 1, tensors);
        }
    } catch (...) {  // the destructor doesn't run for a throwing constructor
        for (auto& mapping : files) munmap(mapping.base, mapping.size);
        throw;
    }
}

Safetensors::~Safetensors() {
    for (auto& mapping : files) munmap(mapping.base, mapping.size);
}

const Tensor& Safetensors::at(const std::string& name) const {
    auto found = tensors.find(name);
    if (found == tensors.end()) throw std::runtime_error("no tensor " + name);
    return found->second;
}

}  // namespace mllm
