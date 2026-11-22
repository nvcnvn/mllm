// Safetensors reader: mmap each file, parse the header, keep a view of each tensor. No copies.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mllm {

constexpr size_t PAGE = 16384;  // OS page; mappings are rounded up to it so Metal can wrap them

struct Tensor {
    std::string dtype;  // safetensors names: BF16, F16, F32, U32, I32, I64, ...
    std::vector<int64_t> shape;
    int file;       // index into Safetensors::files
    size_t offset;  // from the start of that file's mapping
    size_t nbytes;
    size_t numel() const;
};

// path: one .safetensors file -> {path}; a directory -> every *.safetensors in it, sorted by name
std::vector<std::string> files_in(const std::string& path);

struct Mapping {
    std::string path;
    void* base;
    size_t size;  // file size rounded up to PAGE (the tail of the last page reads as zeros)
};

class Safetensors {
public:
    explicit Safetensors(const std::vector<std::string>& paths);
    explicit Safetensors(const std::string& path) : Safetensors(files_in(path)) {}
    ~Safetensors();
    Safetensors(const Safetensors&) = delete;
    Safetensors& operator=(const Safetensors&) = delete;

    const Tensor& at(const std::string& name) const;
    const uint8_t* data(const Tensor& t) const { return (const uint8_t*)files[t.file].base + t.offset; }

    std::vector<Mapping> files;
    std::map<std::string, Tensor> tensors;
};

size_t dtype_size(const std::string& dtype);

}  // namespace mllm
