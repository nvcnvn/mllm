// Golden data from py/dump_golden.py (its docstring has the key layout), and the comparison every test uses.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "safetensors.h"
#include "spec.h"

namespace golden {

// $MLLM_MODEL picks another model's golden data (e.g. mlx-community/Qwen3.6-27B-4bit); make test runs the 0.8B
inline const char* MODEL = getenv("MLLM_MODEL") ? getenv("MLLM_MODEL") : "mlx-community/Qwen3.5-0.8B-4bit";
inline bool quiet = false;  // no mismatch lines (tests that expect mismatches)

// <cache>/golden/<model_key>/, as dump_golden.py writes it. Exits with a hint if it isn't there.
inline std::string dir(const mllm::ModelSpec& model) {
    std::string directory = mllm::spec::cache_dir() + "/golden/" + mllm::spec::model_key(model.path);
    if (!std::filesystem::is_directory(directory)) {
        fprintf(stderr, "no golden data at %s\nrun: HF_HOME=$HF_HOME .venv/bin/python py/dump_golden.py %s\n",
                directory.c_str(), model.path.c_str());
        exit(1);
    }
    return directory;
}

// Golden tensors sit at any byte offset, so each element is memcpy'd out (no unaligned typed reads).
template <class T>
inline T load(const uint8_t* data, size_t index) {
    T value;
    memcpy(&value, data + index * sizeof(T), sizeof(T));
    return value;
}

inline double to_double(const std::string& dtype, const uint8_t* data, size_t index) {
    if (dtype == "BF16") {
        uint32_t bits = (uint32_t)load<uint16_t>(data, index) << 16;
        float value;
        memcpy(&value, &bits, 4);
        return value;
    }
    if (dtype == "F16") return (double)load<__fp16>(data, index);
    if (dtype == "F32") return load<float>(data, index);
    if (dtype == "I32") return load<int32_t>(data, index);
    if (dtype == "I64") return (double)load<int64_t>(data, index);
    if (dtype == "U32") return load<uint32_t>(data, index);
    fprintf(stderr, "golden: unsupported dtype %s\n", dtype.c_str());
    abort();
}

// got has want's dtype and shape. Integers: exact. Floats: cosine >= min_cos and max |got - want| <= max_abs.
// max_abs is absolute: callers comparing large-valued tensors (next-token scores) pass their own thresholds.
// Prints one line on a mismatch unless quiet.
inline bool matches(const std::string& name, const mllm::Tensor& want, const uint8_t* want_data, const void* got,
                    double min_cos = 0.9999, double max_abs = 1e-2) {
    const uint8_t* got_data = (const uint8_t*)got;
    size_t count = want.numel();
    if (want.dtype[0] != 'F' && want.dtype != "BF16") {
        if (!memcmp(got_data, want_data, want.nbytes)) return true;
        size_t first_difference = 0;
        while (first_difference < count &&
               to_double(want.dtype, got_data, first_difference) == to_double(want.dtype, want_data, first_difference))
            first_difference++;
        if (!quiet && first_difference < count)
            fprintf(stderr, "%s: [%zu] = %.0f, want %.0f\n", name.c_str(), first_difference,
                    to_double(want.dtype, got_data, first_difference), to_double(want.dtype, want_data, first_difference));
        return false;
    }
    double dot = 0, got_squared = 0, want_squared = 0, worst = 0;
    for (size_t i = 0; i < count; i++) {
        double got_value = to_double(want.dtype, got_data, i), want_value = to_double(want.dtype, want_data, i);
        dot += got_value * want_value, got_squared += got_value * got_value, want_squared += want_value * want_value;
        worst = std::fmax(worst, std::fabs(got_value - want_value));
    }  // a NaN anywhere makes cos NaN, which fails below
    double cos = got_squared == 0 && want_squared == 0 ? 1 : dot / std::sqrt(got_squared * want_squared);
    if (cos >= min_cos && worst <= max_abs) return true;
    if (!quiet) fprintf(stderr, "%s: cos %.6f (min %g), max abs %g (max %g)\n", name.c_str(), cos, min_cos, worst, max_abs);
    return false;
}

}  // namespace golden
