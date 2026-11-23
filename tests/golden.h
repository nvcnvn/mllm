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
inline std::string dir(const mllm::ModelSpec& s) {
    std::string d = mllm::spec::cache_dir() + "/golden/" + mllm::spec::model_key(s.path);
    if (!std::filesystem::is_directory(d)) {
        fprintf(stderr, "no golden data at %s\nrun: HF_HOME=$HF_HOME .venv/bin/python py/dump_golden.py %s\n",
                d.c_str(), s.path.c_str());
        exit(1);
    }
    return d;
}

// Golden tensors sit at any byte offset, so each element is memcpy'd out (no unaligned typed reads).
template <class T>
inline T load(const uint8_t* p, size_t i) {
    T v;
    memcpy(&v, p + i * sizeof(T), sizeof(T));
    return v;
}

inline double to_double(const std::string& dtype, const uint8_t* p, size_t i) {
    if (dtype == "BF16") {
        uint32_t u = (uint32_t)load<uint16_t>(p, i) << 16;
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    if (dtype == "F16") return (double)load<__fp16>(p, i);
    if (dtype == "F32") return load<float>(p, i);
    if (dtype == "I32") return load<int32_t>(p, i);
    if (dtype == "I64") return (double)load<int64_t>(p, i);
    if (dtype == "U32") return load<uint32_t>(p, i);
    fprintf(stderr, "golden: unsupported dtype %s\n", dtype.c_str());
    abort();
}

// got has want's dtype and shape. Integers: exact. Floats: cosine >= min_cos and max |got - want| <= max_abs.
// max_abs is absolute: callers comparing large-valued tensors (logits) pass their own thresholds.
// Prints one line on a mismatch unless quiet.
inline bool matches(const std::string& name, const mllm::Tensor& want, const uint8_t* want_data, const void* got,
                    double min_cos = 0.9999, double max_abs = 1e-2) {
    const uint8_t* g = (const uint8_t*)got;
    size_t n = want.numel();
    if (want.dtype[0] != 'F' && want.dtype != "BF16") {
        if (!memcmp(g, want_data, want.nbytes)) return true;
        for (size_t i = 0; i < n; i++)
            if (to_double(want.dtype, g, i) != to_double(want.dtype, want_data, i)) {
                if (!quiet)
                    fprintf(stderr, "%s: [%zu] = %.0f, want %.0f\n", name.c_str(), i, to_double(want.dtype, g, i),
                        to_double(want.dtype, want_data, i));
                break;
            }
        return false;
    }
    double dot = 0, gg = 0, ww = 0, worst = 0;
    for (size_t i = 0; i < n; i++) {
        double a = to_double(want.dtype, g, i), b = to_double(want.dtype, want_data, i);
        dot += a * b, gg += a * a, ww += b * b;
        worst = std::fmax(worst, std::fabs(a - b));
    }  // a NaN anywhere makes cos NaN, which fails below
    double cos = gg == 0 && ww == 0 ? 1 : dot / std::sqrt(gg * ww);
    if (cos >= min_cos && worst <= max_abs) return true;
    if (!quiet) fprintf(stderr, "%s: cos %.6f (min %g), max abs %g (max %g)\n", name.c_str(), cos, min_cos, worst, max_abs);
    return false;
}

}  // namespace golden
