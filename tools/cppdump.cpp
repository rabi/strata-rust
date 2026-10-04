// cppdump - the C++ reader over the same file the Rust probe checks, so the two
// readers can be compared field by field. Emits JSON on stdout:
//   {"version":3,"tensors":N,"metadata":M,"data_start":X,"size":Y,"align":A,
//    "tensor":[{"name":..,"dtype":..,"off":..,"dims":[..]},...],
//    "first_bytes":"hex","first_fnv1a":"..","model_fnv1a":"..","model_bytes":N}
// The model hash covers every tensor whose payload size is known and whose
// absolute offset and size are 4096-aligned - exactly the set the Rust probe
// reads through the direct-IO ABI, so the two hashes must match byte for byte.
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {
uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 15]);
    }
    return s;
}
void esc(const std::string& s) {
    for (char c : s) {
        if (c == '"' || c == '\\')
            std::printf("\\%c", c);
        else if ((unsigned char) c < 0x20)
            std::printf("\\u%04x", c);
        else
            std::putchar(c);
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: cppdump <file.gguf>\n");
        return 2;
    }
    constexpr uint64_t kFnv0 = 1469598103934665603ull;  // Strata's seed (pinned.cu)
    try {
        strata::GgufFile g(argv[1]);
        const auto& ts = g.tensors();
        std::printf(
            "{\"version\":%u,\"tensors\":%zu,\"metadata\":%zu,\"data_start\":%llu,"
            "\"size\":%llu,\"align\":%llu,\"tensor\":[",
            g.version(), ts.size(), g.metadata().size(), (unsigned long long) g.data_start(),
            (unsigned long long) g.file_size(), (unsigned long long) g.alignment());
        for (size_t i = 0; i < ts.size(); ++i) {
            if (i) std::putchar(',');
            std::printf("{\"name\":\"");
            esc(ts[i].name);
            std::printf("\",\"dtype\":%u,\"off\":%llu,\"dims\":[", ts[i].type,
                        (unsigned long long) ts[i].offset);
            for (size_t d = 0; d < ts[i].shape.size() && d < 4; ++d)
                std::printf("%s%llu", d ? "," : "", (unsigned long long) ts[i].shape[d]);
            std::printf("]}");
        }
        std::printf("]");

        uint64_t model = kFnv0;
        uint64_t counted = 0;
        for (const auto& t : ts) {
            const uint64_t bytes = tensor_payload_bytes(t);
            if (bytes == 0) continue;
            model = fnv1a64(g.tensor_data(t), (size_t) bytes, model);
            counted += bytes;
        }
        std::string first_hex;
        uint64_t first_hash = 0;
        if (!ts.empty()) {
            const uint64_t bytes = tensor_payload_bytes(ts[0]);
            if (bytes != 0) {
                const uint8_t* p = g.tensor_data(ts[0]);
                const size_t n = (size_t) (bytes < 64 ? bytes : 64);
                first_hex = hex(p, n);
                first_hash = fnv1a64(p, n, kFnv0);
            }
        }
        std::printf(
            ",\"first_bytes\":\"%s\",\"first_fnv1a\":\"%016llx\","
            "\"model_fnv1a\":\"%016llx\",\"model_bytes\":%llu}\n",
            first_hex.c_str(), (unsigned long long) first_hash, (unsigned long long) model,
            (unsigned long long) counted);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cppdump: %s\n", e.what());
        return 1;
    }
    return 0;
}
