// Synthetic GGUF writer for the native corpus: just enough of the format for
// `GgufFile` to open a file, read metadata keys and a tensor directory, and hand
// out payload bytes. Shapes and types come from the caller; each tensor's
// payload is one deterministic pattern, so a Rust replay can rebuild a shard
// byte-for-byte without a reader of its own.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixture {

using strata::MetaType;

inline uint8_t pattern(uint64_t seed, uint64_t i) {
    uint64_t x = seed * 0x9e3779b97f4a7c15ull + i * 0x100000001b3ull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    return (uint8_t)(x >> 24);
}

struct Buf {
    std::vector<uint8_t> v;
    void u8(uint8_t x) { v.push_back(x); }
    void u32(uint32_t x) { for (int i = 0; i < 4; ++i) u8((uint8_t)(x >> (8 * i))); }
    void u64(uint64_t x) { for (int i = 0; i < 8; ++i) u8((uint8_t)(x >> (8 * i))); }
    void i32(int32_t x) { u32((uint32_t) x); }
    void f32(float x) { uint32_t b; std::memcpy(&b, &x, 4); u32(b); }
    void str(const std::string& s) {
        u64(s.size());
        for (unsigned char c : s) u8(c);
    }
};

struct Kv {
    std::string key;
    MetaType type;
    std::vector<std::string> strings;
    std::vector<uint64_t> nums;
    std::vector<Kv> items;  // array payload, all of one string type

    void write(Buf& b) const {
        b.str(key);
        b.u32((uint32_t) type);
        if (type == MetaType::ARRAY) {
            b.u32((uint32_t)(uint32_t) MetaType::STRING);
            b.u64(items.size());
            for (const auto& it : items) it.write(b);
            return;
        }
        switch (type) {
            case MetaType::BOOL: b.u8((uint8_t) nums[0]); break;
            case MetaType::U32: b.u32((uint32_t) nums[0]); break;
            case MetaType::I32: b.i32((int32_t) nums[0]); break;
            case MetaType::F32: b.f32((float) nums[0]); break;
            case MetaType::U64: b.u64(nums[0]); break;
            case MetaType::STRING: b.str(strings[0]); break;
            default: throw std::runtime_error("fixture: unsupported metadata type");
        }
    }
};

inline Kv str(const std::string& k, const std::string& v) {
    return Kv{k, MetaType::STRING, {v}, {}, {}};
}
inline Kv u32(const std::string& k, uint64_t v) {
    return Kv{k, MetaType::U32, {}, {v}, {}};
}
inline Kv str_array(const std::string& k, std::vector<std::string> vs) {
    Kv out{k, MetaType::ARRAY, {}, {}, {}};
    for (auto& s : vs) out.items.push_back(str("", s));
    return out;
}
/// the split metadata `native_dense` arbitrates on. These are the keys the
/// engine reads (`gguf_reader.hpp:541-557`), not GGUF's own split keys.
inline std::vector<Kv> split_keys(uint64_t no, uint64_t count, uint64_t tensors) {
    return {u32("split.count", count), u32("split.no", no), u32("split.tensors.count", tensors)};
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t seed;
};

inline uint64_t payload_bytes(uint32_t dtype, const std::vector<uint64_t>& shape) {
    int elems = 0, bytes = 0;
    if (shape.empty() || !strata::block_geometry(dtype, elems, bytes) ||
        shape[0] % (uint64_t) elems)
        throw std::runtime_error("fixture: no block geometry for the requested dtype/shape");
    uint64_t elements = 1;
    for (auto d : shape) elements *= d;
    return elements / (uint64_t) elems * (uint64_t) bytes;
}

inline std::string hex(const uint8_t* p, uint64_t n) {
    static const char* D = "0123456789abcdef";
    std::string out;
    for (uint64_t i = 0; i < n; ++i) {
        out += D[p[i] >> 4];
        out += D[p[i] & 15];
    }
    return out;
}

/// Write a shard: `keys` as metadata, `tensors` laid out at `alignment` (GGUF's
/// own default is 32) with each payload filled from that tensor's seed pattern.
inline void write(const std::string& path, const std::vector<Kv>& keys, const std::vector<Tensor>& tensors,
                  uint64_t alignment = 32) {
    Buf b;
    b.v.insert(b.v.end(), {'G', 'G', 'U', 'F'});
    b.u32(3);
    b.u64(tensors.size());
    b.u64(keys.size());  // n_kv is a uint64 in the reader's Cursor
    for (const auto& k : keys) k.write(b);

    struct Row {
        const Tensor* t;
        uint64_t off;
        uint64_t bytes;
    };
    std::vector<Row> rows;
    uint64_t cursor = 0;
    for (const auto& t : tensors) {
        cursor = (cursor + alignment - 1) / alignment * alignment;
        const uint64_t n = payload_bytes(t.dtype, t.shape);
        rows.push_back({&t, cursor, n});
        cursor += n;
    }
    for (const auto& r : rows) {
        b.str(r.t->name);
        b.u32((uint32_t) r.t->shape.size());
        for (auto d : r.t->shape) b.u64(d);
        b.u32(r.t->dtype);
        b.u64(r.off);
    }
    uint64_t data_start = (b.v.size() + alignment - 1) / alignment * alignment;
    b.u64(data_start);

    std::vector<uint8_t> payload(cursor, 0);
    for (const auto& r : rows)
        for (uint64_t i = 0; i < r.bytes; ++i) payload[r.off + i] = pattern(r.t->seed, i);

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("fixture: cannot write " + path);
    out.write((const char*) b.v.data(), (std::streamsize) b.v.size());
    for (size_t i = b.v.size(); i < data_start; ++i) out.put(0);
    out.write((const char*) payload.data(), (std::streamsize) payload.size());
}

/// Same file, but the caller's offsets and total payload are used verbatim —
/// no alignment, no capacity check. This is the only way to build the shards
/// the span checks in `native_dense::load` refuse: a directory that lies about
/// where the bytes are, or how many there are.
struct Raw {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;
};

inline void write_raw(const std::string& path, const std::vector<Kv>& keys,
                      const std::vector<Raw>& tensors, uint64_t payload_size,
                      uint64_t alignment = 32) {
    Buf b;
    b.v.insert(b.v.end(), {'G', 'G', 'U', 'F'});
    b.u32(3);
    b.u64(tensors.size());
    b.u64(keys.size());
    for (const auto& k : keys) k.write(b);
    for (const auto& t : tensors) {
        b.str(t.name);
        b.u32((uint32_t) t.shape.size());
        for (auto d : t.shape) b.u64(d);
        b.u32(t.dtype);
        b.u64(t.offset);
    }
    uint64_t data_start = (b.v.size() + alignment - 1) / alignment * alignment;
    b.u64(data_start);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("fixture: cannot write " + path);
    out.write((const char*) b.v.data(), (std::streamsize) b.v.size());
    for (size_t i = b.v.size(); i < data_start; ++i) out.put(0);
    for (uint64_t i = 0; i < payload_size; ++i) out.put((char) fixture::pattern(77, i));
}

}  // namespace fixture
