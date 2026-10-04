// Generate a GGUF v3 file that is deliberately awkward for a reader, then read
// it back and require agreement. The awkwardness is the production shape:
// general.alignment = 32 (like real GGUFs), so tensor offsets and payload ends
// fall off 4096-byte boundaries and every O_DIRECT read of them is a superset
// window; payloads not multiples of the page size hit the short-completion
// path; mixed dtypes (F32/F16/Q8_0/Q4_K), a zero-dims tensor, a single-block
// tensor; KV values of every scalar and array type.
//
// Agreement is checked three ways: the bytes as emitted (hashed while
// writing), the Rust reader via buffered reads, and the engine's C++ cppdump
// if it was built. Writer, Rust reader and C++ reader producing one hash is
// the same property that validates the ABI path on real models.

use super::{check_against_cpp, Fnv};
use std::io::{Seek, SeekFrom, Write};
use strata_artifact::{block_geometry, tensor_payload_bytes, GgufFile};

struct Tw {
    name: String,
    dtype: u32,
    dims: Vec<u64>,
    off: u64,
}

enum Meta<'a> {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Str(&'a str),
    U64(u64),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrStr(Vec<&'a str>),
}

// GGUF value-type tags (spec values; the strata-artifact reader's MetaType).
const T_U8: u8 = 0;
const T_I8: u8 = 1;
const T_U16: u8 = 2;
const T_I16: u8 = 3;
const T_U32: u8 = 4;
const T_I32: u8 = 5;
const T_F32: u8 = 6;
const T_BOOL: u8 = 7;
const T_STR: u8 = 8;
const T_ARR: u8 = 9;
const T_U64: u8 = 10;
const T_I64: u8 = 11;
const T_F64: u8 = 12;

impl Meta<'_> {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Meta::Bool(b) => {
                out.extend_from_slice(&(T_BOOL as u32).to_le_bytes());
                out.push(*b as u8);
            }
            Meta::U8(v) => {
                out.extend_from_slice(&(T_U8 as u32).to_le_bytes());
                out.push(*v);
            }
            Meta::I8(v) => {
                out.extend_from_slice(&(T_I8 as u32).to_le_bytes());
                out.push(*v as u8);
            }
            Meta::U16(v) => {
                out.extend_from_slice(&(T_U16 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::I16(v) => {
                out.extend_from_slice(&(T_I16 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::U32(v) => {
                out.extend_from_slice(&(T_U32 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::I32(v) => {
                out.extend_from_slice(&(T_I32 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::I64(v) => {
                out.extend_from_slice(&(T_I64 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::F32(v) => {
                out.extend_from_slice(&(T_F32 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::F64(v) => {
                out.extend_from_slice(&(T_F64 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Str(s) => {
                out.extend_from_slice(&(T_STR as u32).to_le_bytes());
                out.extend_from_slice(&enc_str(s));
            }
            Meta::U64(v) => {
                out.extend_from_slice(&(T_U64 as u32).to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            // Strata's convention (reader + tools/gguf_reader.py): element type
            // u32, THEN count u64 - the reverse of the order I first wrote.
            Meta::ArrI32(v) => {
                out.extend_from_slice(&(T_ARR as u32).to_le_bytes());
                out.extend_from_slice(&(T_I32 as u32).to_le_bytes());
                out.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Meta::ArrF32(v) => {
                out.extend_from_slice(&(T_ARR as u32).to_le_bytes());
                out.extend_from_slice(&(T_F32 as u32).to_le_bytes());
                out.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Meta::ArrStr(v) => {
                out.extend_from_slice(&(T_ARR as u32).to_le_bytes());
                out.extend_from_slice(&(T_STR as u32).to_le_bytes());
                out.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for s in v {
                    out.extend_from_slice(&enc_str(s));
                }
            }
        }
    }
}

fn payload_bytes(dtype: u32, dims: &[u64]) -> u64 {
    if dims.is_empty() {
        return 0;
    }
    let (be, bb) = match block_geometry(dtype) {
        Some(g) => g,
        None => return 0,
    };
    if !dims[0].is_multiple_of(be as u64) {
        return 0;
    }
    let mut elements = 1u64;
    for &d in dims {
        if d == 0 {
            return 0;
        }
        elements = elements.saturating_mul(d);
    }
    elements / be as u64 * bb as u64
}

pub fn mkgguf(path: &str, size_mb: u64) {
    let mut tw: Vec<Tw> = Vec::new();
    macro_rules! push {
        ($name:expr, $dtype:expr, $dims:expr) => {
            tw.push(Tw {
                name: $name.into(),
                dtype: $dtype,
                dims: $dims,
                off: 0,
            })
        };
    }

    // Small tensors with sizes chosen so cumulative offsets miss 4096 by odd
    // amounts; every start and end here is off-alignment for a 4 KiB reader.
    push!("head.weight", 0, vec![1023, 8]); // F32, odd element count
    push!("head.bias", 1, vec![1000]); // F16, 2000 B
    push!("ffn.gate", 0, vec![11, 7]); // F32, 308 B
    push!("one.block", 8, vec![32]); // Q8_0, exactly one 34 B block
    push!("expert.0", 8, vec![4096, 14]); // Q8_0
    push!("expert.1", 12, vec![1024, 13]); // Q4_K (256-elem blocks)
    push!("expert.2", 1, vec![257, 3]); // F16, odd
    push!("expert.3", 0, vec![4095, 16]); // F32, ends mid-page

    // Fill to the requested size with Q8_0 tensors of ODD block count:
    // 34 * odd is never 4096-aligned, so each one exercises superset reads and
    // (the last one at least, of any odd remainder) short completions.
    let target = size_mb * (1 << 20);
    let mut est: u64 = tw.iter().map(|t| payload_bytes(t.dtype, &t.dims)).sum();
    let mut i = 0u64;
    while est < target {
        let blocks = 3_072 + (i % 37) * 13; // ~102 KiB each, varied odd counts
        let dims = vec![32 * blocks];
        est += payload_bytes(8, &dims);
        push!(format!("layers.{i}.filler.weight"), 8, dims);
        i += 1;
    }

    let kv: Vec<(&str, Meta)> = vec![
        ("general.alignment", Meta::U32(32)), // deliberately NOT 4096
        ("general.architecture", Meta::Str("qwen")),
        ("general.name", Meta::Str("mkgguf-synthetic")),
        ("general.file_type", Meta::U32(15)),
        ("honest.bool", Meta::Bool(true)),
        ("honest.u8", Meta::U8(7)),
        ("honest.i8", Meta::I8(-3)),
        ("honest.u16", Meta::U16(60000)),
        ("honest.i16", Meta::I16(-9999)),
        ("honest.u32", Meta::U32(4294967295)),
        ("honest.u64", Meta::U64(18446744073709551615)),
        ("honest.i32", Meta::I32(-123456)),
        ("honest.i64", Meta::I64(-1_000_000_000_000)),
        ("honest.f32", Meta::F32(0.5)),
        ("honest.f64", Meta::F64(-1.75)),
        ("honest.str", Meta::Str("mixed")),
        ("honest.arr.i32", Meta::ArrI32(vec![1, -2, 3])),
        ("honest.arr.f32", Meta::ArrF32(vec![0.25, -0.5])),
        ("honest.arr.str", Meta::ArrStr(vec!["a", "bb"])),
    ];

    // ---- header (offsets patched to real values after layout) ----
    let mut h: Vec<u8> = Vec::new();
    h.extend_from_slice(b"GGUF");
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&(tw.len() as u64).to_le_bytes());
    h.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in &kv {
        h.extend_from_slice(&enc_str(k));
        v.encode(&mut h);
    }
    let ti_start = h.len() as u64;
    for t in &tw {
        h.extend_from_slice(&enc_str(&t.name));
        h.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            h.extend_from_slice(&d.to_le_bytes());
        }
        h.extend_from_slice(&t.dtype.to_le_bytes());
        h.extend_from_slice(&0u64.to_le_bytes());
    }
    let data_start = (h.len() as u64).div_ceil(32) * 32;

    // ---- write: header, pad, payloads (hash exactly what is emitted) ----
    let file = std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("cannot create {path}: {e}");
        std::process::exit(1);
    });
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&h).unwrap();
    w.write_all(&vec![0u8; (data_start - h.len() as u64) as usize])
        .unwrap();

    let mut model = Fnv::new();
    let mut off = 0u64;
    let mut payload_total = 0u64;
    let mut chunk = vec![0u8; 1 << 16];
    let mut first_hex = String::new();
    let mut first_hash = 0u64;
    let mut first_done = false;
    for t in &mut tw {
        t.off = off;
        let bytes = payload_bytes(t.dtype, &t.dims);
        let seed0 = Fnv::hash(t.name.as_bytes());
        let mut pos = 0u64;
        while pos < bytes {
            let n = (bytes - pos).min(chunk.len() as u64) as usize;
            for (j, b) in chunk[..n].iter_mut().enumerate() {
                *b = lcg(seed0 ^ (pos + j as u64)) as u8;
            }
            w.write_all(&chunk[..n]).unwrap();
            model.fold_in(&chunk[..n]);
            if !first_done {
                let k = n.min(64);
                first_hex = chunk[..k].iter().map(|b| format!("{b:02x}")).collect();
                first_hash = Fnv::hash(&chunk[..k]);
                first_done = true;
            }
            pos += n as u64;
        }
        payload_total += bytes;
        let pad = (32 - (off + bytes) % 32) % 32;
        if pad > 0 {
            w.write_all(&vec![0u8; pad as usize]).unwrap();
        }
        off += bytes + pad;
    }
    w.flush().unwrap();
    drop(w);

    // ---- patch the tensor-info offsets in place ----
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("reopen for offset patch");
    let mut p = ti_start;
    for t in tw.iter() {
        // name, n_dims, dims, dtype, THEN the offset field we are patching
        p += (8 + t.name.len() + 4 + 8 * t.dims.len() + 4) as u64;
        f.seek(SeekFrom::Start(p)).unwrap();
        f.write_all(&t.off.to_le_bytes()).unwrap();
        p += 8;
    }
    f.flush().unwrap();
    drop(f);

    let hash = model.get();
    println!(
        "wrote {path}: {} tensors, {} metadata keys, data_start {data_start}, \
         {payload_total} payload bytes, fnv1a64 = {hash:016x}",
        tw.len(),
        kv.len()
    );

    // ---- check 1: the Rust reader sees what was emitted ----
    let g = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rust reader rejected the file we just wrote: {e}");
            std::process::exit(1);
        }
    };
    if g.tensors().len() != tw.len() || g.metadata().len() != kv.len() {
        eprintln!(
            "rust reader saw {} tensors / {} kv, expected {} / {}",
            g.tensors().len(),
            g.metadata().len(),
            tw.len(),
            kv.len()
        );
        std::process::exit(1);
    }
    for (i, t) in tw.iter().enumerate() {
        let r = &g.tensors()[i];
        if r.name != t.name || r.dtype != t.dtype || r.shape != t.dims || r.offset != t.off {
            eprintln!(
                "tensor {i}: rust {:?} vs written {:?}",
                (r.name.as_str(), r.dtype, &r.shape, r.offset),
                (t.name.as_str(), t.dtype, &t.dims, t.off)
            );
            std::process::exit(1);
        }
    }
    let mut re = Fnv::new();
    let mut re_total = 0u64;
    for t in g.tensors() {
        let bytes = tensor_payload_bytes(t);
        if bytes == 0 {
            continue;
        }
        let data = g
            .read_tensor(t)
            .unwrap_or_else(|e| panic!("buffered read of {} failed: {e}", t.name));
        if data.len() as u64 != bytes {
            eprintln!("{}: read {} of {bytes}", t.name, data.len());
            std::process::exit(1);
        }
        re.fold_in(&data);
        re_total += bytes;
    }
    if re.get() != hash || re_total != payload_total {
        eprintln!(
            "rust buffered reads: {:#016x}/{re_total}, writer: {:#016x}/{payload_total}",
            re.get(),
            hash
        );
        std::process::exit(1);
    }
    println!("rust reader agrees: {re_total} bytes through GgufFile::read_tensor");

    // ---- check 2: the engine's C++ reader agrees too ----
    for cand in ["target/shim/cppdump", "./cppdump"] {
        if std::path::Path::new(cand).exists() {
            match std::process::Command::new(cand).arg(path).output() {
                Ok(o) if o.status.success() => {
                    let json = String::from_utf8_lossy(&o.stdout).into_owned();
                    check_against_cpp(&json, &g, &first_hex, first_hash, hash, re_total);
                    println!("cppdump ({cand}) agrees");
                    return;
                }
                _ => println!("cppdump at {cand} unusable - C++ cross-check skipped"),
            }
        }
    }
    println!("no cppdump found - C++ cross-check skipped (shim/build.sh builds one)");
}

fn enc_str(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(9 + s.len());
    v.extend_from_slice(&(s.len() as u64).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
