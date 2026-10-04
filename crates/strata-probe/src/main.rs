// Read a GGUF's tensor payloads through the REAL C++ direct-IO path: dlopen the
// shim, open the file with strata::platform::DirectFile over the vtable, stream
// every tensor through one pinned 1 MiB buffer in aligned chunks (the shape of
// the expert pager's load loop), and hash what lands. With a cppdump binary as
// the second argument, every header field and the whole-model hash are compared
// against the C++ mmap'd reader - two independent readers of the same file.
//
//   STRATA_KERNELS_LIB=../target/shim/libstrata_kernels.so \
//   cargo run --release -p strata-probe -- model.gguf [path-to-cppdump]

use std::process::Command;
use strata_artifact::{tensor_payload_bytes, GgufFile};
use strata_device::{DeviceFile, IoCompletion, Pinned, Shim};

mod mkgguf;

const CHUNK: usize = 1 << 20; // one pinned buffer, reused for every tensor

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: strata-probe <file.gguf> [cppdump]");
        eprintln!("       strata-probe mkgguf <out.gguf> [size_mb=64]");
        std::process::exit(2);
    };
    if path == "mkgguf" {
        let Some(out) = args.next() else {
            eprintln!("mkgguf needs an output path");
            std::process::exit(2);
        };
        let mb = args.next().map_or(64, |s| s.parse::<u64>().unwrap_or(64));
        mkgguf::mkgguf(&out, mb);
        return;
    }
    let cppdump = args.next();

    let Some(res) = Shim::try_load() else {
        eprintln!(
            "no shim: set STRATA_KERNELS_LIB to the built libstrata_kernels.so (shim/build.sh)"
        );
        std::process::exit(3);
    };
    let shim = match res {
        Ok(s) => s,
        Err(e) => {
            eprintln!("shim failed to load: {e}");
            std::process::exit(3);
        }
    };
    let k = shim.kernels();
    println!(
        "shim loaded: {} of {} slots filled",
        shim.filled_slots(),
        strata_device::STRATA_SLOT_COUNT
    );
    let n = shim.device_count();
    println!("device_count: {n}");
    for i in 0..n.max(0) {
        if let Some(d) = shim.device_info(i) {
            println!(
                "  gpu {i}: {} [{}] cc {}.{} {} SMs, {:.1} GiB of {:.1} GiB free",
                d.name_str(),
                d.arch_str(),
                d.cc_major,
                d.cc_minor,
                d.multi_processor_count,
                d.free_bytes as f64 / (1u64 << 30) as f64,
                d.total_bytes as f64 / (1u64 << 30) as f64
            );
        }
    }

    let f = match DeviceFile::open(k, &path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("direct-IO open failed: {e}");
            std::process::exit(1);
        }
    };
    let align = f.alignment() as u64;
    let g = match GgufFile::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rust gguf open failed: {e}");
            std::process::exit(1);
        }
    };
    if f.size() != g.file_size() {
        eprintln!("size disagrees: cpp {} rust {}", f.size(), g.file_size());
        std::process::exit(1);
    }
    println!(
        "opened {} ({} tensors, {} metadata keys, data_start {}, size {}, direct-IO align {})",
        path,
        g.tensors().len(),
        g.metadata().len(),
        g.data_start(),
        f.size(),
        align
    );

    // One pinned buffer, many reads: submit a chunk, wait its completion, fold
    // the in-range bytes, repeat. Memory stays at one chunk whatever the model.
    let mut buf = Pinned::new(k, CHUNK + align as usize).expect("pinned alloc");
    if !buf.align_ok(align as u32) {
        eprintln!("shim returned an unaligned pinned buffer");
        std::process::exit(1);
    }
    let mut done = [IoCompletion::default(); 1];
    let mut model_hash = Fnv::new();
    let mut total = 0u64;
    let mut tensors_read = 0usize;
    let mut first_hex = String::new();
    let mut first_hash = 0u64;

    for (i, t) in g.tensors().iter().enumerate() {
        let bytes = tensor_payload_bytes(t);
        if bytes == 0 {
            continue; // unknown quantization: no trustworthy length to compare
        }
        let off = g.tensor_file_offset(t);
        let mut pos = 0u64;
        let mut tag = 0u64;
        while pos < bytes {
            // The alignment contract: file offset, length and buffer address all
            // multiples of `align`. Read the aligned window around the chunk and
            // use only the bytes that belong to the tensor.
            let want = (bytes - pos).min(CHUNK as u64);
            let a = off + pos;
            let lo = a / align * align;
            let hi = (a + want).div_ceil(align) * align;
            let len = (hi - lo) as usize;
            let pre = (a - lo) as usize;
            if len == 0 || len > CHUNK + align as usize {
                eprintln!("chunk arithmetic broke on {}", t.name);
                std::process::exit(1);
            }
            let slice = &mut buf.bytes_mut()[..len];
            f.submit(lo, slice, tag)
                .unwrap_or_else(|e| panic!("submit failed for {}: {e}", t.name));
            let n = f.wait(&mut done, 10_000);
            // A completion may legitimately be short only at end of file (the
            // shim's contract); what must always hold is that the bytes this
            // chunk needed - `pre` skipped + `want` kept - actually landed.
            let got = done.first().filter(|_| n == 1);
            match got {
                Some(c)
                    if c.ok != 0
                        && c.tag == tag
                        && (c.bytes as usize) <= len
                        && (c.bytes as usize) >= pre + want as usize => {}
                other => {
                    eprintln!(
                        "bad completion for {} at {pos}: {:?} (wanted tag {tag}, {len} bytes)",
                        t.name,
                        other.map(|c| (c.tag, c.bytes, c.ok))
                    );
                    std::process::exit(1);
                }
            }
            let data = &buf.bytes_mut()[pre..pre + want as usize];
            if i == 0 && pos == 0 {
                let n = data.len().min(64);
                first_hex = data[..n].iter().map(|b| format!("{b:02x}")).collect();
                first_hash = Fnv::hash(&data[..n]);
            }
            model_hash.fold_in(data);
            pos += want;
            tag += 1;
        }
        total += bytes;
        tensors_read += 1;
    }
    println!(
        "read {total} bytes across {tensors_read} tensors through the C++ DirectFile; \
         whole-payload fnv1a64 = {:016x}",
        model_hash.get()
    );

    if let Some(dump) = cppdump {
        let out = Command::new(&dump).arg(&path).output();
        match out {
            Ok(o) if o.status.success() => {
                check_against_cpp(
                    &String::from_utf8_lossy(&o.stdout),
                    &g,
                    &first_hex,
                    first_hash,
                    model_hash.get(),
                    total,
                );
            }
            Ok(o) => {
                eprintln!(
                    "cppdump failed: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("cannot run cppdump {dump}: {e}");
                std::process::exit(1);
            }
        }
    }
}

pub(crate) struct Fnv(u64);
impl Fnv {
    // Strata's own seed (src/core/pinned.cu): the FNV-1a basis with its last
    // digit dropped. Parity with the engine's checksums matters more than the
    // RFC's constant - both readers must start from the same place.
    pub(crate) fn new() -> Fnv {
        Fnv(1469598103934665603)
    }
    pub(crate) fn hash(data: &[u8]) -> u64 {
        let mut h = Fnv::new();
        h.fold_in(data);
        h.0
    }
    pub(crate) fn fold_in(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    pub(crate) fn get(&self) -> u64 {
        self.0
    }
}

/// Compare the C++ reader's dump against the Rust reader and the ABI bytes.
pub(crate) fn check_against_cpp(
    json: &str,
    g: &GgufFile,
    rust_first_hex: &str,
    rust_first_hash: u64,
    rust_model_hash: u64,
    rust_total: u64,
) {
    let v = match parse_dump(json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot parse cppdump json: {e}");
            std::process::exit(1);
        }
    };
    let mut fails: Vec<String> = Vec::new();
    let eq = |name: &str, a: u64, b: u64| -> Option<String> {
        (a != b).then(|| format!("{name}: cpp {a} rust {b}"))
    };
    fails.extend(eq("version", v.version, g.version() as u64));
    fails.extend(eq("tensors", v.tensors, g.tensors().len() as u64));
    fails.extend(eq("metadata", v.metadata, g.metadata().len() as u64));
    fails.extend(eq("data_start", v.data_start, g.data_start()));
    fails.extend(eq("size", v.size, g.file_size()));
    fails.extend(eq("align", v.align, g.alignment()));
    for (a, b) in v.tensor.iter().zip(g.tensors()) {
        if a.name != b.name || a.dtype as u32 != b.dtype || a.off != b.offset || a.dims != b.shape {
            fails.push(format!(
                "tensor: cpp {:?} rust {:?}",
                (a.name.as_str(), a.dtype, a.off, &a.dims),
                (b.name.as_str(), b.dtype, b.offset, &b.shape)
            ));
        }
    }
    if let Some(h) = &v.first_bytes {
        if !h.is_empty() {
            if h != rust_first_hex {
                fails.push(format!("first-64-bytes: cpp {h} rust {rust_first_hex}"));
            }
            fails.extend(eq("first-64-hash", v.first_fnv1a, rust_first_hash));
        }
    }
    fails.extend(eq("model-bytes", v.model_bytes, rust_total));
    fails.extend(eq("model-fnv1a", v.model_fnv1a, rust_model_hash));
    if fails.is_empty() {
        println!(
            "cpp/rust agree: {} tensors, {} metadata keys, and {} bytes of payload \
             read two different ways hash identically",
            v.tensors, v.metadata, rust_total
        );
    } else {
        for f in &fails {
            println!("  MISMATCH {f}");
        }
        println!("{} mismatches", fails.len());
        std::process::exit(1);
    }
}

// ---- a hand-rolled JSON reader: the workspace stays dependency-free
#[derive(Default)]
struct CppDump {
    version: u64,
    tensors: u64,
    metadata: u64,
    data_start: u64,
    size: u64,
    align: u64,
    tensor: Vec<CppTensor>,
    first_bytes: Option<String>,
    first_fnv1a: u64,
    model_fnv1a: u64,
    model_bytes: u64,
}
#[derive(Default)]
struct CppTensor {
    name: String,
    dtype: u64,
    off: u64,
    dims: Vec<u64>,
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Result<u8, String> {
        self.ws();
        self.b.get(self.i).copied().ok_or("eof".to_string())
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek()? == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            if self.b[self.i] == b'\\' {
                self.i += 1;
            }
            self.i += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.i]).into_owned();
        self.eat(b'"')?;
        Ok(s)
    }
    fn number(&mut self) -> Result<u64, String> {
        self.ws();
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "bad number".to_string())?
            .parse()
            .map_err(|_| "bad number".to_string())
    }
    fn hex64(&mut self) -> Result<u64, String> {
        let s = self.string()?;
        u64::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|_| "bad hex".to_string())
    }
    fn numbers(&mut self) -> Result<Vec<u64>, String> {
        self.eat(b'[')?;
        let mut v = Vec::new();
        if self.peek()? != b']' {
            loop {
                v.push(self.number()?);
                if self.peek()? == b',' {
                    self.i += 1;
                } else {
                    break;
                }
            }
        }
        self.eat(b']')?;
        Ok(v)
    }
    fn dump(&mut self) -> Result<CppDump, String> {
        let mut d = CppDump::default();
        self.eat(b'{')?;
        loop {
            let key = self.string()?;
            self.eat(b':')?;
            match key.as_str() {
                "version" => d.version = self.number()?,
                "tensors" => d.tensors = self.number()?,
                "metadata" => d.metadata = self.number()?,
                "data_start" => d.data_start = self.number()?,
                "size" => d.size = self.number()?,
                "align" => d.align = self.number()?,
                "model_bytes" => d.model_bytes = self.number()?,
                "first_bytes" => d.first_bytes = Some(self.string()?),
                "first_fnv1a" => d.first_fnv1a = self.hex64()?,
                "model_fnv1a" => d.model_fnv1a = self.hex64()?,
                "tensor" => {
                    self.eat(b'[')?;
                    if self.peek()? != b']' {
                        loop {
                            d.tensor.push(self.tensor()?);
                            if self.peek()? == b',' {
                                self.i += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    self.eat(b']')?;
                }
                other => return Err(format!("unknown key {other}")),
            }
            if self.peek()? == b',' {
                self.i += 1;
            } else {
                break;
            }
        }
        self.eat(b'}')?;
        Ok(d)
    }
    fn tensor(&mut self) -> Result<CppTensor, String> {
        let mut t = CppTensor::default();
        self.eat(b'{')?;
        loop {
            let key = self.string()?;
            self.eat(b':')?;
            match key.as_str() {
                "name" => t.name = self.string()?,
                "dtype" => t.dtype = self.number()?,
                "off" => t.off = self.number()?,
                "dims" => t.dims = self.numbers()?,
                other => return Err(format!("unknown tensor key {other}")),
            }
            if self.peek()? == b',' {
                self.i += 1;
            } else {
                break;
            }
        }
        self.eat(b'}')?;
        Ok(t)
    }
}

fn parse_dump(s: &str) -> Result<CppDump, String> {
    Parser {
        b: s.as_bytes(),
        i: 0,
    }
    .dump()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_full_dump() {
        let json = r#"{"version":3,"tensors":2,"metadata":1,"data_start":2240,"size":48,
            "align":32,"tensor":[{"name":"blk.0.w","dtype":42,"off":0,"dims":[2560,4]},
            {"name":"t.2","dtype":0,"off":64,"dims":[8]}],"first_bytes":"00ff",
            "first_fnv1a":"00000000000000ff","model_fnv1a":"deadbeefcafe0001","model_bytes":72}"#;
        let d = parse_dump(json).expect("parses");
        assert_eq!(d.version, 3);
        assert_eq!(d.tensors, 2);
        assert_eq!(d.tensor.len(), 2);
        assert_eq!(d.tensor[0].name, "blk.0.w");
        assert_eq!(d.tensor[0].dims, vec![2560, 4]);
        assert_eq!(d.first_bytes.as_deref(), Some("00ff"));
        assert_eq!(d.first_fnv1a, 0xff);
        assert_eq!(d.model_fnv1a, 0xdeadbeefcafe0001);
        assert_eq!(d.model_bytes, 72);
    }

    #[test]
    fn rejects_truncated_and_unknown() {
        assert!(parse_dump(r#"{"version":3,"tensors":1"#).is_err());
        assert!(parse_dump(r#"{"nope":1}"#).is_err());
    }

    #[test]
    fn fnv_matches_strata_seed_and_streams() {
        // The probe hashes chunk-by-chunk into one accumulator; that must equal
        // hashing the whole buffer, and start from Strata's seed, not the RFC's.
        assert_eq!(Fnv::new().get(), 1469598103934665603);
        let whole = Fnv::hash(b"the quick brown fox");
        let mut h = Fnv::new();
        h.fold_in(b"the quick ");
        h.fold_in(b"brown fox");
        assert_eq!(whole, h.get());
    }
}
