// @generated: ported from Strata include/strata/artifact/gguf_reader.hpp

//! GGUF v3 reader. Reference implementation: `tools/gguf_reader.py`, written
//! because gguf-py cannot represent type 42 / Q2_0.
//!
//! Refuses, with a precise error, what it cannot arbitrate: a non-v3 file, an
//! unknown metadata type, a duplicate tensor name (GGUF has no index to say which
//! of two same-named tensors a lookup means, and `find` is first-match), and a
//! data section starting past EOF.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::split::gguf_split_paths;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_V3: u32 = 3;

/// ggml type names we care about. 42 = Q2_0, the PrismML ternary 2-bit encoding
/// this engine targets.
pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        _ => "?",
    }
}

/// Block geometry: (elements per block, bytes per block). Q2_0 is 64/18 - proven
/// from this artifact's own offset brackets in P0.S6 and recorded in
/// docs/q2_0-contract.md.
pub fn block_geometry(t: u32) -> Option<(u32, u32)> {
    let g = match t {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        20 => (32, 18),
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136),
        29 => (256, 56), // IQ1_M
        24 => (1, 1),    // I8: raw bytes (the FP8 PLE table of tools/ple_fp8_pack.py)
        42 => (64, 18),  // Q2_0
        _ => return None,
    };
    Some(g)
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: dim 0 varies fastest
    pub shape: Vec<u64>,
    pub dtype: u32,
    /// offset from the file's data_start
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
    pub fn type_name(&self) -> &'static str {
        ggml_type_name(self.dtype)
    }
}

/// Bytes of a tensor's payload from its shape and block geometry; 0 when the type
/// is unknown, a row is not whole blocks, or the count overflows.
pub fn tensor_payload_bytes(t: &TensorInfo) -> u64 {
    if t.shape.is_empty() {
        return 0;
    }
    let (be, bb) = match block_geometry(t.dtype) {
        Some(g) => g,
        None => return 0,
    };
    if !t.shape[0].is_multiple_of(be as u64) {
        return 0;
    }
    let mut elements: u64 = 1;
    for &d in &t.shape {
        if d == 0 || elements > u64::MAX / d {
            return 0;
        }
        elements *= d;
    }
    let blocks = elements / be as u64;
    if blocks > u64::MAX / bb as u64 {
        return 0;
    }
    blocks * bb as u64
}

/// A bounds-checked reader over a file: every read is checked, so a truncated or
/// corrupt file produces a precise error rather than a panic. Only the header
/// region is read; the data section is addressed by file offset.
struct Cursor<'a> {
    src: Src<'a>,
    pos: u64,
    size: u64,
}

enum Src<'a> {
    File(File),
    /// a GGUF image already in memory, read from `from_bytes`
    Mem(&'a [u8]),
}

impl<'a> Cursor<'a> {
    fn new(path: &Path) -> Result<(Cursor<'static>, u64), String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let size = file
            .metadata()
            .map_err(|e| format!("stat failed: {e}"))?
            .len();
        Ok((
            Cursor {
                src: Src::File(file),
                pos: 0,
                size,
            },
            size,
        ))
    }

    fn mem(buf: &'a [u8]) -> Cursor<'a> {
        Cursor {
            src: Src::Mem(buf),
            pos: 0,
            size: buf.len() as u64,
        }
    }

    fn take(&mut self, n: usize) -> Result<Vec<u8>, String> {
        if self.pos + n as u64 > self.size {
            return Err("GGUF: unexpected end of file in header".into());
        }
        let buf = match &mut self.src {
            Src::File(f) => {
                f.seek(SeekFrom::Start(self.pos))
                    .map_err(|e| format!("seek failed: {e}"))?;
                let mut b = vec![0u8; n];
                f.read_exact(&mut b)
                    .map_err(|e| format!("read failed: {e}"))?;
                b
            }
            Src::Mem(b) => b[self.pos as usize..][..n].to_vec(),
        };
        self.pos += n as u64;
        Ok(buf)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, String> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_f64(&mut self) -> Result<f64, String> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.take(1)?[0] as i8)
    }
    fn read_u16(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_i16(&mut self) -> Result<i16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]) as i16)
    }
    fn read_i32(&mut self) -> Result<i32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i32)
    }
    fn read_i64(&mut self) -> Result<i64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()) as i64)
    }
    fn read_str(&mut self) -> Result<String, String> {
        let n = self.read_u64()?;
        if self.pos + n > self.size {
            return Err("GGUF: unexpected end of file in header".into());
        }
        let b = self.take(n as usize)?;
        String::from_utf8(b).map_err(|_| "GGUF: metadata string is not UTF-8".to_string())
    }
    fn pos(&self) -> u64 {
        self.pos
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetaType {
    fn from_u32(v: u32) -> Result<MetaType, String> {
        Ok(match v {
            0 => MetaType::U8,
            1 => MetaType::I8,
            2 => MetaType::U16,
            3 => MetaType::I16,
            4 => MetaType::U32,
            5 => MetaType::I32,
            6 => MetaType::F32,
            7 => MetaType::Bool,
            8 => MetaType::String,
            9 => MetaType::Array,
            10 => MetaType::U64,
            11 => MetaType::I64,
            12 => MetaType::F64,
            _ => return Err("GGUF: unknown metadata value type".into()),
        })
    }
}

#[derive(Clone, Debug)]
pub struct MetaValue {
    pub vtype: MetaType,
    /// integer payloads
    pub u: u64,
    /// float payloads
    pub f: f64,
    /// string payloads
    pub s: String,
    /// arrays: element type
    pub elem: MetaType,
    pub count: u64,
    /// a sample of the first 64 items; the count is what matters
    pub items: Vec<MetaValue>,
}

impl MetaValue {
    fn empty() -> MetaValue {
        MetaValue {
            vtype: MetaType::U32,
            u: 0,
            f: 0.0,
            s: String::new(),
            elem: MetaType::U32,
            count: 0,
            items: Vec::new(),
        }
    }
    pub fn is_num(&self) -> bool {
        self.vtype != MetaType::String && self.vtype != MetaType::Array
    }
    pub fn num(&self) -> f64 {
        if self.vtype == MetaType::F32 || self.vtype == MetaType::F64 {
            self.f
        } else {
            self.u as f64
        }
    }
}

fn read_value(c: &mut Cursor, t: MetaType, depth: u32) -> Result<MetaValue, String> {
    if depth > 2 {
        return Err("GGUF: array nesting too deep".into());
    }
    let mut v = MetaValue::empty();
    v.vtype = t;
    match t {
        MetaType::U8 => v.u = c.read_u8()? as u64,
        MetaType::I8 => v.u = c.read_i8()? as i64 as u64,
        MetaType::U16 => v.u = c.read_u16()? as u64,
        MetaType::I16 => v.u = c.read_i16()? as i64 as u64,
        MetaType::U32 => v.u = c.read_u32()? as u64,
        MetaType::I32 => v.u = c.read_i32()? as i64 as u64,
        MetaType::F32 => {
            v.f = c.read_f32()? as f64;
            v.u = 0;
        }
        MetaType::Bool => v.u = u64::from(c.read_u8()? != 0),
        MetaType::String => v.s = c.read_str()?,
        MetaType::U64 => v.u = c.read_u64()?,
        MetaType::I64 => v.u = c.read_i64()? as u64,
        MetaType::F64 => v.f = c.read_f64()?,
        MetaType::Array => {
            v.elem = MetaType::from_u32(c.read_u32()?)?;
            v.count = c.read_u64()?;
            if v.count > (1 << 24) {
                return Err("GGUF: implausible array length".into());
            }
            v.items.reserve(v.count.min(64) as usize);
            for i in 0..v.count {
                let e = read_value(c, v.elem, depth + 1)?;
                if i < 64 {
                    v.items.push(e);
                }
            }
        }
    }
    Ok(v)
}

/// One GGUF file (one shard of a model).
#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    size: u64,
    data_start: u64,
    alignment: u64,
    version: u32,
    tensors: Vec<TensorInfo>,
    meta: BTreeMap<String, MetaValue>,
    /// the whole file, when `open_memory` read it in; `None` for `open`, which
    /// keeps only the header and re-opens the file for every payload
    image: Option<Vec<u8>>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<GgufFile, String> {
        let path = path.as_ref();
        let (mut c, size) = Cursor::new(path)?;
        let head = read_header(&mut c, size, &path.display().to_string())?;
        Ok(GgufFile {
            path: path.to_path_buf(),
            size,
            data_start: head.data_start,
            alignment: head.alignment,
            version: head.version,
            tensors: head.tensors,
            meta: head.meta,
            image: None,
        })
    }

    /// `open`, but the whole file is kept in memory so `tensor_bytes` can hand
    /// out a borrowed slice. For a shard small enough to hold; a real model's
    /// shards are memory-mapped by the loader instead.
    pub fn open_memory(path: impl AsRef<Path>) -> Result<GgufFile, String> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let size = bytes.len() as u64;
        let mut c = Cursor::mem(&bytes);
        let head = read_header(&mut c, size, &path.display().to_string())?;
        Ok(GgufFile {
            path: path.to_path_buf(),
            size,
            data_start: head.data_start,
            alignment: head.alignment,
            version: head.version,
            tensors: head.tensors,
            meta: head.meta,
            image: Some(bytes),
        })
    }

    /// A tensor's payload as bytes, not an address: `None` unless the file was
    /// opened by `open_memory`, or its span runs past the end of what was read.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Option<&[u8]> {
        let n = tensor_payload_bytes(t) as usize;
        let start = (self.data_start + t.offset) as usize;
        self.image.as_ref()?.get(start..start + n)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
    pub fn file_size(&self) -> u64 {
        self.size
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn metadata(&self) -> &BTreeMap<String, MetaValue> {
        &self.meta
    }
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }
    /// First tensor named `name`. A file with a duplicate is refused at open, so
    /// first-match is unambiguous here.
    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    /// Read `t`'s payload from the file. Used to cross-check bytes that
    /// arrived through some other path (the ABI's DirectFile) against plain
    /// buffered reads of the same file.
    pub fn read_tensor(&self, t: &TensorInfo) -> Result<Vec<u8>, String> {
        let bytes = tensor_payload_bytes(t);
        if bytes == 0 {
            return Err("unknown quantization: no payload size".into());
        }
        let off = self.tensor_file_offset(t);
        let mut f = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
        let mut out = vec![0u8; bytes as usize];
        f.read_exact(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }

    pub fn count_type(&self, type_name: &str) -> usize {
        self.tensors
            .iter()
            .filter(|t| t.type_name() == type_name)
            .count()
    }
    /// Absolute file offset of a tensor's payload, for the direct-IO loader.
    pub fn tensor_file_offset(&self, t: &TensorInfo) -> u64 {
        self.data_start + t.offset
    }
}

/// Everything a GGUF header holds except the file it came from.
struct Header {
    version: u32,
    data_start: u64,
    alignment: u64,
    tensors: Vec<TensorInfo>,
    meta: BTreeMap<String, MetaValue>,
}

/// Magic, version, metadata and tensor directory — the whole header, from either
/// cursor. `src_name` only appears in the duplicate-tensor message.
fn read_header(c: &mut Cursor<'_>, size: u64, src_name: &str) -> Result<Header, String> {
    let magic = c
        .read_u32()
        .map_err(|_| "not a GGUF file (bad magic)".to_string())?;
    if magic != GGUF_MAGIC {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = c.read_u32()?;
    if version != GGUF_V3 {
        return Err(format!("GGUF v{version}, this reader handles v3"));
    }
    let n_tensors = c.read_u64()?;
    let n_kv = c.read_u64()?;

    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.read_str()?;
        let t = MetaType::from_u32(c.read_u32()?)?;
        let v = read_value(c, t, 0)?;
        meta.insert(key, v);
    }

    // GGUF has no index to arbitrate between two tensors of one name: find() is
    // first-match, so a duplicate would silently win by position. Refuse the file
    // at open instead, naming the tensor and the file.
    let mut tensors = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for _ in 0..n_tensors {
        let name = c.read_str()?;
        if !names.insert(name.clone()) {
            return Err(format!(
                "GGUF: duplicate tensor name '{name}' in {src_name}"
            ));
        }
        let nd = c.read_u32()?;
        if nd == 0 || nd > 4 {
            return Err(format!("GGUF: bad n_dims for {name}"));
        }
        let mut shape = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            shape.push(c.read_u64()?);
        }
        let dtype = c.read_u32()?;
        let offset = c.read_u64()?;
        tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    let mut alignment = 32;
    if let Some(a) = meta.get("general.alignment") {
        if a.u != 0 {
            alignment = a.u;
        }
    }
    let data_start = c.pos().div_ceil(alignment) * alignment;
    if data_start > size {
        return Err("GGUF: data section starts past EOF".into());
    }
    Ok(Header {
        version,
        data_start,
        alignment,
        tensors,
        meta,
    })
}

/// The shards of one model, opened together; tensors are looked up across all of
/// them. A split GGUF carries the model's metadata (general.architecture and the
/// qwen4exp keys) in its FIRST shard only; later shards just declare
/// split.count / split.no / split.tensors.count.
///
/// Refused at construction: a later shard whose split keys disagree with shard 1's
/// (another model's shard, or a shard renamed into the family), a split model
/// whose tensor directories do not add up to split.tensors.count, and a tensor
/// name present in two shards.
pub struct GgufModel {
    shards: Vec<GgufFile>,
    index: BTreeMap<String, (usize, usize)>, // name -> (shard, tensor idx)
}

impl GgufModel {
    pub fn new(paths: Vec<PathBuf>) -> Result<GgufModel, String> {
        if paths.is_empty() {
            return Err("GGUF: a model needs at least one shard".into());
        }
        let mut shards = Vec::with_capacity(paths.len());
        for p in &paths {
            shards.push(GgufFile::open(p)?);
        }
        let mut model = GgufModel {
            shards,
            index: BTreeMap::new(),
        };
        model.validate_split()?;
        for (i, g) in model.shards.iter().enumerate() {
            for (j, t) in g.tensors().iter().enumerate() {
                if let Some((first, _)) = model.index.get(&t.name).copied() {
                    return Err(format!(
                        "GGUF: tensor {} is in two shards ({} and {})",
                        t.name,
                        model.shards[first].path().display(),
                        g.path().display()
                    ));
                }
                model.index.insert(t.name.clone(), (i, j));
            }
        }
        Ok(model)
    }

    /// Opens every shard of the model that `any_shard` belongs to (errors when one is missing).
    pub fn open(any_shard: impl AsRef<Path>) -> Result<GgufModel, String> {
        GgufModel::new(gguf_split_paths(any_shard.as_ref())?)
    }

    pub fn size(&self) -> usize {
        self.shards.len()
    }
    pub fn shard(&self, i: usize) -> &GgufFile {
        &self.shards[i]
    }
    /// The metadata shard: general.architecture and the model's keys.
    pub fn meta(&self) -> &GgufFile {
        &self.shards[0]
    }
    pub fn find(&self, name: &str) -> Option<(&TensorInfo, usize)> {
        self.index
            .get(name)
            .map(|(s, j)| (&self.shards[*s].tensors()[*j], *s))
    }
    /// Whether `t` (a tensor of shard `s`) has a known byte count that lies inside its file.
    pub fn in_bounds(&self, t: &TensorInfo, s: usize) -> bool {
        let g = &self.shards[s];
        let bytes = tensor_payload_bytes(t);
        let payload = g.file_size() - g.data_start();
        bytes != 0 && t.offset <= payload && bytes <= payload - t.offset
    }

    fn validate_split(&self) -> Result<(), String> {
        let n = self.shards.len();
        let count0 = self.shards[0].get("split.count");
        if n == 1 {
            if let Some(c) = count0 {
                if c.u > 1 {
                    return Err(format!(
                        "GGUF: {} is shard 1 of {}, but it was opened as a whole model",
                        self.shards[0].path().display(),
                        c.u
                    ));
                }
            }
            return Ok(());
        }
        if self.shards[0].get("general.architecture").is_none() {
            return Err(format!(
                "GGUF: {} has no general.architecture; the first shard of a split model carries the metadata",
                self.shards[0].path().display()
            ));
        }
        let total = self.shards[0].get("split.tensors.count");
        let mut tensors = 0u64;
        for (i, g) in self.shards.iter().enumerate() {
            let count = g.get("split.count");
            let no = g.get("split.no");
            let tc = g.get("split.tensors.count");
            let ok = count.is_some()
                && no.is_some()
                && count.unwrap().u == n as u64
                && no.unwrap().u == i as u64
                && match (total, tc) {
                    (Some(t), Some(c)) => t.u == c.u,
                    (None, None) => true,
                    _ => false,
                };
            if !ok {
                return Err(format!(
                    "GGUF: {} does not declare itself shard {} of {} of this model (split.count / split.no / split.tensors.count)",
                    g.path().display(),
                    i + 1,
                    n
                ));
            }
            tensors += g.tensors().len() as u64;
        }
        if let Some(total) = total {
            if tensors != total.u {
                return Err(format!(
                    "GGUF: the {n} shards hold {tensors} tensors, but split.tensors.count is {}",
                    total.u
                ));
            }
        }
        Ok(())
    }
}

/// The engine is specialised to ONE model; anything else must be refused with a
/// precise error rather than silently mis-run.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpGuard {
    pub block_count: u64,
    pub hidden: u64,
    /// 0 = presence-only: pruned variants (GSQ-RCO Coder) legitimately ship fewer
    /// experts than the canonical 512; the graph reads the true value.
    pub experts: u64,
    pub experts_used: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
}

impl Default for Qwen4ExpGuard {
    fn default() -> Self {
        Qwen4ExpGuard {
            block_count: 48,
            hidden: 2560,
            experts: 0,
            experts_used: 0,
            head_count: 24,
            head_count_kv: 2,
        }
    }
}

/// Empty Ok string == pass.
pub fn check_architecture(g: &GgufFile, want: &Qwen4ExpGuard) -> String {
    let arch = match g.get("general.architecture") {
        Some(a) => a,
        None => return "missing general.architecture".into(),
    };
    if arch.s != "qwen4exp" {
        return format!(
            "architecture is '{}', this engine requires 'qwen4exp'",
            arch.s
        );
    }
    let reqs: [(&str, u64); 6] = [
        ("qwen4exp.block_count", want.block_count),
        ("qwen4exp.embedding_length", want.hidden),
        ("qwen4exp.expert_count", want.experts),
        ("qwen4exp.expert_used_count", want.experts_used),
        ("qwen4exp.attention.head_count", want.head_count),
        ("qwen4exp.attention.head_count_kv", want.head_count_kv),
    ];
    for (key, want_v) in reqs {
        match g.get(key) {
            None => return format!("missing {key}"),
            Some(v) => {
                if want_v != 0 && v.u != want_v {
                    return format!("{key} = {}, expected {want_v}", v.u);
                }
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ported from Strata src/artifact/gguf_reader_test.cpp: the reader refuses what
    // it cannot arbitrate (synthetic GGUF, no model, no GPU).
    fn write_gguf(names: &[&str]) -> PathBuf {
        fn put_u32(b: &mut Vec<u8>, v: u32) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn put_u64(b: &mut Vec<u8>, v: u64) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn put_str(b: &mut Vec<u8>, s: &str) {
            put_u64(b, s.len() as u64);
            b.extend_from_slice(s.as_bytes());
        }
        // A GGUF v3 file of F32[8] tensors named `names`, laid out 32 bytes apart
        // from a 32-byte-aligned data start.
        let mut b: Vec<u8> = Vec::new();
        put_u32(&mut b, 0x46554747); // "GGUF"
        put_u32(&mut b, 3);
        put_u64(&mut b, names.len() as u64); // n_tensors
        put_u64(&mut b, 0); // n_kv
        for (i, name) in names.iter().enumerate() {
            put_str(&mut b, name);
            put_u32(&mut b, 1); // n_dims
            put_u64(&mut b, 8);
            put_u32(&mut b, 0); // F32
            put_u64(&mut b, (32 * i) as u64); // offset from data_start
        }
        b.resize(b.len().div_ceil(32) * 32 + 32 * names.len(), 0);
        // one fixture per distinct call: parallel tests must not delete each other's file
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "strata_rust_gguf_test_{}.gguf",
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, &b).unwrap();
        path
    }

    #[test]
    fn distinct_names_open_and_both_are_found() {
        let path = write_gguf(&["blk.0.attn_q.weight", "blk.0.attn_k.weight"]);
        let r = GgufFile::open(&path);
        std::fs::remove_file(&path).ok();
        let g = r.expect("the file opens");
        assert_eq!(g.tensors().len(), 2);
        assert!(g.find("blk.0.attn_q.weight").is_some() && g.find("blk.0.attn_k.weight").is_some());
    }

    #[test]
    fn the_same_name_twice_is_refused_at_open_naming_tensor_and_file() {
        let path = write_gguf(&["blk.0.attn_q.weight", "blk.0.attn_q.weight"]);
        let err = GgufFile::open(&path).expect_err("a duplicate must be refused");
        std::fs::remove_file(&path).ok();
        assert!(
            err.contains("blk.0.attn_q.weight"),
            "the error names the tensor: {err}"
        );
        assert!(
            err.contains("strata_rust_gguf_test"),
            "the error names the file: {err}"
        );
    }

    #[test]
    fn bad_magic_and_wrong_version_are_refused() {
        let path = std::env::temp_dir().join("strata_rust_gguf_bad.gguf");
        std::fs::write(&path, [0u8; 64]).unwrap();
        assert!(GgufFile::open(&path).unwrap_err().contains("bad magic"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn q2_0_geometry_is_64_over_18() {
        assert_eq!(block_geometry(42), Some((64, 18)));
        assert_eq!(block_geometry(29), Some((256, 56))); // IQ1_M, present in C++ but missing from the
                                                         // earlier Python reference
        assert_eq!(ggml_type_name(42), "Q2_0");
        assert_eq!(block_geometry(4), None);
    }

    #[test]
    fn payload_bytes_match_the_block_geometry() {
        let t = TensorInfo {
            name: "w".into(),
            shape: vec![4096, 2560],
            dtype: 42,
            offset: 0,
        };
        // 4096*2560 elements / 64 per block * 18 bytes
        assert_eq!(tensor_payload_bytes(&t), 4096 * 2560 / 64 * 18);
        let ragged = TensorInfo {
            name: "w".into(),
            shape: vec![4095, 2560],
            dtype: 42,
            offset: 0,
        };
        assert_eq!(tensor_payload_bytes(&ragged), 0);
        let unknown = TensorInfo {
            name: "w".into(),
            shape: vec![64],
            dtype: 5,
            offset: 0,
        };
        assert_eq!(tensor_payload_bytes(&unknown), 0);
        let zero = TensorInfo {
            name: "w".into(),
            shape: vec![0],
            dtype: 42,
            offset: 0,
        };
        assert_eq!(tensor_payload_bytes(&zero), 0);
    }

    #[test]
    fn data_start_is_aligned_past_the_header() {
        let path = write_gguf(&["a"]);
        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.data_start() % g.alignment(), 0);
        assert!(g.data_start() <= g.file_size());
        std::fs::remove_file(&path).ok();
    }
}
