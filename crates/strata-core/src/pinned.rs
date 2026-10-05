// @generated: ported from Strata src/core/pinned.cu @ 99f3dbd0 - the
// deterministic third of it. The rest (mmap, madvise, hugepages,
// cudaHostRegister, the working-set lock, the timed probes) is the platform
// boundary and stays where it is.

//! The pinned arena's decision logic: the pin cap, the slice bounds, and the
//! read plan with its per-layer checksums and refusal messages.
//!
//! `pinned.cu` has no `__global__` or `__device__` in it - it is host code that
//! calls the runtime - so the corpus harness compiles it with g++ against stub
//! headers and calls the real functions. What is deterministic ports; what
//! measures the machine does not, and the timing fields are excluded from the
//! golden for that reason.
//!
//! The properties the C++ got right and this keeps:
//!
//! * A garbage `STRATA_ARENA_PIN_GIB` is not "unset". `atoi("abc")` is `0`, and
//!   `0` is not negative, so a typo pins zero GiB. Only an empty value means
//!   unset; `auto` means the Windows shared-memory budget decides.
//! * A short read is a WRONG load, not a slow one. The refusal says where and how
//!   short, and the checksums come back empty on failure - they are only moved on
//!   the success path, so a refused load cannot report per-layer hashes as if the
//!   tail had been written.
//! * `bytes` and `layers` are filled before the read starts, so they survive a
//!   failure. The error path does not zero them.
//! * Seeking past EOF on a regular file succeeds; the read that follows returns
//!   zero bytes and that is the short read. There is no seek error to report.
//! * The per-layer checksum is independent of which thread ran which layer, so
//!   the plan is deterministic under any thread count. The corpus proves it: the
//!   one-thread and four-thread cases hash identically.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{fnv_seeded, FNV_OFFSET};

/// The seed every layer hash starts from (and the one `fnv1a64` defaults to).
/// What `load_experts_ranges` decided. The timing fields exist because the C++
/// struct has them; they are not part of the golden.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadStats {
    /// Wall clock of the whole loop, in seconds; `-1.0` on failure. Not in the golden.
    pub seconds: f64,
    /// Summed over the reader threads. Not in the golden.
    pub read_seconds: f64,
    /// memcpy + FNV-1a only. Not in the golden.
    pub copy_seconds: f64,
    /// False: the load failed, see `error`.
    pub ok: bool,
    /// Why it failed, for the caller's message.
    pub error: String,
    pub bytes: u64,
    pub layers: u64,
    /// One FNV-1a per layer. Empty when the load failed - it is only filled on
    /// the success path.
    pub layer_checksums: Vec<u64>,
}

/// `std::atoi`: skip whitespace, optional sign, leading digits, stop. No digits
/// is zero, not an error - which is why a garbage env value pins zero.
fn c_atoi(s: &str) -> i64 {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && (b[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    let neg = match b.get(i) {
        Some(b'+') => {
            i += 1;
            false
        }
        Some(b'-') => {
            i += 1;
            true
        }
        _ => false,
    };
    let start = i;
    let mut acc: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        acc = acc.saturating_mul(10).saturating_add((b[i] - b'0') as i64);
        i += 1;
    }
    if i == start {
        return 0;
    }
    if neg {
        -acc
    } else {
        acc
    }
}

/// How much of the arena the sliced registration may pin, in GiB. `-1`: unset.
/// `-2`: `auto` - the Windows shared-memory budget sets it. Anything else is
/// `atoi` of the value, and a value that is not a number is therefore zero.
#[must_use]
pub fn arena_pin_cap_gib(env: Option<&str>) -> i64 {
    let Some(e) = env else { return -1 };
    if e.is_empty() {
        return -1;
    }
    if e == "auto" {
        return -2;
    }
    let v = c_atoi(e);
    if v < 0 {
        -1
    } else {
        v
    }
}

/// Slice boundaries for a uniform sliced registration. The trailing partial
/// slice is included as the end of the last full one - a lone remainder is not a
/// slice. `slice == 0` is no bounds at all.
#[must_use]
pub fn uniform_bounds(bytes: u64, slice: u64) -> Vec<u64> {
    let mut b = Vec::new();
    if slice == 0 {
        return b;
    }
    let mut off = 0u64;
    while off.wrapping_add(slice) <= bytes {
        b.push(off);
        off += slice;
    }
    if !b.is_empty() {
        b.push(b[b.len() - 1] + slice);
    }
    b
}

/// `#633`: whether the experts must be read unbuffered. On Linux the answer is
/// always no unless the env says otherwise; the timed probe is Windows-only.
/// `env` is `STRATA_UNBUFFERED_LOAD`.
#[must_use]
pub fn experts_unbuffered(env: Option<&str>) -> (bool, String) {
    if let Some(e) = env.filter(|e| !e.is_empty()) {
        return (
            e.as_bytes()[0] != b'0',
            format!("STRATA_UNBUFFERED_LOAD={e}"),
        );
    }
    (false, "buffered (not Windows)".to_string())
}

/// Read the layers into `dst` and hash each one. `dst` must cover every
/// `layer_off[i] + layer_bytes[i]`.
///
/// `threads` is the reader-pool size (clamped to at least 1); the result does not
/// depend on it, only the timing does.
///
/// # Panics
///
/// Panics if `dst` is too small for a layer that was actually read.
pub fn load_experts_ranges(
    path: &Path,
    dst: &mut [u8],
    layer_off: &[u64],
    layer_bytes: &[u64],
    threads: usize,
    chunk: u64,
) -> LoadStats {
    let mut st = LoadStats {
        ok: true,
        ..Default::default()
    };
    st.layers = layer_off.len() as u64;
    st.bytes = layer_bytes.iter().fold(0u64, |a, b| a.wrapping_add(*b));
    let mut layer_hash = vec![FNV_OFFSET; layer_off.len()];
    let threads = threads.max(1);

    // The C++ runs this on a thread pool, one file handle per worker, seeked once
    // per layer - a shared handle would need a lock around the seek and would
    // serialise the very thing the threads are here to parallelise. The plan is
    // thread-count independent (the corpus proves it: the one-thread and
    // four-thread cases hash identically), so the port runs it serially and the
    // parallelism stays with whatever owns the destination buffer. `threads` is
    // kept in the signature because the C++ has it; it changes the timing, not the
    // answer.
    let _ = threads;
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            st.ok = false;
            st.error = format!("cannot open {}", path.display());
            st.seconds = -1.0;
            return st; // `layer_checksums` stays empty: it is only moved on success
        }
    };
    let mut buf = vec![0u8; chunk as usize];
    for l in 0..layer_off.len() {
        let off = layer_off[l];
        let mut remaining = layer_bytes[l];
        let mut pos = 0u64;
        let mut h = FNV_OFFSET;
        // 64-bit seek: the pack is 42.9 GB, so a 32-bit seek would wrap past
        // 4 GiB. Seeking past EOF succeeds; the read is what reports it.
        if file.seek(SeekFrom::Start(off)).is_err() {
            st.ok = false;
            st.error = format!("seek to {off} B failed in layer {l}");
            st.seconds = -1.0;
            return st;
        }
        while remaining > 0 {
            let n = remaining.min(chunk) as usize;
            // A short read is EOF or an I/O error, never a silent zero fill: say
            // WHERE and HOW SHORT, and keep no further - the caller turns this into
            // a refused load, not a wrong answer.
            let got = file.read(&mut buf[..n]).unwrap_or(0);
            if got != n {
                // `ferror` is not set at a clean EOF, which is the only thing that
                // gets here on Linux; a real error would add " (ferror set)".
                st.ok = false;
                st.error = format!(
                    "short read in layer {l}: got {got} of {n} B at offset {}",
                    off + pos
                );
                st.seconds = -1.0;
                return st;
            }
            dst[(off + pos) as usize..(off + pos) as usize + n].copy_from_slice(&buf[..n]);
            h = fnv_seeded(&buf[..n], h);
            pos += n as u64;
            remaining -= n as u64;
        }
        layer_hash[l] = h;
    }
    st.layer_checksums = layer_hash;
    st.seconds = 0.0;
    st
}

/// The unbuffered loader. On Linux it is not the path taken, and the C++ says so
/// by returning an unmodified `LoadStats` with `ok = false`.
#[must_use]
pub fn load_experts_direct(
    path: &Path,
    dst: &mut [u8],
    layer_off: &[u64],
    layer_bytes: &[u64],
) -> LoadStats {
    let _ = (path, dst, layer_off, layer_bytes);
    LoadStats {
        ok: false,
        ..Default::default()
    }
}
