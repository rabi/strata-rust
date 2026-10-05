// @generated: ported from Strata src/core/expert_source.cpp (the `detail`
// namespace - the CPU-only policy the header exposes so selection and byte
// accounting can be tested without initializing a GPU) @ 99f3dbd0.

//! Expert admission policy: host RAM (with cgroup limits), the cache-complement
//! plan, and the resident-keep decision.
//!
//! These are the parts of `expert_source.cpp` that are pure - no CUDA call is
//! reachable from them - so they port cleanly and the corpus harness can compile
//! the real C++ and compare. What the C++ got right and this keeps:
//!
//! * The cgroup walk is a *string* prefix test against the mount root, not a
//!   component test, and a group whose limit file is unreadable fails the whole
//!   read - except the root group, which legitimately has no `memory.max` when
//!   `cgroup.controllers` says it is a real cgroup root.
//! * `inactive_file` can race `current`, so it is bounded by the charged usage
//!   before any of it is counted as reclaimable, and `file_dirty` /
//!   `file_writeback` are subtracted from it in order, each clamped at zero.
//! * The plan's error strings are the contract - the caller shows them.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

/// Sentinel used by the pure complement planner for a blob that remains in the
/// mmap fallback.
pub const NO_CACHE_COMPLEMENT: u64 = u64::MAX;

/// The prefix every `FileExpertSource` refusal carries.
const ERR: &str = "FileExpertSource: ";

// ------------------------------------------------------------------ C++ stream parsing

/// `stream >> uint64_t`: skip whitespace, optional sign, at least one digit, stop
/// at the first non-digit. Out of range is a failed extraction. A `-` negates the
/// magnitude, which wraps - it is not an error.
fn stream_u64(s: &str) -> Option<u64> {
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
    let mut acc: u64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        acc = acc.checked_mul(10)?.checked_add((b[i] - b'0') as u64)?;
        i += 1;
    }
    if i == start {
        return None;
    }
    Some(if neg { acc.wrapping_neg() } else { acc })
}

/// `stream >> std::string`: skip whitespace, take up to the next whitespace.
fn stream_string(s: &str) -> Option<&str> {
    let trimmed = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if trimmed.is_empty() {
        return None;
    }
    trimmed.split_whitespace().next()
}

/// `std::stoull(s, &consumed)` for a caller that requires the whole string to be
/// consumed (`consumed == s.len()`) - so leading whitespace is allowed, a trailing
/// suffix is not. `None` is the throw: no conversion, or out of range.
fn stoull(s: &str) -> Option<u64> {
    let trimmed = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (neg, digits) = match trimmed.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let value = match digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
        .filter(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        Some(hex) => u64::from_str_radix(hex, 16).ok()?,
        None => digits.parse::<u64>().ok()?,
    };
    Some(if neg { value.wrapping_neg() } else { value })
}

// ------------------------------------------------------------------ cgroup arithmetic

/// Required cgroup-v2 usage counters for the conservative cache-reclaim
/// allowance. `None` from [`cgroup_available_bytes`] means they were unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CgroupMemoryStat {
    pub current: u64,
    pub inactive_file: u64,
    pub file_dirty: u64,
    pub file_writeback: u64,
    pub valid: bool,
}

/// Calculate additional bytes under a finite cgroup limit after reclaiming only
/// clean inactive file cache. `None` when the required `memory.stat` counters
/// were unavailable.
#[must_use]
pub fn cgroup_available_bytes(limit: u64, stat: &CgroupMemoryStat) -> Option<u64> {
    if !stat.valid {
        return None;
    }
    // memory.stat's inactive_file can race memory.current, so bound it to
    // charged usage first.
    let mut reclaimable = std::cmp::min(stat.inactive_file, stat.current);
    reclaimable = reclaimable.saturating_sub(stat.file_dirty);
    reclaimable = reclaimable.saturating_sub(stat.file_writeback);
    // Reclaiming clean file pages reduces usage; saturating subtraction also
    // handles a transient over-limit read.
    let usage_after_reclaim = stat.current - reclaimable;
    Some(limit.saturating_sub(usage_after_reclaim))
}

/// #633: the RAM this process can get. `available`: MemAvailable, lowered to the
/// room under the tightest cgroup limit; `cgroup_limit`: that tightest limit
/// itself, `u64::MAX` when there is none - what a container can never exceed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostMemory {
    pub available: u64,
    pub cgroup_limit: u64,
}

impl HostMemory {
    fn new() -> HostMemory {
        HostMemory {
            available: 0,
            cgroup_limit: u64::MAX,
        }
    }
}

/// `std::filesystem::lexically_normal` - purely lexical, no symlink resolution.
fn lexically_normal(path: &Path) -> PathBuf {
    let absolute = path.is_absolute();
    let mut parts: Vec<&OsStr> = Vec::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.last().is_some_and(|p| *p != OsStr::new("..")) {
                    parts.pop();
                } else if !absolute {
                    parts.push(OsStr::new(".."));
                }
            }
            Component::Normal(s) => parts.push(s),
            Component::RootDir => {}
            other => parts.push(other.as_os_str()),
        }
    }
    let mut out = PathBuf::new();
    if absolute {
        out.push(std::path::MAIN_SEPARATOR_STR);
    }
    for p in parts {
        out.push(p);
    }
    out
}

/// `path.string().rfind(root.string(), 0) == 0` - a string prefix test, not a
/// component test.
fn string_prefixed(path: &Path, root: &Path) -> bool {
    let root = root.to_string_lossy();
    path.to_string_lossy().starts_with(&*root)
}

fn is_directory(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_dir())
}

fn path_exists(path: &Path) -> bool {
    std::fs::metadata(path).is_ok()
}

fn read_to_string(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// `read_cgroup_memory_stat`: `memory.stat` must parse as exactly `key value` per
/// line and must carry all three counters. A duplicate is a contradiction, not a
/// second sample.
fn read_cgroup_memory_stat(dir: &Path, current: u64) -> Option<CgroupMemoryStat> {
    let text = read_to_string(&dir.join("memory.stat"))?;
    let mut stat = CgroupMemoryStat {
        current,
        valid: true,
        ..Default::default()
    };
    let (mut saw_inactive, mut saw_dirty, mut saw_writeback) = (false, false, false);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
            return None;
        };
        // `fields >> std::ws; if (!fields.eof()) return false` - no trailing junk.
        if fields.next().is_some() {
            return None;
        }
        let value = stream_u64(value)?;
        match key {
            "inactive_file" => {
                if saw_inactive {
                    return None;
                }
                saw_inactive = true;
                stat.inactive_file = value;
            }
            "file_dirty" => {
                if saw_dirty {
                    return None;
                }
                saw_dirty = true;
                stat.file_dirty = value;
            }
            "file_writeback" => {
                if saw_writeback {
                    return None;
                }
                saw_writeback = true;
                stat.file_writeback = value;
            }
            _ => {}
        }
    }
    if !(saw_inactive && saw_dirty && saw_writeback) {
        return None;
    }
    Some(stat)
}

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root`.
/// cgroup v2 (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: limit - usage); no cgroup line at
/// all is MemAvailable alone. `None` when the RAM cannot be determined.
#[must_use]
pub fn host_available_memory(
    meminfo: &Path,
    self_cgroup: &Path,
    cgroup_root: &Path,
) -> Option<HostMemory> {
    let mut m = HostMemory::new();

    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    // The last well-formed line wins; a malformed one is skipped, not fatal.
    let mut bytes = 0u64;
    if let Some(text) = read_to_string(meminfo) {
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let (Some(key), Some(value), Some(unit)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            if key == "MemAvailable:" && unit == "kB" {
                if let Some(kb) = stream_u64(value).filter(|kb| *kb <= u64::MAX / 1024) {
                    bytes = kb * 1024;
                }
            }
        }
    }
    if bytes == 0 {
        return None;
    }

    // Account for the tightest cgroup ancestor limit when its normal mount is
    // visible. This is a point-in-time guard, not a reservation.
    let groups = read_to_string(self_cgroup)?;
    let mut v2 = false;
    let mut v1_path = String::new();
    for line in groups.lines() {
        if !line.starts_with("0::/") {
            // v1: "N:controller[,controller]:/path"
            let Some(c1) = line.find(':') else { continue };
            let Some(c2) = line[c1 + 1..].find(':').map(|p| p + c1 + 1) else {
                continue;
            };
            if line[c1 + 1..c2].split(',').any(|c| c == "memory") {
                v1_path = line[c2 + 1..].to_string();
            }
            continue;
        }
        let mut path = lexically_normal(&cgroup_root.join(&line[4..]));
        if !string_prefixed(&path, cgroup_root) || !is_directory(&path) {
            return None;
        }
        v2 = true;
        while string_prefixed(&path, cgroup_root) {
            let limit_text = read_to_string(&path.join("memory.max")).unwrap_or_default();
            let current_text = read_to_string(&path.join("memory.current")).unwrap_or_default();
            let limit = stream_string(&limit_text);
            let current = stream_u64(&current_text);
            let readable = limit.is_some() && current.is_some();
            // The host's root cgroup has no memory.max; ordinary child groups
            // must expose their limits.
            if !readable
                && !(path == cgroup_root
                    && !path_exists(&path.join("memory.max"))
                    && path_exists(&path.join("cgroup.controllers")))
            {
                return None;
            }
            if let (Some(limit), Some(current)) = (limit, current) {
                if limit != "max" {
                    let cap = stoull(limit)?;
                    let stat = read_cgroup_memory_stat(&path, current)?;
                    let available = cgroup_available_bytes(cap, &stat)?;
                    bytes = std::cmp::min(bytes, available);
                    m.cgroup_limit = std::cmp::min(m.cgroup_limit, cap);
                }
            }
            if path == cgroup_root {
                break;
            }
            path = path.parent().map_or_else(PathBuf::new, |p| p.to_path_buf());
        }
    }

    if !v2 && !v1_path.is_empty() {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory
        // controller's group and its ancestors. An unlimited group says a number
        // near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone,
        // as with no cgroup at all.
        let mroot = cgroup_root.join("memory");
        let tail = v1_path.strip_prefix('/').unwrap_or(&v1_path);
        let mut path = lexically_normal(&mroot.join(tail));
        while string_prefixed(&path, &mroot) {
            let cap = stream_u64(
                &read_to_string(&path.join("memory.limit_in_bytes")).unwrap_or_default(),
            );
            let usage = stream_u64(
                &read_to_string(&path.join("memory.usage_in_bytes")).unwrap_or_default(),
            );
            if let (Some(cap), Some(usage)) = (cap, usage) {
                if cap < 1u64 << 62 {
                    bytes = std::cmp::min(bytes, cap.saturating_sub(usage));
                    m.cgroup_limit = std::cmp::min(m.cgroup_limit, cap);
                }
            }
            if path == mroot {
                break;
            }
            path = path.parent().map_or_else(PathBuf::new, |p| p.to_path_buf());
        }
    }

    m.available = bytes;
    Some(m)
}

// ------------------------------------------------------------------ the complement plan

/// Resolve one blob through the compact copy when present, otherwise preserve its
/// exact mapped-file fallback.
/// `Some(offset)` = read `offset` bytes into the compact complement copy;
/// `None` = the exact mapped-file pointer. `have_complement` stands in for the
/// C++'s null `complement_host` check.
#[must_use]
pub fn cache_complement_blob_or_fallback(
    index: usize,
    offsets: &[u64],
    have_complement: bool,
) -> Option<u64> {
    if have_complement && index < offsets.len() && offsets[index] != NO_CACHE_COMPLEMENT {
        return Some(offsets[index]);
    }
    None
}

/// Build compact offsets for experts absent from both the primary GPU cache and
/// an optional second GPU tier. `Err` carries the exact message the C++ writes.
pub fn make_cache_complement_plan(
    n_layers: i64,
    n_expert: i64,
    layer_blob_bytes: &[u64],
    primary_gpu_pairs: &[(i32, i32)],
    additional_gpu_pairs: &[(i32, i32)],
) -> Result<(Vec<u64>, u64), String> {
    let mut bytes = 0u64;
    if n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.len() as i64 != n_layers {
        return Err(format!(
            "{ERR}invalid geometry for the cache complement plan"
        ));
    }
    if n_layers as u64 > u64::MAX / n_expert as u64 {
        return Err(format!("{ERR}cache complement index table is too large"));
    }
    let count = n_layers as usize * n_expert as usize;
    let mut omitted = vec![0u8; count];
    let mut mark = |pairs: &[(i32, i32)], bit: u8, label: &str| -> Result<(), String> {
        for &(a, b) in pairs {
            if a < 0 || b < 0 || a as i64 >= n_layers || b as i64 >= n_expert {
                return Err(format!("{ERR}{label} pair is outside the expert geometry"));
            }
            let index = a as usize * n_expert as usize + b as usize;
            if omitted[index] & bit != 0 {
                return Err(format!(
                    "{ERR}duplicate {label} pair in the cache complement plan"
                ));
            }
            if bit == 2 && omitted[index] & 1 != 0 {
                return Err(format!(
                    "{ERR}the primary and additional GPU expert tiers overlap"
                ));
            }
            omitted[index] |= bit;
        }
        Ok(())
    };
    mark(primary_gpu_pairs, 1, "primary GPU")?;
    mark(additional_gpu_pairs, 2, "additional GPU")?;
    for &blob_bytes in layer_blob_bytes {
        if blob_bytes == 0 {
            return Err(format!(
                "{ERR}cache complement layer has zero-sized expert blobs"
            ));
        }
    }

    let mut offsets = vec![NO_CACHE_COMPLEMENT; count];
    for (layer, &blob_bytes) in layer_blob_bytes.iter().enumerate() {
        for expert in 0..n_expert as usize {
            let index = layer * n_expert as usize + expert;
            if omitted[index] != 0 {
                continue;
            }
            if bytes > u64::MAX - blob_bytes {
                return Err(format!("{ERR}cache complement size overflows"));
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if bytes > usize::MAX as u64 {
        return Err(format!(
            "{ERR}cache complement exceeds the host address space"
        ));
    }
    Ok((offsets, bytes))
}

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too. The
/// prompt path lends the cache's LAST slots (from `lend_from` on; a short prompt
/// lends only the last few), and a lent slot's expert is streamed from RAM during
/// the prompt and copied back into its slot after it. `base_bytes` (every expert
/// no slot holds) must fit `budget`; slots are then added from the end down to
/// `lend_from` while they still fit. Returns the first slot kept in RAM
/// (`slot_bytes.len()` = none), or -1 when `base_bytes` alone exceeds `budget`.
#[must_use]
pub fn choose_resident_keep_from(
    slot_bytes: &[u64],
    base_bytes: u64,
    budget: u64,
    lend_from: i64,
) -> i64 {
    if base_bytes > budget {
        return -1;
    }
    let slots = slot_bytes.len() as i64;
    let lend_from = if lend_from < 0 || lend_from > slots {
        slots
    } else {
        lend_from
    }; // no lend region
    let mut keep = slots;
    let mut bytes = base_bytes;
    while keep > lend_from {
        let b = slot_bytes[(keep - 1) as usize];
        if b > budget - bytes {
            break;
        }
        bytes += b;
        keep -= 1;
    }
    keep
}

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes
/// `in`'s place in the compact copy (the caller copies out's bytes there). False,
/// and nothing changed, unless `in` is in the copy and `out` is not.
pub fn exchange_cache_complement(offsets: &mut [u64], in_index: usize, out_index: usize) -> bool {
    if in_index == out_index
        || in_index >= offsets.len()
        || out_index >= offsets.len()
        || offsets[in_index] == NO_CACHE_COMPLEMENT
        || offsets[out_index] != NO_CACHE_COMPLEMENT
    {
        return false;
    }
    offsets[out_index] = offsets[in_index];
    offsets[in_index] = NO_CACHE_COMPLEMENT;
    true
}
