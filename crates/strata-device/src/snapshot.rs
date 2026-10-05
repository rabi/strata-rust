//! The shim-backed [`Device`] for the conversation-snapshot contract.
//!
//! `strata-core`'s snapshot modules move bytes through a pointer-free trait —
//! they name a layer, a pool set and a region, never an address. This module is
//! where addresses live: the session's device buffers are named once, and every
//! transfer resolves its destination exactly the way the C++ engine did — from
//! the state's own layout flags — and then goes through the `memcpy_default` and
//! `device_sync` slots.
//!
//! The module is the workspace's second (and last) home for `unsafe`: the
//! vtable calls, same kind as `shim.rs`.
//!
//! What this type cannot express is rejected at build time, not at transfer
//! time: a null pointer becomes `None`, and the two residency hooks have no
//! slots at all, so `new` refuses any state that needs them (mode 1 needs the
//! stream map, mode 2 the ring restore) rather than admitting a snapshot that
//! could never be made readable. A fully-resident session (mode 0) covers every
//! pool with plain copies, which is why it is the first state this seam serves.

use std::os::raw::c_char;
use std::os::raw::c_void;

use strata_core::conversation_kv::{Device, Layer, PoolSet, Region, Running, RunningPart};

use crate::abi::StrataKernels;

/// One device buffer the session owns: base address and size, allocated by the
/// engine (`device_alloc`) or handed to it by the loader. Borrowed, not owned —
/// dropping [`SnapshotDevice`] never frees anything.
#[derive(Clone, Copy, Debug)]
pub struct DeviceRegion {
    /// a null pointer is a *missing* buffer here, never a live address
    addr: usize,
    bytes: usize,
}

impl DeviceRegion {
    /// `p == nullptr` means the state does not have this buffer.
    pub fn new(addr: *const c_void, bytes: usize) -> Option<DeviceRegion> {
        if addr.is_null() {
            None
        } else {
            Some(DeviceRegion {
                addr: addr as usize,
                bytes,
            })
        }
    }
}

/// The device buffers of one layer's K/V, resolved once from its layout flags
/// (`pool_ptrs`) so a transfer is a lookup and a range check, never a format
/// branch. A set whose region is absent carries `None`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerKv {
    regions: [Option<DeviceRegion>; 5],
}

impl LayerKv {
    pub fn regions(regions: [Option<DeviceRegion>; 5]) -> LayerKv {
        LayerKv { regions }
    }
}

/// Where one checkpoint byte range lives on the device. The C++ reaches these by
/// `st.gdn_state`, `st.ple_hist` and the per-layer indexer arrays; a session
/// builds this table once and hands it to [`SnapshotDevice::running`].
#[derive(Clone, Debug)]
pub struct RunningTarget {
    /// `SessionState::gdn_state`
    pub gdn: Option<DeviceRegion>,
    /// `SessionState::ple_hist`
    pub ple: Option<DeviceRegion>,
    /// one entry per QSA layer, indexed by the same ordinal the contract names
    pub indexer: Vec<IndexerBuffers>,
}

/// The four indexer arrays of one QSA layer, as device buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexerBuffers {
    pub tail: Option<DeviceRegion>,
    pub dead: Option<DeviceRegion>,
    pub block_pos: Option<DeviceRegion>,
    /// the moving spare row reads from inside the pooled array, so this is the
    /// whole array, not one row
    pub pooled: Option<DeviceRegion>,
}

impl RunningTarget {
    pub fn empty() -> RunningTarget {
        RunningTarget {
            gdn: None,
            ple: None,
            indexer: Vec::new(),
        }
    }
}

/// The K/V residency of one layer, as far as this seam must refuse or accept it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// every page resident: plain copies cover the whole contract
    Resident,
    /// streamed (`mode 1`): restoring needs `kv_stream_reset`, which this shim
    /// build does not offer
    Streamed,
    /// a ring (`mode 2`): restoring needs `kv_ring_restore`
    Ring,
}

/// The `QsaState` device memory of one layer, resolved to addresses. `layout()`
/// decides which set a transfer lands in (the same format rules the contract's
/// presence bits describe); this says whether the seam can serve the state at
/// all.
#[derive(Clone, Copy, Debug)]
pub struct LayerState {
    pub residency: Residency,
    /// the VRAM slots the readers see
    pub slots: LayerKv,
    /// the authoritative copy; `Default::default()` = no host set
    pub host: LayerKv,
}

impl LayerState {
    pub fn resident(slots: LayerKv, host: LayerKv) -> LayerState {
        LayerState {
            residency: Residency::Resident,
            slots,
            host,
        }
    }
}

/// A session's device-side state: the drafter's ring, the QSA layers in the
/// contract's ordinal order, and the running state. Addresses only — the engine
/// owns the allocations; [`SnapshotDevice::new`] consumes this descriptor.
#[derive(Clone, Debug)]
pub struct SessionDevice {
    pub draft: LayerState,
    pub qsa: Vec<LayerState>,
    pub running: RunningTarget,
}

/// Errors from building the seam: missing slots, states it cannot serve, or
/// transfers that would run past a buffer.
fn err(msg: impl Into<String>) -> String {
    msg.into()
}

/// The shim's `cudaGetErrorString` bytes, NUL-terminated, into a String.
fn cstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().map(|&b| b as u8).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// The snapshot contract's device side over a shim vtable. Borrows the shim;
/// the session's buffers are `Copy` addresses the engine keeps alive.
#[derive(Debug)]
pub struct SnapshotDevice<'a> {
    k: &'a StrataKernels,
    draft: LayerKv,
    draft_host: LayerKv,
    qsa: Vec<(LayerKv, LayerKv)>,
    running: RunningTarget,
}

impl<'a> SnapshotDevice<'a> {
    /// The vtable must carry the snapshot slots; and no layer may need residency
    /// work, because until this ABI grows those slots a shim cannot make an
    /// evicted image readable. Checking both here means the first transfer never
    /// fails for a reason known at construction.
    pub fn new(k: &'a StrataKernels, session: SessionDevice) -> Result<SnapshotDevice<'a>, String> {
        if k.memcpy_default.is_none() {
            return Err(err(
                "this shim has no memcpy_default slot (CPU-only build?)",
            ));
        }
        if k.device_sync.is_none() {
            return Err(err("this shim has no device_sync slot (CPU-only build?)"));
        }
        let mut qsa = Vec::with_capacity(session.qsa.len());
        for (i, l) in session.qsa.iter().enumerate() {
            if l.residency != Residency::Resident {
                return Err(err(format!(
                    "QSA layer {i} is {:?}: this build cannot reset residency or restore a ring, \
                     so its snapshot could never be verified readable",
                    l.residency
                )));
            }
            qsa.push((l.slots, l.host));
        }
        Ok(SnapshotDevice {
            k,
            draft: session.draft.slots,
            draft_host: session.draft.host,
            qsa,
            running: session.running,
        })
    }

    pub fn can_serve(k: &StrataKernels) -> bool {
        k.memcpy_default.is_some() && k.device_sync.is_some()
    }

    /// One `cudaMemcpy(..., cudaMemcpyDefault)` between a host slice and a
    /// device buffer, at `at` into that buffer. The raw `cudaErrorString` comes
    /// back for the call site to prefix — the prefix differs between the K/V and
    /// running-state helpers, and both messages are part of the contract.
    ///
    /// # Safety
    /// `host` must stay live for `len` bytes across the synchronous call.
    unsafe fn copy(
        &self,
        region: Option<DeviceRegion>,
        at: usize,
        host: *mut c_void,
        len: usize,
        to_state: bool,
    ) -> Result<(), String> {
        let Some(f) = self.k.memcpy_default else {
            return Err(err("no memcpy_default slot"));
        };
        let Some(region) = region else {
            return Err(err("state has no buffer for this transfer"));
        };
        if at
            .checked_add(len)
            .ok_or_else(|| err("device range overflows"))?
            > region.bytes
        {
            return Err(err(format!(
                "transfer of {len} bytes at {at} runs past a {}-byte device buffer",
                region.bytes
            )));
        }
        if len == 0 {
            return Ok(());
        }
        // The range check above keeps the device side inside live memory; the
        // caller's slice lives for the whole synchronous call.
        let dst = if to_state {
            (region.addr + at) as *mut c_void
        } else {
            host
        };
        let src = if to_state {
            host as *const c_void
        } else {
            (region.addr + at) as *const c_void
        };
        let mut errbuf = [0 as c_char; 256];
        let status = f(dst, src, len as u64, errbuf.as_mut_ptr(), errbuf.len());
        if status.is_ok() {
            return Ok(());
        }
        let raw = cstr(&errbuf);
        Err(if raw.is_empty() {
            "cudaMemcpy failed".to_string()
        } else {
            raw
        })
    }

    fn layer_kv(&self, layer: Layer, set: PoolSet, region: Region) -> Option<DeviceRegion> {
        let (slots, host) = match layer {
            Layer::Draft => (self.draft, self.draft_host),
            Layer::Qsa { ordinal } => *self.qsa.get(ordinal)?,
        };
        match set {
            PoolSet::Slots => slots,
            // mode 0 has no separate host copy: the slots ARE authoritative, and
            // the contract says so by never naming the host set for a resident
            // layer — a session with a real host set hands it here.
            PoolSet::Authoritative => host,
        }
        .region(region)
    }

    fn running_target(&self, target: Running) -> Option<DeviceRegion> {
        match target {
            Running::Gdn => self.running.gdn,
            Running::Ple => self.running.ple,
            Running::Indexer { ordinal, part } => {
                let l = self.running.indexer.get(ordinal)?;
                Some(match part {
                    RunningPart::Tail => l.tail?,
                    RunningPart::Dead => l.dead?,
                    RunningPart::BlockPos => l.block_pos?,
                    RunningPart::PooledRow => l.pooled?,
                })
            }
        }
    }
}

impl LayerKv {
    fn region(&self, region: Region) -> Option<DeviceRegion> {
        self.regions[region as usize]
    }
}

impl Device for SnapshotDevice<'_> {
    fn read(
        &mut self,
        layer: Layer,
        set: PoolSet,
        region: Region,
        at: usize,
        host: &mut [u8],
    ) -> Result<(), String> {
        let Some(r) = self.layer_kv(layer, set, region) else {
            return Err(err("conversation snapshot: missing state buffer"));
        };
        // SAFETY: host is a live slice for the synchronous copy.
        unsafe {
            self.copy(
                Some(r),
                at,
                host.as_mut_ptr() as *mut c_void,
                host.len(),
                false,
            )
        }
    }

    fn write(
        &mut self,
        layer: Layer,
        set: PoolSet,
        region: Region,
        at: usize,
        host: &[u8],
    ) -> Result<(), String> {
        let Some(r) = self.layer_kv(layer, set, region) else {
            return Err(err("conversation snapshot: missing state buffer"));
        };
        // SAFETY: host is a live slice for the synchronous copy.
        unsafe { self.copy(Some(r), at, host.as_ptr() as *mut c_void, host.len(), true) }
    }

    fn read_running(&mut self, target: Running, at: usize, host: &mut [u8]) -> Result<(), String> {
        // SAFETY: host is a live slice for the synchronous copy.
        unsafe {
            self.copy(
                self.running_target(target),
                at,
                host.as_mut_ptr() as *mut c_void,
                host.len(),
                false,
            )
        }
    }

    fn write_running(&mut self, target: Running, at: usize, host: &[u8]) -> Result<(), String> {
        // SAFETY: host is a live slice for the synchronous copy.
        unsafe {
            self.copy(
                self.running_target(target),
                at,
                host.as_ptr() as *mut c_void,
                host.len(),
                true,
            )
        }
    }

    fn sync(&mut self) -> Result<(), String> {
        let Some(f) = self.k.device_sync else {
            return Err(err("no device_sync slot"));
        };
        // SAFETY: errbuf is live and correctly sized for the callee.
        let mut errbuf = [0 as c_char; 256];
        let st = unsafe { f(errbuf.as_mut_ptr(), errbuf.len()) };
        if st.is_ok() {
            return Ok(());
        }
        let raw = cstr(&errbuf);
        Err(if raw.is_empty() {
            "cudaDeviceSynchronize failed".to_string()
        } else {
            raw
        })
    }
}
