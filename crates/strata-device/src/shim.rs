//! Load a C++ shim `libstrata_kernels.so` and fill a [`StrataKernels`].
//!
//! The loader is the only place in the workspace that talks to the dynamic
//! linker. `dlopen`/`dlsym` are declared here rather than pulled from a crate;
//! the shim is found via `STRATA_KERNELS_LIB` (a path) or the usual search
//! path under `strata_kernels`.
//!
//! This module holds the workspace's non-test `unsafe`: every raw pointer,
//! dlopen handle and vtable call stops here, and the types it hands out
//! (`Shim`, `DeviceFile`, `Pinned`) are safe to use.
#![allow(unsafe_code)]

use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use crate::abi::{StrataKernels, STRATA_ABI_VERSION};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

/// `strata_kernels_load` as exported by the shim.
// The C function returns StrataStatus (enum : int); modelling it as c_int keeps
// the FFI signature to integers only, which is the guaranteed-representable form.
type LoadFn = unsafe extern "C" fn(u32, *mut StrataKernels, *mut c_char, usize) -> c_int;

/// A loaded shim. The `StrataKernels` it filled stays valid only while this
/// guard is alive (dropping it unmaps the library), so kernels() borrows it.
pub struct Shim {
    handle: *mut c_void,
    kernels: StrataKernels,
}

impl Shim {
    /// `None` when `STRATA_KERNELS_LIB` is unset and the default name cannot be
    /// found — the same absence `StrataKernels::none()` models, so callers use
    /// `Shim::try_load(..)?.unwrap_or_none()` in one line.
    pub fn try_load() -> Option<Result<Shim, String>> {
        let name =
            env::var("STRATA_KERNELS_LIB").unwrap_or_else(|_| "libstrata_kernels.so".to_string());
        let cpath = match CString::new(name.clone()) {
            Ok(p) => p,
            Err(_) => return Some(Err(format!("lib name {name} contains a NUL"))),
        };
        // SAFETY: dlopen takes a valid NUL-terminated path; RTLD_NOW resolves
        // every symbol now, so a half-loaded shim fails here, not later.
        let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            let why = last_dlerror();
            if env::var("STRATA_KERNELS_LIB").is_err() && why.contains("No such file") {
                return None; // default name absent: no shim installed, not an error
            }
            return Some(Err(format!("dlopen {name}: {why}")));
        }
        // SAFETY: dlsym on a live handle; the symbol type is checked by the
        // signature we cast to, and the shim's own ABI-version argument settles
        // the struct layout question at call time.
        let sym = unsafe { dlsym(handle, c"strata_kernels_load".as_ptr()) };
        if sym.is_null() {
            let why = last_dlerror();
            unsafe { crate::shim::dlclose_maybe(handle) };
            return Some(Err(format!("dlsym strata_kernels_load: {why}")));
        }
        let load: LoadFn = unsafe { std::mem::transmute::<*mut c_void, LoadFn>(sym) };
        let mut kernels = StrataKernels::none();
        let mut err = [0 as c_char; 256];
        // SAFETY: `load` is the shim's exported entry point; `kernels` and
        // `err` are live, correctly sized, and the callee fills only through them.
        let status = unsafe {
            load(
                STRATA_ABI_VERSION,
                &mut kernels as *mut StrataKernels,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if status != 0 {
            let why = cstr(&err);
            unsafe { crate::shim::dlclose_maybe(handle) };
            return Some(Err(format!("strata_kernels_load: {why} (status {status})")));
        }
        Some(Ok(Shim { handle, kernels }))
    }

    pub fn kernels(&self) -> &StrataKernels {
        &self.kernels
    }

    /// GPUs the shim reports (0 when it has no device slots at all).
    pub fn device_count(&self) -> i32 {
        match self.kernels.device_count {
            Some(f) => unsafe { f() },
            None => 0,
        }
    }

    /// Fill an ABI `DeviceInfo` for one ordinal.
    pub fn device_info(&self, ordinal: i32) -> Option<crate::abi::DeviceInfo> {
        let f = self.kernels.device_info?;
        let mut out = crate::abi::DeviceInfo::default();
        // SAFETY: `out` is a live, correctly sized ABI struct the callee fills.
        if unsafe { f(ordinal as std::os::raw::c_int, &mut out) }.is_ok() {
            Some(out)
        } else {
            None
        }
    }

    /// The file the shim would report for O_DIRECT (`file_alignment` slot).
    pub fn file_alignment(&self) -> u32 {
        match self.kernels.file_alignment {
            Some(f) => unsafe { f() },
            None => 0,
        }
    }

    /// "" when the device can run this binary's PTX/JIT, else the reason.
    /// `None` = slot missing, `Some("")` = no problem.
    pub fn gpu_arch_problem(&self, ordinal: i32) -> Option<String> {
        let f = self.kernels.gpu_arch_problem?;
        let mut buf = [0 as c_char; 128];
        // SAFETY: buf is a live 128-byte out buffer; the callee writes at most
        // out_len bytes including the NUL.
        let st = unsafe { f(ordinal as c_int, buf.as_mut_ptr(), buf.len()) };
        if st.is_ok() {
            Some(cstr(&buf))
        } else {
            let why = cstr(&buf);
            Some(if why.is_empty() {
                "gpu_arch_problem failed silently".to_string()
            } else {
                why
            })
        }
    }

    /// How many of the [`STRATA_SLOT_COUNT`] vtable slots this shim build filled.
    pub fn filled_slots(&self) -> usize {
        self.kernels.filled_slots()
    }
}

impl Drop for Shim {
    fn drop(&mut self) {
        // SAFETY: handle came from dlopen in try_load and was not closed before.
        unsafe { crate::shim::dlclose_maybe(self.handle) };
    }
}

// dlclose is only declared so Drop can call it; leaking a shim on exit is
// harmless, so it lives behind one helper and one allow.
unsafe fn dlclose_maybe(handle: *mut c_void) {
    unsafe extern "C" {
        fn dlclose(handle: *mut c_void) -> c_int;
    }
    // SAFETY: called only with a handle from dlopen that is not otherwise in use.
    unsafe { dlclose(handle) };
}

fn last_dlerror() -> String {
    // SAFETY: dlerror returns a thread-local C string or null.
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown".to_string()
    } else {
        // SAFETY: just-returned dlerror string is valid until the next dl* call.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn cstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().map(|&b| b as u8).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// A file opened through the shim's direct-IO slots. Reads land in [`Pinned`]
/// buffers aligned to `file_alignment()`, the pager's contract.
pub struct DeviceFile<'a> {
    k: &'a StrataKernels,
    file: *mut c_void,
}

impl<'a> DeviceFile<'a> {
    pub fn open(
        k: &'a StrataKernels,
        path: impl AsRef<std::path::Path>,
    ) -> Result<DeviceFile<'a>, String> {
        let path = path
            .as_ref()
            .to_str()
            .ok_or_else(|| "path is not utf-8".to_string())?;
        let Some(open) = k.file_open else {
            return Err("no file_open slot in this shim".into());
        };
        let cpath = CString::new(path).map_err(|_| "path contains a NUL")?;
        let mut file = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: cpath is a valid C string; file/err are live and sized; the
        // callee writes the handle only on success.
        let st = unsafe { open(cpath.as_ptr(), &mut file, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(DeviceFile { k, file })
    }

    pub fn size(&self) -> u64 {
        match self.k.file_size {
            Some(f) => unsafe { f(self.file) },
            None => 0,
        }
    }

    pub fn alignment(&self) -> u32 {
        match self.k.file_alignment {
            Some(f) => unsafe { f() },
            None => 0,
        }
    }

    /// Queue one aligned read into `buf` (a `Pinned` slice); `tag` comes back on
    /// the completion. Submit never blocks.
    pub fn submit(&self, offset: u64, buf: &mut [u8], tag: u64) -> Result<(), String> {
        let Some(submit) = self.k.file_submit else {
            return Err("no file_submit slot".into());
        };
        let align = self.alignment().max(1) as usize;
        if buf.is_empty()
            || !buf.len().is_multiple_of(align)
            || !offset.is_multiple_of(align as u64)
        {
            return Err("read is not aligned/ sized to the file's alignment".into());
        }
        if !(buf.as_ptr() as usize).is_multiple_of(align) {
            return Err("buffer address not aligned to the file's alignment".into());
        }
        let mut err = [0 as c_char; 256];
        // SAFETY: the shim only touches buf[..length] and err[..err_len]; buf
        // stays borrowed for the call, which is all the contract needs (the
        // completion may land later, and the caller keeps buf alive until then).
        let st = unsafe {
            submit(
                self.file,
                offset,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as u32,
                tag,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    /// Wait for completions (up to `out.len()`), `timeout_ms` < 0 waits forever.
    /// Returns how many completed.
    pub fn wait(&self, out: &mut [crate::abi::IoCompletion], timeout_ms: i32) -> usize {
        let Some(wait) = self.k.file_wait else {
            return 0;
        };
        if out.is_empty() {
            return 0;
        }
        // SAFETY: out is a live array of the ABI completion struct; max matches
        // its length.
        unsafe { wait(self.file, out.as_mut_ptr(), out.len() as c_int, timeout_ms) as usize }
    }
}

impl Drop for DeviceFile<'_> {
    fn drop(&mut self) {
        if let Some(close) = self.k.file_close {
            // SAFETY: the handle came from file_open and is closed exactly once,
            // here, after every submitted read has been waited on by the caller
            // (DirectFile::close drains the queue either way).
            unsafe { close(self.file) };
        }
    }
}

/// Page-aligned host memory from the shim's pinned slot. On a CUDA shim this is
/// cudaHostAlloc; on a CPU-only shim, aligned malloc.
pub struct Pinned<'a> {
    k: &'a StrataKernels,
    p: *mut c_void,
    bytes: usize,
}

impl<'a> Pinned<'a> {
    pub fn new(k: &'a StrataKernels, bytes: usize) -> Result<Pinned<'a>, String> {
        let Some(alloc) = k.pinned_alloc else {
            return Err("no pinned_alloc slot in this shim".into());
        };
        if bytes == 0 {
            return Err("zero-sized pinned".into());
        }
        let mut p = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: p and err are live and correctly sized for the callee.
        let st = unsafe { alloc(bytes as u64, &mut p, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(Pinned { k, p, bytes })
    }

    /// Every read submitted to a `DeviceFile` lands here.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the shim returned `p` for exactly `bytes` of writable memory.
        unsafe { std::slice::from_raw_parts_mut(self.p as *mut u8, self.bytes) }
    }

    pub fn align_ok(&self, alignment: u32) -> bool {
        alignment == 0 || (self.p as usize).is_multiple_of(alignment as usize)
    }
}

impl Drop for Pinned<'_> {
    fn drop(&mut self) {
        if let Some(free) = self.k.pinned_free {
            // SAFETY: p came from this shim's pinned_alloc and is freed once.
            unsafe { free(self.p, self.bytes as u64) };
        }
    }
}

/// A CUDA stream from the shim. Dropping it destroys the stream, so any graph
/// captured on it must be ended first (graph_end does not need the stream to
/// outlive the capture window, only to exist during it).
pub struct Stream<'a> {
    k: &'a StrataKernels,
    s: crate::abi::Stream,
}

impl<'a> Stream<'a> {
    pub fn create(k: &'a StrataKernels) -> Result<Stream<'a>, String> {
        let Some(create) = k.stream_create else {
            return Err("no stream_create slot in this shim (CPU-only build?)".into());
        };
        let mut s = std::ptr::null_mut();
        // SAFETY: s is a live out-pointer the callee writes on success only.
        let st = unsafe { create(&mut s) };
        if !st.is_ok() {
            return Err("stream_create failed".into());
        }
        Ok(Stream { k, s })
    }

    /// The raw handle, for ABI calls that take a stream.
    pub fn raw(&self) -> crate::abi::Stream {
        self.s
    }

    /// Non-blocking poll: 1 = nothing queued or everything finished.
    pub fn query_done(&self) -> bool {
        match self.k.stream_query_done {
            Some(f) => (unsafe { f(self.s) }) != 0,
            None => false,
        }
    }

    /// Poll until the stream drains or `budget_ms` runs out; the ABI's own
    /// completion rule (poll, never a blocking sync call) applied from Rust.
    pub fn drain(&self, budget_ms: u128) -> bool {
        let start = std::time::Instant::now();
        loop {
            if self.query_done() {
                return true;
            }
            if start.elapsed().as_millis() >= budget_ms {
                return false;
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.k.stream_destroy {
            // SAFETY: s came from this shim's stream_create, destroyed once, and
            // nothing else in this process still references it.
            unsafe { f(self.s) };
        }
    }
}

/// Device memory from the shim's device_alloc. `dev_ptr()` is a DEVICE pointer:
/// hand it only to ABI slots whose parameter is named `*_dev`.
pub struct DeviceBuf<'a> {
    k: &'a StrataKernels,
    p: *mut c_void,
    bytes: usize,
}

impl<'a> DeviceBuf<'a> {
    pub fn alloc(k: &'a StrataKernels, bytes: usize) -> Result<DeviceBuf<'a>, String> {
        let Some(alloc) = k.device_alloc else {
            return Err("no device_alloc slot in this shim (CPU-only build?)".into());
        };
        if bytes == 0 {
            return Err("zero-sized device alloc".into());
        }
        let mut p = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: p and err are live and correctly sized for the callee.
        let st = unsafe { alloc(bytes as u64, &mut p, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(DeviceBuf { k, p, bytes })
    }

    pub fn dev_ptr(&self) -> *mut c_void {
        self.p
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Async host->device into this buffer from pinned host memory (only
    /// pinned sources actually run async). Stream-ordered: `stream.drain()`
    /// settles it.
    pub fn copy_h2d(&self, src: &[u8], stream: &Stream) -> Result<(), String> {
        let Some(f) = self.k.memcpy_h2d_async else {
            return Err("no memcpy_h2d_async slot".into());
        };
        if src.len() > self.bytes {
            return Err(format!(
                "h2d {} bytes into a {}-byte buffer",
                src.len(),
                self.bytes
            ));
        }
        // SAFETY: dst has `bytes` of live device memory, src.len() <= bytes;
        // src stays alive across the async copy because the caller drains the
        // stream before this borrow ends.
        let st = unsafe {
            f(
                self.p,
                src.as_ptr() as *const c_void,
                src.len() as u64,
                stream.raw(),
            )
        };
        if !st.is_ok() {
            return Err("memcpy_h2d_async failed".into());
        }
        Ok(())
    }

    /// Async device->host into `dst` (pinned for real async; any slice works,
    /// the drain keeps it alive). The slice must be page-aligned if you care
    /// that the copy is not silently staged through an extra bounce.
    pub fn copy_d2h(&self, dst: &mut [u8], stream: &Stream) -> Result<(), String> {
        let Some(f) = self.k.memcpy_d2h_async else {
            return Err("no memcpy_d2h_async slot".into());
        };
        if dst.len() > self.bytes {
            return Err(format!(
                "d2h {} bytes from a {}-byte buffer",
                dst.len(),
                self.bytes
            ));
        }
        // SAFETY: src is `bytes` of live device memory, dst.len() <= bytes,
        // and dst is borrowed until after the caller's stream drain.
        let st = unsafe {
            f(
                dst.as_mut_ptr() as *mut c_void,
                self.p,
                dst.len() as u64,
                stream.raw(),
            )
        };
        if !st.is_ok() {
            return Err("memcpy_d2h_async failed".into());
        }
        Ok(())
    }

    /// Synchronous read-back into an owned Vec (drain included).
    pub fn to_vec(&self) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; self.bytes];
        // The copy needs a stream; borrow one only for the call. Every shim
        // slot accepts the default stream handle, and CUDA null means "default
        // stream"; but the shim created streams, so drain an explicit one here.
        {
            let s = Stream::create(self.k)?;
            self.copy_d2h(&mut out, &s)?;
            if !s.drain(10_000) {
                return Err("d2h never completed".into());
            }
        }
        Ok(out)
    }
}

impl Drop for DeviceBuf<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.k.device_free {
            // SAFETY: p came from this shim's device_alloc, freed once. The
            // free is stream-ordered by the driver, so queued reads settle
            // before the memory can be reused.
            unsafe { f(self.p, self.bytes as u64) };
        }
    }
}

/// A captured CUDA graph: record -> (submit work via other slots) -> end ->
/// replay. Same rule as C++: every buffer the captured calls touch must exist
/// (this shim's device_alloc) and not move until the graph is destroyed.
pub struct CapturedGraph<'a> {
    k: &'a StrataKernels,
    g: crate::abi::CapturedGraphHandle,
}

impl<'a> CapturedGraph<'a> {
    /// Start capturing on `stream`. The caller must issue the body's ABI calls
    /// on the same stream, then call `end`.
    pub fn begin(k: &'a StrataKernels, stream: &Stream) -> Result<CapturedGraph<'a>, String> {
        let Some(record) = k.graph_record else {
            return Err("no graph_record slot in this shim (CPU-only build?)".into());
        };
        let mut g = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: g/err live and sized; stream is a live handle from Stream::create.
        let st = unsafe { record(&mut g, stream.raw(), err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(CapturedGraph { k, g })
    }

    /// Close the capture window and instantiate. Errors leave the graph
    /// destroyed by nobody — dropping this guard still frees it.
    pub fn end(self) -> Result<ReplayableGraph<'a>, String> {
        let Some(end) = self.k.graph_end else {
            return Err("no graph_end slot".into());
        };
        let mut err = [0 as c_char; 256];
        let (k, g) = (self.k, self.g);
        // SAFETY: g is the handle begin() returned and no one else touched it.
        let st = unsafe { end(g, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        std::mem::forget(self); // ownership moves into the replayable wrapper
        Ok(ReplayableGraph { k, g })
    }
}

impl Drop for CapturedGraph<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.k.graph_destroy {
            // SAFETY: g from graph_record, destroyed once; a graph that ended
            // successfully is dropped via ReplayableGraph instead (forget).
            unsafe { f(self.g) };
        }
    }
}

/// A graph that finished capture+instantiate and can be replayed.
pub struct ReplayableGraph<'a> {
    k: &'a StrataKernels,
    g: crate::abi::CapturedGraphHandle,
}

impl<'a> ReplayableGraph<'a> {
    pub fn replay(&self, stream: &Stream) -> Result<(), String> {
        let Some(replay) = self.k.graph_replay else {
            return Err("no graph_replay slot".into());
        };
        let mut err = [0 as c_char; 256];
        // SAFETY: g came from a successful capture+end; stream is live.
        let st = unsafe { replay(self.g, stream.raw(), err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }
}

impl Drop for ReplayableGraph<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.k.graph_destroy {
            // SAFETY: exactly one graph_destroy for this handle, after every
            // replay was submitted by the caller (replay is queued; the
            // driver keeps exec alive for in-flight launches).
            unsafe { f(self.g) };
        }
    }
}

// ---- appended slots: the pinned arena, the pager, layer, mtp, session --------
//
// Thin safe wrappers, one per vtable slot. A missing slot is an error rather than
// a panic, so a shim build without CUDA degrades the way the C++ build without
// CUDA does: the caller asks, gets told, and falls back.

impl Shim {
    pub fn get_device(&self) -> Result<i32, String> {
        let f = self.kernels.get_device.ok_or("no get_device slot")?;
        let mut out = 0 as c_int;
        let mut err = [0 as c_char; 256];
        // SAFETY: out and err are live and correctly sized for the callee.
        let st = unsafe { f(&mut out, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(out)
    }

    /// `(free, total)` as the runtime reports them.
    pub fn mem_get_info(&self) -> Result<(u64, u64), String> {
        let f = self.kernels.mem_get_info.ok_or("no mem_get_info slot")?;
        let mut free = 0u64;
        let mut total = 0u64;
        let mut err = [0 as c_char; 256];
        // SAFETY: both out-pointers and err are live.
        let st = unsafe { f(&mut free, &mut total, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok((free, total))
    }

    /// cudaGetLastError: reports the sticky error AND CLEARS it. `None` when this
    /// shim build has no such slot, so the caller cannot tell a clean runtime from
    /// a shim that cannot ask.
    pub fn take_last_error(&self) -> Option<Option<String>> {
        let f = self.kernels.get_last_error?;
        let mut err = [0 as c_char; 256];
        // SAFETY: err is a live out buffer of the size passed.
        let st = unsafe { f(err.as_mut_ptr(), err.len()) };
        Some(if st.is_ok() { None } else { Some(cstr(&err)) })
    }

    /// cudaPeekAtLastError: reports WITHOUT clearing. The two are not
    /// interchangeable and the engine uses both, which is why both are slots.
    pub fn peek_last_error(&self) -> Option<Option<String>> {
        let f = self.kernels.peek_last_error?;
        let mut err = [0 as c_char; 256];
        // SAFETY: err is a live out buffer of the size passed.
        let st = unsafe { f(err.as_mut_ptr(), err.len()) };
        Some(if st.is_ok() { None } else { Some(cstr(&err)) })
    }

    /// cudaHostRegister over memory this process already owns (the arena the C++
    /// mmaps under STRATA_ARENA_MMAP). `flags` is the runtime's flag word.
    pub fn host_register(&self, host: &mut [u8], flags: u32) -> Result<(), String> {
        let f = self.kernels.host_register.ok_or("no host_register slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: host is live for the call and `host.len()` is its real length.
        let st = unsafe {
            f(
                host.as_mut_ptr() as *mut c_void,
                host.len() as u64,
                flags,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    pub fn host_unregister(&self, host: &mut [u8]) -> Result<(), String> {
        let f = self
            .kernels
            .host_unregister
            .ok_or("no host_unregister slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: this slice covers exactly the registration being undone.
        let st = unsafe {
            f(
                host.as_mut_ptr() as *mut c_void,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    /// cudaHostGetDevicePointer: the device address that aliases a registration.
    /// The pointer is only valid while the registration is alive, so the caller
    /// keeps the slice, not just the pointer.
    pub fn host_device_pointer(&self, host: &mut [u8]) -> Result<*mut c_void, String> {
        let f = self
            .kernels
            .host_get_device_pointer
            .ok_or("no host_get_device_pointer slot")?;
        let mut out = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: out and err are live; host stays alive for the call.
        let st = unsafe {
            f(
                host.as_mut_ptr() as *mut c_void,
                &mut out,
                0,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(out)
    }
}

impl DeviceBuf<'_> {
    /// cudaMemset — synchronous, value is the byte written.
    pub fn memset(&self, value: u8) -> Result<(), String> {
        let f = self.k.memset_dev.ok_or("no memset_dev slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: p is live device memory of self.bytes.
        let st = unsafe {
            f(
                self.p,
                c_int::from(value),
                self.bytes as u64,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    pub fn memset_async(&self, value: u8, stream: &Stream) -> Result<(), String> {
        let f = self.k.memset_async.ok_or("no memset_async slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: p is live device memory of self.bytes; the stream is live.
        let st = unsafe {
            f(
                self.p,
                c_int::from(value),
                self.bytes as u64,
                stream.raw(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    /// cudaMemcpy2DAsync, host->device: `src` covers `height` rows of `spitch`
    /// bytes with `width` useful in each, landing on `self` at its own pitch.
    pub fn copy_h2d_2d(
        &self,
        src: &[u8],
        spitch: u64,
        width: u64,
        height: u64,
        dpitch: u64,
        stream: &Stream,
    ) -> Result<(), String> {
        let f = self.k.memcpy2d_async.ok_or("no memcpy2d_async slot")?;
        if width * height > spitch * height || width * height > dpitch * height {
            return Err(format!(
                "2d copy {width}x{height} does not fit pitches {spitch}/{dpitch}"
            ));
        }
        let need = spitch * height;
        if need > src.len() as u64 || dpitch * height > self.bytes as u64 {
            return Err(format!(
                "2d copy needs {need} source bytes in {} and {} destination bytes in {}",
                src.len(),
                dpitch * height,
                self.bytes
            ));
        }
        let mut err = [0 as c_char; 256];
        // SAFETY: both regions cover the copy (checked above); src stays alive
        // because the caller drains the stream before this borrow ends.
        let st = unsafe {
            f(
                self.p,
                dpitch,
                src.as_ptr() as *const c_void,
                spitch,
                width,
                height,
                1, // cudaMemcpyHostToDevice
                stream.raw(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }
}

impl Stream<'_> {
    /// cudaStreamCreateWithFlags. `create(..)` is the flags-0 case.
    pub fn create_with_flags(k: &StrataKernels, flags: u32) -> Result<Stream<'_>, String> {
        let f = k
            .stream_create_with_flags
            .ok_or("no stream_create_with_flags slot")?;
        let mut s = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: s is a live out-pointer written on success only.
        let st = unsafe { f(&mut s, flags, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(Stream { k, s })
    }

    /// cudaStreamSynchronize — blocks this thread until the stream drains.
    pub fn sync(&self) -> Result<(), String> {
        let f = self.k.stream_sync.ok_or("no stream_sync slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: this stream is live (Drop has not run).
        let st = unsafe { f(self.s, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), String> {
        let f = self
            .k
            .stream_wait_event
            .ok_or("no stream_wait_event slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: both handles are live and from this shim.
        let st = unsafe { f(self.s, event.e, 0, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }
}

/// A CUDA event. Dropping it destroys it, so nothing may still be recorded on it.
pub struct Event<'a> {
    k: &'a StrataKernels,
    e: crate::abi::Event,
}

impl<'a> Event<'a> {
    /// cudaEventCreateWithFlags; flags 0 is cudaEventCreate.
    pub fn create(k: &'a StrataKernels, flags: u32) -> Result<Event<'a>, String> {
        let f = k.event_create.ok_or("no event_create slot")?;
        let mut e = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: e is a live out-pointer written on success only.
        let st = unsafe { f(&mut e, flags, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(Event { k, e })
    }

    pub fn record(&self, stream: &Stream) -> Result<(), String> {
        let f = self.k.event_record.ok_or("no event_record slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: both handles are live and from this shim.
        let st = unsafe { f(self.e, stream.raw(), err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        let f = self.k.event_sync.ok_or("no event_sync slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: this event is live.
        let st = unsafe { f(self.e, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }

    /// Non-blocking poll, same 1 = done convention as `Stream::query_done`.
    pub fn query_done(&self) -> bool {
        match self.k.event_query_done {
            Some(f) => (unsafe { f(self.e) }) != 0,
            None => false,
        }
    }

    /// cudaEventElapsedTime between two recorded events, in milliseconds.
    pub fn elapsed_ms(&self, other: &Event) -> Result<f32, String> {
        let f = self.k.event_elapsed_ms.ok_or("no event_elapsed_ms slot")?;
        let mut ms = 0f32;
        let mut err = [0 as c_char; 256];
        // SAFETY: out is live and both events are live handles from this shim.
        let st = unsafe { f(&mut ms, self.e, other.e, err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(ms)
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.k.event_destroy {
            // SAFETY: exactly one destroy for this handle, after the caller has
            // stopped recording on it.
            unsafe { f(self.e) };
        }
    }
}

impl ReplayableGraph<'_> {
    /// cudaGraphUpload on the instantiated exec: pay the upload cost before the
    /// first replay instead of inside it.
    pub fn upload(&self, stream: &Stream) -> Result<(), String> {
        let f = self.k.graph_upload.ok_or("no graph_upload slot")?;
        let mut err = [0 as c_char; 256];
        // SAFETY: g came from a successful capture+end; stream is live.
        let st = unsafe { f(self.g, stream.raw(), err.as_mut_ptr(), err.len()) };
        if !st.is_ok() {
            return Err(cstr(&err));
        }
        Ok(())
    }
}
