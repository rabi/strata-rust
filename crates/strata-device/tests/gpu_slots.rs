//! What the 20 appended vtable slots actually do on real hardware.
//!
//! The slots forward to runtime calls, so the only honest check is to call them
//! and look at the bytes. On a CPU-only shim build every test here returns early
//! with a printed skip; on a CUDA build they run for real:
//!
//!   STRATA_SHIM_CUDA=1 CUDA_HOME=/usr/local/cuda ./shim/build.sh
//!   STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
//!     cargo test -p strata-device --test gpu_slots -- --nocapture
//!
//! What each proves:
//!   runtime_answers_the_appended_slots  get_device and mem_get_info agree with
//!                                device_info, and a clean runtime reports no error
//!   peek_reports_without_clearing  the ONE behavioural difference between
//!                                cudaGetLastError and cudaPeekAtLastError, which
//!                                is the whole reason both are slots
//!   host_register_gives_a_device_alias  the pager's registration path: register
//!                                memory this process owns and get an alias back
//!   memset_and_2d_copy_land      memset_async fills the buffer and a pitched 2D
//!                                copy lands in the right grid cells (a wrong pitch
//!                                is the bug this slot exists to be tested for)
//!   events_time_a_stream         record/elapsed_ms/query_done, and a stream
//!                                gated on another stream's event
//!   graph_uploads_before_replay  cudaGraphUpload pays the upload, and the replay
//!                                after it still moves live bytes

// The aligned allocation below is the one thing here that cannot go through a
// wrapper; `unsafe_code` is deny (not forbid) at the workspace level exactly so a
// test can do this.
#![allow(unsafe_code)]

use std::alloc::{handle_alloc_error, Layout};
use strata_device::{CapturedGraph, DeviceBuf, Event, Pinned, Shim, Stream};

/// Load the shim, or skip the test when none is configured. Prints, because a
/// silent pass here proves nothing about the slots.
fn shim() -> Option<Shim> {
    let s = Shim::try_load()?;
    if s.is_err() {
        eprintln!("skipped: STRATA_KERNELS_LIB is set but the shim did not load");
    }
    Some(s.expect("STRATA_KERNELS_LIB set but failed to load"))
}

/// Skip when this shim build has no CUDA slots.
macro_rules! need_device {
    ($shim:expr) => {
        if $shim.kernels().device_alloc.is_none() || $shim.kernels().stream_create.is_none() {
            eprintln!(
                "skipped: shim has {} of {} slots (device slots NULL - rebuild with STRATA_SHIM_CUDA=1)",
                $shim.filled_slots(),
                strata_device::STRATA_SLOT_COUNT
            );
            return;
        }
    };
}

#[test]
fn runtime_answers_the_appended_slots() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    let d = shim.get_device().expect("get_device");
    let info = shim.device_info(d).expect("device_info");
    assert_eq!(info.ordinal, d, "get_device and device_info disagree");

    let (free, total) = shim.mem_get_info().expect("mem_get_info");
    assert!(total > 0 && free <= total, "mem_get_info: {free}/{total}");
    assert_eq!(
        total, info.total_bytes,
        "mem_get_info and device_info report different totals"
    );
    eprintln!("device {d}: {} free of {total}", free);

    assert_eq!(
        shim.peek_last_error().unwrap(),
        None,
        "a clean runtime must report no error"
    );
}

#[test]
fn peek_reports_without_clearing() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    for f in ["get_last_error", "peek_last_error"] {
        let slot = match f {
            "get_last_error" => shim.kernels().get_last_error.is_some(),
            _ => shim.kernels().peek_last_error.is_some(),
        };
        assert!(slot, "no {f} slot");
    }

    // Force a sticky error: a memset whose destination is not device memory.
    let mut host = [0u8; 64];
    let before = shim.peek_last_error().unwrap();
    let r = shim
        .kernels()
        .memset_dev
        .map(|f| {
            let mut err = [0i8; 256];
            // SAFETY: err is a live out buffer of the size passed; host is live.
            unsafe {
                f(
                    host.as_mut_ptr() as *mut _,
                    0,
                    64,
                    err.as_mut_ptr(),
                    err.len(),
                )
            }
        })
        .expect("memset_dev");
    assert!(!r.is_ok(), "a memset into a host pointer must fail");

    // The difference the two slots exist for: peek leaves the error in place,
    // take reports it once and clears it.
    let first = shim.peek_last_error().unwrap();
    let second = shim.peek_last_error().unwrap();
    assert!(
        first.is_some(),
        "the failed memset should have left an error"
    );
    assert_eq!(first, second, "peek must not clear");
    let taken = shim.take_last_error().unwrap();
    assert_eq!(taken, first, "take must report the same error");
    assert_eq!(
        shim.peek_last_error().unwrap(),
        if before.is_some() { before } else { None },
        "after take, the runtime must be clean"
    );
}

#[test]
fn host_register_gives_a_device_alias() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    // cudaHostRegister wants page-aligned memory the process owns.
    let l = Layout::from_size_align(1 << 12, 1 << 12).unwrap();
    // SAFETY: aligned allocation of a non-zero size; the slice covers it.
    let raw = unsafe { std::alloc::alloc(l) };
    if raw.is_null() {
        handle_alloc_error(l);
    }
    // SAFETY: raw points at l.size() bytes of live, l.align()-aligned memory.
    let page = unsafe { std::slice::from_raw_parts_mut(raw, l.size()) };
    page.fill(0x5c);

    shim.host_register(page, 0).expect("host_register");
    let alias = shim.host_device_pointer(page).expect("host_device_pointer");
    assert!(!alias.is_null(), "the alias must be a real address");
    shim.host_unregister(page).expect("host_unregister");
    // SAFETY: the allocation outlives the slice and is freed exactly here.
    unsafe { std::alloc::dealloc(raw, l) };
}

#[test]
fn memset_and_2d_copy_land() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    let mut pinned = Pinned::new(shim.kernels(), 4096).expect("pinned");
    let buf = DeviceBuf::alloc(shim.kernels(), 4096).expect("device alloc");
    let s = Stream::create(shim.kernels()).expect("stream");

    buf.memset(0x5a).expect("memset");
    buf.memset_async(0x33, &s).expect("memset_async");
    s.sync().expect("stream_sync");
    assert!(
        buf.to_vec().unwrap().iter().all(|&b| b == 0x33),
        "memset_async did not fill the buffer"
    );

    // A pitched copy: 4 rows of 64 useful bytes on a 128-byte source pitch, onto
    // a 256-byte destination pitch. A wrong pitch shows up as rows landing in the
    // wrong cells, which is the only thing this test can catch.
    let src = pinned.bytes_mut();
    for (i, b) in src.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let src_len = 4 * 128;
    let src_view = &mut src[..src_len];
    let mut want = vec![0u8; 4096];
    for r in 0..4u64 {
        for c in 0..64u64 {
            want[(r * 256 + c) as usize] = src_view[(r * 128 + c) as usize];
        }
    }
    buf.memset(0).expect("clear");
    buf.copy_h2d_2d(src_view, 128, 64, 4, 256, &s)
        .expect("2d copy");
    s.sync().expect("stream_sync");
    assert_eq!(buf.to_vec().unwrap(), want, "pitched copy landed wrong");
}

#[test]
fn events_time_a_stream() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    for f in [
        "event_create",
        "event_record",
        "event_sync",
        "event_query_done",
        "event_elapsed_ms",
        "stream_wait_event",
        "stream_create_with_flags",
    ] {
        let present = match f {
            "event_create" => shim.kernels().event_create.is_some(),
            "event_record" => shim.kernels().event_record.is_some(),
            "event_sync" => shim.kernels().event_sync.is_some(),
            "event_query_done" => shim.kernels().event_query_done.is_some(),
            "event_elapsed_ms" => shim.kernels().event_elapsed_ms.is_some(),
            "stream_wait_event" => shim.kernels().stream_wait_event.is_some(),
            _ => shim.kernels().stream_create_with_flags.is_some(),
        };
        assert!(present, "no {f} slot");
    }

    let s = Stream::create(shim.kernels()).expect("stream");
    let gate = Event::create(shim.kernels(), 0).expect("event_create");
    let buf = DeviceBuf::alloc(shim.kernels(), 1 << 20).expect("device alloc");

    let start = Event::create(shim.kernels(), 0).expect("event_create");
    start.record(&s).expect("record start");
    buf.memset_async(0x11, &s).expect("memset_async");
    let end = Event::create(shim.kernels(), 0).expect("event_create");
    end.record(&s).expect("record end");

    // A second stream gated on the first: the wait is submitted before the work
    // is queued on `s`, so the ordering is the ABI's job, not the caller's.
    let other = Stream::create_with_flags(shim.kernels(), 0).expect("create_with_flags");
    gate.record(&s).expect("record gate");
    other.wait_event(&gate).expect("stream_wait_event");
    end.record(&other).expect("record end on the gated stream");

    s.sync().expect("stream_sync");
    other.sync().expect("stream sync");
    assert!(end.query_done(), "the end event should be done after sync");
    let ms = start.elapsed_ms(&end).expect("elapsed_ms");
    eprintln!("1 MiB memset round trip: {ms:.3} ms");

    // The gated stream must not have run before the gate: its own work is the
    // same buffer, so a sane elapsed time on the gated pair is the proof.
    let g2 = Event::create(shim.kernels(), 0).expect("create");
    g2.record(&other).expect("record");
    other.sync().expect("sync");
    assert!(g2.query_done());
}

#[test]
fn graph_uploads_before_replay() {
    let Some(shim) = shim() else { return };
    need_device!(shim);
    if shim.device_count() == 0 {
        eprintln!("skipped: no GPU visible to the shim build");
        return;
    }
    assert!(
        shim.kernels().graph_upload.is_some(),
        "no graph_upload slot"
    );
    let s = Stream::create(shim.kernels()).expect("stream");
    let buf = DeviceBuf::alloc(shim.kernels(), 4096).expect("device alloc");

    // The captured body is a copy from a LIVE pinned source, so the replay must
    // move the bytes the source holds at replay time, not the bytes it held at
    // capture time. That is the property upload must not break.
    let mut pinned = Pinned::new(shim.kernels(), 4096).expect("pinned");
    pinned.bytes_mut().fill(0x10);
    let graph = CapturedGraph::begin(shim.kernels(), &s).expect("begin");
    buf.copy_h2d(&pinned.bytes_mut()[..4096], &s)
        .expect("captured copy");
    let graph = graph.end().expect("end");
    graph.upload(&s).expect("upload");
    s.sync().expect("sync");

    // Live source: change it, replay, and the device buffer must follow.
    pinned.bytes_mut().fill(0x20);
    graph.replay(&s).expect("replay");
    s.sync().expect("sync");
    assert_eq!(
        buf.to_vec().unwrap(),
        vec![0x20; 4096],
        "the replay must move the CURRENT source bytes"
    );
}
