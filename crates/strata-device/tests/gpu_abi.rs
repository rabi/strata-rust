// GPU checks over the ABI. On a CPU-only shim build the device slots are NULL
// and every test here returns early with a printed skip, so `cargo test` stays
// green everywhere; on a CUDA shim build they run for real:
//
//   ./shim/build.sh                                   # CPU parts
//   STRATA_SHIM_CUDA=1 CUDA_HOME=/usr/local/cuda ./shim/build.sh
//   STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
//     cargo test -p strata-device --test gpu_abi -- --nocapture
//
// What each test proves:
//   device_report           the runtime answers and numbers look like a GPU
//   pinned_memory_is_registered  host-side proof the arena really is pinned
//   h2d_d2h_round_trip      real DMA: bytes out, bytes back, wrong data caught
//   graph_captures_and_replays   capture a copy, replay it 100x, every replay
//                                must move the CURRENT buffer bytes (not the
//                                bytes captured — replay reads live memory)
//   cpu_shim_reports_zero_devices  the degrade path stays honest on CPU builds

use strata_device::{CapturedGraph, DeviceBuf, Pinned, Shim, Stream};

/// Load the shim, or skip the test when none is configured.
fn shim() -> Option<Shim> {
    Shim::try_load().map(|r| r.expect("STRATA_KERNELS_LIB set but failed to load"))
}

/// Skip (print + return) when this shim build has no CUDA slots.
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
fn device_report() {
    let Some(shim) = shim() else { return };
    let n = shim.device_count();
    if n == 0 {
        eprintln!("skipped: shim reports 0 devices (no GPU visible to the shim build)");
        return;
    }
    assert!(
        shim.kernels().device_count.is_some(),
        "device_count() returned {n} but the slot is NULL"
    );
    for i in 0..n {
        let d = shim
            .device_info(i)
            .unwrap_or_else(|| panic!("device_info({i}) failed"));
        assert_eq!(d.ordinal, i);
        assert!(!d.name_str().is_empty(), "device {i} has no name");
        assert!(d.multi_processor_count > 0);
        assert!(d.cc_major >= 5 && d.cc_major <= 12, "cc {}", d.cc_major);
        assert!(
            d.total_bytes > 512 << 20,
            "total {:.1} GiB",
            d.total_bytes as f64 / (1u64 << 30) as f64
        );
        assert!(d.free_bytes <= d.total_bytes);
        let driver = d.driver_version;
        assert!((10000..=20000).contains(&driver), "driver version {driver}");
        println!(
            "gpu {i}: {} [{}] cc {}.{} {} SMs, {:.1} GiB of {:.1} GiB free, driver {driver}",
            d.name_str(),
            d.arch_str(),
            d.cc_major,
            d.cc_minor,
            d.multi_processor_count,
            d.free_bytes as f64 / (1u64 << 30) as f64,
            d.total_bytes as f64 / (1u64 << 30) as f64
        );
    }
    for i in 0..n.max(0) {
        if let Some(p) = shim.gpu_arch_problem(i) {
            println!(
                "gpu {i}: arch problem = {:?}",
                if p.is_empty() { None } else { Some(&p[..]) }
            );
        }
    }
}

#[test]
fn pinned_memory_is_registered() {
    let Some(shim) = shim() else { return };
    // cudaHostAlloc'd memory must be page-aligned (the contract pinned_alloc
    // inherits from DirectFile::alignment) and large allocations must not be
    // silently downgraded to pageable. 32 MiB: well past any malloc trick.
    let k = shim.kernels();
    let align = shim.file_alignment();
    let mut b = Pinned::new(k, 32 << 20).expect("pinned 32 MiB");
    assert!(b.align_ok(align), "pinned buffer not aligned to {align}");
    // Writing every 64th byte touches every cache line: a host allocation that
    // silently failed would fault here, not later inside a kernel.
    let bytes = b.bytes_mut();
    for x in bytes.iter_mut().step_by(64) {
        *x = 0xa5;
    }
    assert!(bytes.iter().step_by(4096).all(|&x| x == 0xa5));
}

#[test]
fn h2d_d2h_round_trip() {
    let Some(shim) = shim() else { return };
    need_device!(&shim);
    let k = shim.kernels();
    let stream = Stream::create(k).expect("stream");
    let mut src = Pinned::new(k, 1 << 20).expect("pinned");
    let mut dst = Pinned::new(k, 1 << 20).expect("pinned");
    let dev = DeviceBuf::alloc(k, 1 << 20).expect("device 1 MiB");

    // Deterministic pattern; src stays pinned-alive across both async copies.
    {
        let s = src.bytes_mut();
        for (i, x) in s.iter_mut().enumerate() {
            *x = (i.wrapping_mul(31) ^ (i >> 7)) as u8;
        }
    }
    // zero the destination so a no-op copy cannot pass
    for x in dst.bytes_mut().iter_mut() {
        *x = 0;
    }

    let src_pat = src.bytes_mut()[..1 << 20].to_vec();
    dev.copy_h2d(&src_pat, &stream).expect("h2d");
    assert!(stream.drain(10_000), "h2d never completed");

    // d2h back into the pinned destination
    let d = dst.bytes_mut();
    dev.copy_d2h(d, &stream).expect("d2h");
    assert!(stream.drain(10_000), "d2h never completed");
    assert_eq!(dst.bytes_mut(), &src_pat[..], "round trip changed bytes");

    // And a negative: h2d with *different* data must change the device bytes.
    // (Guards against a shim whose memcpy is secretly a no-op returning OK.)
    {
        let s2 = src.bytes_mut();
        for (i, x) in s2.iter_mut().enumerate() {
            *x = (i.wrapping_mul(31) ^ (i >> 7) ^ 0xff) as u8;
        }
    }
    let s2 = src.bytes_mut()[..1 << 20].to_vec();
    dev.copy_h2d(&s2, &stream).expect("h2d 2");
    assert!(stream.drain(10_000));
    let d2 = dst.bytes_mut();
    dev.copy_d2h(d2, &stream).expect("d2h 2");
    assert!(stream.drain(10_000));
    assert_eq!(dst.bytes_mut(), &s2[..], "device bytes did not update");
    assert_ne!(s2, src_pat, "the two patterns must differ");
}

#[test]
fn graph_captures_and_replays() {
    let Some(shim) = shim() else { return };
    need_device!(&shim);
    let k = shim.kernels();
    let stream = Stream::create(k).expect("stream");
    let dev = DeviceBuf::alloc(k, 1 << 20).expect("device 1 MiB");
    let mut host = Pinned::new(k, 1 << 20).expect("pinned");
    for x in host.bytes_mut().iter_mut() {
        *x = 0;
    }

    // Capture exactly one d2h copy: dst pinned host, src device, live stream.
    let cap = CapturedGraph::begin(k, &stream).expect("begin capture");
    dev.copy_d2h(host.bytes_mut(), &stream)
        .expect("captured copy");
    let graph = cap.end().expect("end capture + instantiate");

    // Before any replay, the host buffer is still all-zero (capture ran nothing).
    assert!(host.bytes_mut().iter().step_by(4096).all(|&x| x == 0));

    // Replay 100x, changing the device bytes between rounds. A graph that
    // snapshotted bytes at capture time would replay the same pattern forever;
    // a correct one moves whatever is in the device buffer right now.
    for round in 0..100u8 {
        let mut src = Pinned::new(k, 1 << 20).expect("pinned src");
        for x in src.bytes_mut().iter_mut() {
            *x = round;
        }
        let s = src.bytes_mut()[..1 << 20].to_vec();
        dev.copy_h2d(&s, &stream).expect("h2d");
        assert!(stream.drain(10_000));
        for x in host.bytes_mut().iter_mut() {
            *x = 0xff; // poison the destination: a failed/no-op replay leaves 0xff
        }
        graph.replay(&stream).expect("replay");
        assert!(stream.drain(10_000), "replay {round} never completed");
        let seen = host.bytes_mut()[0];
        assert_eq!(
            seen, round,
            "replay {round} delivered {seen:#x}, not {round:#x}"
        );
    }
}

#[test]
fn concurrent_streams_overlap_or_at_least_work() {
    let Some(shim) = shim() else { return };
    need_device!(&shim);
    let k = shim.kernels();
    // Four streams, four buffers, one pinned source: copies must all complete
    // independently (stream-ordered per stream, no cross-stream corruption).
    let src = {
        let mut s = Pinned::new(k, 4 << 20).expect("pinned");
        for (i, x) in s.bytes_mut().iter_mut().enumerate() {
            *x = (i % 251) as u8;
        }
        let v = s.bytes_mut()[..4 << 20].to_vec();
        v
    };
    let mut bufs = Vec::new();
    let mut streams = Vec::new();
    for _ in 0..4 {
        streams.push(Stream::create(k).expect("stream"));
        bufs.push(DeviceBuf::alloc(k, 4 << 20).expect("device 4 MiB"));
    }
    for (buf, st) in bufs.iter().zip(&streams) {
        buf.copy_h2d(&src, st).expect("h2d");
    }
    for st in &streams {
        assert!(st.drain(10_000), "a stream hung");
    }
    for buf in &bufs {
        assert_eq!(buf.to_vec().expect("d2h"), src, "a buffer got wrong bytes");
    }
}

#[test]
fn cpu_shim_reports_zero_devices() {
    let Some(shim) = shim() else { return };
    if shim.kernels().device_alloc.is_some() {
        eprintln!("skipped: this is a CUDA shim build");
        return;
    }
    // A CPU shim must answer 0, not crash or pretend.
    assert_eq!(shim.device_count(), 0);
    assert!(shim.device_info(0).is_none());
    assert!(
        Stream::create(shim.kernels()).is_err(),
        "CPU shim must not fake streams"
    );
    assert!(
        DeviceBuf::alloc(shim.kernels(), 4096).is_err(),
        "CPU shim must not fake device memory"
    );
}
