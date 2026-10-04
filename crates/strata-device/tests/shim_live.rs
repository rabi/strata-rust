// End-to-end ABI test against a real shim build. Skips cleanly when
// STRATA_KERNELS_LIB is unset so `cargo test` stays green on boxes with no
// shim; on a machine that built shim/build.sh, this exercises the whole
// contract: load, negotiate, O_DIRECT submit/wait with completion tags,
// short-read-at-EOF tolerance, pinned alignment, and the file-size oracle.
//
//   ./shim/build.sh
//   STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so cargo test -p strata-device
//   STRATA_MODEL=/path/any.gguf ...        # also streams every tensor

use strata_device::{DeviceFile, IoCompletion, Pinned, Shim};

fn shim() -> Option<Shim> {
    Shim::try_load().map(|r| r.expect("shim present but failed to load"))
}

#[test]
fn load_negotiate_and_fill_slots() {
    let Some(shim) = shim() else {
        eprintln!("STRATA_KERNELS_LIB unset - skipped");
        return;
    };
    // The CPU shim must at least speak the IO path it was built to expose.
    let filled = shim.filled_slots();
    assert!(
        filled >= 8,
        "only {filled} of {} slots filled",
        strata_device::STRATA_SLOT_COUNT
    );
    assert!(shim.kernels().file_open.is_some());
    assert!(shim.kernels().file_submit.is_some());
    assert!(shim.kernels().pinned_alloc.is_some());
    assert!(shim.device_count() >= 0);
}

#[test]
fn direct_io_round_trip_and_alignment_contract() {
    let Some(shim) = shim() else { return };
    let k = shim.kernels();

    // A file of known bytes, deliberately not a multiple of the alignment.
    let path = std::env::temp_dir().join("strata_abi_io.bin");
    let data: Vec<u8> = (0..20000u32)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 4)) as u8)
        .collect();
    std::fs::write(&path, &data).unwrap();

    let f = DeviceFile::open(k, &path).expect("open");
    assert_eq!(f.size() as usize, data.len());
    let align = f.alignment();
    assert!(align.is_multiple_of(4096) || align == 0);

    let mut buf = Pinned::new(k, 8192).expect("pinned");
    assert!(buf.align_ok(align));
    let mut done = [IoCompletion::default(); 1];

    // aligned first block
    f.submit(0, &mut buf.bytes_mut()[..4096], 7).unwrap();
    assert_eq!(f.wait(&mut done, 5_000), 1);
    assert!(done[0].ok != 0 && done[0].tag == 7);
    assert_eq!(&buf.bytes_mut()[..4096], &data[..4096]);

    // unaligned requests must be refused, not silently serviced
    assert!(f.submit(1, &mut buf.bytes_mut()[..4096], 1).is_err());
    assert!(f.submit(0, &mut buf.bytes_mut()[..4095], 1).is_err());

    // tail: the aligned window overruns EOF and must come back short-but-ok
    let tail_off = (data.len() as u64 / 4096) * 4096;
    let win = 4096 + 4096; // one page of slack so the window covers the tail
    let mut tail = Pinned::new(k, win).unwrap();
    let slice = &mut tail.bytes_mut()[..win];
    f.submit(tail_off, slice, 42).unwrap();
    assert_eq!(f.wait(&mut done, 5_000), 1);
    let got = done[0];
    assert!(got.ok != 0 && got.tag == 42);
    assert!((got.bytes as usize) <= win);
    let in_range = data.len() - tail_off as usize;
    assert_eq!(
        got.bytes as usize, in_range,
        "short read must land exactly at EOF"
    );
    assert_eq!(&tail.bytes_mut()[..in_range], &data[tail_off as usize..]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn real_model_streaming_if_available() {
    let Some(shim) = shim() else { return };
    let Ok(model) = std::env::var("STRATA_MODEL") else {
        eprintln!("STRATA_MODEL unset - skipped");
        return;
    };
    // cargo test runs from the crate dir; resolve relative paths against the
    // workspace root the caller was standing in.
    let model = {
        let p = std::path::PathBuf::from(&model);
        if p.is_absolute() {
            p
        } else {
            std::env::current_dir()
                .ok()
                .and_then(|d| {
                    let mut d = d.as_path();
                    loop {
                        let c = d.join(&model);
                        if c.exists() {
                            return Some(c);
                        }
                        match d.parent() {
                            Some(pp) => d = pp,
                            None => break None,
                        }
                    }
                })
                .unwrap_or(p)
        }
    };
    assert!(
        model.exists(),
        "STRATA_MODEL points at nothing: {}",
        model.display()
    );
    let model = model.to_string_lossy().into_owned();
    let k = shim.kernels();
    let f = DeviceFile::open(k, &model).expect("open model");
    let g = strata_artifact::GgufFile::open(&model).expect("rust parse");
    assert_eq!(f.size(), g.file_size());

    let mut buf = Pinned::new(k, 1 << 20).unwrap();
    let mut done = [IoCompletion::default(); 1];
    let mut checked = 0u64;
    let mut tag = 0u64;
    let mut tensors = 0u64;
    for t in g.tensors() {
        let bytes = strata_artifact::tensor_payload_bytes(t);
        if bytes == 0 || checked > 1 << 22 {
            continue; // bounded: one MiB per tensor is enough to prove the path
        }
        let want = bytes.min(1 << 20);
        let off = g.tensor_file_offset(t);
        let lo = off / 4096 * 4096;
        let len = ((off + want).div_ceil(4096) * 4096 - lo) as usize;
        let pre = (off - lo) as usize;
        f.submit(lo, &mut buf.bytes_mut()[..len], tag).unwrap();
        assert_eq!(f.wait(&mut done, 10_000), 1);
        let got = done[0];
        assert!(
            got.ok != 0 && got.tag == tag && (got.bytes as usize) >= pre + want as usize,
            "completion {:?} for {} at tag {tag}",
            (got.tag, got.bytes, got.ok),
            t.name
        );
        tag += 1;
        // and the bytes must agree with plain buffered reads of the same file
        let cpp = g.read_tensor(t).unwrap();
        assert_eq!(
            &buf.bytes_mut()[pre..pre + want as usize],
            &cpp[..want as usize]
        );
        checked += want;
        tensors += 1;
    }
    eprintln!(
        "streamed {checked} bytes across {tensors} tensors over the ABI, byte-compared \
         against buffered reads"
    );
    assert!(checked > 0, "model had no readable tensors");
}
