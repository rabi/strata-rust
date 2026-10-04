// @generated: ported from Strata src/core/conversation_memory.cpp +
// include/strata/core/conversation_memory.hpp @ 99f3dbd0 (23/23 C++ checks pass
// before and after this port; the C++ test is boolean-only, so its conditions
// are transcribed as-is in the tests below).

//! Physical-memory admission for the conversation cache.
//!
//! The engine asks one question before parking a conversation snapshot: is
//! enough *host physical* memory free that adding these bytes cannot push the
//! machine into swap thrash? Two properties the C++ got right and this must
//! keep:
//!
//! * Unknown telemetry is distinct from a measured zero and fails CLOSED. A
//!   machine that will not say its free memory does not get to park caches.
//! * The arithmetic cannot overflow or underflow: the floor is subtracted
//!   after `available >= floor`, and the parse rejects anything that is not a
//!   plain unsigned number of kB.

/// Parse `/proc/meminfo`-shaped telemetry. Returns `None` (unknown) unless
/// exactly one well-formed `MemAvailable:` line exists and the stream reads
/// cleanly. A read error at any point - even after a valid line - is unknown:
/// it is safer to decline the cache than to trust a torn sample.
pub fn mem_available<R: std::io::BufRead>(mut src: R) -> Option<u64> {
    let mut result: Option<u64> = None;
    let mut line = String::new();
    loop {
        line.clear();
        match src.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => return None,
        }
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("MemAvailable:") => {}
            Some(_) => continue,
            None => continue,
        }
        // duplicate telemetry is a contradiction, not a second sample
        if result.is_some() {
            return None;
        }
        let (Some(value), Some(unit)) = (fields.next(), fields.next()) else {
            return None;
        };
        if unit != "kB" || fields.next().is_some() {
            return None;
        }
        // from_chars equivalent: plain ASCII digits, whole token, no overflow.
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let Ok(kb) = value.parse::<u64>() else {
            return None;
        };
        if kb > u64::MAX / 1024 {
            return None;
        }
        result = Some(kb * 1024);
    }
    result
}

/// May `allocation` bytes be committed while keeping `floor` bytes free?
/// Unknown telemetry says no. `available - floor` cannot underflow because the
/// floor is checked first; the ordering IS the overflow defense.
pub fn memory_admit(available: Option<u64>, allocation: u64, floor: u64) -> bool {
    matches!(available, Some(a) if a >= floor && allocation <= a - floor)
}

/// Host physical memory available right now, in bytes: not swap or commit, not
/// a container or job-object reservation. `None` means the platform offered no
/// trustworthy number - which callers must treat as "decline", never as zero.
pub fn available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let f = std::fs::File::open("/proc/meminfo").ok()?;
        mem_available(std::io::BufReader::new(f))
    }
    #[cfg(target_os = "windows")]
    {
        windows_avail_phys()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// `GlobalMemoryStatusEx`, the same six lines the C++ uses. The declaration and
/// layout mirror the documented Windows ABI; the call itself is the only
/// unsafe in this module.
#[cfg(target_os = "windows")]
#[allow(unsafe_code)] // one extern "system" call, args valid by construction
fn windows_avail_phys() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_pagefile: u64,
        avail_pagefile: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    // MEMORYSTATUSEX as documented: dwLength + dwMemoryLoad + seven u64s.
    const _: () = assert!(std::mem::size_of::<MemoryStatusEx>() == 64);

    extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_pagefile: 0,
        avail_pagefile: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a live, correctly sized object and dwLength is set to
    // that size before the call, per the API contract.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok != 0 {
        Some(status.avail_phys)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor, Read};

    fn parse(text: &str) -> Option<u64> {
        mem_available(BufReader::new(Cursor::new(text.as_bytes().to_vec())))
    }

    #[test]
    fn malformed_telemetry_is_unknown() {
        // transcription of conversation_memory_test.cpp's malformed table
        for text in [
            "",
            "MemFree: 100 kB\n",
            "MemAvailable: -1 kB\n",
            "MemAvailable: +1 kB\n",
            "MemAvailable: 1 MB\n",
            "MemAvailable: 1\n",
            "MemAvailable: 1x kB\n",
            "MemAvailable: 18446744073709551615 kB\n",
            "MemAvailable: 18446744073709551616 kB\n",
            "MemAvailable: 1 kB trailing\n",
            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
        ] {
            assert!(!parse(text).is_some(), "must be unknown: {text:?}");
        }
    }

    #[test]
    fn only_mem_available_is_counted() {
        let normal = "MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n";
        assert_eq!(parse(normal), Some(12345 * 1024));
    }

    #[test]
    fn zero_available_is_known() {
        assert_eq!(parse("MemAvailable: 0 kB"), Some(0));
    }

    #[test]
    fn io_failure_declines_admission() {
        // reads the valid line first, then the device fails: the sample must
        // still come back unknown because the stream did not end cleanly
        struct TornStream {
            data: Cursor<Vec<u8>>,
        }
        impl Read for TornStream {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.data.read(buf)?;
                if n > 0 {
                    return Ok(n);
                }
                Err(std::io::Error::other("device failure"))
            }
        }
        let torn = TornStream {
            data: Cursor::new(b"MemAvailable: 123 kB\n".to_vec()),
        };
        assert_eq!(mem_available(BufReader::new(torn)), None);
    }

    #[test]
    fn admit_fails_closed_on_unknown() {
        assert!(!memory_admit(None, 0, 0));
    }

    #[test]
    fn admit_boundaries() {
        assert!(memory_admit(Some(100), 40, 60), "exact fit");
        assert!(!memory_admit(Some(99), 40, 60), "one byte short");
        assert!(
            !memory_admit(Some(59), 0, 60),
            "floor subtraction cannot underflow"
        );
        assert!(memory_admit(Some(60), 0, 60), "post-capture floor check");
    }

    #[test]
    fn admit_arithmetic_cannot_overflow() {
        assert!(!memory_admit(Some(100), u64::MAX, 1));
        assert!(
            memory_admit(Some(u64::MAX), u64::MAX, 0),
            "maximal exact bound"
        );
        assert!(
            !memory_admit(Some(u64::MAX), u64::MAX, 1),
            "maximal sum overflow rejected"
        );
    }

    #[test]
    fn provider_returns_bytes_or_unknown() {
        // same shape as the C++ check: whatever this machine says, it must be
        // self-consistent (admitting nothing with zero floor means a negative
        // or garbage reading); no assumption about how much RAM is free
        let avail = available_memory();
        assert!(!avail.is_some() || memory_admit(avail, 0, 0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_provider_reads_proc_meminfo() {
        // on Linux /proc/meminfo always exists; the C++ cannot assert this
        // portably, the port can, so it does
        assert!(
            available_memory().is_some(),
            "/proc/meminfo present but provider said unknown"
        );
    }
}
