//! The byte-format half of the native MMVQ path: which GGML quantizations the
//! native kernels read, how many bytes one of their weight matrices occupies,
//! and the two fp16 conversions the weight loader performs. Ports the parts of
//! `src/kernels/cuda/native_mmvq.cu` and `include/strata/kernels/f16_bits.hpp`
//! that are pure integer/bit logic; the kernels themselves stay CUDA.

/// The types `native_mmvq_supported` accepts.
pub fn native_mmvq_supported(ggml_type: i32) -> bool {
    matches!(
        ggml_type,
        2 | 6 | 7 | 8 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 20 | 21 | 22 | 23 | 29 | 42
    )
}

/// `(block_elems, block_bytes)` for every type the MMVQ switch knows, `None`
/// otherwise. Public so the corpus replay can compare the table itself against
/// the C++ `native_block_size` probe line by line, rather than only through the
/// products `native_mmvq_weight_bytes` returns.
pub fn weight_block(ggml_type: i32) -> Option<(u64, u64)> {
    let elems;
    let bytes: u64;
    match ggml_type {
        2 => {
            elems = 32;
            bytes = 18; // Q4_0
        }
        6 => {
            elems = 32;
            bytes = 22; // Q5_0
        }
        7 => {
            elems = 32;
            bytes = 24; // Q5_1
        }
        8 => {
            elems = 32;
            bytes = 34; // Q8_0
        }
        20 => {
            elems = 32;
            bytes = 18; // IQ4_NL
        }
        11 => {
            elems = 256;
            bytes = 110; // Q3_K
        }
        12 => {
            elems = 256;
            bytes = 144; // Q4_K
        }
        13 => {
            elems = 256;
            bytes = 176; // Q5_K
        }
        14 => {
            elems = 256;
            bytes = 210; // Q6_K
        }
        23 => {
            elems = 256;
            bytes = 136; // IQ4_XS
        }
        42 => {
            elems = 64;
            bytes = 18; // Q2_0
        }
        16 => {
            elems = 256;
            bytes = 66; // IQ2_XXS
        }
        17 => {
            elems = 256;
            bytes = 74; // IQ2_XS
        }
        18 => {
            elems = 256;
            bytes = 98; // IQ3_XXS
        }
        21 => {
            elems = 256;
            bytes = 110; // IQ3_S
        }
        22 => {
            elems = 256;
            bytes = 82; // IQ2_S
        }
        29 => {
            elems = 256;
            bytes = 56; // IQ1_M
        }
        _ => return None,
    }
    Some((elems, bytes))
}

/// `native_mmvq_weight_bytes(type, n_in, n_out)`: a row is `n_in / block_elems`
/// blocks, the matrix is `n_out` rows. The C++ throws on an unsupported type, a
/// non-divisible `n_in`, or a byte count that would overflow; the error strings are
/// the ones `NativeDense::load` surfaces through its `catch`.
pub fn native_mmvq_weight_bytes(ggml_type: i32, n_in: i32, n_out: i32) -> Result<u64, String> {
    let (elems, bytes) = match weight_block(ggml_type) {
        Some(g) => g,
        None => return Err("unsupported native MMVQ GGML type".into()),
    };
    if n_in <= 0 || !(n_in as u64).is_multiple_of(elems) {
        return Err(
            "native MMVQ requires n_in > 0 and divisible by its block element count".into(),
        );
    }
    if n_out <= 0 {
        return Err("native MMVQ requires n_out > 0".into());
    }
    let row_bytes = (n_in as u64 / elems) * bytes;
    row_bytes
        .checked_mul(n_out as u64)
        .ok_or_else(|| "native MMVQ weight byte count overflows size_t".to_string())
}

/// `native_q8_1_bytes(n_in, ncols)`: the shared scratch every native projection
/// quantizes its input into. `Q81Block` is `half2 ds` + `int8_t qs[32]` = 36 B.
pub fn native_q8_1_bytes(n_in: i32, ncols: i32) -> Result<u64, String> {
    if n_in <= 0 || !(n_in as u64).is_multiple_of(32) {
        return Err(
            "native MMVQ requires n_in > 0 and divisible by its block element count".into(),
        );
    }
    if !(1..=8).contains(&ncols) {
        return Err("native MMVQ requires 1 <= ncols <= 8".into());
    }
    Ok(ncols as u64 * (n_in as u64 / 32) * 36)
}

/// Round-to-nearest-even f32 -> fp16, returned as raw bits.
///
/// The rounding regimes and the inf-vs-overflow split are documented in
/// `include/strata/kernels/f16_bits.hpp`: a finite value too large for fp16
/// saturates to inf, only an f32 inf/NaN (raw exponent 255) produces a NaN.
pub fn f16_from_f32(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let rawexp = (x >> 23) & 0xFF;
    let exp = rawexp as i32 - 127 + 15;
    let man = x & 0x7F_FFFF;
    if rawexp == 0xFF {
        return sign | 0x7C00 | (if man != 0 { 0x200 } else { 0 });
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let man = man | 0x80_0000; // restore the implicit 1
        let sh = (14 - exp) as u32;
        let mut h = ((man >> sh) & 0x3FF) as u16;
        let rem = man & ((1u32 << sh) - 1);
        if rem > (1u32 << (sh - 1)) || (rem == (1u32 << (sh - 1)) && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = sign | ((exp as u32) << 10) as u16 | ((man >> 13) as u16);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h = h.wrapping_add(1); // the carry may bump the exponent
    }
    h
}

/// Exact fp16 bits -> f32. No rounding: every one of the 65,536 inputs maps to a
/// value f32 represents exactly, subnormals included, which is why widening a
/// pack's fp16 scale plane to f32 changes no weight.
pub fn f32_from_f16(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let ex = ((h >> 10) & 0x1F) as u32;
    let man = u32::from(h & 0x3FF);
    let out = if ex == 0 {
        if man == 0 {
            sign
        } else {
            let mut m = man;
            let mut s = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                s += 1;
            }
            sign | ((113 - s) << 23) | ((m & 0x3FF) << 13)
        }
    } else if ex == 31 {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((ex + 112) << 23) | (man << 13) // ex-15+127, kept unsigned
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_set_matches_the_kernel() {
        for t in [
            2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42,
        ] {
            assert!(native_mmvq_supported(t), "type {t}");
        }
        for t in [
            0, 1, 3, 4, 5, 9, 10, 15, 19, 24, 25, 26, 27, 28, 30, 31, 41, 43, -1,
        ] {
            assert!(!native_mmvq_supported(t), "type {t} must be unsupported");
        }
    }

    #[test]
    fn weight_bytes_are_the_row_sum() {
        // Q8_0, 256 in x 4 out: 8 blocks/row * 34 B * 4 rows.
        assert_eq!(native_mmvq_weight_bytes(8, 256, 4), Ok(8 * 34 * 4));
        // Q2_0 is 64-element blocks: 256/64 = 4 blocks/row * 18 B.
        assert_eq!(native_mmvq_weight_bytes(42, 256, 2), Ok(4 * 18 * 2));
        // IQ3_XXS: 256/256 = 1 block/row * 98 B.
        assert_eq!(native_mmvq_weight_bytes(18, 256, 2), Ok(98 * 2));
        assert_eq!(
            native_mmvq_weight_bytes(1, 256, 4),
            Err("unsupported native MMVQ GGML type".into())
        );
        // n_in must be divisible by the block element count.
        assert!(native_mmvq_weight_bytes(8, 250, 4).is_err());
        assert_eq!(native_mmvq_weight_bytes(42, 128, 4), Ok(2 * 18 * 4)); // Q2_0: 128 % 64 == 0
        assert!(native_mmvq_weight_bytes(8, 256, 0).is_err());
        // the largest i32 matrix of the largest-block type stays inside u64
        assert!(native_mmvq_weight_bytes(14, i32::MAX - 255, i32::MAX).is_ok());
    }

    #[test]
    fn scratch_size_is_36_bytes_per_32_inputs() {
        assert_eq!(native_q8_1_bytes(256, 1), Ok(8 * 36));
        assert_eq!(native_q8_1_bytes(2560, 8), Ok(8 * 80 * 36));
        assert!(native_q8_1_bytes(255, 1).is_err());
        assert!(native_q8_1_bytes(256, 0).is_err());
        assert!(native_q8_1_bytes(256, 9).is_err());
    }

    #[test]
    fn f16_conversion_hits_the_regimes() {
        assert_eq!(f16_from_f32(1.0), 0x3C00);
        assert_eq!(f16_from_f32(-1.0), 0xBC00);
        assert_eq!(f16_from_f32(0.0), 0x0000);
        assert_eq!(f16_from_f32(-0.0), 0x8000);
        // finite overflow saturates to inf, NOT a NaN (round 198's numpy finding)
        assert_eq!(f16_from_f32(1e30), 0x7C00);
        assert_eq!(f16_from_f32(-1e30), 0xFC00);
        assert_eq!(f16_from_f32(f32::INFINITY), 0x7C00);
        assert_eq!(f16_from_f32(f32::NAN) & 0x7C00, 0x7C00);
        assert_ne!(f16_from_f32(f32::NAN), 0x7C00); // a quiet NaN keeps a mantissa bit
                                                    // underflow to signed zero
        assert_eq!(f16_from_f32(-1e-10), 0x8000);
        // smallest fp16 subnormal is exact
        assert_eq!(f16_from_f32(5.960_464_5e-8), 0x0001);
        // round-to-nearest-even on the mantissa
        assert_eq!(f16_from_f32(1.000_000_1), 0x3C00);
    }

    #[test]
    fn f16_round_trip_is_exact() {
        for bits in 0u32..=0xFFFF {
            let h = bits as u16;
            if (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0 {
                assert!(
                    f32_from_f16(h).is_nan(),
                    "bits {h:#06x} must decode to a NaN"
                );
                continue; // NaNs cannot round-trip: f16_from_f32 emits a quiet NaN
            }
            let f = f32_from_f16(h);
            assert_eq!(
                f16_from_f32(f),
                h,
                "bits {h:#06x} -> {f} -> {:#06x}",
                f16_from_f32(f)
            );
        }
    }

    #[test]
    fn f16_subnormals_decode_to_the_right_power_of_two() {
        // 0x0001 is 2^-24, the value the two private C++ copies flushed to zero.
        assert_eq!(f32_from_f16(0x0001), 5.960_464_5e-8);
        assert_eq!(f32_from_f16(0x0002), 1.192_092_9e-7);
        // the largest subnormal (0x3FF) is (2^10-1) * 2^-24
        assert_eq!(f32_from_f16(0x03FF), (1023.0f64 * 2f64.powi(-24)) as f32);
        assert!(f32_from_f16(0x8000).is_sign_negative() && f32_from_f16(0x8000) == 0.0);
        assert!(f32_from_f16(0x7C00).is_infinite());
        assert!(f32_from_f16(0x7C01).is_nan());
    }
}
