// @generated: ported from Strata include/strata/artifact/gguf_split.hpp

//! The shards of a split model.
//!
//! Split models: `<name>-00001-of-0000N.gguf` ... `<name>-0000N-of-0000N.gguf`,
//! as llama.cpp's gguf-split names them. Given ANY shard's path, the shards in
//! split order (the metadata shard, 00001, first); a file without that suffix is
//! a model of one shard. Only the five digits of the shard number are rewritten,
//! so the directory and the stem are the caller's own spelling (a symlinked
//! Hugging Face snapshot keeps its split names). Throws when a shard is missing —
//! a download that was still running must not produce a model with some layers
//! missing and an error much later, or none.

use std::path::{Path, PathBuf};

pub fn gguf_split_paths(any: &Path) -> Result<Vec<PathBuf>, String> {
    let name = any.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let at = match name.rfind("-of-") {
        Some(i) => i,
        None => return Ok(vec![any.to_path_buf()]),
    };
    // "...-NNNNN-of-MMMMM.gguf": five digits before the tag behind a '-', five after
    if at < 6 || name.as_bytes()[at - 6] != b'-' {
        return Ok(vec![any.to_path_buf()]);
    }
    let (no_s, rest) = (&name[at - 5..at], &name[at + 4..]);
    if rest.len() != 10
        || !rest[..5].bytes().all(|b| b.is_ascii_digit())
        || !rest[5..].eq_ignore_ascii_case(".gguf")
    {
        return Ok(vec![any.to_path_buf()]);
    }
    if !no_s.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(vec![any.to_path_buf()]);
    }
    let no: u32 = no_s.parse().unwrap();
    let n: u32 = rest[..5].parse().unwrap();
    if n < 1 || no < 1 || no > n {
        return Ok(vec![any.to_path_buf()]);
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 1..=n {
        // only the five digits of the shard number are rewritten; the stem, the
        // caller's spelling of the total and the extension are kept
        let p = any.with_file_name(format!("{}{:05}{}", &name[..at - 5], i, &name[at..]));
        if !p.exists() {
            return Err(format!(
                "missing model shard {} (shard {} of {n}; is the download complete?)",
                p.display(),
                i
            ));
        }
        out.push(p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn touch(dir: &Path, name: &str) {
        fs::write(dir.join(name), b"").unwrap();
    }

    #[test]
    fn a_plain_path_is_a_model_of_one_shard() {
        let dir = std::env::temp_dir();
        let p = gguf_split_paths(&dir.join("model.gguf")).unwrap();
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn a_name_that_merely_ends_like_a_shard_is_one_shard() {
        let dir = std::env::temp_dir();
        // no digits before "-of-"
        let p = gguf_split_paths(&dir.join("model-of-00003.gguf")).unwrap();
        assert_eq!(p.len(), 1);
        // five digits before but four after
        let p = gguf_split_paths(&dir.join("model-00001-of-0003.gguf")).unwrap();
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn all_shards_resolve_in_order() {
        let dir = std::env::temp_dir().join(format!("strata_split_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for i in 1..=3 {
            touch(&dir, &format!("m-{i:05}-of-00003.gguf"));
        }
        // open from the LAST shard: the order must still start with 00001
        let p = gguf_split_paths(&dir.join("m-00003-of-00003.gguf")).unwrap();
        assert_eq!(p.len(), 3);
        assert!(p[0].to_str().unwrap().ends_with("m-00001-of-00003.gguf"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_shard_is_an_error_that_names_it() {
        let dir = std::env::temp_dir().join(format!("strata_split_missing_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        touch(&dir, "m-00001-of-00002.gguf");
        let err = gguf_split_paths(&dir.join("m-00001-of-00002.gguf")).unwrap_err();
        assert!(
            err.contains("m-00002-of-00002"),
            "the error names the missing shard: {err}"
        );
        assert!(err.contains("shard 2 of 2"));
        fs::remove_dir_all(&dir).ok();
    }
}
