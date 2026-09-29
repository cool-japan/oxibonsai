//! A synthetic, weighted, dense `qwen3` GGUF + byte-level tokenizer fixture
//! for `/v1/embeddings` tests, built on [`crate::gguf_fixture::GgufFixtureBuilder`].
//!
//! The standalone `oxibonsai-serve`
//! binary's own composition tests (`main.rs`) and its integration test suite
//! (`tests/server_integration_tests.rs`) each hand-rolled a byte-for-byte
//! near-duplicate of this same fixture (a small `qwen3` model with a real
//! ternary-quantized FFN/attention stack and a real, non-degenerate token
//! embedding table, so a served pool actually answers `/v1/embeddings` with
//! distinguishable vectors) — one canonical copy here, shared by both.
//!
//! Every value is fully deterministic
//! ([`crate::gguf_fixture::deterministic_weights`]'s contract): the same
//! call always produces byte-identical output, across processes and runs.

use crate::gguf_fixture::{FixtureQuant, GgufFixtureBuilder};

/// Hidden size.
pub const HIDDEN: usize = 128;
/// Feed-forward intermediate size.
pub const INTER: usize = 256;
/// Transformer block count.
pub const LAYERS: usize = 2;
/// Attention query head count.
pub const N_Q: usize = 4;
/// Attention key/value head count.
pub const N_KV: usize = 2;
/// Per-head dimension.
pub const HEAD_DIM: usize = 32;
/// Vocabulary size — one id per byte, matching [`byte_tokenizer_json`].
pub const VOCAB: usize = 256;
/// A generous max sequence length every consumer's engine pool can share.
pub const MAX_SEQ: usize = 256;

/// A small dense (`qwen3`) GGUF with real ternary-quantized weights: a
/// `TQ2_0_g128` FFN/attention stack, an FP32 token-embedding table and
/// output norm, loadable by `BonsaiModel::from_gguf` on any CPU tier.
///
/// # Panics
///
/// Never in practice: every tensor's element count is a multiple of the
/// `TQ2_0_g128` block size (128) by construction (every dimension here is
/// itself a multiple of 128 or the tensor is `F32`, which has no block-size
/// constraint), so [`GgufFixtureBuilder::tensor`] cannot fail; a change to
/// the constants above that broke this invariant would be caught
/// immediately by this crate's own tests, not silently at a call site.
#[must_use]
pub fn weighted_dense_gguf() -> Vec<u8> {
    let mut b = GgufFixtureBuilder::new();
    b.metadata_str("general.architecture", "qwen3")
        .metadata_str("general.name", "OxibonsaiServeDenseEmbeddingFixture")
        .metadata_u32("qwen3.embedding_length", HIDDEN as u32)
        .metadata_u32("qwen3.block_count", LAYERS as u32)
        .metadata_u32("qwen3.attention.head_count", N_Q as u32)
        .metadata_u32("qwen3.attention.head_count_kv", N_KV as u32)
        .metadata_u32("qwen3.feed_forward_length", INTER as u32)
        .metadata_u32("qwen3.vocab_size", VOCAB as u32)
        .metadata_u32("qwen3.context_length", 512)
        .metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_f32("qwen3.rope.freq_base", 10_000.0);

    b.tensor(
        "token_embd.weight",
        &[HIDDEN as u64, VOCAB as u64],
        FixtureQuant::F32,
        1,
    )
    .expect("token_embd.weight: F32 has no block-size constraint, so this cannot fail");
    b.tensor("output_norm.weight", &[HIDDEN as u64], FixtureQuant::F32, 2)
        .expect("output_norm.weight: F32 has no block-size constraint, so this cannot fail");
    b.tensor(
        "output.weight",
        &[HIDDEN as u64, VOCAB as u64],
        FixtureQuant::TQ2_0_g128,
        0xBEEF,
    )
    .expect("output.weight: HIDDEN and VOCAB are multiples of the TQ2_0_g128 block size (128)");

    for layer in 0..LAYERS {
        let prefix = format!("blk.{layer}");
        for (index, (name, dim)) in [
            ("attn_norm.weight", HIDDEN),
            ("ffn_norm.weight", HIDDEN),
            ("attn_q_norm.weight", HEAD_DIM),
            ("attn_k_norm.weight", HEAD_DIM),
        ]
        .into_iter()
        .enumerate()
        {
            b.tensor(
                &format!("{prefix}.{name}"),
                &[dim as u64],
                FixtureQuant::F32,
                ((layer as u64) << 8) + 0x100 + index as u64,
            )
            .unwrap_or_else(|e| panic!("{prefix}.{name}: {e}"));
        }
        for (bump, (name, in_dim, out_dim)) in [
            ("attn_q.weight", HIDDEN, N_Q * HEAD_DIM),
            ("attn_k.weight", HIDDEN, N_KV * HEAD_DIM),
            ("attn_v.weight", HIDDEN, N_KV * HEAD_DIM),
            ("attn_output.weight", N_Q * HEAD_DIM, HIDDEN),
            ("ffn_gate.weight", HIDDEN, INTER),
            ("ffn_up.weight", HIDDEN, INTER),
            ("ffn_down.weight", INTER, HIDDEN),
        ]
        .into_iter()
        .enumerate()
        {
            b.tensor(
                &format!("{prefix}.{name}"),
                &[in_dim as u64, out_dim as u64],
                FixtureQuant::TQ2_0_g128,
                ((layer as u64) << 8) + bump as u64,
            )
            .unwrap_or_else(|e| panic!("{prefix}.{name}: {e}"));
        }
    }
    b.build().expect("serialize the dense embedding fixture")
}

/// GPT-2's byte-level alphabet (`bytes_to_unicode`): printable bytes map to
/// themselves, the other 68 to `U+0100 + n` in byte order.
fn byte_to_unicode(byte: u8) -> char {
    let printable = |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable(byte) {
        return char::from(byte);
    }
    let rank = (0..byte).filter(|&b| !printable(b)).count();
    char::from_u32(256 + u32::try_from(rank).unwrap_or(0)).unwrap_or('?')
}

/// A byte-level `tokenizer.json` whose 256 ids are the 256 bytes, so every
/// text encodes to ids inside [`weighted_dense_gguf`]'s vocabulary
/// ([`VOCAB`]). A pure string — the caller decides whether to write it to
/// disk (for path-based auto-detection) or build a `TokenizerBridge`
/// directly from it (`TokenizerBridge::native_from_json_str`).
#[must_use]
pub fn byte_tokenizer_json() -> String {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..=255u8)
        .map(|byte| (byte_to_unicode(byte).to_string(), u32::from(byte).into()))
        .collect();
    serde_json::json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": [],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false
        },
    })
    .to_string()
}

/// Write `bytes` to `path` atomically: a sibling temp file (unique per call,
/// via [`std::process::id`] + [`std::time::SystemTime`]) written in full,
/// then renamed into place. `std::fs::write` alone is not atomic — a reader
/// racing the write (e.g. a concurrently-running test binary that happens to
/// probe the same auto-detection candidate path) can observe a partial
/// file; `rename` within the same filesystem is atomic on every platform
/// this workspace targets, so a reader only ever sees "not there yet" or
/// "fully there".
///
/// # Errors
///
/// Any [`std::io::Error`] from writing the temp file or renaming it.
pub fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let unique = format!(
        "{}.tmp-{}-{}",
        path.display(),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp = std::path::PathBuf::from(unique);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_dense_gguf_is_a_well_formed_gguf() {
        let bytes = weighted_dense_gguf();
        assert!(bytes.starts_with(b"GGUF"), "must start with the magic");
        let parsed =
            oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("must parse cleanly");
        assert_eq!(
            parsed
                .metadata
                .get_string("general.architecture")
                .expect("architecture"),
            "qwen3"
        );
        // token_embd + output_norm + output + LAYERS * (4 norm + 7 weight).
        assert_eq!(parsed.tensors.len(), 3 + LAYERS * 11);
    }

    #[test]
    fn weighted_dense_gguf_is_deterministic() {
        assert_eq!(weighted_dense_gguf(), weighted_dense_gguf());
    }

    #[test]
    fn byte_tokenizer_json_covers_every_byte() {
        let json = byte_tokenizer_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let vocab = parsed["model"]["vocab"].as_object().expect("vocab map");
        assert_eq!(vocab.len(), VOCAB);
        let mut ids: Vec<u64> = vocab
            .values()
            .filter_map(serde_json::Value::as_u64)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..VOCAB as u64).collect::<Vec<_>>());
    }

    #[test]
    fn write_atomic_leaves_no_temp_file_behind_and_writes_the_exact_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_testkit_write_atomic_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        let path = dir.join("fixture.bin");
        write_atomic(&path, b"hello fixture").expect("atomic write");
        assert_eq!(std::fs::read(&path).expect("read back"), b"hello fixture");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no temp file must survive a successful write"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
