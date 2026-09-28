//! Shared, hermetic test fixtures for the CLI's unit tests (compiled only
//! under `#[cfg(test)]`): tiny GGUF images built with the real core writer,
//! a byte-level BPE `tokenizer.json`, a compact template mirroring the
//! Bonsai 2 template's thinking construct, and the environment-variable
//! lookups the real-model tests use (they SKIP with a capability report
//! when a variable is unset — never a hardcoded path, never `#[ignore]`).

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

/// Every `qwen35.*` key `HybridConfig::from_metadata` needs, with the real
/// Bonsai 2 27B values (design Appendix A.4), plus `extra` metadata.
pub(crate) fn qwen35_27b_metadata(w: &mut GgufWriter) {
    let s = |v: &str| MetadataWriteValue::Str(v.to_string());
    w.add_metadata("general.architecture", s("qwen35"));
    w.add_metadata("general.name", s("Ternary-Bonsai-2-27B"));
    for (key, value) in [
        ("qwen35.block_count", 64u32),
        ("qwen35.context_length", 262_144),
        ("qwen35.embedding_length", 5120),
        ("qwen35.feed_forward_length", 17_408),
        ("qwen35.attention.head_count", 24),
        ("qwen35.attention.head_count_kv", 4),
        ("qwen35.attention.key_length", 256),
        ("qwen35.attention.value_length", 256),
        ("qwen35.rope.dimension_count", 64),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 128),
        ("qwen35.ssm.group_count", 16),
        ("qwen35.ssm.time_step_rank", 48),
        ("qwen35.ssm.inner_size", 6144),
        ("qwen35.full_attention_interval", 4),
        ("qwen35.vocab_size", 248_320),
    ] {
        w.add_metadata(key, MetadataWriteValue::U32(value));
    }
    w.add_metadata(
        "qwen35.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen35.rope.freq_base", MetadataWriteValue::F32(1.0e7));
    w.add_metadata(
        "qwen35.rope.dimension_sections",
        MetadataWriteValue::ArrayI32(vec![11, 11, 10, 0]),
    );
}

/// A compact template using the real Bonsai 2 template's own
/// generation-prompt construct (`enable_thinking is defined and
/// enable_thinking is false` → a closed empty block, else an open one) and
/// its assistant re-rendering (`<think>\n{reasoning}\n</think>\n\n{content}`).
pub(crate) const THINKING_TEMPLATE: &str = "{%- for message in messages %}\
{%- if message.role == 'assistant' and message.reasoning_content is defined %}\
{{- '<|im_start|>assistant\\n<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' + message.content + '<|im_end|>\\n' }}\
{%- else %}\
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\
{%- endif %}\
{%- endfor %}\
{%- if add_generation_prompt %}\
{{- '<|im_start|>assistant\\n' }}\
{%- if enable_thinking is defined and enable_thinking is false %}\
{{- '<think>\\n\\n</think>\\n\\n' }}\
{%- else %}\
{{- '<think>\\n' }}\
{%- endif %}\
{%- endif %}";

/// GPT-2's byte → printable-unicode map (the ByteLevel pre-tokenizer's
/// alphabet), so a vocabulary of these 256 characters encodes any text.
pub(crate) fn byte_level_alphabet() -> Vec<char> {
    let mut printable: Vec<u32> = ('!' as u32..='~' as u32).collect();
    printable.extend('¡' as u32..='¬' as u32);
    printable.extend('®' as u32..='ÿ' as u32);
    let mut out = Vec::with_capacity(256);
    let mut next_extra = 256u32;
    for byte in 0u32..256 {
        if printable.contains(&byte) {
            out.push(char::from_u32(byte).unwrap_or('?'));
        } else {
            out.push(char::from_u32(next_extra).unwrap_or('?'));
            next_extra += 1;
        }
    }
    out
}

/// The ids of [`byte_level_tokenizer_json`]'s added special tokens.
pub(crate) const IM_START_ID: u32 = 256;
pub(crate) const IM_END_ID: u32 = 257;
pub(crate) const THINK_OPEN_ID: u32 = 258;
pub(crate) const THINK_CLOSE_ID: u32 = 259;
pub(crate) const TOOL_CALL_OPEN_ID: u32 = 260;
pub(crate) const TOOL_CALL_CLOSE_ID: u32 = 261;

/// A byte-level BPE `tokenizer.json` (256 byte tokens, no merges) with the
/// chat-contract added tokens `<|im_start|>`, `<|im_end|>`, `<think>`,
/// `</think>`, `<tool_call>`, `</tool_call>` — 262 tokens in all.
pub(crate) fn byte_level_tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for (id, ch) in byte_level_alphabet().into_iter().enumerate() {
        vocab.insert(ch.to_string(), serde_json::Value::from(id as u64));
    }
    let added = [
        (IM_START_ID, "<|im_start|>", true),
        (IM_END_ID, "<|im_end|>", true),
        (THINK_OPEN_ID, "<think>", false),
        (THINK_CLOSE_ID, "</think>", false),
        (TOOL_CALL_OPEN_ID, "<tool_call>", false),
        (TOOL_CALL_CLOSE_ID, "</tool_call>", false),
    ];
    let added_tokens: Vec<serde_json::Value> = added
        .iter()
        .map(|(id, content, special)| {
            serde_json::json!({ "id": id, "content": content, "special": special })
        })
        .collect();
    serde_json::json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": added_tokens,
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    })
    .to_string()
}

/// Vocabulary size of [`byte_level_tokenizer_json`].
#[cfg(feature = "server")]
pub(crate) const BYTE_LEVEL_VOCAB: u64 = 262;

/// A metadata+tensor GGUF a tokenizer can be checked against: `qwen3`
/// architecture, a `token_embd.weight` of `vocab` rows (so
/// `model_vocab_size` resolves), optionally a `tokenizer.chat_template` and
/// optionally an embedded `tokenizer.ggml.*` vocabulary.
#[cfg(feature = "server")]
pub(crate) fn tokenizer_host_gguf(
    vocab: u64,
    chat_template: Option<&str>,
    embedded_vocab: bool,
) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    if let Some(template) = chat_template {
        w.add_metadata(
            "tokenizer.chat_template",
            MetadataWriteValue::Str(template.to_string()),
        );
    }
    if embedded_vocab {
        let mut tokens: Vec<String> = byte_level_alphabet()
            .into_iter()
            .map(String::from)
            .collect();
        let mut types: Vec<i32> = vec![1; tokens.len()];
        for special in [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
        ] {
            tokens.push(special.to_string());
            types.push(3);
        }
        w.add_metadata(
            "tokenizer.ggml.model",
            MetadataWriteValue::Str("gpt2".to_string()),
        );
        w.add_metadata(
            "tokenizer.ggml.pre",
            MetadataWriteValue::Str("qwen2".to_string()),
        );
        w.add_metadata(
            "tokenizer.ggml.tokens",
            MetadataWriteValue::ArrayStr(tokens),
        );
        w.add_metadata(
            "tokenizer.ggml.token_type",
            MetadataWriteValue::ArrayI32(types),
        );
        w.add_metadata(
            "tokenizer.ggml.merges",
            MetadataWriteValue::ArrayStr(Vec::new()),
        );
        w.add_metadata(
            "tokenizer.ggml.eos_token_id",
            MetadataWriteValue::U32(IM_END_ID),
        );
    }
    let hidden = 4u64;
    let data: Vec<u8> = (0..hidden * vocab)
        .flat_map(|_| 0.0f32.to_le_bytes())
        .collect();
    w.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![hidden, vocab],
        tensor_type: TensorType::F32,
        data,
    });
    w.to_bytes().expect("serialize tokenizer-host fixture")
}

/// `n` weights of `Q1_0_g128` data (all bits set, scale 1.0 as f16 `0x3C00`).
fn q1_0_g128_data(num_weights: usize) -> Vec<u8> {
    let scale = 0x3C00u16.to_le_bytes();
    let mut data = Vec::with_capacity(num_weights / 128 * 18);
    for _ in 0..num_weights / 128 {
        data.extend_from_slice(&scale);
        data.extend_from_slice(&[0xFFu8; 16]);
    }
    data
}

/// A minimal dense `qwen3` GGUF the real engine loads on any CPU tier (the
/// runtime's own pool-test shape: 2 layers, hidden 128, FFN 256, 4/2 heads
/// of 32, vocab 32, context 512; attention/FFN/LM head `Q1_0_g128`, token
/// embedding F32), plus `extra` metadata. Every output row is identical, so
/// every logit ties: greedy decoding is deterministic and a sampled one is
/// uniform over the kept set — ideal for seed/routing assertions.
pub(crate) fn tiny_dense_gguf(extra: Vec<(&str, MetadataWriteValue)>) -> Vec<u8> {
    let (h, inter, layers, nq, nkv, hd, vocab) =
        (128usize, 256usize, 2usize, 4usize, 2usize, 32usize, 32usize);
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".into()),
    );
    w.add_metadata("general.name", MetadataWriteValue::Str("TinyCli".into()));
    for (key, value) in [
        ("qwen3.embedding_length", h),
        ("qwen3.block_count", layers),
        ("qwen3.attention.head_count", nq),
        ("qwen3.attention.head_count_kv", nkv),
        ("qwen3.feed_forward_length", inter),
        ("qwen3.vocab_size", vocab),
        ("qwen3.context_length", 512),
    ] {
        w.add_metadata(key, MetadataWriteValue::U32(value as u32));
    }
    w.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    for (key, value) in extra {
        w.add_metadata(key, value);
    }
    let ones = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
    let f32_tensor = |name: String, shape: Vec<u64>, n: usize| TensorEntry {
        name,
        shape,
        tensor_type: TensorType::F32,
        data: ones(n),
    };
    let q1_tensor = |name: String, shape: Vec<u64>, n: usize| TensorEntry {
        name,
        shape,
        tensor_type: TensorType::Q1_0G128,
        data: q1_0_g128_data(n),
    };
    w.add_tensor(f32_tensor(
        "token_embd.weight".into(),
        vec![h as u64, vocab as u64],
        vocab * h,
    ));
    w.add_tensor(f32_tensor("output_norm.weight".into(), vec![h as u64], h));
    w.add_tensor(q1_tensor(
        "output.weight".into(),
        vec![h as u64, vocab as u64],
        vocab * h,
    ));
    for layer in 0..layers {
        let p = format!("blk.{layer}");
        for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
            w.add_tensor(f32_tensor(format!("{p}.{suffix}"), vec![h as u64], h));
        }
        for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            w.add_tensor(f32_tensor(format!("{p}.{suffix}"), vec![hd as u64], hd));
        }
        let (q, kv, ffn) = ((nq * hd) as u64, (nkv * hd) as u64, inter as u64);
        let hh = h as u64;
        w.add_tensor(q1_tensor(
            format!("{p}.attn_q.weight"),
            vec![hh, q],
            nq * hd * h,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.attn_k.weight"),
            vec![hh, kv],
            nkv * hd * h,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.attn_v.weight"),
            vec![hh, kv],
            nkv * hd * h,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.attn_output.weight"),
            vec![q, hh],
            h * nq * hd,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.ffn_gate.weight"),
            vec![hh, ffn],
            inter * h,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.ffn_up.weight"),
            vec![hh, ffn],
            inter * h,
        ));
        w.add_tensor(q1_tensor(
            format!("{p}.ffn_down.weight"),
            vec![ffn, hh],
            h * inter,
        ));
    }
    w.to_bytes().expect("serialize the tiny dense fixture")
}

/// A unique scratch directory under the system temp dir.
pub(crate) fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// A real-model resource located through `var`, or `None` after printing a
/// capability report (the test then returns early — it is never
/// `#[ignore]`d, so the gate always compiles and runs it).
pub(crate) fn env_path(var: &str, what: &str) -> Option<std::path::PathBuf> {
    match std::env::var_os(var) {
        Some(value) if !value.is_empty() => {
            let path = std::path::PathBuf::from(value);
            if path.exists() {
                Some(path)
            } else {
                eprintln!(
                    "SKIPPED (capability): {var}={} does not exist ({what})",
                    path.display()
                );
                None
            }
        }
        _ => {
            eprintln!("SKIPPED (capability): set {var} to run this test ({what})");
            None
        }
    }
}

/// A file inside `OXIBONSAI_MODELS_DIR` (the real `models/` directory),
/// or `None` with a capability report.
pub(crate) fn models_dir_file(name: &str) -> Option<std::path::PathBuf> {
    let dir = env_path(
        "OXIBONSAI_MODELS_DIR",
        "the directory holding the real GGUFs",
    )?;
    let path = dir.join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "SKIPPED (capability): {} is not present in OXIBONSAI_MODELS_DIR",
            path.display()
        );
        None
    }
}

/// Serializes the real-model unit tests (one real GGUF load at a time — the
/// 27B memory discipline, and one model's worth of RSS per test binary).
pub(crate) fn real_model_lock() -> std::sync::MutexGuard<'static, ()> {
    static REAL_MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    REAL_MODEL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A minimal `tracing` subscriber that records every event as
/// `"field=value ..."` text, so a test can assert that a code path logged
/// what it promises (e.g. the hybrid-embeddings 501 info line) without a
/// `tracing-subscriber` dependency. Install it with
/// `tracing::subscriber::with_default` around synchronous code.
#[cfg(feature = "server")]
#[derive(Clone, Default)]
pub(crate) struct CapturedEvents {
    events: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

#[cfg(feature = "server")]
impl CapturedEvents {
    /// Every recorded event, in order.
    pub(crate) fn events(&self) -> Vec<String> {
        self.events
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default()
    }
}

#[cfg(feature = "server")]
struct EventText(String);

#[cfg(feature = "server")]
impl tracing::field::Visit for EventText {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let _ = write!(self.0, "{}={:?} ", field.name(), value);
    }
}

#[cfg(feature = "server")]
impl tracing::Subscriber for CapturedEvents {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut text = EventText(format!("[{}] ", event.metadata().level()));
        event.record(&mut text);
        if let Ok(mut events) = self.events.lock() {
            events.push(text.0);
        }
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}
