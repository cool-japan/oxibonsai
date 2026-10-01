//! Metal encoder for the Qwen3.5 / PrismML Bonsai 2 hybrid stack
//! (`general.architecture = "qwen35"`; MET-09, design §4.2).
//!
//! # Why a separate encoder
//!
//! The dense Qwen3 encoder (`FullForwardLayerParams*`) is a flat set of
//! eight weight handles per layer with one layer kind and a KV cache sized
//! `n_layers × n_kv × max_seq × head_dim` per buffer. A `qwen35` stack
//! interleaves two structurally different layers — full attention on layer
//! `i` iff `(i + 1) % full_attention_interval == 0`, a recurrent Gated
//! DeltaNet on the others — so here:
//!
//! * every layer is a [`LayerKind`] (`FullAttention` / `LinearAttention`)
//!   and the encoder matches on it;
//! * the KV cache has one **slot** per full-attention layer, found through
//!   the dense index `layer_kv_slot: Vec<Option<u32>>` (27B: layer 3 → slot
//!   0, layer 7 → slot 1, …), and the recurrent state one slot per linear
//!   layer through `layer_rec_slot`. For the 27B at its 262 144-token
//!   context a layer-indexed cache would be 34.36 GB per buffer — above the
//!   M3's 14.30 GB `maxBufferLength` — while the 16-slot cache is 8.59 GB,
//!   so slot indexing is what makes the allocation possible at all.
//!
//! # Weight residency
//!
//! Matrices are bound in their on-disk block layout (see
//! `kernel_sources::qwen35`). When the caller hands over the page-aligned
//! file mapping the weights live in ([`Qwen35MappedRegion`]), the whole
//! mapping is wrapped in **one no-copy Metal buffer** and every matrix is
//! bound at its byte offset: a 7.2 GB model costs no extra memory and no
//! upload time. Otherwise each matrix is copied into its own device buffer.
//!
//! # Context capacity on a 24 GB M3
//!
//! [`Qwen35GpuModel::max_context`] derives the longest context a model fits
//! from the device's own limits, and construction refuses a longer
//! `max_seq_len` with a typed error. With the `f16` sparse KV cache a
//! Bonsai 2 27B position costs `16 slots × 4 kv-heads × 256 dims × 2 B × 2
//! (K, V) = 64 KiB`, one K (or V) buffer 32 KiB of it:
//!
//! * `maxBufferLength` (14 302 248 960 B on this M3): 436 470 positions per
//!   K or V buffer — not the binding limit;
//! * `recommendedMaxWorkingSetSize` (19 069 665 280 B) minus everything the
//!   device keeps resident besides the KV cache — the weights it reads
//!   (6 893 936 640 B for `PQ2_0`, 5 694 013 440 B for `PTQ1_0`: every
//!   matrix, the LM head and the widened `ssm_alpha`/`ssm_beta`; the token
//!   embedding stays a host-side lookup), the 149.6 MiB recurrent state, the
//!   logits and the activation scratch at `max_batch` tokens (274.2 MiB at
//!   the default 512-token prefill chunk) — over 65 888 B per position (the
//!   K + V cache, the rope angles and one score row): **178 034** positions
//!   for `PQ2_0` and **196 246** for `PTQ1_0`.
//!
//! That is the ceiling for a process in which the runner is the only large
//! resident. The GPU shares the machine's 24 GiB with the host, so the
//! practical window is what the runtime's RAM-derived budget (design A.3:
//! total RAM − model file − 25 % OS reserve − 256 MiB activations − the
//! recurrent state) leaves per position: 178 176 / 197 632 positions for
//! the CPU model alone (what `oxibonsai info` reports), ≈ 171 K / ≈ 190 K
//! for the runner alone, and — for a process that holds the CPU
//! `HybridModel` beside the runner, as the parity gates and a CPU fallback
//! do — two KV caches and two recurrent states: 131 680 B per position
//! (the CPU's 64 KiB KV + 256 B of rope angles, plus the runner's 65 888 B)
//! after both states, the runner's gates, logits and prefill scratch, i.e.
//! **83 968** positions for `PQ2_0` and **93 184** for `PTQ1_0` (each
//! floored to 1024). The runtime engine picks the working window from those
//! budgets and this ceiling (`oxibonsai_runtime::engine_hybrid_gpu`, reading
//! the same [`qwen35_footprint`]); this check only guarantees an allocation
//! never exceeds what the device can keep resident.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use metal::foreign_types::ForeignType;
use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use oxibonsai_core::quant_prism::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64};
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxibonsai_core::tensor::BlockQ1_0G128;

use crate::gpu_backend::kernel_sources::ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY;
use crate::gpu_backend::metal_graph::{alloc_buf, commit_and_wait, MetalGraph, MetalGraphError};

// ═══════════════════════════════════════════════════════════════════════════
// Host-side description of a model
// ═══════════════════════════════════════════════════════════════════════════

/// One weight matrix, `y = W x` with `W` of `rows × cols`, borrowed in the
/// exact block layout its GGUF tensor stores (row-major, `cols / 128`
/// super-blocks of 128 input elements per row).
#[derive(Debug, Clone, Copy)]
pub enum Qwen35MatrixData<'a> {
    /// PrismML `PQ2_0` (ggml 142), or the `d`-first reading of ggml 42 @ g128.
    Pq2_0(&'a [BlockPQ2_0]),
    /// PrismML `PTQ1_0` (ggml 143).
    Ptq1_0(&'a [BlockPTQ1_0]),
    /// Legacy OxiBonsai ternary (`qs` first, ggml 42 @ g128).
    Tq2_0G128(&'a [BlockTQ2_0_g128]),
    /// Mainline `Q2_0` at group 64.
    Q2_0G64(&'a [BlockQ2_0G64]),
    /// 1-bit `Q1_0_g128`.
    Q1_0G128(&'a [BlockQ1_0G128]),
    /// Unquantized, row-major `f32`.
    F32(&'a [f32]),
}

impl Qwen35MatrixData<'_> {
    /// Bytes one row of `cols` input elements occupies.
    fn row_bytes(&self, cols: usize) -> usize {
        let per_128 = match self {
            Self::Pq2_0(_) | Self::Tq2_0G128(_) => 34,
            Self::Ptq1_0(_) => 28,
            Self::Q2_0G64(_) => 36,
            Self::Q1_0G128(_) => 18,
            Self::F32(_) => 512,
        };
        cols / 128 * per_128
    }

    /// The matrix's bytes, exactly as stored.
    fn bytes(&self) -> &[u8] {
        // SAFETY: every block type is `#[repr(C)]` with its byte size
        // asserted in `oxibonsai-core` (34 / 28 / 34 / 18 / 18 bytes, no
        // padding) and `f32` has no invalid bit patterns, so viewing the
        // slice's storage as bytes of the same total length is valid for
        // reads; the view borrows the slice and cannot outlive it.
        unsafe {
            match self {
                Self::Pq2_0(b) => as_bytes(b),
                Self::Ptq1_0(b) => as_bytes(b),
                Self::Tq2_0G128(b) => as_bytes(b),
                Self::Q2_0G64(b) => as_bytes(b),
                Self::Q1_0G128(b) => as_bytes(b),
                Self::F32(v) => as_bytes(v),
            }
        }
    }

    /// Bytes the matrix occupies as stored — what binding it reads from the
    /// file mapping, or copies when it is not mapped.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes().len()
    }

    /// Whether the encoder always copies this format (unquantized `f32`
    /// rows are never bound in place, even from a mapping).
    #[must_use]
    pub fn always_copied(&self) -> bool {
        matches!(self, Self::F32(_))
    }

    /// The GEMV entry point for this format.
    fn kernel(&self) -> &'static str {
        match self {
            Self::Pq2_0(_) => "q35_gemv_pq2",
            Self::Ptq1_0(_) => "q35_gemv_ptq1",
            Self::Tq2_0G128(_) => "q35_gemv_tq2",
            Self::Q2_0G64(_) => "q35_gemv_q2g64",
            Self::Q1_0G128(_) => "q35_gemv_q1",
            Self::F32(_) => "q35_gemv_f32",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Pq2_0(_) => "PQ2_0",
            Self::Ptq1_0(_) => "PTQ1_0",
            Self::Tq2_0G128(_) => "TQ2_0_g128",
            Self::Q2_0G64(_) => "Q2_0_g64",
            Self::Q1_0G128(_) => "Q1_0_g128",
            Self::F32(_) => "F32",
        }
    }
}

/// View a slice of plain-old-data values as its bytes.
///
/// # Safety
///
/// `T` must have no padding bytes (every byte of its storage initialised).
unsafe fn as_bytes<T>(values: &[T]) -> &[u8] {
    std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
}

/// One weight matrix with its logical shape.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35Matrix<'a> {
    /// Output features (`W`'s rows).
    pub rows: usize,
    /// Input features (`W`'s columns); a multiple of 128.
    pub cols: usize,
    /// The stored blocks.
    pub data: Qwen35MatrixData<'a>,
}

/// A full-attention layer's weights.
#[derive(Debug, Clone)]
pub struct Qwen35FullAttentionWeights<'a> {
    /// Pre-attention RMSNorm, `[hidden]`.
    pub attn_norm: &'a [f32],
    /// Pre-FFN RMSNorm (GGUF `post_attention_norm`), `[hidden]`.
    pub post_attention_norm: &'a [f32],
    /// `[q | gate]` per head, `hidden → 2 · n_heads · head_dim`.
    pub attn_q: Qwen35Matrix<'a>,
    /// `hidden → n_kv_heads · head_dim`.
    pub attn_k: Qwen35Matrix<'a>,
    /// `hidden → n_kv_heads · head_dim`.
    pub attn_v: Qwen35Matrix<'a>,
    /// `n_heads · head_dim → hidden`.
    pub attn_output: Qwen35Matrix<'a>,
    /// Per-head query RMSNorm, `[head_dim]`.
    pub attn_q_norm: &'a [f32],
    /// Per-head key RMSNorm, `[head_dim]`.
    pub attn_k_norm: &'a [f32],
    /// SwiGLU gate projection.
    pub ffn_gate: Qwen35Matrix<'a>,
    /// SwiGLU up projection.
    pub ffn_up: Qwen35Matrix<'a>,
    /// SwiGLU down projection.
    pub ffn_down: Qwen35Matrix<'a>,
}

/// A Gated-DeltaNet (linear-attention) layer's weights.
#[derive(Debug, Clone)]
pub struct Qwen35LinearAttentionWeights<'a> {
    /// Pre-attention RMSNorm, `[hidden]`.
    pub attn_norm: &'a [f32],
    /// Pre-FFN RMSNorm, `[hidden]`.
    pub post_attention_norm: &'a [f32],
    /// `[q | k | v]` projection, `hidden → conv_dim` (v rows in tiled order).
    pub attn_qkv: Qwen35Matrix<'a>,
    /// Output gate `z`, `hidden → n_v_heads · head_v_dim` (tiled order).
    pub attn_gate: Qwen35Matrix<'a>,
    /// `ssm_alpha` rows then `ssm_beta` rows, `[2 · n_v_heads][hidden]`
    /// row-major `f32`, in the GGUF's tiled v-head order. Not folded: it
    /// consumes the un-rotated normed activation.
    pub ssm_alpha_beta: &'a [f32],
    /// Depthwise conv taps, `[conv_dim][4]`, oldest tap first.
    pub ssm_conv1d: &'a [f32],
    /// `ssm_a` (`-exp(A_log)`), `[n_v_heads]`, **grouped** v-head order.
    pub a_neg: &'a [f32],
    /// `ssm_dt.bias`, `[n_v_heads]`, **grouped** v-head order.
    pub dt_bias: &'a [f32],
    /// Gated-RMSNorm weight shared by every v-head, `[head_v_dim]`.
    pub ssm_norm: &'a [f32],
    /// `n_v_heads · head_v_dim → hidden`, columns in grouped v-head order.
    pub ssm_out: Qwen35Matrix<'a>,
    /// SwiGLU gate projection.
    pub ffn_gate: Qwen35Matrix<'a>,
    /// SwiGLU up projection.
    pub ffn_up: Qwen35Matrix<'a>,
    /// SwiGLU down projection.
    pub ffn_down: Qwen35Matrix<'a>,
}

/// One layer's weights, by kind.
#[derive(Debug, Clone)]
pub enum Qwen35LayerWeights<'a> {
    /// Full attention with a `q | gate` interleave and partial RoPE.
    FullAttention(Box<Qwen35FullAttentionWeights<'a>>),
    /// Gated DeltaNet.
    LinearAttention(Box<Qwen35LinearAttentionWeights<'a>>),
}

/// Geometry and hyper-parameters of a `qwen35` stack.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35GpuConfig {
    /// Residual width (27B: 5120).
    pub hidden: usize,
    /// FFN width (27B: 17408).
    pub intermediate: usize,
    /// Query heads (27B: 24).
    pub n_heads: usize,
    /// Key/value heads (27B: 4).
    pub n_kv_heads: usize,
    /// Attention head width (27B: 256).
    pub head_dim: usize,
    /// Rotated dims per head (27B: 64).
    pub n_rot: usize,
    /// Gated-DeltaNet k/q heads (27B: 16).
    pub n_k_heads: usize,
    /// Gated-DeltaNet v heads (27B: 48).
    pub n_v_heads: usize,
    /// Key/query channels per GDN head (27B: 128).
    pub head_k_dim: usize,
    /// Value channels per GDN head (27B: 128).
    pub head_v_dim: usize,
    /// Depthwise conv taps (must be 4).
    pub conv_kernel: usize,
    /// RMSNorm / L2-norm epsilon.
    pub rms_eps: f32,
    /// `prism.hadamard.block_size` when the checkpoint is folded.
    pub hadamard_block: Option<usize>,
    /// Vocabulary (LM head rows).
    pub vocab: usize,
    /// KV window, in positions.
    pub max_seq_len: usize,
    /// Largest token count one [`Qwen35GpuModel::forward`] call accepts.
    pub max_batch: usize,
}

impl Qwen35GpuConfig {
    /// `[q | k | v]` width of a linear layer's projection.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.n_k_heads * self.head_k_dim + self.n_v_heads * self.head_v_dim
    }

    /// `n_v_heads · head_v_dim`.
    #[must_use]
    pub fn inner(&self) -> usize {
        self.n_v_heads * self.head_v_dim
    }

    /// `n_heads · head_dim`.
    #[must_use]
    pub fn heads_width(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Reject a geometry the kernels cannot serve, naming the constraint.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for the first violated
    /// constraint.
    pub fn validate(&self) -> Result<(), MetalGraphError> {
        let bad = |what: String| {
            Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {what}"
            )))
        };
        let widths = [
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("n_heads * head_dim", self.heads_width()),
            ("n_v_heads * head_v_dim", self.inner()),
        ];
        for (name, width) in widths {
            if width == 0 || !width.is_multiple_of(128) {
                return bad(format!(
                    "{name} = {width} must be a non-zero multiple of 128"
                ));
            }
        }
        if self.hidden > MAX_NORM_ROW {
            return bad(format!(
                "hidden = {} exceeds the single-threadgroup RMSNorm row ({MAX_NORM_ROW})",
                self.hidden
            ));
        }
        if self.head_dim == 0 || self.head_dim > ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY as usize {
            return bad(format!(
                "head_dim = {} must be in 1..={ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY}",
                self.head_dim
            ));
        }
        if self.n_rot == 0 || !self.n_rot.is_multiple_of(2) || self.n_rot > self.head_dim {
            return bad(format!(
                "n_rot = {} must be even and in 2..=head_dim ({})",
                self.n_rot, self.head_dim
            ));
        }
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return bad(format!(
                "n_heads = {} must be a multiple of n_kv_heads = {}",
                self.n_heads, self.n_kv_heads
            ));
        }
        if self.n_k_heads == 0 || !self.n_v_heads.is_multiple_of(self.n_k_heads) {
            return bad(format!(
                "n_v_heads = {} must be a multiple of n_k_heads = {}",
                self.n_v_heads, self.n_k_heads
            ));
        }
        if self.head_k_dim == 0 || self.head_k_dim > MAX_GDN_HEAD {
            return bad(format!(
                "head_k_dim = {} must be in 1..={MAX_GDN_HEAD}",
                self.head_k_dim
            ));
        }
        if self.head_v_dim == 0 || !self.head_v_dim.is_multiple_of(32) || self.head_v_dim > MAX_SPAN
        {
            return bad(format!(
                "head_v_dim = {} must be a multiple of 32 no larger than {MAX_SPAN}",
                self.head_v_dim
            ));
        }
        // A square Gated-DeltaNet state, as the reference implementation
        // requires (`S_k == S_v`): the output scale `1/sqrt(head_v_dim)` is
        // then also its `1/sqrt(S_k)`.
        if self.head_k_dim != self.head_v_dim {
            return bad(format!(
                "head_k_dim = {} must equal head_v_dim = {} (a square Gated-DeltaNet state)",
                self.head_k_dim, self.head_v_dim
            ));
        }
        if self.conv_kernel != 4 {
            return bad(format!(
                "ssm.conv_kernel = {} (the GPU conv is the 4-tap kernel)",
                self.conv_kernel
            ));
        }
        if let Some(block) = self.hadamard_block {
            if !block.is_power_of_two() || !(2..=MAX_SPAN).contains(&block) {
                return bad(format!(
                    "Hadamard block {block} must be a power of two in 2..={MAX_SPAN}"
                ));
            }
            for (name, width) in widths {
                if !width.is_multiple_of(block) {
                    return bad(format!(
                        "{name} = {width} is not a whole number of Hadamard blocks ({block})"
                    ));
                }
            }
            if !(block.is_multiple_of(self.head_v_dim) || self.head_v_dim.is_multiple_of(block)) {
                return bad(format!(
                    "Hadamard block {block} and head_v_dim {} must divide one another",
                    self.head_v_dim
                ));
            }
        }
        // `q35_gated_norm_rotate` sums each v-head over the simdgroups of one
        // threadgroup of `gated_norm_span` threads, so a span must hold whole
        // heads (else a head straddles two threadgroups and its sum reads
        // partial slots no simdgroup wrote) and the spans must tile the row.
        let span = self.gated_norm_span();
        if !span.is_multiple_of(self.head_v_dim) || !self.inner().is_multiple_of(span) {
            return bad(format!(
                "the gated-norm span {span} must be a whole number of v-heads (head_v_dim {}) \
                 and divide n_v_heads * head_v_dim = {}",
                self.head_v_dim,
                self.inner()
            ));
        }
        if self.vocab == 0 || self.max_seq_len == 0 || self.max_batch == 0 {
            return bad("vocab, max_seq_len and max_batch must be non-zero".to_string());
        }
        if u32::try_from(self.max_seq_len).is_err() {
            return bad(format!(
                "max_seq_len = {} does not fit u32",
                self.max_seq_len
            ));
        }
        Ok(())
    }

    /// Span (threads, and elements rotated together) of the rotating
    /// producers for one width.
    fn span(&self) -> usize {
        self.hadamard_block.unwrap_or(UNFOLDED_SPAN)
    }

    /// Span of the gated norm: a whole number of v-heads and of Hadamard
    /// blocks.
    fn gated_norm_span(&self) -> usize {
        self.span().max(self.head_v_dim)
    }
}

/// Largest RMSNorm row (`Q35_MAX_ROW` in the MSL).
const MAX_NORM_ROW: usize = 6144;
/// Largest rotated span (`Q35_MAX_SPAN` in the MSL).
const MAX_SPAN: usize = 1024;
/// Largest Gated-DeltaNet key head (`Q35_GDN_MAX_HEAD` in the MSL).
const MAX_GDN_HEAD: usize = 128;
/// Span of the producers when the checkpoint is not folded.
const UNFOLDED_SPAN: usize = 128;
/// Rows per GEMV threadgroup (`Q35_GEMV_NSG × Q35_GEMV_R` in the MSL).
const GEMV_ROWS_PER_TG: usize = 8;
/// Threads per GEMV threadgroup.
const GEMV_THREADS: u64 = 128;
/// Threads of the Gated-DeltaNet kernel (8 simdgroups).
const GDN_THREADS: u64 = 256;
/// Positions each `batched_attention_scores_v2` threadgroup scores.
const SCORES_BATCH_STRIDE: u32 = 16;
/// Threads of the single-threadgroup RMSNorm.
const NORM_THREADS: u64 = 1024;

/// Everything [`Qwen35GpuModel::new`] needs, borrowed for the call.
#[derive(Debug, Clone)]
pub struct Qwen35ModelWeights<'a> {
    /// Geometry.
    pub config: Qwen35GpuConfig,
    /// Every layer, in stack order.
    pub layers: Vec<Qwen35LayerWeights<'a>>,
    /// Final RMSNorm, `[hidden]`.
    pub output_norm: &'a [f32],
    /// LM head, `hidden → vocab` (folded when the checkpoint is).
    pub lm_head: Qwen35Matrix<'a>,
    /// `(width, signs)` for every rotated width (hidden, heads width, inner,
    /// intermediate); ignored when `hadamard_block` is `None`.
    pub signs: Vec<(usize, &'a [f32])>,
    /// Partial-RoPE angles, `[max_seq_len][n_rot / 2]` each.
    pub rope_cos: &'a [f32],
    /// See [`Self::rope_cos`].
    pub rope_sin: &'a [f32],
}

/// A page-aligned, live memory mapping that the model's weight blocks point
/// into — the whole GGUF file mapped read-only.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MappedRegion<'m> {
    bytes: &'m [u8],
}

impl<'m> Qwen35MappedRegion<'m> {
    /// Wrap a file mapping.
    ///
    /// # Safety
    ///
    /// `bytes` must be the start of a memory mapping (so `bytes.as_ptr()` is
    /// page-aligned) whose pages up to `bytes.len()` rounded up to the page
    /// size stay mapped for `'m` — which a `memmap2::Mmap` of the whole file
    /// guarantees for as long as it is borrowed. The GPU reads the pages
    /// directly; nothing is copied.
    #[must_use]
    pub unsafe fn from_mapping(bytes: &'m [u8]) -> Self {
        Self { bytes }
    }

    /// Wrap `bytes` for zero-copy binding when it starts on a host page
    /// boundary; `None` for an empty or unaligned slice (the caller then
    /// copies the weights instead).
    ///
    /// No caller contract is needed here, unlike [`Self::from_mapping`]:
    /// memory is mapped and protected in whole pages, so every page from the
    /// first byte of a live, readable slice through the page holding its
    /// last byte is mapped and readable for as long as the slice is
    /// borrowed. For a slice that starts on a page boundary those pages are
    /// exactly `bytes.len()` rounded up to the page size — the span the
    /// no-copy buffer covers — whether the slice is a whole-file mapping
    /// (always page-aligned) or an allocation that happens to be.
    #[must_use]
    pub fn page_aligned(bytes: &'m [u8]) -> Option<Self> {
        let aligned = (bytes.as_ptr() as usize).is_multiple_of(host_page_size());
        (!bytes.is_empty() && aligned).then_some(Self { bytes })
    }

    /// Bytes the region spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the region is empty (never true for one built by
    /// [`Self::page_aligned`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn contains(&self, inner: &[u8]) -> Option<u64> {
        let base = self.bytes.as_ptr() as usize;
        let start = inner.as_ptr() as usize;
        let end = start.checked_add(inner.len())?;
        if start >= base && end <= base + self.bytes.len() {
            Some((start - base) as u64)
        } else {
            None
        }
    }
}

/// How a model's weight blocks reach the GPU.
#[derive(Debug, Clone, Copy)]
pub enum Qwen35Residency<'m> {
    /// Copy every matrix into its own device buffer.
    Copied,
    /// Alias the file mapping zero-copy; matrices outside it are copied.
    Mapped(Qwen35MappedRegion<'m>),
}

// ═══════════════════════════════════════════════════════════════════════════
// Device-side model
// ═══════════════════════════════════════════════════════════════════════════

/// One matrix bound on the device: its buffer (the shared mapping or its own
/// copy), the byte offset of its first block, its shape and its kernel.
#[derive(Clone)]
struct DeviceMatrix {
    buffer: Buffer,
    offset: u64,
    rows: usize,
    cols: usize,
    pso: ComputePipelineState,
}

struct DeviceFullLayer {
    attn_norm: Buffer,
    post_attention_norm: Buffer,
    attn_q: DeviceMatrix,
    attn_k: DeviceMatrix,
    attn_v: DeviceMatrix,
    attn_output: DeviceMatrix,
    attn_q_norm: Buffer,
    attn_k_norm: Buffer,
    ffn: DeviceFfn,
}

struct DeviceLinearLayer {
    attn_norm: Buffer,
    post_attention_norm: Buffer,
    attn_qkv: DeviceMatrix,
    attn_gate: DeviceMatrix,
    ssm_alpha_beta: DeviceMatrix,
    ssm_conv1d: Buffer,
    a_neg: Buffer,
    dt_bias: Buffer,
    ssm_norm: Buffer,
    ssm_out: DeviceMatrix,
    ffn: DeviceFfn,
}

struct DeviceFfn {
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
}

/// A layer on the device, by kind — the discriminant the encoder matches on.
enum LayerKind {
    FullAttention(Box<DeviceFullLayer>),
    LinearAttention(Box<DeviceLinearLayer>),
}

/// Pipelines of the combined metallib this encoder dispatches.
struct Pipelines {
    rmsnorm_rotate: ComputePipelineState,
    swiglu_rotate: ComputePipelineState,
    sigmoid_gate_rotate: ComputePipelineState,
    gated_norm_rotate: ComputePipelineState,
    fwht_signed: ComputePipelineState,
    conv1d_silu: ComputePipelineState,
    gdn: ComputePipelineState,
    kv_store: ComputePipelineState,
    qk_norm_rope: ComputePipelineState,
    scores: ComputePipelineState,
    softmax: ComputePipelineState,
    weighted_sum: ComputePipelineState,
}

/// Activation buffers for up to `capacity` tokens.
struct Scratch {
    capacity: usize,
    resid: Buffer,
    normed: Buffer,
    rotated: Buffer,
    qkv: Buffer,
    conv_out: Buffer,
    z: Buffer,
    ab: Buffer,
    gdn_out: Buffer,
    gdn_rot: Buffer,
    q_all: Buffer,
    k: Buffer,
    v: Buffer,
    q_rope: Buffer,
    k_rope: Buffer,
    attn: Buffer,
    attn_rot: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    ffn_act: Buffer,
}

impl Scratch {
    /// Floats one token occupies across every buffer [`Self::allocate`]
    /// creates — the per-token cost [`Qwen35GpuModel::max_context`] charges
    /// for `max_batch` tokens.
    fn floats_per_token(cfg: &Qwen35GpuConfig) -> usize {
        let kv = cfg.n_kv_heads * cfg.head_dim;
        3 * cfg.hidden // resid, normed, rotated
            + 2 * cfg.conv_dim() // qkv, conv_out
            + 3 * cfg.inner() // z, gdn_out, gdn_rot
            + 2 * cfg.n_v_heads // ab
            + 2 * cfg.heads_width() // q_all
            + 3 * kv // k, v, k_rope
            + 3 * cfg.heads_width() // q_rope, attn, attn_rot
            + 3 * cfg.intermediate // ffn_gate, ffn_up, ffn_act
    }

    fn allocate(
        graph: &MetalGraph,
        cfg: &Qwen35GpuConfig,
        capacity: usize,
    ) -> Result<Self, MetalGraphError> {
        let f = |floats: usize| -> Result<Buffer, MetalGraphError> {
            alloc_buf(
                &graph.device,
                (floats.max(1) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let t = capacity;
        Ok(Self {
            capacity,
            resid: f(t * cfg.hidden)?,
            normed: f(t * cfg.hidden)?,
            rotated: f(t * cfg.hidden)?,
            qkv: f(t * cfg.conv_dim())?,
            conv_out: f(t * cfg.conv_dim())?,
            z: f(t * cfg.inner())?,
            ab: f(t * 2 * cfg.n_v_heads)?,
            gdn_out: f(t * cfg.inner())?,
            gdn_rot: f(t * cfg.inner())?,
            q_all: f(t * 2 * cfg.heads_width())?,
            k: f(t * cfg.n_kv_heads * cfg.head_dim)?,
            v: f(t * cfg.n_kv_heads * cfg.head_dim)?,
            q_rope: f(t * cfg.heads_width())?,
            k_rope: f(t * cfg.n_kv_heads * cfg.head_dim)?,
            attn: f(t * cfg.heads_width())?,
            attn_rot: f(t * cfg.heads_width())?,
            ffn_gate: f(t * cfg.intermediate)?,
            ffn_up: f(t * cfg.intermediate)?,
            ffn_act: f(t * cfg.intermediate)?,
        })
    }
}

/// Named intermediate activations of one layer, captured stage by stage by
/// [`Qwen35GpuModel::trace_layer`] (each `[t_len][width]`).
#[derive(Debug, Clone, Default)]
pub struct Qwen35LayerTrace {
    /// `(stage name, values)` in execution order.
    pub stages: Vec<(&'static str, Vec<f32>)>,
}

impl Qwen35LayerTrace {
    /// The values of `stage`, if it was captured.
    #[must_use]
    pub fn stage(&self, stage: &str) -> Option<&[f32]> {
        self.stages
            .iter()
            .find(|(name, _)| *name == stage)
            .map(|(_, v)| v.as_slice())
    }
}

/// A `qwen35` hybrid model resident on the GPU: weights (mapped or copied),
/// the sparse KV cache, the recurrent state and the activation scratch,
/// driven through its own Metal session.
///
/// `'m` is the lifetime of the file mapping the weights may alias (see
/// [`Qwen35Residency::Mapped`]).
pub struct Qwen35GpuModel<'m> {
    graph: Arc<MetalGraph>,
    cfg: Qwen35GpuConfig,
    pipes: Pipelines,
    layers: Vec<LayerKind>,
    layer_kv_slot: Vec<Option<u32>>,
    layer_rec_slot: Vec<Option<u32>>,
    output_norm: Buffer,
    lm_head: DeviceMatrix,
    signs: BTreeMap<usize, Buffer>,
    rope_cos: Buffer,
    rope_sin: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,
    ssm_state: Buffer,
    conv_state: Buffer,
    scores: Buffer,
    logits: Buffer,
    scratch: Scratch,
    weight_bytes: u64,
    mapped: bool,
    last_gpu_seconds: f64,
    _mapping: PhantomData<&'m [u8]>,
}

// Every Metal object type the `metal` crate exposes is declared `Sync + Send`
// (its `foreign_obj_type!` macro), and the remaining fields are plain data, so
// the compiler derives `Send` for the model on its own — no `unsafe impl` is
// needed. This assertion keeps that property from silently disappearing if a
// non-`Send` field is ever added (the hybrid engine seam moves the model
// between threads).
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Qwen35GpuModel<'static>>();
};

impl std::fmt::Debug for Qwen35GpuModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35GpuModel")
            .field("config", &self.cfg)
            .field("layers", &self.layers.len())
            .field("layer_kv_slot", &self.layer_kv_slot)
            .field("weight_bytes", &self.weight_bytes)
            .field("mapped", &self.mapped)
            .finish()
    }
}

/// The page size of the host's virtual memory (16 KiB on Apple silicon).
const fn host_page_size() -> usize {
    if cfg!(target_arch = "aarch64") {
        16_384
    } else {
        4_096
    }
}

/// Wrap `region` (a page-aligned mapping) in one no-copy shared buffer.
fn no_copy_buffer(
    graph: &MetalGraph,
    region: &Qwen35MappedRegion<'_>,
) -> Result<Buffer, MetalGraphError> {
    let page = host_page_size();
    let base = region.bytes.as_ptr();
    if !(base as usize).is_multiple_of(page) {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "qwen35 GPU: mapped region base {base:p} is not {page}-byte page aligned"
        )));
    }
    let length = region.bytes.len().div_ceil(page) * page;
    if length == 0 || length as u64 > graph.device.max_buffer_length() {
        return Err(MetalGraphError::BufferTooLarge {
            what: "qwen35 mapped weights",
            requested: length as u64,
            max: graph.device.max_buffer_length(),
        });
    }
    let options = MTLResourceOptions::StorageModeShared;
    // SAFETY: `base` is page-aligned and `[base, base + length)` is mapped
    // for the region's lifetime (the `Qwen35MappedRegion::from_mapping`
    // contract); the GPU only reads it. The deallocator is nil, so Metal
    // never frees the mapping. A nil return is checked before it is wrapped.
    let raw: *mut metal::MTLBuffer = unsafe {
        msg_send![&*graph.device, newBufferWithBytesNoCopy: base.cast::<std::ffi::c_void>()
                                  length: length as u64
                                  options: options
                                  deallocator: std::ptr::null::<std::ffi::c_void>()]
    };
    if raw.is_null() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    // SAFETY: `raw` is a non-null, +1-retained MTLBuffer returned by a `new…`
    // method, which `Buffer` takes ownership of.
    Ok(unsafe { Buffer::from_ptr(raw) })
}

fn upload_f32(graph: &MetalGraph, data: &[f32]) -> Result<Buffer, MetalGraphError> {
    // SAFETY: `f32` has no padding; the byte view borrows `data`.
    upload_bytes(graph, unsafe { as_bytes(data) })
}

fn upload_bytes(graph: &MetalGraph, data: &[u8]) -> Result<Buffer, MetalGraphError> {
    let buf = alloc_buf(
        &graph.device,
        data.len().max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )?;
    // SAFETY: `alloc_buf` returned a shared buffer of at least `data.len()`
    // bytes with non-null contents; the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents().cast::<u8>(), data.len());
    }
    Ok(buf)
}

fn zeroed(graph: &MetalGraph, bytes: usize) -> Result<Buffer, MetalGraphError> {
    let buf = alloc_buf(
        &graph.device,
        bytes.max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )?;
    // SAFETY: shared buffer of at least `bytes` bytes, not yet used by the GPU.
    unsafe { std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, bytes.max(4)) };
    Ok(buf)
}

fn check_len(what: &str, got: usize, want: usize) -> Result<(), MetalGraphError> {
    if got == want {
        Ok(())
    } else {
        Err(MetalGraphError::InvalidDimensions(format!(
            "qwen35 GPU: {what} has {got} elements, expected {want}"
        )))
    }
}

/// Builds device matrices, sharing one mapped buffer where possible.
struct Binder<'g, 'm> {
    graph: &'g MetalGraph,
    region: Option<(Qwen35MappedRegion<'m>, Buffer)>,
    copied_bytes: u64,
    mapped_bytes: u64,
}

impl Binder<'_, '_> {
    fn bind(
        &mut self,
        what: &str,
        m: &Qwen35Matrix<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceMatrix, MetalGraphError> {
        if m.rows != rows || m.cols != cols {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {what} is {}x{}, expected {rows}x{cols}",
                m.rows, m.cols
            )));
        }
        if !cols.is_multiple_of(128) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {what} has {cols} input features, not a multiple of 128"
            )));
        }
        let bytes = m.data.bytes();
        let want = rows * m.data.row_bytes(cols);
        if bytes.len() != want {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {what} ({}) holds {} bytes, expected {want}",
                m.data.name(),
                bytes.len()
            )));
        }
        let pso = self.graph.pipeline_for(m.data.kernel())?;
        let mapped = match (&self.region, &m.data) {
            (Some((region, buffer)), data) if !data.always_copied() => region
                .contains(bytes)
                .filter(|offset| offset % 4 == 0)
                .map(|offset| (buffer.clone(), offset)),
            _ => None,
        };
        let (buffer, offset) = match mapped {
            Some(found) => {
                self.mapped_bytes += bytes.len() as u64;
                found
            }
            None => {
                self.copied_bytes += bytes.len() as u64;
                (upload_bytes(self.graph, bytes)?, 0)
            }
        };
        Ok(DeviceMatrix {
            buffer,
            offset,
            rows,
            cols,
            pso,
        })
    }

    fn norm(&mut self, what: &str, w: &[f32], len: usize) -> Result<Buffer, MetalGraphError> {
        check_len(what, w.len(), len)?;
        upload_f32(self.graph, w)
    }

    fn ffn(
        &mut self,
        layer: usize,
        gate: &Qwen35Matrix<'_>,
        up: &Qwen35Matrix<'_>,
        down: &Qwen35Matrix<'_>,
        cfg: &Qwen35GpuConfig,
    ) -> Result<DeviceFfn, MetalGraphError> {
        Ok(DeviceFfn {
            gate: self.bind(
                &format!("blk.{layer}.ffn_gate"),
                gate,
                cfg.intermediate,
                cfg.hidden,
            )?,
            up: self.bind(
                &format!("blk.{layer}.ffn_up"),
                up,
                cfg.intermediate,
                cfg.hidden,
            )?,
            down: self.bind(
                &format!("blk.{layer}.ffn_down"),
                down,
                cfg.hidden,
                cfg.intermediate,
            )?,
        })
    }
}

impl<'m> Qwen35GpuModel<'m> {
    /// Upload (or map) a model and allocate its caches on a new Metal
    /// session of the process-shared device.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a geometry the kernels
    /// cannot serve ([`Qwen35GpuConfig::validate`]), a weight whose shape or
    /// byte length disagrees with the config, or a `max_seq_len` past
    /// [`Self::max_context`]; [`MetalGraphError::DeviceNotFound`] without a
    /// Metal device; allocation errors from the device.
    pub fn new(
        weights: &Qwen35ModelWeights<'_>,
        residency: Qwen35Residency<'m>,
    ) -> Result<Self, MetalGraphError> {
        let cfg = weights.config.clone();
        cfg.validate()?;
        let graph = MetalGraph::new_session()?;

        let n_layers = weights.layers.len();
        let mut layer_kv_slot = Vec::with_capacity(n_layers);
        let mut layer_rec_slot = Vec::with_capacity(n_layers);
        let (mut n_full, mut n_linear) = (0u32, 0u32);
        for layer in &weights.layers {
            match layer {
                Qwen35LayerWeights::FullAttention(_) => {
                    layer_kv_slot.push(Some(n_full));
                    layer_rec_slot.push(None);
                    n_full += 1;
                }
                Qwen35LayerWeights::LinearAttention(_) => {
                    layer_kv_slot.push(None);
                    layer_rec_slot.push(Some(n_linear));
                    n_linear += 1;
                }
            }
        }

        let (region, mapped) = match residency {
            Qwen35Residency::Copied => (None, false),
            Qwen35Residency::Mapped(region) => {
                let buffer = no_copy_buffer(&graph, &region)?;
                (Some((region, buffer)), true)
            }
        };
        let mut binder = Binder {
            graph: &graph,
            region,
            copied_bytes: 0,
            mapped_bytes: 0,
        };

        let mut layers = Vec::with_capacity(n_layers);
        for (i, layer) in weights.layers.iter().enumerate() {
            layers.push(match layer {
                Qwen35LayerWeights::FullAttention(w) => {
                    LayerKind::FullAttention(Box::new(Self::bind_full(&mut binder, i, w, &cfg)?))
                }
                Qwen35LayerWeights::LinearAttention(w) => LayerKind::LinearAttention(Box::new(
                    Self::bind_linear(&mut binder, i, w, &cfg)?,
                )),
            });
        }
        let output_norm = binder.norm("output_norm", weights.output_norm, cfg.hidden)?;
        let lm_head = binder.bind("output", &weights.lm_head, cfg.vocab, cfg.hidden)?;

        let mut signs = BTreeMap::new();
        if cfg.hadamard_block.is_some() {
            let mut widths = vec![cfg.hidden, cfg.intermediate];
            if n_full > 0 {
                widths.push(cfg.heads_width());
            }
            if n_linear > 0 {
                widths.push(cfg.inner());
            }
            for width in widths {
                let values = weights
                    .signs
                    .iter()
                    .find(|(w, _)| *w == width)
                    .map(|(_, s)| *s)
                    .ok_or_else(|| {
                        MetalGraphError::InvalidDimensions(format!(
                            "qwen35 GPU: no Hadamard sign vector for width {width}"
                        ))
                    })?;
                check_len("signs", values.len(), width)?;
                signs.entry(width).or_insert(upload_f32(&graph, values)?);
            }
        }

        let half_rot = cfg.n_rot / 2;
        check_len(
            "rope_cos",
            weights.rope_cos.len(),
            cfg.max_seq_len * half_rot,
        )?;
        check_len(
            "rope_sin",
            weights.rope_sin.len(),
            cfg.max_seq_len * half_rot,
        )?;
        let rope_cos = upload_f32(&graph, weights.rope_cos)?;
        let rope_sin = upload_f32(&graph, weights.rope_sin)?;

        let weight_bytes = binder.copied_bytes + binder.mapped_bytes;
        let cap = Self::max_context(
            &cfg,
            n_full as usize,
            n_linear as usize,
            weight_bytes,
            &graph,
        );
        if cfg.max_seq_len > cap {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: max_seq_len {} exceeds this device's capacity of {cap} positions \
                 (maxBufferLength {} B, recommendedMaxWorkingSetSize {} B, {} B of weights)",
                cfg.max_seq_len,
                graph.device.max_buffer_length(),
                graph.device.recommended_max_working_set_size(),
                weight_bytes
            )));
        }

        let kv_elems = n_full.max(1) as usize * cfg.n_kv_heads * cfg.max_seq_len * cfg.head_dim;
        let k_cache = alloc_buf(
            &graph.device,
            (kv_elems * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )?;
        let v_cache = alloc_buf(
            &graph.device,
            (kv_elems * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )?;
        let state_elems =
            n_linear.max(1) as usize * cfg.n_v_heads * cfg.head_v_dim * cfg.head_k_dim;
        let ssm_state = zeroed(&graph, state_elems * 4)?;
        let conv_state = zeroed(&graph, n_linear.max(1) as usize * cfg.conv_dim() * 3 * 4)?;
        let scores = zeroed(&graph, cfg.n_heads * cfg.max_seq_len * 4)?;
        let logits = zeroed(&graph, cfg.vocab * 4)?;
        let scratch = Scratch::allocate(&graph, &cfg, 1)?;

        let pipes = Pipelines {
            rmsnorm_rotate: graph.pipeline_for("q35_rmsnorm_rotate")?,
            swiglu_rotate: graph.pipeline_for("q35_swiglu_rotate")?,
            sigmoid_gate_rotate: graph.pipeline_for("q35_sigmoid_gate_rotate")?,
            gated_norm_rotate: graph.pipeline_for("q35_gated_norm_rotate")?,
            fwht_signed: graph.pipeline_for("q35_fwht_signed")?,
            conv1d_silu: graph.pipeline_for("q35_conv1d_silu")?,
            gdn: graph.pipeline_for("q35_gdn")?,
            kv_store: graph.pipeline_for("q35_kv_store")?,
            qk_norm_rope: graph.pipeline_for("fused_qk_norm_rope_partial")?,
            scores: graph.pipelines.batched_attention_scores_v2.clone(),
            softmax: graph.pipelines.batched_softmax.clone(),
            weighted_sum: graph.pipelines.batched_attention_weighted_sum.clone(),
        };
        let span = cfg.span() as u64;
        for (name, pso, threads) in [
            ("q35_rmsnorm_rotate", &pipes.rmsnorm_rotate, NORM_THREADS),
            ("q35_swiglu_rotate", &pipes.swiglu_rotate, span),
            ("q35_sigmoid_gate_rotate", &pipes.sigmoid_gate_rotate, span),
            (
                "q35_gated_norm_rotate",
                &pipes.gated_norm_rotate,
                cfg.gated_norm_span() as u64,
            ),
            ("q35_fwht_signed", &pipes.fwht_signed, span),
            ("q35_gdn", &pipes.gdn, GDN_THREADS),
            ("fused_qk_norm_rope_partial", &pipes.qk_norm_rope, 256),
        ] {
            if pso.max_total_threads_per_threadgroup() < threads {
                return Err(MetalGraphError::InvalidDimensions(format!(
                    "qwen35 GPU: {name} needs {threads} threads per threadgroup, this device \
                     allows {}",
                    pso.max_total_threads_per_threadgroup()
                )));
            }
        }

        Ok(Self {
            graph,
            cfg,
            pipes,
            layers,
            layer_kv_slot,
            layer_rec_slot,
            output_norm,
            lm_head,
            signs,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            ssm_state,
            conv_state,
            scores,
            logits,
            scratch,
            weight_bytes,
            mapped,
            last_gpu_seconds: 0.0,
            _mapping: PhantomData,
        })
    }

    fn bind_full(
        b: &mut Binder<'_, '_>,
        i: usize,
        w: &Qwen35FullAttentionWeights<'_>,
        cfg: &Qwen35GpuConfig,
    ) -> Result<DeviceFullLayer, MetalGraphError> {
        let kv = cfg.n_kv_heads * cfg.head_dim;
        Ok(DeviceFullLayer {
            attn_norm: b.norm("attn_norm", w.attn_norm, cfg.hidden)?,
            post_attention_norm: b.norm(
                "post_attention_norm",
                w.post_attention_norm,
                cfg.hidden,
            )?,
            attn_q: b.bind(
                &format!("blk.{i}.attn_q"),
                &w.attn_q,
                2 * cfg.heads_width(),
                cfg.hidden,
            )?,
            attn_k: b.bind(&format!("blk.{i}.attn_k"), &w.attn_k, kv, cfg.hidden)?,
            attn_v: b.bind(&format!("blk.{i}.attn_v"), &w.attn_v, kv, cfg.hidden)?,
            attn_output: b.bind(
                &format!("blk.{i}.attn_output"),
                &w.attn_output,
                cfg.hidden,
                cfg.heads_width(),
            )?,
            attn_q_norm: b.norm("attn_q_norm", w.attn_q_norm, cfg.head_dim)?,
            attn_k_norm: b.norm("attn_k_norm", w.attn_k_norm, cfg.head_dim)?,
            ffn: b.ffn(i, &w.ffn_gate, &w.ffn_up, &w.ffn_down, cfg)?,
        })
    }

    fn bind_linear(
        b: &mut Binder<'_, '_>,
        i: usize,
        w: &Qwen35LinearAttentionWeights<'_>,
        cfg: &Qwen35GpuConfig,
    ) -> Result<DeviceLinearLayer, MetalGraphError> {
        let ab = Qwen35Matrix {
            rows: 2 * cfg.n_v_heads,
            cols: cfg.hidden,
            data: Qwen35MatrixData::F32(w.ssm_alpha_beta),
        };
        check_len(
            "ssm_conv1d",
            w.ssm_conv1d.len(),
            cfg.conv_dim() * cfg.conv_kernel,
        )?;
        Ok(DeviceLinearLayer {
            attn_norm: b.norm("attn_norm", w.attn_norm, cfg.hidden)?,
            post_attention_norm: b.norm(
                "post_attention_norm",
                w.post_attention_norm,
                cfg.hidden,
            )?,
            attn_qkv: b.bind(
                &format!("blk.{i}.attn_qkv"),
                &w.attn_qkv,
                cfg.conv_dim(),
                cfg.hidden,
            )?,
            attn_gate: b.bind(
                &format!("blk.{i}.attn_gate"),
                &w.attn_gate,
                cfg.inner(),
                cfg.hidden,
            )?,
            ssm_alpha_beta: b.bind(
                &format!("blk.{i}.ssm_alpha|ssm_beta"),
                &ab,
                2 * cfg.n_v_heads,
                cfg.hidden,
            )?,
            ssm_conv1d: upload_f32(b.graph, w.ssm_conv1d)?,
            a_neg: b.norm("ssm_a", w.a_neg, cfg.n_v_heads)?,
            dt_bias: b.norm("ssm_dt.bias", w.dt_bias, cfg.n_v_heads)?,
            ssm_norm: b.norm("ssm_norm", w.ssm_norm, cfg.head_v_dim)?,
            ssm_out: b.bind(
                &format!("blk.{i}.ssm_out"),
                &w.ssm_out,
                cfg.hidden,
                cfg.inner(),
            )?,
            ffn: b.ffn(i, &w.ffn_gate, &w.ffn_up, &w.ffn_down, cfg)?,
        })
    }

    /// The longest KV window this device can hold for a model of this
    /// geometry and weight size (see the module docs for the 27B numbers):
    /// [`qwen35_context_capacity`] with this device's `maxBufferLength` and
    /// `recommendedMaxWorkingSetSize`.
    #[must_use]
    pub fn max_context(
        cfg: &Qwen35GpuConfig,
        n_full: usize,
        n_linear: usize,
        weight_bytes: u64,
        graph: &MetalGraph,
    ) -> usize {
        let device = &graph.device;
        qwen35_context_capacity(
            cfg,
            n_full,
            n_linear,
            weight_bytes,
            device.max_buffer_length(),
            device.recommended_max_working_set_size(),
        )
    }

    /// The geometry this model was built with.
    #[must_use]
    pub fn config(&self) -> &Qwen35GpuConfig {
        &self.cfg
    }

    /// Weight bytes bound on the device (mapped plus copied).
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Whether the weights alias the file mapping zero-copy.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// GPU execution time of the last [`Self::forward`]'s command buffer
    /// (`GPUEndTime − GPUStartTime`, seconds), `0.0` before the first one —
    /// what separates GPU-bound time from host encode/wait time.
    #[must_use]
    pub fn last_gpu_seconds(&self) -> f64 {
        self.last_gpu_seconds
    }

    /// KV slot of every layer (`None` for a recurrent layer): the dense
    /// index the encoder addresses the sparse KV cache through.
    #[must_use]
    pub fn layer_kv_slots(&self) -> &[Option<u32>] {
        &self.layer_kv_slot
    }

    /// Recurrent-state slot of every layer (`None` for full attention).
    #[must_use]
    pub fn layer_rec_slots(&self) -> &[Option<u32>] {
        &self.layer_rec_slot
    }

    /// Bytes of the device KV cache (K and V together).
    #[must_use]
    pub fn kv_cache_bytes(&self) -> u64 {
        self.k_cache.length() + self.v_cache.length()
    }

    /// Zero the recurrent state (conv windows and Gated-DeltaNet state).
    ///
    /// The KV cache needs no clearing: a position is always written before
    /// any query at or after it reads it.
    pub fn reset(&mut self) {
        for buf in [&self.ssm_state, &self.conv_state] {
            // SAFETY: shared buffers owned by this model; every forward waits
            // for its command buffer, so no GPU work is in flight.
            unsafe {
                std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, buf.length() as usize);
            }
        }
    }

    fn ensure_capacity(&mut self, t_len: usize) -> Result<(), MetalGraphError> {
        if t_len > self.scratch.capacity {
            self.scratch = Scratch::allocate(&self.graph, &self.cfg, t_len)?;
        }
        Ok(())
    }

    fn check_window(&self, t_len: usize, start_pos: usize) -> Result<(), MetalGraphError> {
        if t_len == 0 || t_len > self.cfg.max_batch {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {t_len} tokens per call (1..={})",
                self.cfg.max_batch
            )));
        }
        let end = start_pos.checked_add(t_len).ok_or_else(|| {
            MetalGraphError::InvalidDimensions("qwen35 GPU: position overflow".to_string())
        })?;
        if end > self.cfg.max_seq_len {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: positions {start_pos}..{end} run past the KV window of {}",
                self.cfg.max_seq_len
            )));
        }
        Ok(())
    }

    fn load_rows(&mut self, rows: &[f32]) -> Result<usize, MetalGraphError> {
        let h = self.cfg.hidden;
        if rows.is_empty() || !rows.len().is_multiple_of(h) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {} input floats is not a whole number of {h}-wide rows",
                rows.len()
            )));
        }
        let t_len = rows.len() / h;
        self.ensure_capacity(t_len)?;
        // SAFETY: `resid` holds `capacity * hidden >= rows.len()` floats and
        // no GPU work is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr(),
                self.scratch.resid.contents().cast::<f32>(),
                rows.len(),
            );
        }
        Ok(t_len)
    }

    /// Run the whole stack over `t_len = hidden_rows.len() / hidden` tokens
    /// at positions `start_pos..start_pos + t_len` (their embeddings, after
    /// any inverse rotation, in `hidden_rows`), advancing the recurrent
    /// state once per token and storing every key/value; when `logits` is
    /// `Some`, write the last token's `[vocab]` logits into it.
    ///
    /// One command buffer, one encoder, one wait, inside one autorelease
    /// pool: `commandBuffer` and `computeCommandEncoder` hand back
    /// autoreleased objects, and without a pool they would pile up on the
    /// calling thread — about 1.8 KiB per call, i.e. per decoded token of a
    /// long-lived server thread — until that thread exits.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a bad row count, a window
    /// past `max_seq_len` or a short `logits`;
    /// [`MetalGraphError::EncodingFailed`] if a layer lacks the cache slot or
    /// sign vector it needs (nothing is committed then); a failed command
    /// buffer.
    pub fn forward(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
        logits: Option<&mut [f32]>,
    ) -> Result<(), MetalGraphError> {
        autoreleasepool(|| self.forward_unpooled(hidden_rows, start_pos, logits))
    }

    fn forward_unpooled(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
        logits: Option<&mut [f32]>,
    ) -> Result<(), MetalGraphError> {
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, start_pos)?;
        if let Some(out) = logits.as_ref() {
            if out.len() < self.cfg.vocab {
                return Err(MetalGraphError::InvalidDimensions(format!(
                    "qwen35 GPU: logits buffer holds {} < vocab {}",
                    out.len(),
                    self.cfg.vocab
                )));
            }
        }
        let want_logits = logits.is_some();
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let encoded = (0..self.layers.len())
            .try_for_each(|layer| self.encode_layer(enc, layer, t_len, start_pos))
            .and_then(|()| {
                if want_logits {
                    self.encode_head(enc, t_len)
                } else {
                    Ok(())
                }
            });
        // The encoder is closed on every path; an encoding error drops the
        // command buffer uncommitted.
        enc.end_encoding();
        encoded?;
        commit_and_wait(cmd, "qwen35_forward")?;
        // SAFETY: `commit_and_wait` returned after `wait_until_completed`,
        // so the command buffer is in a terminal state and both timestamp
        // properties are readable.
        let (gpu_start, gpu_end): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        self.last_gpu_seconds = (gpu_end - gpu_start).max(0.0);
        if let Some(out) = logits {
            let vocab = self.cfg.vocab;
            // SAFETY: `logits` holds `vocab` floats written by the command
            // buffer that has just completed.
            let src =
                unsafe { std::slice::from_raw_parts(self.logits.contents().cast::<f32>(), vocab) };
            out[..vocab].copy_from_slice(src);
        }
        Ok(())
    }

    /// [`Self::forward`] one layer at a time, returning the residual stream
    /// after every layer (`[layer][t_len * hidden]`) and the last token's
    /// logits — the GPU counterpart of the CPU forward's per-layer dump.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`]; a layer that fails to encode is not committed,
    /// though the layers before it already ran.
    pub fn forward_with_dump(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), MetalGraphError> {
        autoreleasepool(|| self.forward_with_dump_unpooled(hidden_rows, start_pos))
    }

    fn forward_with_dump_unpooled(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), MetalGraphError> {
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, start_pos)?;
        let n = t_len * self.cfg.hidden;
        let mut dump = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            let cmd = self.graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let encoded = self.encode_layer(enc, layer, t_len, start_pos);
            enc.end_encoding();
            encoded?;
            commit_and_wait(cmd, "qwen35_forward_layer")?;
            dump.push(read_buffer(&self.scratch.resid, 0, n));
        }
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let encoded = self.encode_head(enc, t_len);
        enc.end_encoding();
        encoded?;
        commit_and_wait(cmd, "qwen35_forward_head")?;
        Ok((dump, read_buffer(&self.logits, 0, self.cfg.vocab)))
    }

    /// Run layer `layer` alone on `hidden_rows` (its input residual rows),
    /// one kernel per command buffer, capturing every intermediate
    /// activation — the per-kernel parity harness. The layer's KV slot or
    /// recurrent state advances exactly as in [`Self::forward`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], plus an out-of-range `layer`.
    pub fn trace_layer(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<Qwen35LayerTrace, MetalGraphError> {
        autoreleasepool(|| self.trace_layer_unpooled(layer, hidden_rows, start_pos))
    }

    fn trace_layer_unpooled(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<Qwen35LayerTrace, MetalGraphError> {
        if layer >= self.layers.len() {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: layer {layer} of {}",
                self.layers.len()
            )));
        }
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, start_pos)?;
        let mut trace = Qwen35LayerTrace::default();
        let stages = self.layer_stages(layer);
        for &stage in stages {
            let cmd = self.graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let encoded = self.encode_stage(enc, layer, stage, t_len, start_pos);
            enc.end_encoding();
            encoded?;
            commit_and_wait(cmd, "qwen35_trace_stage")?;
            for (name, buffer, width) in self.stage_outputs(stage) {
                trace
                    .stages
                    .push((name, read_buffer(buffer, 0, t_len * width)));
            }
        }
        Ok(trace)
    }

    /// `y = W x` for one matrix of layer `layer` (or the LM head, `layer ==
    /// n_layers`), on the GPU — with a one-hot `x` this returns column `j`
    /// of the dequantized matrix exactly (`d · code`), the bitwise check of
    /// every weight decoder.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for an unknown matrix or a
    /// wrong-length `x`; a failed command buffer.
    pub fn gemv_probe(
        &mut self,
        layer: usize,
        matrix: Qwen35MatrixId,
        x: &[f32],
    ) -> Result<Vec<f32>, MetalGraphError> {
        autoreleasepool(|| self.gemv_probe_unpooled(layer, matrix, x))
    }

    fn gemv_probe_unpooled(
        &mut self,
        layer: usize,
        matrix: Qwen35MatrixId,
        x: &[f32],
    ) -> Result<Vec<f32>, MetalGraphError> {
        let m = self.matrix(layer, matrix)?.clone();
        check_len("gemv_probe input", x.len(), m.cols)?;
        let xb = upload_f32(&self.graph, x)?;
        let yb = zeroed(&self.graph, m.rows * 4)?;
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_gemv(enc, &m, (&xb, 0), (&yb, 0), 1, false);
        enc.end_encoding();
        commit_and_wait(cmd, "qwen35_gemv_probe")?;
        Ok(read_buffer(&yb, 0, m.rows))
    }

    /// Blockwise rotation of `rows` rows of `x` on the GPU with the model's
    /// signs for `x`'s row width: forward (`signs`, then FWHT) or inverse
    /// (FWHT, then `signs`) — the standalone transform the producers fuse.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] when the model is not folded,
    /// has no signs for the width, or `x` is not `rows` whole rows.
    pub fn rotate(
        &mut self,
        x: &[f32],
        width: usize,
        inverse: bool,
    ) -> Result<Vec<f32>, MetalGraphError> {
        autoreleasepool(|| self.rotate_unpooled(x, width, inverse))
    }

    fn rotate_unpooled(
        &mut self,
        x: &[f32],
        width: usize,
        inverse: bool,
    ) -> Result<Vec<f32>, MetalGraphError> {
        let block = self.cfg.hadamard_block.ok_or_else(|| {
            MetalGraphError::InvalidDimensions(
                "qwen35 GPU: the model is not Hadamard-folded".to_string(),
            )
        })?;
        let signs = self.signs.get(&width).cloned().ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!("qwen35 GPU: no signs for width {width}"))
        })?;
        if width == 0 || x.is_empty() || !x.len().is_multiple_of(width) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {} floats is not a whole number of {width}-wide rows",
                x.len()
            )));
        }
        let rows = x.len() / width;
        let buf = upload_f32(&self.graph, x)?;
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipes.fwht_signed);
        enc.set_buffer(0, Some(&buf), 0);
        enc.set_buffer(1, Some(&signs), 0);
        set_u32(enc, 2, width as u32);
        set_u32(enc, 3, block as u32);
        set_u32(enc, 4, u32::from(inverse));
        set_f32(enc, 5, rotation_scale(block));
        enc.dispatch_thread_groups(
            MTLSize::new((width / block) as u64, rows as u64, 1),
            MTLSize::new(block.min(MAX_SPAN) as u64, 1, 1),
        );
        enc.end_encoding();
        commit_and_wait(cmd, "qwen35_rotate")?;
        Ok(read_buffer(&buf, 0, x.len()))
    }

    fn matrix(&self, layer: usize, id: Qwen35MatrixId) -> Result<&DeviceMatrix, MetalGraphError> {
        if layer == self.layers.len() {
            return match id {
                Qwen35MatrixId::LmHead => Ok(&self.lm_head),
                _ => Err(MetalGraphError::InvalidDimensions(format!(
                    "qwen35 GPU: {id:?} is not the LM head"
                ))),
            };
        }
        let missing = || {
            MetalGraphError::InvalidDimensions(format!("qwen35 GPU: layer {layer} has no {id:?}"))
        };
        match (self.layers.get(layer), id) {
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::AttnQ) => Ok(&l.attn_q),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::AttnK) => Ok(&l.attn_k),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::AttnV) => Ok(&l.attn_v),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::AttnOutput) => Ok(&l.attn_output),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::AttnQkv) => Ok(&l.attn_qkv),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::AttnGate) => Ok(&l.attn_gate),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::SsmAlphaBeta) => {
                Ok(&l.ssm_alpha_beta)
            }
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::SsmOut) => Ok(&l.ssm_out),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::FfnGate) => Ok(&l.ffn.gate),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::FfnGate) => Ok(&l.ffn.gate),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::FfnUp) => Ok(&l.ffn.up),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::FfnUp) => Ok(&l.ffn.up),
            (Some(LayerKind::FullAttention(l)), Qwen35MatrixId::FfnDown) => Ok(&l.ffn.down),
            (Some(LayerKind::LinearAttention(l)), Qwen35MatrixId::FfnDown) => Ok(&l.ffn.down),
            _ => Err(missing()),
        }
    }
}

/// A matrix of one layer, for [`Qwen35GpuModel::gemv_probe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MatrixId {
    /// `attn_q` (full attention).
    AttnQ,
    /// `attn_k` (full attention).
    AttnK,
    /// `attn_v` (full attention).
    AttnV,
    /// `attn_output` (full attention).
    AttnOutput,
    /// `attn_qkv` (linear attention).
    AttnQkv,
    /// `attn_gate` (linear attention).
    AttnGate,
    /// The stacked `ssm_alpha | ssm_beta` rows (linear attention).
    SsmAlphaBeta,
    /// `ssm_out` (linear attention).
    SsmOut,
    /// `ffn_gate`.
    FfnGate,
    /// `ffn_up`.
    FfnUp,
    /// `ffn_down`.
    FfnDown,
    /// `output` (pass `layer == n_layers`).
    LmHead,
}

/// The longest KV window a device with `max_buffer_length` /
/// `recommended_working_set` bytes can hold for a `qwen35` model of
/// geometry `cfg` with `n_full` full-attention and `n_linear` Gated-DeltaNet
/// layers whose device-read weights take `weight_bytes`.
///
/// The minimum of two bounds:
///
/// * one K (or V) buffer — every full-attention slot's keys — must fit in
///   `maxBufferLength`;
/// * everything [`Qwen35GpuModel`] keeps resident must fit in
///   `recommendedMaxWorkingSetSize`: the weights, the recurrent state
///   (Gated-DeltaNet slabs and conv windows), the logits and the activation
///   scratch at `cfg.max_batch` tokens (the most one forward grows it to),
///   plus per position the `f16` K + V cache, the rope angles and one row
///   of attention scores.
///
/// Both bounds read [`qwen35_footprint`], the one place the resident
/// arithmetic lives.
#[must_use]
pub fn qwen35_context_capacity(
    cfg: &Qwen35GpuConfig,
    n_full: usize,
    n_linear: usize,
    weight_bytes: u64,
    max_buffer_length: u64,
    recommended_working_set: u64,
) -> usize {
    let footprint = qwen35_footprint(cfg, n_full, n_linear);
    let by_buffer = max_buffer_length / footprint.kv_buffer_bytes_per_position.max(1);
    let fixed = weight_bytes.saturating_add(footprint.fixed_bytes);
    let by_working_set =
        recommended_working_set.saturating_sub(fixed) / footprint.per_position_bytes.max(1);
    usize::try_from(by_buffer.min(by_working_set)).unwrap_or(usize::MAX)
}

/// `1/sqrt(block)` exactly as `hadamard::fwht_forward_signed` computes it.
fn rotation_scale(block: usize) -> f32 {
    1.0_f32 / (block as f32).sqrt()
}

fn read_buffer(buf: &Buffer, offset_floats: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    // SAFETY: every caller sized `buf` for `offset_floats + n` floats and
    // waited for the command buffer that wrote them.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().cast::<f32>().add(offset_floats),
            out.as_mut_ptr(),
            n,
        );
    }
    out
}

fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, v: u32) {
    enc.set_bytes(index, 4, (&v as *const u32).cast());
}

fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, v: f32) {
    enc.set_bytes(index, 4, (&v as *const f32).cast());
}

fn set_u64(enc: &ComputeCommandEncoderRef, index: u64, v: u64) {
    enc.set_bytes(index, 8, (&v as *const u64).cast());
}

/// `y = W x` (or `y += W x`) for `t_len` columns.
fn encode_gemv(
    enc: &ComputeCommandEncoderRef,
    m: &DeviceMatrix,
    x: (&Buffer, u64),
    y: (&Buffer, u64),
    t_len: usize,
    accumulate: bool,
) {
    // One column (token) per threadgroup row of the grid: a prefill is the
    // decode GEMV for every token, in one dispatch.
    enc.set_compute_pipeline_state(&m.pso);
    enc.set_buffer(0, Some(&m.buffer), m.offset);
    enc.set_buffer(1, Some(x.0), x.1);
    enc.set_buffer(2, Some(y.0), y.1);
    set_u32(enc, 3, m.rows as u32);
    set_u32(enc, 4, m.cols as u32);
    set_u32(enc, 5, t_len as u32);
    set_u32(enc, 6, u32::from(accumulate));
    enc.dispatch_thread_groups(
        MTLSize::new(m.rows.div_ceil(GEMV_ROWS_PER_TG) as u64, t_len as u64, 1),
        MTLSize::new(GEMV_THREADS, 1, 1),
    );
}

#[path = "qwen35_encode.rs"]
mod encode;

#[path = "qwen35_state.rs"]
mod state;

pub use state::{qwen35_footprint, Qwen35DeviceLimits, Qwen35Footprint, Qwen35RecurrentSnapshot};

#[cfg(test)]
#[path = "qwen35_tests.rs"]
mod tests;
