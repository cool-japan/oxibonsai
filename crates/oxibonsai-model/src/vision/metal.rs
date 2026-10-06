//! The Qwen3-VL vision tower on the Metal GPU: [`VisionTowerMetal`] loads a
//! Bonsai 2 `mmproj` into `oxibonsai_kernels`' `VisionGpuModel` and serves
//! the CPU [`VisionTower`](super::VisionTower)'s contract — same input
//! boundary, same merged grid, same row order, rows in the same unrotated
//! embedding basis.
//!
//! # What runs where
//!
//! The host does the per-image bookkeeping the CPU tower does the same way:
//! normalise the pixels ([`normalize_planar`]), gather the patches in 2 × 2
//! merge-window order ([`window_order`]), resize the learned position grid
//! to the patch grid ([`resize_position_grid`]; the rows of the most recently
//! used grids are kept under a hard bound, so the memory held does not follow
//! the shapes clients send — see "The grid-rows cache") and build the 2-D
//! rotary rows with the kernels crate's `mrope_vision_build_tables` — the CPU
//! tower's own builder, so every angle is bit-identical. The device runs every
//! matrix product, norm, rotation, the attention and the merger in one command
//! buffer.
//!
//! # Weights
//!
//! Every matrix stays in the file's own storage, so the device computes
//! with exactly the weights the CPU tower dequantises: the `Q8_0` blocks
//! are copied as they are (the kernels read them exactly), an `F16` matrix
//! is held as `f16`, anything else as `f32` (`VisionWeightFormat`, chosen
//! per matrix kind from the file's types — every block's matrix of a kind
//! must share one). For Bonsai 2 that is 0.61 GB of weights against the CPU
//! tower's 1.84 GB of `f32`, each written into its device buffer as it is
//! read, so no whole-tower `f32` copy ever exists.
//!
//! # The grid-rows cache
//!
//! The host rows of a patch grid (the resized position rows and the rotary
//! rows) are cached by the grid of the image, so a stream of equal-sized
//! images builds them once. The cache has two hard bounds, both held at every
//! instant: at most 8 grids **and** at most 128 MiB of rows over all of them
//! (`GRID_CACHE_ENTRIES` and `GRID_CACHE_BYTES`), whichever binds first,
//! whatever shapes clients send. One grid's rows are `n_patches × (hidden +
//! head_dim) × 4` bytes — 4 896 bytes per patch for Bonsai 2's projector, so
//! the largest grid of the default 1 024-token budget is 20 MB and six of them
//! fit. A grid whose rows alone exceed the 128 MiB (more than 6 853 merged
//! tokens for that projector; a 16 384-token image is 321 MB) is not cached at
//! all: its rows are built again, from the tower's weights alone, for every
//! encode of that shape.
//!
//! The cache is **not** part of [`VisionTowerMetal::footprint`] or
//! [`VisionTowerMetal::resident_bytes`], which stay the device weights and
//! scratch plus the host position grid: a memory plan adds the cache's ceiling,
//! 128 MiB, to them. Beyond the cache, memory is held only by encodes in
//! flight — each keeps the rows of its one grid alive until it finishes, even
//! after the cache evicted them or when they were too large to be cached — so
//! `n` concurrent encodes can hold up to `n` further grids.
//!
//! # Sharing
//!
//! [`VisionTowerMetal::encode`] takes `&self`: the device model sits behind
//! a mutex, so one tower serves every engine replica of a process and
//! concurrent encodes run one after another (the GPU is serial anyway).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::GgufTensorType;
use oxibonsai_kernels::gpu_backend::metal_vision::{
    VisionGpuBuilder, VisionGpuConfig, VisionGpuModel, VisionMatrixFormats, VisionTensor,
    VisionWeightFormat, VIT_ATTN_MAX_SEQ,
};
use oxibonsai_kernels::rope_mrope::mrope_vision_build_tables;
use oxibonsai_kernels::KernelError;
use oxibonsai_kernels::MetalGraphError;

use super::clip_loader::{load_tensor_f32, names, VisionConfig, SPATIAL_MERGE};
use super::patch_embed::{normalize_planar, resize_position_grid, window_order};
use super::tower::merged_grid_of;
use super::{GridSize, ImageRgb8};
use crate::error::{ModelError, ModelResult};

/// The per-block tensor suffixes of the `qwen3vl_merger` projector.
const BLOCK_SUFFIXES: [&str; 12] = [
    "ln1.weight",
    "ln1.bias",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
];

/// Most merged tokens one Metal encode can take: the flash attention's
/// patch bound over the four patches of a merge window.
pub const METAL_VISION_MAX_TOKENS: usize = VIT_ATTN_MAX_SEQ / (SPATIAL_MERGE * SPATIAL_MERGE);

// Every image budget the preprocessing accepts is one the Metal tower
// serves.
const _: () = assert!(METAL_VISION_MAX_TOKENS >= super::MAX_IMAGE_MAX_TOKENS);

/// The host inputs of one patch grid that depend only on its shape: the
/// resized position rows and the rotary rows, both in window order.
#[derive(Debug)]
struct GridRows {
    pos: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl GridRows {
    /// Bytes the three row sets hold: the one size the grid cache budgets an
    /// entry at. The arithmetic saturates, so a size that overflows `usize`
    /// reads as `usize::MAX` — over any budget — rather than wrapping.
    fn byte_len(&self) -> usize {
        self.pos
            .len()
            .saturating_add(self.cos.len())
            .saturating_add(self.sin.len())
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

/// The most patch grids whose host rows a [`VisionTowerMetal`] keeps at
/// once: the entry bound of the grid cache, beside its byte bound
/// ([`GRID_CACHE_BYTES`]). Together they cap the cache at the smaller of 8
/// entries and 128 MiB.
///
/// The key of an entry is the patch grid of a client's image, so the cache is
/// bounded in entries — a client that sends images of ever new shapes must
/// not grow the process. An entry is a pure function of its `(rows,
/// columns)` and the tower's weights, so a grid that was evicted is simply
/// rebuilt: host arithmetic only, small beside the encode it feeds. The entry
/// count alone says nothing about bytes, since an entry's size follows the
/// image; that is what [`GRID_CACHE_BYTES`] bounds.
const GRID_CACHE_ENTRIES: usize = 8;

// A cache that keeps no entries would never cache anything.
const _: () = assert!(GRID_CACHE_ENTRIES >= 1);

/// The most bytes of rows ([`GridRows::byte_len`], summed over the entries)
/// the grid cache keeps at once: 128 MiB.
///
/// An entry's size follows the image: one entry holds `n_patches × (hidden +
/// head_dim) × 4` bytes — the resized position rows and the cosine and sine
/// rotary rows — and the largest grid a tower admits has `max_tokens × 4`
/// patches. For Bonsai 2's projector (hidden 1152, head width 72: 4 896 bytes
/// per patch) the largest grid of the default budget of 1 024 tokens is
/// 20 054 016 bytes, so eight of them would be 160 MB and this budget holds
/// six; at the Metal tower's own ceiling, [`METAL_VISION_MAX_TOKENS`] tokens,
/// one grid is 320 864 256 bytes, so eight would have been 2.57 GB. A grid
/// whose rows alone exceed this budget (more than 6 853 merged tokens for
/// that projector) is never cached: its rows are built again for each encode
/// of that shape.
///
/// So the cache holds at most the smaller of [`GRID_CACHE_ENTRIES`] entries
/// and this many bytes. Beyond that, memory is held only by encodes in
/// flight: an encode keeps the rows it is using alive past their eviction (an
/// `Arc`), so each concurrent encode adds one grid, up to the tower's
/// largest. The cache is not counted by [`VisionTowerMetal::footprint`] or
/// [`VisionTowerMetal::resident_bytes`]; this constant is its ceiling.
const GRID_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// A patch grid `(rows, columns)`, the key of a [`GridCache`].
type GridKey = (usize, usize);

/// The [`GridRows`] of the most recently used patch grids, under two hard
/// bounds that hold at every instant (the tower locks the cache around every
/// call): at most `max_entries` entries and at most `max_bytes` bytes of rows
/// ([`GridRows::byte_len`]) over all of them. The tower's cache takes
/// [`GRID_CACHE_ENTRIES`] and [`GRID_CACHE_BYTES`], so it holds at most the
/// smaller of 8 entries and 128 MiB whatever shapes the images take.
///
/// A lookup that hits and an insert both make their grid the most recently
/// used one. An insert looks the key up first (a racing builder's duplicate
/// keeps the entry already cached), then refuses rows that alone exceed the
/// byte bound — they are returned uncached and nothing is evicted on their
/// behalf — and otherwise drops the least recently used grids until both
/// bounds hold with the new entry included. Rows are shared as `Arc`s, so rows
/// handed out stay valid after their entry is evicted, or when they were never
/// cached.
#[derive(Debug)]
struct GridCache {
    /// Least recently used first, most recently used last; at most
    /// `max_entries` entries, one per grid.
    entries: VecDeque<(GridKey, Arc<GridRows>)>,
    /// The sum of [`GridRows::byte_len`] over `entries`, added to when an
    /// entry is inserted and subtracted from when one is evicted (never
    /// recomputed): at most `max_bytes`.
    held_bytes: usize,
    /// The most entries kept.
    max_entries: usize,
    /// The most bytes of rows kept in all.
    max_bytes: usize,
}

impl Default for GridCache {
    /// The tower's cache: [`GRID_CACHE_ENTRIES`] entries and
    /// [`GRID_CACHE_BYTES`] bytes.
    fn default() -> Self {
        Self::with_limits(GRID_CACHE_ENTRIES, GRID_CACHE_BYTES)
    }
}

impl GridCache {
    /// An empty cache keeping at most `max_entries` grids and `max_bytes`
    /// bytes of rows. The tower takes the constants; the limits are a
    /// parameter so that a test can drive the byte bound with small rows. A
    /// cache of no entries caches nothing.
    fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            held_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    /// The rows of `key`, which become the most recently used entry.
    fn get(&mut self, key: GridKey) -> Option<Arc<GridRows>> {
        let at = self.entries.iter().position(|(k, _)| *k == key)?;
        let entry = self.entries.remove(at)?;
        let rows = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(rows)
    }

    /// Keep `rows` for `key` as the most recently used entry and return the
    /// rows now cached for `key` — or `rows` itself when they are not cached.
    ///
    /// 1. A key that is already cached (two encodes of one new grid raced and
    ///    the other built it first) keeps its entry, refreshed, and that entry
    ///    is returned: a grid never takes two entries and its bytes are
    ///    counted once.
    /// 2. Rows larger than the byte bound — or any rows, when the cache keeps
    ///    no entries — are not cached: they are returned as given and the cache
    ///    is left exactly as it was, since evicting for an entry that cannot
    ///    be kept would only empty the cache.
    /// 3. Otherwise the least recently used entries are evicted, one at a time
    ///    and no more than it takes, until both bounds hold with the new entry
    ///    included, and the new entry is pushed as the most recently used.
    fn insert(&mut self, key: GridKey, rows: Arc<GridRows>) -> Arc<GridRows> {
        if let Some(cached) = self.get(key) {
            return cached;
        }
        let size = rows.byte_len();
        if self.max_entries == 0 || size > self.max_bytes {
            return rows;
        }
        while self.entries.len() >= self.max_entries
            || self.held_bytes.saturating_add(size) > self.max_bytes
        {
            let Some((_, evicted)) = self.entries.pop_front() else {
                break;
            };
            self.held_bytes = self.held_bytes.saturating_sub(evicted.byte_len());
        }
        self.entries.push_back((key, Arc::clone(&rows)));
        self.held_bytes = self.held_bytes.saturating_add(size);
        rows
    }

    /// The grids held, least recently used first.
    #[cfg(test)]
    fn keys(&self) -> Vec<GridKey> {
        self.entries.iter().map(|(key, _)| *key).collect()
    }
}

/// A poisoned grid-cache lock as a model error.
fn grid_cache_poisoned<T>(_: PoisonError<T>) -> ModelError {
    ModelError::Internal("vision grid cache poisoned".to_string())
}

/// The Qwen3-VL vision tower of a Bonsai 2 `mmproj`, on the Metal GPU (see
/// the module docs).
pub struct VisionTowerMetal {
    config: VisionConfig,
    gpu: Mutex<VisionGpuModel>,
    /// `v.position_embd.weight`, `[side² × hidden]`, raster order.
    pos_embd: Vec<f32>,
    /// The [`GridRows`] of the most recently used patch grids: at most
    /// [`GRID_CACHE_ENTRIES`] of them and [`GRID_CACHE_BYTES`] of rows in
    /// all. Not counted by [`Self::resident_bytes`].
    grids: Mutex<GridCache>,
    bound_tensors: usize,
    max_tokens: usize,
}

impl std::fmt::Debug for VisionTowerMetal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionTowerMetal")
            .field("config", &self.config)
            .field("bound_tensors", &self.bound_tensors)
            .field("max_tokens", &self.max_tokens)
            .field("resident_bytes", &self.resident_bytes())
            .finish()
    }
}

/// A device error as a model error (a refused geometry stays an
/// unsupported operation).
fn from_gpu(e: MetalGraphError) -> ModelError {
    match e {
        MetalGraphError::InvalidDimensions(msg) => {
            ModelError::Kernel(KernelError::UnsupportedOperation(msg))
        }
        other => ModelError::Kernel(KernelError::GpuError(format!(
            "Metal vision tower: {other}"
        ))),
    }
}

/// How the device holds a matrix the file stores as `tensor_type`: `Q8_0`
/// blocks as they are, `F16` as `f16`, anything else as `f32` (the loader
/// refuses a type it cannot read when the values are loaded).
fn weight_format(tensor_type: GgufTensorType) -> VisionWeightFormat {
    match tensor_type {
        GgufTensorType::Q8_0 => VisionWeightFormat::Q8_0,
        GgufTensorType::F16 => VisionWeightFormat::F16,
        _ => VisionWeightFormat::F32,
    }
}

/// The storage of each matrix kind, from the file's tensor types: every
/// block's matrix of a kind must share one (a mixed file is refused, naming
/// the tensor that differs).
fn matrix_formats(gguf: &GgufFile<'_>, config: &VisionConfig) -> ModelResult<VisionMatrixFormats> {
    let type_of = |name: &str| -> ModelResult<GgufTensorType> {
        gguf.tensors
            .get(name)
            .map(|info| info.tensor_type)
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_string(),
            })
    };
    let per_block = |suffix: &str| -> ModelResult<VisionWeightFormat> {
        let first = type_of(&names::block(0, suffix))?;
        for layer in 1..config.blocks {
            let name = names::block(layer, suffix);
            let this = type_of(&name)?;
            if this != first {
                return Err(ModelError::InvalidTensor(format!(
                    "{name}: stored as {}, but block 0's is {} — the Metal tower holds every \
                     block's {suffix} in one format",
                    this.name(),
                    first.name()
                )));
            }
        }
        Ok(weight_format(first))
    };
    Ok(VisionMatrixFormats {
        qkv: per_block("attn_qkv.weight")?,
        out: per_block("attn_out.weight")?,
        up: per_block("ffn_up.weight")?,
        down: per_block("ffn_down.weight")?,
        mm0: weight_format(type_of(names::MM0_WEIGHT)?),
        mm2: weight_format(type_of(names::MM2_WEIGHT)?),
    })
}

/// The raw `Q8_0` blocks of the `[inner, rows]` matrix `name` (GGUF
/// dimension order), after checking its shape, its row blocking and its
/// byte count.
fn q8_0_blocks<'g>(
    gguf: &'g GgufFile<'_>,
    name: &str,
    inner: usize,
    rows: usize,
) -> ModelResult<&'g [u8]> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_string(),
        })?;
    let actual: Vec<usize> = info
        .shape
        .iter()
        .map(|&d| d as usize)
        .filter(|&d| d != 1)
        .collect();
    let expected: Vec<usize> = [inner, rows].into_iter().filter(|&d| d != 1).collect();
    if actual != expected {
        return Err(ModelError::ShapeMismatch {
            name: name.to_string(),
            expected,
            actual,
        });
    }
    info.validate_row_blocking()?;
    let data = gguf.tensor_data(name)?;
    let bytes = VisionWeightFormat::Q8_0.bytes(inner * rows);
    if data.len() != bytes {
        return Err(ModelError::InvalidTensor(format!(
            "{name}: {} bytes of Q8_0 blocks, the shape needs {bytes}",
            data.len()
        )));
    }
    Ok(data)
}

/// Write one matrix into the builder in its format: `Q8_0` blocks as they
/// are, anything else through the CPU loader's values.
fn set_matrix(
    builder: &mut VisionGpuBuilder,
    gguf: &GgufFile<'_>,
    tensor: VisionTensor,
    name: &str,
    inner: usize,
    rows: usize,
    format: VisionWeightFormat,
) -> ModelResult<()> {
    if format == VisionWeightFormat::Q8_0 {
        let blocks = q8_0_blocks(gguf, name, inner, rows)?;
        builder.set_q8_0(tensor, blocks).map_err(from_gpu)
    } else {
        let values = load_tensor_f32(gguf, name, &[inner, rows])?;
        builder.set(tensor, &values).map_err(from_gpu)
    }
}

/// The device geometry of a projector with `config` and matrix `formats`,
/// sized for images of up to `max_tokens` merged tokens.
fn gpu_config(
    config: &VisionConfig,
    formats: VisionMatrixFormats,
    max_tokens: usize,
) -> ModelResult<VisionGpuConfig> {
    if max_tokens == 0 || max_tokens > METAL_VISION_MAX_TOKENS {
        return Err(ModelError::ShapeInvariant {
            tensor: "image token budget".to_string(),
            expected: format!(
                "1..={METAL_VISION_MAX_TOKENS} merged tokens per image on the Metal tower"
            ),
            actual: max_tokens.to_string(),
        });
    }
    Ok(VisionGpuConfig {
        hidden: config.hidden,
        heads: config.heads,
        head_dim: config.head_dim,
        ffn: config.ffn,
        blocks: config.blocks,
        eps: config.eps,
        patch_len: config.patch_len(),
        merger_hidden: config.merger_hidden,
        projection_dim: config.projection_dim,
        max_patches: max_tokens * SPATIAL_MERGE * SPATIAL_MERGE,
        formats,
    })
}

/// Every tensor name of a projector with `blocks` blocks.
fn expected_names(blocks: usize) -> Vec<String> {
    let mut out: Vec<String> = [
        names::PATCH_EMBD,
        names::PATCH_EMBD_1,
        names::PATCH_BIAS,
        names::POSITION_EMBD,
        names::POST_LN_WEIGHT,
        names::POST_LN_BIAS,
        names::MM0_WEIGHT,
        names::MM0_BIAS,
        names::MM2_WEIGHT,
        names::MM2_BIAS,
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    for layer in 0..blocks {
        out.extend(BLOCK_SUFFIXES.iter().map(|s| names::block(layer, s)));
    }
    out
}

/// The file carries exactly the projector's tensors — the CPU loader's
/// inventory rule: a missing one first, then every stray one at once.
fn check_inventory(gguf: &GgufFile<'_>, config: &VisionConfig) -> ModelResult<()> {
    let expected = expected_names(config.blocks);
    for name in &expected {
        if gguf.tensors.get(name).is_none() {
            return Err(ModelError::MissingTensor { name: name.clone() });
        }
    }
    let known: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let mut stray: Vec<&str> = gguf
        .tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| !known.contains(name))
        .collect();
    if stray.is_empty() {
        return Ok(());
    }
    stray.sort_unstable();
    Err(ModelError::ShapeInvariant {
        tensor: "mmproj tensor inventory".to_string(),
        expected: format!(
            "exactly the {} tensors of the qwen3vl_merger graph",
            expected.len()
        ),
        actual: format!("{} unexpected tensor(s): {}", stray.len(), stray.join(", ")),
    })
}

impl VisionTowerMetal {
    /// Load a `qwen3vl_merger` projector onto the GPU, sized for images of
    /// up to `max_tokens` merged tokens (`--image-max-tokens`).
    ///
    /// Reads and validates the metadata exactly as the CPU tower does,
    /// checks the tensor inventory, and writes every tensor into its device
    /// buffer as it is read — each matrix in the file's own storage (see
    /// the module docs, "Weights").
    ///
    /// # Errors
    ///
    /// The CPU loader's metadata and tensor errors (each naming the key or
    /// tensor); [`ModelError::ShapeInvariant`] for a budget outside
    /// `1..=`[`METAL_VISION_MAX_TOKENS`]; [`ModelError::Kernel`] for a
    /// geometry the kernels do not serve, no Metal device, or a failed
    /// allocation.
    pub fn from_mmproj(gguf: &GgufFile<'_>, max_tokens: usize) -> ModelResult<Self> {
        let config = VisionConfig::from_gguf(gguf)?;
        check_inventory(gguf, &config)?;
        let formats = matrix_formats(gguf, &config)?;
        let gpu_cfg = gpu_config(&config, formats, max_tokens)?;
        let mut builder = VisionGpuBuilder::new(gpu_cfg).map_err(from_gpu)?;
        let (h, f) = (config.hidden, config.ffn);
        for layer in 0..config.blocks {
            let load = |suffix: &str, shape: &[usize]| {
                load_tensor_f32(gguf, &names::block(layer, suffix), shape)
            };
            let vectors: [(VisionTensor, &str, usize); 8] = [
                (VisionTensor::Ln1Weight(layer), "ln1.weight", h),
                (VisionTensor::Ln1Bias(layer), "ln1.bias", h),
                (VisionTensor::QkvBias(layer), "attn_qkv.bias", 3 * h),
                (VisionTensor::OutBias(layer), "attn_out.bias", h),
                (VisionTensor::Ln2Weight(layer), "ln2.weight", h),
                (VisionTensor::Ln2Bias(layer), "ln2.bias", h),
                (VisionTensor::UpBias(layer), "ffn_up.bias", f),
                (VisionTensor::DownBias(layer), "ffn_down.bias", h),
            ];
            for (tensor, suffix, len) in vectors {
                builder
                    .set(tensor, &load(suffix, &[len])?)
                    .map_err(from_gpu)?;
            }
            // Each matrix as `[inner, rows]` in GGUF order.
            let matrices: [(VisionTensor, &str, usize, usize, VisionWeightFormat); 4] = [
                (
                    VisionTensor::QkvWeight(layer),
                    "attn_qkv.weight",
                    h,
                    3 * h,
                    formats.qkv,
                ),
                (
                    VisionTensor::OutWeight(layer),
                    "attn_out.weight",
                    h,
                    h,
                    formats.out,
                ),
                (
                    VisionTensor::UpWeight(layer),
                    "ffn_up.weight",
                    h,
                    f,
                    formats.up,
                ),
                (
                    VisionTensor::DownWeight(layer),
                    "ffn_down.weight",
                    f,
                    h,
                    formats.down,
                ),
            ];
            for (tensor, suffix, inner, rows, format) in matrices {
                let name = names::block(layer, suffix);
                set_matrix(&mut builder, gguf, tensor, &name, inner, rows, format)?;
            }
        }
        let p = config.patch_size;
        let patch_shape = [p, p, 3, h];
        let mut kernel = load_tensor_f32(gguf, names::PATCH_EMBD, &patch_shape)?;
        let slice1 = load_tensor_f32(gguf, names::PATCH_EMBD_1, &patch_shape)?;
        // Both temporal slices, summed once (a still image is a two-frame
        // clip of itself) — `PatchEmbed::new`'s own sum.
        for (k0, k1) in kernel.iter_mut().zip(&slice1) {
            *k0 += *k1;
        }
        drop(slice1);
        builder
            .set(VisionTensor::PatchKernel, &kernel)
            .map_err(from_gpu)?;
        drop(kernel);
        let merged = config.merged_width();
        let globals: [(VisionTensor, &str, usize); 5] = [
            (VisionTensor::PatchBias, names::PATCH_BIAS, h),
            (VisionTensor::PostLnWeight, names::POST_LN_WEIGHT, h),
            (VisionTensor::PostLnBias, names::POST_LN_BIAS, h),
            (VisionTensor::Mm0Bias, names::MM0_BIAS, config.merger_hidden),
            (
                VisionTensor::Mm2Bias,
                names::MM2_BIAS,
                config.projection_dim,
            ),
        ];
        for (tensor, name, len) in globals {
            builder
                .set(tensor, &load_tensor_f32(gguf, name, &[len])?)
                .map_err(from_gpu)?;
        }
        set_matrix(
            &mut builder,
            gguf,
            VisionTensor::Mm0Weight,
            names::MM0_WEIGHT,
            merged,
            config.merger_hidden,
            formats.mm0,
        )?;
        set_matrix(
            &mut builder,
            gguf,
            VisionTensor::Mm2Weight,
            names::MM2_WEIGHT,
            config.merger_hidden,
            config.projection_dim,
            formats.mm2,
        )?;
        let pos_embd = load_tensor_f32(
            gguf,
            names::POSITION_EMBD,
            &[h, config.pos_grid * config.pos_grid],
        )?;
        let gpu = builder.build().map_err(from_gpu)?;
        Ok(Self {
            bound_tensors: config.expected_tensor_count(),
            config,
            gpu: Mutex::new(gpu),
            pos_embd,
            grids: Mutex::new(GridCache::default()),
            max_tokens,
        })
    }

    /// Bytes a Metal tower for the projector `gguf`, sized for
    /// `max_tokens`, keeps resident — its device weights (in the file's own
    /// storage) and scratch plus the host position grid — read from the
    /// file's metadata and tensor types without building one or touching a
    /// device (what a caller budgets memory with first).
    ///
    /// Like [`Self::resident_bytes`], this does not include the tower's
    /// grid-rows cache (see the module docs): the cache's ceiling is
    /// `GRID_CACHE_BYTES`, 128 MiB, which a memory plan adds on top.
    ///
    /// # Errors
    ///
    /// [`VisionConfig::from_gguf`]'s errors, a missing matrix or blocks of
    /// mixed storage, and [`ModelError::ShapeInvariant`] for a budget
    /// outside `1..=`[`METAL_VISION_MAX_TOKENS`].
    pub fn footprint(gguf: &GgufFile<'_>, max_tokens: usize) -> ModelResult<u64> {
        let config = VisionConfig::from_gguf(gguf)?;
        let formats = matrix_formats(gguf, &config)?;
        let gpu_cfg = gpu_config(&config, formats, max_tokens)?;
        let host = (config.pos_grid * config.pos_grid * config.hidden * 4) as u64;
        Ok(VisionGpuModel::footprint(&gpu_cfg)
            .total_bytes()
            .saturating_add(host))
    }

    /// The projector's hyper-parameters.
    #[must_use]
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// The number of ViT blocks bound.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.config.blocks
    }

    /// The number of GGUF tensors bound (the file's whole inventory).
    #[must_use]
    pub fn bound_tensor_count(&self) -> usize {
        self.bound_tensors
    }

    /// The most merged tokens one encode takes (the budget the tower was
    /// built for).
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Bytes the tower keeps resident: device weights and scratch, and the
    /// host position grid.
    ///
    /// This does not include the grid-rows cache (see the module docs): its
    /// ceiling is `GRID_CACHE_BYTES`, 128 MiB, which a memory plan adds on
    /// top, and every encode in flight holds the rows of its one grid besides.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let device = self.gpu.lock().map_or(0, |gpu| {
            usize::try_from(gpu.resident_bytes()).unwrap_or(usize::MAX)
        });
        device.saturating_add(std::mem::size_of_val(self.pos_embd.as_slice()))
    }

    /// GPU time of the last encode's command buffer, in seconds.
    #[must_use]
    pub fn last_gpu_seconds(&self) -> f64 {
        self.gpu.lock().map_or(0.0, |gpu| gpu.last_gpu_seconds())
    }

    /// The merged grid an image of `width x height` pixels produces, after
    /// checking it can be encoded as-is — [`super::VisionTower::merged_grid`]'s
    /// rule, plus the budget this tower was built for.
    ///
    /// # Errors
    ///
    /// As the CPU tower's, and [`ModelError::ShapeInvariant`] past
    /// [`Self::max_tokens`].
    pub fn merged_grid(
        &self,
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<GridSize> {
        merged_grid_of(&self.config, width, height, max_tokens.min(self.max_tokens))
    }

    /// Encode an RGB8 image already sized for the tower into `[n_merged ×
    /// projection_dim]` rows (row-major over the returned merged grid) in
    /// the language model's unrotated embedding basis — the CPU tower's
    /// contract.
    ///
    /// # Errors
    ///
    /// As the CPU tower's `encode`, plus a device failure.
    pub fn encode(&self, img: &ImageRgb8, max_tokens: usize) -> ModelResult<(Vec<f32>, GridSize)> {
        img.validate()?;
        let grid = self.merged_grid(img.width, img.height, max_tokens)?;
        let planar = normalize_planar(img, self.config.image_mean, self.config.image_std);
        let rows = self.run(&planar, img.width, img.height)?;
        Ok((rows, grid))
    }

    /// [`Self::encode`] for pixels already normalised to planar
    /// `[3][height][width]`.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`], plus [`ModelError::ShapeMismatch`] for a
    /// `planar` of the wrong length.
    pub fn encode_normalized(
        &self,
        planar: &[f32],
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<(Vec<f32>, GridSize)> {
        let grid = self.merged_grid(width, height, max_tokens)?;
        let rows = self.run(planar, width, height)?;
        Ok((rows, grid))
    }

    /// The grid cache's lock, mapping a poisoned one to a model error.
    fn lock_grids(&self) -> ModelResult<MutexGuard<'_, GridCache>> {
        self.grids.lock().map_err(grid_cache_poisoned)
    }

    /// The position and rotary rows of a `grid_h x grid_w` patch grid, from
    /// the cache or built now.
    ///
    /// The cache's lock is held only for the lookup and for the insert, never
    /// while the rows are built, so two requests for one new grid may both
    /// build it; the second insert finds the first's entry and returns it.
    /// The rows are shared, so an encode that holds them keeps them alive
    /// after a later request evicts their entry. Rows larger than the cache's
    /// byte budget are returned without being cached, so the next request for
    /// that grid builds them again.
    fn grid_rows(&self, grid_h: usize, grid_w: usize) -> ModelResult<Arc<GridRows>> {
        let key = (grid_h, grid_w);
        let cached = self.lock_grids()?.get(key);
        if let Some(rows) = cached {
            return Ok(rows);
        }
        let rows = Arc::new(self.build_grid_rows(grid_h, grid_w)?);
        Ok(self.lock_grids()?.insert(key, rows))
    }

    /// The position and rotary rows of a `grid_h x grid_w` patch grid, built
    /// from the tower's weights alone.
    fn build_grid_rows(&self, grid_h: usize, grid_w: usize) -> ModelResult<GridRows> {
        let cfg = &self.config;
        let hidden = cfg.hidden;
        let order = window_order(grid_h, grid_w);
        let resized;
        let grid: &[f32] = if grid_h == cfg.pos_grid && grid_w == cfg.pos_grid {
            &self.pos_embd
        } else {
            resized = resize_position_grid(&self.pos_embd, cfg.pos_grid, hidden, grid_h, grid_w)?;
            &resized
        };
        let mut pos = Vec::with_capacity(order.len() * hidden);
        for &(py, px) in &order {
            let row = grid
                .get((py * grid_w + px) * hidden..(py * grid_w + px + 1) * hidden)
                .ok_or_else(|| ModelError::Internal("position grid row out of range".into()))?;
            pos.extend_from_slice(row);
        }
        let half = cfg.head_dim / 2;
        let mut cos = vec![0.0f32; order.len() * half];
        let mut sin = vec![0.0f32; order.len() * half];
        let sections = cfg.rope_sections().map(|s| s as u32);
        for ((&(py, px), c), s) in order
            .iter()
            .zip(cos.chunks_mut(half.max(1)))
            .zip(sin.chunks_mut(half.max(1)))
        {
            let (y, x) = (axis(py)?, axis(px)?);
            mrope_vision_build_tables(
                [y, x, y, x],
                sections,
                cfg.head_dim,
                cfg.rope_freq_base,
                c,
                s,
            )
            .map_err(ModelError::Kernel)?;
        }
        Ok(GridRows { pos, cos, sin })
    }

    /// The whole graph on a validated, normalised planar image.
    fn run(&self, planar: &[f32], width: usize, height: usize) -> ModelResult<Vec<f32>> {
        let cfg = &self.config;
        let p = cfg.patch_size;
        let plane = width.saturating_mul(height);
        if Some(planar.len()) != plane.checked_mul(3) {
            return Err(ModelError::ShapeMismatch {
                name: "normalised image (planar RGB)".to_string(),
                expected: vec![3, height, width],
                actual: vec![planar.len()],
            });
        }
        let (grid_h, grid_w) = (height / p, width / p);
        let order = window_order(grid_h, grid_w);
        let n = order.len();
        let patch_len = cfg.patch_len();
        // The CPU patch embedding's own gather: channel, kernel row, kernel
        // column, per patch in window order.
        let mut patches = vec![0.0f32; n * patch_len];
        for (row, &(py, px)) in patches.chunks_mut(patch_len).zip(&order) {
            for c in 0..3 {
                for ky in 0..p {
                    let src = c * plane + (py * p + ky) * width + px * p;
                    let dst = c * p * p + ky * p;
                    let (Some(d), Some(s)) = (row.get_mut(dst..dst + p), planar.get(src..src + p))
                    else {
                        return Err(ModelError::Internal("patch gather out of range".into()));
                    };
                    d.copy_from_slice(s);
                }
            }
        }
        let grid = self.grid_rows(grid_h, grid_w)?;
        let merged = n / (SPATIAL_MERGE * SPATIAL_MERGE);
        let mut out = vec![0.0f32; merged * cfg.projection_dim];
        let mut gpu = self
            .gpu
            .lock()
            .map_err(|_| ModelError::Internal("Metal vision tower mutex poisoned".to_string()))?;
        gpu.encode(&patches, &grid.pos, &grid.cos, &grid.sin, &mut out)
            .map_err(from_gpu)?;
        Ok(out)
    }
}

fn axis(p: usize) -> ModelResult<i32> {
    i32::try_from(p).map_err(|_| ModelError::ShapeInvariant {
        tensor: "vision rope position".to_string(),
        expected: "a patch coordinate that fits in i32".to_string(),
        actual: p.to_string(),
    })
}

#[cfg(test)]
#[path = "metal_tests.rs"]
mod tests;
