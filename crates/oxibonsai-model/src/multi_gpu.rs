//! Multi-GPU / multi-device inference **simulation** utilities.
//!
//! ## What this module actually is
//!
//! This module runs entirely on the CPU, over rayon thread pools. It does
//! **not** talk to a GPU, does **not** link against or implement any part of
//! NVIDIA's NCCL library, and does **not** probe real hardware for device
//! counts, memory, or compute capability. [`SimulatedDeviceMesh`] and
//! [`SimulatedCollectives`] are exactly what their names say: a simulation
//! useful for developing and testing the *sharding math* (weight
//! partitioning, collective reduction/gather semantics) that a real
//! multi-device backend would need, without requiring the hardware.
//!
//! The crate's public API still re-exports these under the legacy names
//! [`DeviceMesh`] and [`NcclCollectives`] (type aliases to the `Simulated*`
//! types below) for backward compatibility with existing callers; new code
//! should prefer the `Simulated*` names directly, since they do not imply
//! any relationship to real GPU hardware or to NCCL.
//!
//! ## Reachability
//!
//! As of this writing, nothing in `oxibonsai-model`, `oxibonsai-runtime`, or
//! the CLI constructs a [`SimulatedDeviceMesh`] or calls
//! [`SimulatedCollectives`] outside of this module's own tests — there is no
//! production entry point that performs multi-device inference. This module
//! is a tested, documented primitive for that future work, not a wired
//! feature.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────┐
//!  │            SimulatedDeviceMesh (tp × pp)         │
//!  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
//!  │  │ Device 0 │  │ Device 1 │  │ Device 2 │  ...  │
//!  │  │ (tp=0,   │  │ (tp=1,   │  │ (tp=0,   │       │
//!  │  │  pp=0)   │  │  pp=0)   │  │  pp=1)   │       │
//!  │  └──────────┘  └──────────┘  └──────────┘       │
//!  └─────────────────────────────────────────────────┘
//!
//!   SimulatedCollectives ─► all_reduce_sum / all_gather / broadcast …
//!   partition_weights_column / partition_weights_row ─► shards
//! ```

use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceId
// ─────────────────────────────────────────────────────────────────────────────

/// A logical device identifier (CPU thread group simulating a GPU).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

// ─────────────────────────────────────────────────────────────────────────────
// DeviceInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Capabilities reported for one simulated device.
///
/// **Not a hardware probe.** [`compute_units`][Self::compute_units] is the
/// one field that *is* real: it reflects the actual CPU parallelism
/// available to the rayon thread pool this simulation runs on
/// (`std::thread::available_parallelism`), because that genuinely is the
/// resource backing a "device" in this simulation.
/// [`memory_bytes`][Self::memory_bytes] is honestly `None` — this module
/// has no dependency-free, portable way to query real GPU or system
/// memory, so rather than hand out a fabricated number it reports that the
/// value is unknown. Populate it only from a real backend probe, if one is
/// ever wired in.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// The logical device identifier.
    pub id: DeviceId,
    /// Device memory budget, if known. Always `None` in this CPU
    /// simulation — there is no real hardware here to probe, and this
    /// module does not fabricate a number in its place. See the
    /// struct-level docs.
    pub memory_bytes: Option<usize>,
    /// Real CPU thread parallelism available to this simulation, from
    /// `std::thread::available_parallelism()` (falls back to `1` if the
    /// platform cannot report it). This is measured, not fabricated — it is
    /// literally the concurrency rayon will use to simulate this "device".
    pub compute_units: usize,
    /// Human-readable device name (e.g. "SimDevice-0").
    pub name: String,
}

impl DeviceInfo {
    fn simulated(linear_id: usize) -> Self {
        Self {
            id: DeviceId(linear_id),
            memory_bytes: None,
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            name: format!("SimDevice-{linear_id}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedDeviceMesh (public alias: DeviceMesh)
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D logical device mesh: tensor-parallel dimension × pipeline-parallel dimension.
///
/// Devices are stored in row-major order: device at `(tp_rank, pp_rank)` has
/// linear index `tp_rank + pp_rank * tp_size`. See the module docs — this is
/// a CPU simulation, not a real device topology.
pub struct SimulatedDeviceMesh {
    devices: Vec<DeviceInfo>,
    tp_size: usize,
    pp_size: usize,
}

/// Backward-compatible alias for [`SimulatedDeviceMesh`].
///
/// Kept so the crate root's existing `pub use multi_gpu::{.., DeviceMesh,
/// ..}` re-export continues to resolve; prefer `SimulatedDeviceMesh`
/// directly in new code, since `DeviceMesh` on its own does not signal that
/// this is a CPU simulation rather than a real device topology.
pub type DeviceMesh = SimulatedDeviceMesh;

impl SimulatedDeviceMesh {
    /// Create a 1-D tensor-parallel mesh of `n` simulated devices.
    pub fn tensor_parallel(n: usize) -> Self {
        Self::new(n, 1)
    }

    /// Create a 2-D (`tp_size` × `pp_size`) mesh.
    ///
    /// Total device count is `tp_size * pp_size`.
    pub fn new(tp_size: usize, pp_size: usize) -> Self {
        let total = tp_size * pp_size;
        let devices = (0..total).map(DeviceInfo::simulated).collect();
        Self {
            devices,
            tp_size,
            pp_size,
        }
    }

    /// Total number of devices in the mesh.
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// Get the device at tensor-parallel rank `tp_rank` and pipeline-parallel rank `pp_rank`.
    ///
    /// Returns `None` if either rank is out of bounds.
    pub fn get(&self, tp_rank: usize, pp_rank: usize) -> Option<&DeviceInfo> {
        if tp_rank >= self.tp_size || pp_rank >= self.pp_size {
            return None;
        }
        let idx = tp_rank + pp_rank * self.tp_size;
        self.devices.get(idx)
    }

    /// All devices in the tensor-parallel group for a given `pp_rank`.
    ///
    /// Returns an empty vec if `pp_rank` is out of range.
    pub fn tp_group(&self, pp_rank: usize) -> Vec<&DeviceInfo> {
        if pp_rank >= self.pp_size {
            return Vec::new();
        }
        (0..self.tp_size)
            .filter_map(|tp| self.get(tp, pp_rank))
            .collect()
    }

    /// All devices in the pipeline-parallel group for a given `tp_rank`.
    ///
    /// Returns an empty vec if `tp_rank` is out of range.
    pub fn pp_group(&self, tp_rank: usize) -> Vec<&DeviceInfo> {
        if tp_rank >= self.tp_size {
            return Vec::new();
        }
        (0..self.pp_size)
            .filter_map(|pp| self.get(tp_rank, pp))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CollectiveResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a collective communication operation.
#[derive(Debug, Clone)]
pub struct CollectiveResult {
    /// The reduced / gathered data.
    pub data: Vec<f32>,
    /// Number of devices that participated.
    pub participating_devices: usize,
    /// Name tag identifying the operation (e.g. `"all_reduce_sum"`).
    pub op_name: &'static str,
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiGpuError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from the checked collective operations on [`SimulatedCollectives`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MultiGpuError {
    /// A collective that requires every shard to have the same length (e.g.
    /// all-reduce) was called with ragged shards.
    #[error(
        "ragged shards: shard 0 has length {expected}, but shard {index} has length {got} \
         (all-reduce/reduce-scatter require every rank's shard to be the same length)"
    )]
    RaggedShards {
        /// Index of the first shard whose length disagrees with shard 0.
        index: usize,
        /// Length of shard 0 (the length every other shard was expected to match).
        expected: usize,
        /// The actual, differing length found at `index`.
        got: usize,
    },
    /// The collective was called with no shards at all.
    #[error("no shards provided")]
    EmptyShards,
    /// `partition_weights_column`/`partition_weights_row` were called with a
    /// `weights` slice whose length does not equal `rows * cols`.
    #[error("weights length {got} does not equal rows*cols = {expected}")]
    WeightsLengthMismatch {
        /// `rows * cols`, the length `weights` was expected to have.
        expected: usize,
        /// The actual length of the `weights` slice.
        got: usize,
    },
}

/// Verify every shard has the same length as the first. Returns that common
/// length, or `Err(MultiGpuError::RaggedShards)` naming the first mismatch.
fn check_uniform_shard_len(shards: &[Vec<f32>]) -> Result<usize, MultiGpuError> {
    let expected = shards.first().ok_or(MultiGpuError::EmptyShards)?.len();
    for (index, shard) in shards.iter().enumerate() {
        if shard.len() != expected {
            return Err(MultiGpuError::RaggedShards {
                index,
                expected,
                got: shard.len(),
            });
        }
    }
    Ok(expected)
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives (public alias: NcclCollectives)
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-simulated collective communication operations (sum/max/gather/scatter
/// over rayon), modeled on the semantics of NCCL/MPI-style collectives.
///
/// This does **not** implement, wrap, or link against NVIDIA's NCCL library
/// — it is a pure-CPU reference simulation of the same *semantics*, useful
/// for testing sharding logic without GPU hardware.
pub struct SimulatedCollectives;

/// Backward-compatible alias for [`SimulatedCollectives`]. See the module
/// docs and [`SimulatedCollectives`]'s own doc comment: prefer the
/// `Simulated*` name in new code.
pub type NcclCollectives = SimulatedCollectives;

impl SimulatedCollectives {
    /// All-reduce (sum): element-wise sum of tensors from all participating
    /// devices; the result is the same on every device.
    ///
    /// Returns `Err(MultiGpuError::RaggedShards)` if the shards are not all
    /// the same length, and `Err(MultiGpuError::EmptyShards)` if `shards`
    /// is empty — never panics.
    pub fn all_reduce_sum_checked(shards: &[Vec<f32>]) -> Result<CollectiveResult, MultiGpuError> {
        let n = check_uniform_shard_len(shards)?;
        let data: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| shards.iter().map(|s| s[i]).sum::<f32>())
            .collect();
        Ok(CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_reduce_sum",
        })
    }

    /// Infallible convenience wrapper around
    /// [`all_reduce_sum_checked`][Self::all_reduce_sum_checked].
    ///
    /// On ragged or empty shards — caller-supplied conditions that used to
    /// panic via an out-of-bounds index — this now returns a *documented
    /// degenerate result* instead: `data` is empty (never a plausible but
    /// wrong partial sum) and `participating_devices` still reports
    /// `shards.len()`. Callers that need to distinguish success from
    /// failure should use the `_checked` variant instead of inspecting
    /// `data.is_empty()`.
    pub fn all_reduce_sum(shards: &[Vec<f32>]) -> CollectiveResult {
        Self::all_reduce_sum_checked(shards).unwrap_or_else(|_| CollectiveResult {
            data: Vec::new(),
            participating_devices: shards.len(),
            op_name: "all_reduce_sum",
        })
    }

    /// All-reduce (max): element-wise maximum across all device tensors.
    ///
    /// Returns `Err(MultiGpuError::RaggedShards)` / `Err(EmptyShards)` under
    /// the same conditions as
    /// [`all_reduce_sum_checked`][Self::all_reduce_sum_checked] — never
    /// panics.
    pub fn all_reduce_max_checked(shards: &[Vec<f32>]) -> Result<CollectiveResult, MultiGpuError> {
        let n = check_uniform_shard_len(shards)?;
        let data: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                shards
                    .iter()
                    .map(|s| s[i])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        Ok(CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_reduce_max",
        })
    }

    /// Infallible convenience wrapper around
    /// [`all_reduce_max_checked`][Self::all_reduce_max_checked]; see
    /// [`all_reduce_sum`][Self::all_reduce_sum] for the degenerate-result
    /// contract on ragged/empty input.
    pub fn all_reduce_max(shards: &[Vec<f32>]) -> CollectiveResult {
        Self::all_reduce_max_checked(shards).unwrap_or_else(|_| CollectiveResult {
            data: Vec::new(),
            participating_devices: shards.len(),
            op_name: "all_reduce_max",
        })
    }

    /// All-gather: concatenate tensors from all devices in rank order.
    ///
    /// Unlike the all-reduce family, ragged shards are well-defined here —
    /// concatenation does not require equal lengths — so this stays
    /// infallible.
    pub fn all_gather(shards: &[Vec<f32>]) -> CollectiveResult {
        let data: Vec<f32> = shards.iter().flat_map(|s| s.iter().copied()).collect();
        CollectiveResult {
            data,
            participating_devices: shards.len(),
            op_name: "all_gather",
        }
    }

    /// **Scatter only** (despite the NCCL-style name kept for backward
    /// compatibility): splits a single flat `data` buffer into `world_size`
    /// contiguous, near-equal shards. No reduction is performed — there is
    /// only one input buffer, so there is nothing to reduce *across*.
    ///
    /// If `data.len()` is not evenly divisible by `world_size`, the first
    /// `data.len() % world_size` shards get one extra element.
    ///
    /// For a real reduce-then-scatter over **multiple** ranks' buffers
    /// (sum across ranks, then split the sum), use
    /// [`reduce_scatter_sum`][Self::reduce_scatter_sum] instead — that is
    /// the operation NCCL's `ncclReduceScatter` actually performs.
    pub fn scatter_equal_shards(data: &[f32], world_size: usize) -> Vec<Vec<f32>> {
        scatter_equal(data, world_size)
    }

    /// Real reduce-then-scatter: element-wise sums `shards` (one full
    /// buffer per rank, all the same length — see
    /// [`all_reduce_sum_checked`][Self::all_reduce_sum_checked]), then
    /// splits the summed buffer into `world_size` contiguous, near-equal
    /// pieces, exactly like [`scatter_equal_shards`][Self::scatter_equal_shards] does
    /// for a single buffer.
    ///
    /// Satisfies the decomposition identity
    /// `all_gather(reduce_scatter_sum(shards, n)?) == all_reduce_sum_checked(shards)?.data`
    /// (see this module's tests).
    ///
    /// # Errors
    ///
    /// Returns `Err(MultiGpuError::RaggedShards)` / `Err(EmptyShards)` under
    /// the same conditions as
    /// [`all_reduce_sum_checked`][Self::all_reduce_sum_checked].
    pub fn reduce_scatter_sum(
        shards: &[Vec<f32>],
        world_size: usize,
    ) -> Result<Vec<Vec<f32>>, MultiGpuError> {
        let reduced = Self::all_reduce_sum_checked(shards)?.data;
        Ok(scatter_equal(&reduced, world_size))
    }

    /// Broadcast: replicate `data` from device 0 to all `world_size` devices.
    pub fn broadcast(data: &[f32], world_size: usize) -> Vec<Vec<f32>> {
        (0..world_size).map(|_| data.to_vec()).collect()
    }
}

/// Split `data` into `world_size` contiguous, near-equal shards. Shared by
/// [`SimulatedCollectives::scatter_equal_shards`] and
/// [`SimulatedCollectives::reduce_scatter_sum`].
fn scatter_equal(data: &[f32], world_size: usize) -> Vec<Vec<f32>> {
    if world_size == 0 {
        return Vec::new();
    }
    let base = data.len() / world_size;
    let remainder = data.len() % world_size;
    (0..world_size)
        .map(|rank| {
            let start = rank * base + rank.min(remainder);
            let end = start + base + if rank < remainder { 1 } else { 0 };
            data[start..end.min(data.len())].to_vec()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight partition helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Partition a row-major weight matrix `[rows × cols]` into column-parallel shards.
///
/// Splits along the `cols` dimension, giving each device `cols / world_size`
/// (or `cols / world_size + 1` for the first few devices if not evenly divisible).
///
/// If `weights.len() != rows * cols` (a caller-supplied shape mismatch that
/// would otherwise index out of bounds), returns an empty `Vec` rather than
/// panicking; use
/// [`partition_weights_column_checked`] for a `Result` that reports the
/// mismatch.
pub fn partition_weights_column(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Vec<Vec<f32>> {
    partition_weights_column_checked(weights, rows, cols, world_size).unwrap_or_default()
}

/// Checked variant of [`partition_weights_column`].
///
/// # Errors
///
/// Returns `Err(MultiGpuError::WeightsLengthMismatch)` if
/// `weights.len() != rows * cols` — never panics.
pub fn partition_weights_column_checked(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Result<Vec<Vec<f32>>, MultiGpuError> {
    let expected = rows.saturating_mul(cols);
    if weights.len() != expected {
        return Err(MultiGpuError::WeightsLengthMismatch {
            expected,
            got: weights.len(),
        });
    }
    if world_size == 0 {
        return Ok(Vec::new());
    }
    let base_cols = cols / world_size;
    let remainder = cols % world_size;
    Ok((0..world_size)
        .map(|rank| {
            let col_start = rank * base_cols + rank.min(remainder);
            let shard_cols = base_cols + if rank < remainder { 1 } else { 0 };
            let mut shard = Vec::with_capacity(rows * shard_cols);
            for row in 0..rows {
                let row_base = row * cols;
                shard.extend_from_slice(
                    &weights[row_base + col_start..row_base + col_start + shard_cols],
                );
            }
            shard
        })
        .collect())
}

/// Partition a row-major weight matrix `[rows × cols]` into row-parallel shards.
///
/// Splits along the `rows` dimension, giving each device a contiguous block of rows.
///
/// If `weights.len() != rows * cols`, returns an empty `Vec` rather than
/// panicking; use [`partition_weights_row_checked`] for a `Result` that
/// reports the mismatch.
pub fn partition_weights_row(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Vec<Vec<f32>> {
    partition_weights_row_checked(weights, rows, cols, world_size).unwrap_or_default()
}

/// Checked variant of [`partition_weights_row`].
///
/// # Errors
///
/// Returns `Err(MultiGpuError::WeightsLengthMismatch)` if
/// `weights.len() != rows * cols` — never panics.
pub fn partition_weights_row_checked(
    weights: &[f32],
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Result<Vec<Vec<f32>>, MultiGpuError> {
    let expected = rows.saturating_mul(cols);
    if weights.len() != expected {
        return Err(MultiGpuError::WeightsLengthMismatch {
            expected,
            got: weights.len(),
        });
    }
    if world_size == 0 {
        return Ok(Vec::new());
    }
    let base_rows = rows / world_size;
    let remainder = rows % world_size;
    Ok((0..world_size)
        .map(|rank| {
            let row_start = rank * base_rows + rank.min(remainder);
            let shard_rows = base_rows + if rank < remainder { 1 } else { 0 };
            weights[row_start * cols..(row_start + shard_rows) * cols].to_vec()
        })
        .collect())
}

/// Merge column-parallel shards back into a single `[rows × cols]` weight matrix.
///
/// Assumes shards are produced by [`partition_weights_column`] with the same `rows`.
pub fn merge_column_shards(shards: &[Vec<f32>], rows: usize) -> Vec<f32> {
    if shards.is_empty() || rows == 0 {
        return Vec::new();
    }
    // Each shard: rows × (shard_cols)
    let total_cols: usize = shards.iter().map(|s| s.len() / rows).sum();
    let mut result = vec![0.0f32; rows * total_cols];

    let mut col_offset = 0usize;
    for shard in shards {
        let shard_cols = shard.len() / rows;
        for row in 0..rows {
            let dst_start = row * total_cols + col_offset;
            let src_start = row * shard_cols;
            result[dst_start..dst_start + shard_cols]
                .copy_from_slice(&shard[src_start..src_start + shard_cols]);
        }
        col_offset += shard_cols;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (CQ-11's evidence noted zero in-file tests here; CQ-10/CQ-16 fixes
// need direct coverage of the new checked/degenerate-result contracts)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- DeviceInfo / SimulatedDeviceMesh -----------------------------------

    #[test]
    fn device_info_compute_units_is_real_and_positive() {
        let mesh = SimulatedDeviceMesh::tensor_parallel(1);
        let dev = mesh.get(0, 0).expect("device 0 exists");
        // Must agree with the actual OS-reported parallelism, not a fixed
        // fabricated constant (previously always 108).
        let expected = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(dev.compute_units, expected);
    }

    #[test]
    fn device_mesh_alias_is_the_simulated_type() {
        // `DeviceMesh` must be usable exactly like `SimulatedDeviceMesh` —
        // it is a type alias, not a distinct compatible type.
        let via_alias = DeviceMesh::new(2, 2);
        let via_canonical = SimulatedDeviceMesh::new(2, 2);
        assert_eq!(via_alias.size(), via_canonical.size());
    }

    #[test]
    fn device_mesh_2d_indexing() {
        let mesh = SimulatedDeviceMesh::new(2, 3);
        assert_eq!(mesh.size(), 6);
        assert!(mesh.get(1, 2).is_some());
        assert!(
            mesh.get(2, 0).is_none(),
            "tp_rank 2 is out of bounds for tp_size=2"
        );
    }

    // --- SimulatedCollectives: checked all-reduce ---------------------------

    #[test]
    fn all_reduce_sum_checked_uniform_shards_ok() {
        let shards = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];
        let result = SimulatedCollectives::all_reduce_sum_checked(&shards)
            .expect("uniform shards must succeed");
        assert_eq!(result.data, vec![4.0, 6.0]);
    }

    #[test]
    fn all_reduce_sum_checked_ragged_shards_is_err() {
        let shards = vec![vec![1.0f32, 2.0], vec![3.0f32]];
        let err = SimulatedCollectives::all_reduce_sum_checked(&shards)
            .expect_err("ragged shards must be rejected, not panic");
        assert!(matches!(
            err,
            MultiGpuError::RaggedShards {
                index: 1,
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn all_reduce_sum_checked_empty_is_err() {
        let shards: Vec<Vec<f32>> = Vec::new();
        assert!(matches!(
            SimulatedCollectives::all_reduce_sum_checked(&shards),
            Err(MultiGpuError::EmptyShards)
        ));
    }

    #[test]
    fn all_reduce_max_checked_ragged_shards_is_err() {
        let shards = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0]];
        assert!(matches!(
            SimulatedCollectives::all_reduce_max_checked(&shards),
            Err(MultiGpuError::RaggedShards { .. })
        ));
    }

    /// CQ-10/CQ-16 regression: ragged shards used to panic inside a rayon
    /// worker via an out-of-bounds index. The infallible entry point must
    /// now return a documented degenerate (empty-data) result instead of
    /// panicking or fabricating a plausible-looking wrong sum.
    #[test]
    fn all_reduce_sum_infallible_ragged_shards_does_not_panic() {
        let shards = vec![vec![1.0f32, 2.0], vec![3.0f32]];
        let result = SimulatedCollectives::all_reduce_sum(&shards);
        assert!(
            result.data.is_empty(),
            "ragged input must produce an empty (unambiguous) result, got {:?}",
            result.data
        );
        assert_eq!(result.participating_devices, 2);
    }

    #[test]
    fn all_reduce_max_infallible_ragged_shards_does_not_panic() {
        let shards = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32]];
        let result = SimulatedCollectives::all_reduce_max(&shards);
        assert!(result.data.is_empty());
    }

    // --- scatter_equal_shards (scatter-only) / reduce_scatter_sum (real reduce) --

    #[test]
    fn scatter_equal_shards_is_scatter_only_no_reduction_claimed() {
        // Single-buffer scatter: covers all input, no reduction applied.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let shards = SimulatedCollectives::scatter_equal_shards(&data, 4);
        assert_eq!(shards.len(), 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, data.len());
    }

    /// The decomposition identity a real reduce-scatter must satisfy:
    /// gathering the scattered, reduced shards reconstructs exactly the
    /// all-reduced sum.
    #[test]
    fn reduce_scatter_sum_decomposition_identity() {
        let shards = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![10.0f32, 20.0, 30.0, 40.0],
            vec![100.0f32, 200.0, 300.0, 400.0],
        ];
        let world_size = 4;

        let scattered = SimulatedCollectives::reduce_scatter_sum(&shards, world_size)
            .expect("uniform shards must succeed");
        let gathered = SimulatedCollectives::all_gather(&scattered);

        let reduced = SimulatedCollectives::all_reduce_sum_checked(&shards)
            .expect("uniform shards must succeed");

        assert_eq!(
            gathered.data, reduced.data,
            "all_gather(reduce_scatter_sum(shards)) must equal all_reduce_sum(shards)"
        );
    }

    #[test]
    fn reduce_scatter_sum_ragged_shards_is_err() {
        let shards = vec![vec![1.0f32, 2.0], vec![3.0f32]];
        assert!(matches!(
            SimulatedCollectives::reduce_scatter_sum(&shards, 2),
            Err(MultiGpuError::RaggedShards { .. })
        ));
    }

    // --- partition/merge round-trip, including the ragged case -------------

    #[test]
    fn partition_merge_column_round_trip_ragged_world_size() {
        // 3 rows x 7 cols, world_size=3 => cols % world_size = 1 (ragged).
        let rows = 3;
        let cols = 7;
        let original: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let shards = partition_weights_column(&original, rows, cols, 3);
        assert_eq!(shards.len(), 3);
        let merged = merge_column_shards(&shards, rows);
        assert_eq!(merged, original, "ragged column partition must round-trip");
    }

    #[test]
    fn partition_weights_column_checked_rejects_length_mismatch() {
        let weights = vec![1.0f32, 2.0, 3.0]; // 3 elements, but rows*cols = 4
        let err = partition_weights_column_checked(&weights, 2, 2, 2)
            .expect_err("length mismatch must be rejected");
        assert!(matches!(
            err,
            MultiGpuError::WeightsLengthMismatch {
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn partition_weights_column_infallible_length_mismatch_returns_empty() {
        let weights = vec![1.0f32, 2.0, 3.0];
        let shards = partition_weights_column(&weights, 2, 2, 2);
        assert!(
            shards.is_empty(),
            "a shape mismatch must degrade to an empty result, not panic"
        );
    }

    #[test]
    fn partition_weights_row_checked_rejects_length_mismatch() {
        let weights = vec![1.0f32; 5];
        let err = partition_weights_row_checked(&weights, 2, 3, 2)
            .expect_err("length mismatch must be rejected");
        assert!(matches!(
            err,
            MultiGpuError::WeightsLengthMismatch {
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn partition_merge_row_round_trip() {
        let rows = 8;
        let cols = 4;
        let original: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let shards = partition_weights_row(&original, rows, cols, 3);
        assert_eq!(shards.len(), 3);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, original.len());
    }
}
