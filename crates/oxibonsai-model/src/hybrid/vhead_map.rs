//! V-head grouping for Gated-DeltaNet layers — a pure **index map**, never a
//! materialised permutation (design §3.3).
//!
//! # The two index spaces
//!
//! A `qwen35` linear-attention layer has `n_v_heads` value heads sharing
//! `n_k_heads` key/query heads (27B: 48 over 16, so `v_per_k == 3`). Two
//! conventions for "which k-head does v-head *i* read" appear in the same
//! checkpoint:
//!
//! * **Tiled** — the order every v-indexed GGUF *row* is stored in
//!   (`attn_qkv`'s v block, `attn_gate`, `ssm_alpha`/`ssm_beta` outputs,
//!   `ssm_a`, `ssm_dt.bias`, and the v channels of `ssm_conv1d`):
//!   v-head `j` reads k-head `j % n_k_heads`, i.e. the Qwen3.5 layout
//!   `[rep][n_k][head_dim]` = `[k0v0, k1v0, …, k0v1, k1v1, …]`.
//! * **Grouped** — the order `ssm_out`'s *columns* are in, and the order
//!   this crate keeps the recurrent state in: v-head `m` reads k-head
//!   `m / v_per_k`, i.e. `[n_k][rep][head_dim]`.
//!
//! `prism.hadamard.gdn_v_grouped = true` announces the grouped `ssm_out`:
//! the columns of a Hadamard-**folded** matrix cannot be permuted after the
//! fold, so PrismML permuted them *before* folding and left every row tensor
//! tiled.
//!
//! # Why an index map and not a permuted copy
//!
//! `attn_qkv` (10240×5120) and `attn_gate` (6144×5120) are ~22 MB per layer,
//! ×48 layers ≈ 1 GB. Materialising permuted copies would cost that in
//! anonymous RAM and lose page-cache sharing of the mmap'd file on a 24 GB
//! box, to save an index computation that is a single `usize` lookup. So:
//! **permute nothing; re-index at slice time** — the design's own wording.
//!
//! # Integration contract (the silent-failure class)
//!
//! `oxibonsai_kernels::gated_delta_net`'s default entry points
//! (`gdn_step_f32`, `gdn_step`, `gdn_prefill_f32`, `gdn_chunk`) assume
//! [`GdnHeadOrder::Grouped`]. Handing them raw GGUF (tiled) `v`/gate buffers
//! is **not detectable by the kernel** — it silently pairs each v-head with
//! the wrong k-head (a mutation check measured golden cosine
//! dropping to 0.47). This module pins the **grouped** convention:
//!
//! * v-indexed activations are gathered tiled → grouped through
//!   [`VHeadMap::gather_grouped`] / [`VHeadMap::gather_grouped_scalar`]
//!   before the kernel call;
//! * the recurrent state and the kernel output stay grouped, so the write
//!   into `ssm_out`'s input is contiguous and needs no scatter.
//!
//! `vhead_map_matches_grouped_kernel_order` pins that choice
//! against [`GdnHeadOrder`] itself, and
//! `tiled_buffers_pair_the_wrong_k_head_under_grouped_order`
//! pins the failure mode the contract exists to prevent.

use oxibonsai_kernels::gated_delta_net::{GdnDims, GdnHeadOrder};

use crate::error::{ModelError, ModelResult};

/// Maps a **grouped** v-head index (`m = k * rep + r`, the order `ssm_out`
/// and the recurrent state use) to the **tiled** v-head index
/// (`j = r * n_k + k`, the order the GGUF rows use), and back.
///
/// `grouped_to_tiled[m] = (m % rep) * n_k + (m / rep)`, which is exactly
/// PrismML `runtime.py::vperm` with `unit = 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VHeadMap {
    /// `grouped_to_tiled[m]` — the GGUF row index of grouped v-head `m`.
    grouped_to_tiled: Vec<u32>,
    /// `tiled_to_grouped[j]` — the grouped index of GGUF row `j`.
    tiled_to_grouped: Vec<u32>,
    /// Gated-DeltaNet k/q heads (`qwen35.ssm.group_count`).
    n_k_heads: usize,
    /// Gated-DeltaNet v heads (`qwen35.ssm.time_step_rank`).
    n_v_heads: usize,
    /// `n_v_heads / n_k_heads`.
    rep: usize,
}

impl VHeadMap {
    /// Build the map for `n_v_heads` v-heads over `n_k_heads` k-heads.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] when either count is zero or
    /// `n_v_heads` is not a whole multiple of `n_k_heads` (the repeat would
    /// not be an integer and neither direction of the map would be a
    /// bijection).
    pub fn new(n_k_heads: usize, n_v_heads: usize) -> ModelResult<Self> {
        if n_k_heads == 0 || n_v_heads == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "VHeadMap".to_string(),
                expected: "n_k_heads > 0 and n_v_heads > 0".to_string(),
                actual: format!("n_k_heads = {n_k_heads}, n_v_heads = {n_v_heads}"),
            });
        }
        if !n_v_heads.is_multiple_of(n_k_heads) {
            return Err(ModelError::ShapeInvariant {
                tensor: "VHeadMap".to_string(),
                expected: format!("n_v_heads a whole multiple of n_k_heads ({n_k_heads})"),
                actual: format!("n_v_heads = {n_v_heads}"),
            });
        }
        let n_v_u32 = u32::try_from(n_v_heads).map_err(|_| ModelError::ShapeInvariant {
            tensor: "VHeadMap".to_string(),
            expected: "n_v_heads representable as u32".to_string(),
            actual: format!("n_v_heads = {n_v_heads}"),
        })?;
        let rep = n_v_heads / n_k_heads;
        let mut grouped_to_tiled = vec![0u32; n_v_heads];
        let mut tiled_to_grouped = vec![0u32; n_v_heads];
        for (grouped, slot) in grouped_to_tiled.iter_mut().enumerate() {
            let tiled = (grouped % rep) * n_k_heads + (grouped / rep);
            // Both indices are < n_v_heads, which is `u32`-representable.
            *slot = u32::try_from(tiled).unwrap_or(n_v_u32);
            if let Some(inverse) = tiled_to_grouped.get_mut(tiled) {
                *inverse = u32::try_from(grouped).unwrap_or(n_v_u32);
            }
        }
        Ok(Self {
            grouped_to_tiled,
            tiled_to_grouped,
            n_k_heads,
            n_v_heads,
            rep,
        })
    }

    /// [`VHeadMap::new`], refusing the one fold shape that has no supported
    /// index math (design §3.3).
    ///
    /// `gdn_v_grouped == false` means `ssm_out` was folded with its columns
    /// in tiled order, so its input must stay tiled while the state and the
    /// kernel work grouped. With `v_per_k == 1` the two orders coincide and
    /// the flag is immaterial; otherwise this is
    /// [`ModelError::UngroupedFoldedGdnOutput`] — the same refusal PrismML's
    /// `runtime.py` makes, rather than untested index math.
    ///
    /// A model with **no** Hadamard fold at all (gen-1 `qwen35` files) has
    /// no `gdn_v_grouped` key; pass `true` there — nothing was folded, so
    /// nothing constrains the column order and the grouped convention this
    /// package pins applies unchanged.
    ///
    /// # Errors
    ///
    /// [`ModelError::UngroupedFoldedGdnOutput`], or anything
    /// [`VHeadMap::new`] reports.
    pub fn for_fold(n_k_heads: usize, n_v_heads: usize, gdn_v_grouped: bool) -> ModelResult<Self> {
        let map = Self::new(n_k_heads, n_v_heads)?;
        if !gdn_v_grouped && !map.is_identity() {
            return Err(ModelError::UngroupedFoldedGdnOutput {
                n_k_heads,
                n_v_heads,
            });
        }
        Ok(map)
    }

    /// The GGUF (tiled) row index of grouped v-head `grouped`.
    ///
    /// Out-of-range indices return `grouped` unchanged rather than panicking:
    /// the legal domain is `0..n_v_heads`, every caller in this crate proves
    /// its index in range before the loop (see [`VHeadMap::gather_grouped`]),
    /// and keeping the accessor branch-free matters in the per-token inner
    /// loop. Mirrors `GdnHeadOrder::k_head`'s own saturating style. A
    /// `debug_assert!` still flags an out-of-range caller in tests/debug
    /// builds — it is a caller bug, not a data-dependent condition, so
    /// asserting costs nothing in the release build this accessor is tuned
    /// for.
    #[inline]
    #[must_use]
    pub fn tiled(&self, grouped: usize) -> usize {
        debug_assert!(
            grouped < self.n_v_heads,
            "VHeadMap::tiled: grouped index {grouped} is out of range for n_v_heads {}",
            self.n_v_heads
        );
        self.grouped_to_tiled
            .get(grouped)
            .map_or(grouped, |&j| j as usize)
    }

    /// The grouped index of GGUF (tiled) row `tiled` — the inverse of
    /// [`VHeadMap::tiled`], with the same out-of-range behaviour (including
    /// the debug-only assertion).
    #[inline]
    #[must_use]
    pub fn grouped(&self, tiled: usize) -> usize {
        debug_assert!(
            tiled < self.n_v_heads,
            "VHeadMap::grouped: tiled index {tiled} is out of range for n_v_heads {}",
            self.n_v_heads
        );
        self.tiled_to_grouped
            .get(tiled)
            .map_or(tiled, |&m| m as usize)
    }

    /// The k/q head that grouped v-head `grouped` reads (`grouped / rep`).
    ///
    /// Identical to [`GdnHeadOrder::Grouped`]'s own rule by construction;
    /// `vhead_map_matches_grouped_kernel_order` pins the two together.
    #[inline]
    #[must_use]
    pub fn k_head(&self, grouped: usize) -> usize {
        grouped / self.rep.max(1)
    }

    /// `true` when tiled and grouped order coincide (`v_per_k == 1`), so
    /// every gather below is a straight copy.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.rep == 1
    }

    /// Gated-DeltaNet k/q heads.
    #[inline]
    #[must_use]
    pub fn n_k_heads(&self) -> usize {
        self.n_k_heads
    }

    /// Gated-DeltaNet v heads.
    #[inline]
    #[must_use]
    pub fn n_v_heads(&self) -> usize {
        self.n_v_heads
    }

    /// v-heads per k-head (`n_v_heads / n_k_heads`).
    #[inline]
    #[must_use]
    pub fn v_per_k(&self) -> usize {
        self.rep
    }

    /// The kernel geometry this map belongs to.
    #[must_use]
    pub fn gdn_dims(&self, head_k_dim: usize, head_v_dim: usize) -> GdnDims {
        GdnDims::new(self.n_k_heads, self.n_v_heads, head_k_dim, head_v_dim)
    }

    /// Re-index a per-v-head **vector** activation from GGUF (tiled) order
    /// into grouped order: `dst[m * head_dim ..] = src[tiled(m) * head_dim ..]`.
    ///
    /// This is the call that turns a raw `attn_qkv` v block (or `attn_gate`'s
    /// `z`) into something the grouped Gated-DeltaNet entry points may be
    /// handed. `head_dim` is `head_v_dim` (27B: 128), so both buffers are
    /// `n_v_heads * head_dim` long (27B: 6144).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when either buffer is not exactly
    /// `n_v_heads * head_dim` long.
    pub fn gather_grouped(
        &self,
        src_tiled: &[f32],
        head_dim: usize,
        dst_grouped: &mut [f32],
    ) -> ModelResult<()> {
        let width = self.expect_width(src_tiled.len(), dst_grouped.len(), head_dim)?;
        debug_assert_eq!(width, self.n_v_heads * head_dim);
        for grouped in 0..self.n_v_heads {
            let tiled = self.tiled(grouped);
            let (src_lo, dst_lo) = (tiled * head_dim, grouped * head_dim);
            let (src_slice, dst_slice) = (
                src_tiled.get(src_lo..src_lo + head_dim),
                dst_grouped.get_mut(dst_lo..dst_lo + head_dim),
            );
            match (src_slice, dst_slice) {
                (Some(s), Some(d)) => d.copy_from_slice(s),
                // Unreachable: `expect_width` proved both buffers exactly
                // `n_v_heads * head_dim` long and every index above is
                // `< n_v_heads`. Reported rather than `unwrap`ped so the
                // invariant is enforced by the type system, not by comment.
                _ => {
                    return Err(ModelError::ShapeInvariant {
                        tensor: "VHeadMap::gather_grouped".to_string(),
                        expected: format!("head {grouped} -> {tiled} in range"),
                        actual: format!("n_v_heads = {}", self.n_v_heads),
                    })
                }
            }
        }
        Ok(())
    }

    /// [`VHeadMap::gather_grouped`] for a **scalar**-per-v-head tensor
    /// (`ssm_alpha`/`ssm_beta` outputs, `ssm_a`, `ssm_dt.bias`):
    /// `dst[m] = src[tiled(m)]`.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when either buffer is not exactly
    /// `n_v_heads` long.
    pub fn gather_grouped_scalar(
        &self,
        src_tiled: &[f32],
        dst_grouped: &mut [f32],
    ) -> ModelResult<()> {
        self.expect_width(src_tiled.len(), dst_grouped.len(), 1)?;
        for grouped in 0..self.n_v_heads {
            let value = src_tiled.get(self.tiled(grouped)).copied().unwrap_or(0.0);
            if let Some(slot) = dst_grouped.get_mut(grouped) {
                *slot = value;
            }
        }
        Ok(())
    }

    /// Inverse of [`VHeadMap::gather_grouped`]: write a grouped-order
    /// activation back into GGUF (tiled) order.
    ///
    /// Not used on the decode path — the state is kept grouped precisely so
    /// the Gated-DeltaNet output feeds `ssm_out` with no scatter — but it is
    /// what makes the map a provable bijection rather than a one-way
    /// convention, and it is the fallback a future `gdn_v_grouped = false`
    /// checkpoint would need.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when either buffer is not exactly
    /// `n_v_heads * head_dim` long.
    pub fn scatter_tiled(
        &self,
        src_grouped: &[f32],
        head_dim: usize,
        dst_tiled: &mut [f32],
    ) -> ModelResult<()> {
        self.expect_width(src_grouped.len(), dst_tiled.len(), head_dim)?;
        for grouped in 0..self.n_v_heads {
            let tiled = self.tiled(grouped);
            let (src_lo, dst_lo) = (grouped * head_dim, tiled * head_dim);
            let (src_slice, dst_slice) = (
                src_grouped.get(src_lo..src_lo + head_dim),
                dst_tiled.get_mut(dst_lo..dst_lo + head_dim),
            );
            match (src_slice, dst_slice) {
                (Some(s), Some(d)) => d.copy_from_slice(s),
                _ => {
                    return Err(ModelError::ShapeInvariant {
                        tensor: "VHeadMap::scatter_tiled".to_string(),
                        expected: format!("head {grouped} -> {tiled} in range"),
                        actual: format!("n_v_heads = {}", self.n_v_heads),
                    })
                }
            }
        }
        Ok(())
    }

    /// Both buffers must be exactly `n_v_heads * head_dim` long; returns
    /// that width.
    fn expect_width(&self, src_len: usize, dst_len: usize, head_dim: usize) -> ModelResult<usize> {
        let width = self.n_v_heads * head_dim;
        for (role, len) in [("src", src_len), ("dst", dst_len)] {
            if len != width {
                return Err(ModelError::ShapeMismatch {
                    name: format!("VHeadMap {role} buffer"),
                    expected: vec![width],
                    actual: vec![len],
                });
            }
        }
        Ok(width)
    }
}

/// The head order this module hands the Gated-DeltaNet kernels, pinned as a
/// constant so a future edit has to change a named value rather than a call
/// argument (see the module docs' integration contract).
pub const GDN_HEAD_ORDER: GdnHeadOrder = GdnHeadOrder::Grouped;

#[cfg(test)]
mod tests {
    use super::*;

    /// 27B geometry.
    const NK: usize = 16;
    const NV: usize = 48;
    const HD: usize = 128;

    #[test]
    fn round_trips_both_directions() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        for grouped in 0..NV {
            assert_eq!(
                map.grouped(map.tiled(grouped)),
                grouped,
                "grouped {grouped}"
            );
        }
        for tiled in 0..NV {
            assert_eq!(map.tiled(map.grouped(tiled)), tiled, "tiled {tiled}");
        }
        // Both directions are permutations of 0..NV.
        let mut seen = vec![false; NV];
        for grouped in 0..NV {
            let tiled = map.tiled(grouped);
            assert!(tiled < NV);
            assert!(!seen[tiled], "tiled index {tiled} produced twice");
            seen[tiled] = true;
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn matches_runtime_py_vperm_formula() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        assert_eq!(map.v_per_k(), 3);
        for grouped in 0..NV {
            let rep = NV / NK;
            assert_eq!(map.tiled(grouped), (grouped % rep) * NK + (grouped / rep));
        }
        // Spot values: grouped 0,1,2 are k-head 0's three v-heads and land
        // on GGUF rows 0, 16, 32 (`[rep][n_k][hd]`).
        assert_eq!(map.tiled(0), 0);
        assert_eq!(map.tiled(1), 16);
        assert_eq!(map.tiled(2), 32);
        assert_eq!(map.tiled(3), 1);
        assert_eq!(map.k_head(0), 0);
        assert_eq!(map.k_head(2), 0);
        assert_eq!(map.k_head(3), 1);
        assert_eq!(map.k_head(47), 15);
    }

    /// Integration contract, half 1: re-indexing a tiled buffer through
    /// this map and then calling a **grouped** kernel entry point pairs every
    /// v-head with the same physical k-head the fork's tiled convention does.
    #[test]
    fn vhead_map_matches_grouped_kernel_order() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        let dims = map.gdn_dims(HD, HD);
        for grouped in 0..NV {
            // What the map says.
            let k = map.k_head(grouped);
            // What the kernel's grouped order says for the same index.
            assert_eq!(GdnHeadOrder::Grouped.k_head(grouped, &dims), k);
            // What the fork's tiled order says for the GGUF row this
            // grouped head was gathered from — the physical ground truth.
            assert_eq!(GdnHeadOrder::Tiled.k_head(map.tiled(grouped), &dims), k);
        }
        assert_eq!(GDN_HEAD_ORDER, GdnHeadOrder::Grouped);
    }

    /// Integration contract, half 2: the silent failure the gather
    /// exists to prevent. Handing raw (tiled) buffers to a grouped entry
    /// point is not an error the kernel can detect — it just reads the wrong
    /// k-head for most v-heads.
    #[test]
    fn tiled_buffers_pair_the_wrong_k_head_under_grouped_order() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        let dims = map.gdn_dims(HD, HD);
        let coincide: Vec<usize> = (0..NV)
            .filter(|&j| {
                GdnHeadOrder::Grouped.k_head(j, &dims) == GdnHeadOrder::Tiled.k_head(j, &dims)
            })
            .collect();
        // `j / 3 == j % 16` holds for exactly four of the 48 v-heads, so 44
        // of them would silently read the wrong k-head.
        assert_eq!(coincide, vec![0, 23, 24, 47]);
        assert_eq!(
            NV - coincide.len(),
            44,
            "only the 4 v-heads whose two indices coincide would survive a \
             tiled buffer fed to a grouped kernel"
        );
    }

    #[test]
    fn identity_when_one_v_head_per_k_head() {
        let map = VHeadMap::new(8, 8).expect("valid geometry");
        assert!(map.is_identity());
        for h in 0..8 {
            assert_eq!(map.tiled(h), h);
            assert_eq!(map.grouped(h), h);
            assert_eq!(map.k_head(h), h);
        }
    }

    #[test]
    fn gather_and_scatter_are_inverse() {
        let map = VHeadMap::new(4, 12).expect("valid geometry");
        let head_dim = 3;
        let src: Vec<f32> = (0..12 * head_dim).map(|i| i as f32).collect();
        let mut grouped = vec![0.0f32; src.len()];
        map.gather_grouped(&src, head_dim, &mut grouped)
            .expect("gather");
        // Grouped head 1 is tiled head (1 % 3) * 4 + (1 / 3) = 4.
        assert_eq!(map.tiled(1), 4);
        assert_eq!(
            &grouped[head_dim..2 * head_dim],
            &src[4 * head_dim..5 * head_dim]
        );

        let mut back = vec![0.0f32; src.len()];
        map.scatter_tiled(&grouped, head_dim, &mut back)
            .expect("scatter");
        assert_eq!(back, src);
    }

    #[test]
    fn gather_scalar_follows_the_same_map() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        let src: Vec<f32> = (0..NV).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; NV];
        map.gather_grouped_scalar(&src, &mut dst).expect("gather");
        for (grouped, value) in dst.iter().enumerate() {
            assert_eq!(*value, map.tiled(grouped) as f32);
        }
    }

    #[test]
    fn gather_rejects_a_mis_sized_buffer() {
        let map = VHeadMap::new(NK, NV).expect("valid geometry");
        let src = vec![0.0f32; NV * HD];
        let mut dst = vec![0.0f32; NV * HD - 1];
        let err = map
            .gather_grouped(&src, HD, &mut dst)
            .expect_err("short destination must be rejected");
        assert_eq!(err.error_code(), "SHAPE_MISMATCH");
    }

    #[test]
    fn rejects_a_non_multiple_geometry() {
        let err = VHeadMap::new(5, 48).expect_err("48 is not a multiple of 5");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
        assert!(VHeadMap::new(0, 48).is_err());
        assert!(VHeadMap::new(16, 0).is_err());
    }

    /// The design §3.3 guard, and the package's acceptance criterion.
    #[test]
    fn refuses_an_ungrouped_fold_when_v_heads_outnumber_k_heads() {
        let err = VHeadMap::for_fold(NK, NV, false).expect_err("must be refused");
        assert_eq!(err.error_code(), "UNGROUPED_FOLDED_GDN_OUTPUT");
        let message = err.to_string();
        assert!(message.contains("48 v-heads"), "{message}");
        assert!(message.contains("16 k-heads"), "{message}");

        // Grouped folds are accepted, and an ungrouped fold is immaterial
        // when the two orders coincide.
        assert!(VHeadMap::for_fold(NK, NV, true).is_ok());
        assert!(VHeadMap::for_fold(NK, NK, false).is_ok());
    }
}
