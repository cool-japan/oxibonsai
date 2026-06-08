//! Isolated CUDA TQ2 GEMV parity probe (debugging Blackwell garbage output).
//!
//! Compares the CUDA `gemv_tq2_g128_v1` kernel (via the self-contained
//! `CudaGraph::encode_lm_head_gemv_tq2` upload→kernel→download path) against the
//! CPU scalar reference `gemv_tq2_0_g128`, on the same deterministic ternary
//! weights. If they diverge, the bug is in the TQ2 GEMV kernel or the AoS→SoA
//! reformat — not in the fused full-forward orchestration.
//!
//! Run:
//!   cargo test -p oxibonsai-kernels --features native-cuda \
//!     --test cuda_tq2_gemv_parity -- --nocapture

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod cuda_tq2 {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128;
    use oxibonsai_kernels::CudaGraph;

    /// Deterministic ternary blocks: vary qs codes (covering 0,1,2,3) and the
    /// FP16 scale across blocks so the matrix is "interesting".
    fn make_blocks(n_rows: usize, blocks_per_row: usize) -> Vec<BlockTQ2_0_g128> {
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for bk in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (byte_idx, b) in qs.iter_mut().enumerate() {
                    let seed = row * 31 + bk * 17 + byte_idx;
                    let c0 = (seed % 3) as u8;
                    let c1 = ((seed / 3) % 3) as u8;
                    let c2 = ((seed / 9) % 3) as u8;
                    let c3 = ((seed / 27) % 3) as u8;
                    *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                }
                let scale = 0.25_f32 + 0.5_f32 * ((row * 7 + bk * 3) % 101) as f32 / 101.0;
                blocks.push(BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(scale),
                });
            }
        }
        blocks
    }

    fn aos_bytes(blocks: &[BlockTQ2_0_g128]) -> &[u8] {
        let ptr = blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(blocks);
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    fn run_case(n_rows: usize, k: usize, handle_id: u64) {
        let blocks_per_row = k / 128;
        let blocks = make_blocks(n_rows, blocks_per_row);
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 0.3).collect();

        // CPU reference.
        let mut cpu = vec![0f32; n_rows];
        gemv_tq2_0_g128(&blocks, &input, &mut cpu, n_rows, k).expect("cpu gemv");

        // CUDA path.
        let graph = match CudaGraph::global() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no CUDA device ({e}) — skipping case n_rows={n_rows} k={k}");
                return;
            }
        };
        let cuda = graph
            .encode_lm_head_gemv_tq2(&input, handle_id, aos_bytes(&blocks), n_rows, k)
            .expect("cuda gemv");

        // Compare.
        let mut max_abs = 0f32;
        let mut max_idx = 0usize;
        for (i, (a, b)) in cpu.iter().zip(cuda.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        eprintln!(
            "── n_rows={n_rows} k={k} (blocks_per_row={blocks_per_row}) ──\n\
             max|Δ| = {max_abs:.6e} at row {max_idx}\n\
             cpu[0..6]  = {:?}\n\
             cuda[0..6] = {:?}\n\
             cpu[{max_idx}]={} cuda[{max_idx}]={}",
            &cpu[..6.min(n_rows)],
            &cuda[..6.min(n_rows)],
            cpu[max_idx],
            cuda[max_idx],
        );
        assert!(
            max_abs < 1e-2,
            "CUDA TQ2 GEMV diverges from CPU reference: max|Δ|={max_abs:.6e} at row {max_idx} (n_rows={n_rows}, k={k})"
        );
    }

    /// Real-8B inner dimension (hidden=4096 → 32 blocks/row).
    #[test]
    fn cuda_tq2_gemv_parity_k4096() {
        run_case(512, 4096, 0xDEAD_0001);
    }

    /// Small shape (k=256 → 2 blocks/row), like the Metal unit test.
    #[test]
    fn cuda_tq2_gemv_parity_k256() {
        run_case(64, 256, 0xDEAD_0002);
    }

    /// 1.7B-ish inner dim (hidden=2048 → 16 blocks/row).
    #[test]
    fn cuda_tq2_gemv_parity_k2048() {
        run_case(256, 2048, 0xDEAD_0003);
    }
}
