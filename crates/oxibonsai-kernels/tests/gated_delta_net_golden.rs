//! Gated DeltaNet (B2-05) — golden, parity and contract tests.
//!
//! What is checked here, in the order the package's acceptance list names it:
//!
//! 1. **Golden parity.** `golden_matches_fork_transliteration_over_32_steps`
//!    compares 32 recurrence steps (outputs *and* the final state) against a
//!    dump produced by an independent C++ oracle.
//! 2. **Single-pass == three-pass.** Bitwise, which is stronger than the 1e-6
//!    the spec asks for — the fused kernel is built from primitives that leave
//!    exactly the bits the split primitives leave.
//! 3. **Prefill == T × step.** Bitwise, for both the small and the real
//!    Bonsai 2 geometry.
//! 4. **Versus an O(T²) naive reference**, ≤1e-4, including a 64-step
//!    state-drift run.
//! 5. **Softplus cutoff exactly 20.0**, end to end through the kernel.
//! 6. **`ssm_a` consumed as-is**: a positive element is rejected, not logged.
//! 7. The kda per-channel branch, the two head orders, cross-tier parity, the
//!    non-square geometry and the length contract.
//!
//! # How the golden was produced (regeneration recipe)
//!
//! `GOLDEN_OUT` / `GOLDEN_STATE` come from `scratchpad/gdn_golden_gen.cpp`, a
//! literal transliteration of the PrismML llama.cpp fork's CPU kernel
//! `fork/models/gated_delta_net_cpu.cpp.txt:108-160` (the per-token body of
//! `ggml_compute_forward_gated_delta_net_one_chunk`) with ggml's three scalar
//! vector helpers — `ggml_vec_scale_f32`, `ggml_vec_mad_f32` and
//! `ggml_vec_dot_f32` (which accumulates in `ggml_float` = `double`) — spelled
//! out. It is built and run outside the repository with
//!
//! ```text
//! clang++ -O2 -std=c++17 -ffp-contract=off -o gdn_golden_gen gdn_golden_gen.cpp
//! ./gdn_golden_gen > gdn_golden.rs.txt
//! ```
//!
//! (Apple clang 21.0.0.) `-ffp-contract=off` is required: Rust never contracts
//! `a*b + c` into an FMA, so the C++ side must not either, otherwise the
//! deterministic input construction below would not be reproduced bit for bit.
//!
//! The oracle is an independent *implementation*, not a dump from a built
//! `llama.cpp` binary — the fork is staged as sources only, with no CUDA/Metal
//! or ggml build on this machine. It is therefore checked with `cos ≥ 0.9999`
//! (the acceptance threshold) rather than bitwise: its dots accumulate in
//! `double` while the kernel accumulates in `f32` lanes.
//!
//! The inputs are *not* embedded — both sides derive them from the same
//! splitmix64 stream with the same exact-in-binary32 arithmetic, so only the
//! results need to be pinned.
//!
//! The dump uses the fork's own head map (`ik1 = iv1 % nek1`), so it also pins
//! [`GdnHeadOrder::Tiled`]; `grouped_and_tiled_orders_agree_under_the_vhead_map`
//! ties that to the grouped order the model layer uses.

use oxibonsai_kernels::error::KernelError;
use oxibonsai_kernels::gated_delta_net::{
    gdn_chunk, gdn_prefill_f32, gdn_prefill_with, gdn_step, gdn_step_f32, gdn_step_with, softplus,
    validate_a_neg, GdnBeta, GdnDecay, GdnDims, GdnGates, GdnHeadOrder, GdnPath, GdnState,
};
use oxibonsai_kernels::gated_delta_net_chunk::{gdn_prefill_with_tier, GdnTier};

// ─── Golden dump (see the module docs for the regeneration recipe) ───

const GOLDEN_OUT: [u32; 1024] = [
    0x3c188bf9, 0x3c7b85b5, 0x3c3c6b06, 0xbd137ad9, 0x3d018290, 0xbc4a5888, 0xbcc17425, 0xbca45453,
    0xbc09e044, 0xbce28186, 0x3d06b48e, 0xbc8392e9, 0xbca6bf64, 0x3d0f1363, 0xbc42beef, 0x3c5200c1,
    0xbbb6342a, 0x3cd19a3d, 0xbaadd366, 0x3cb98ef8, 0x3c8bdda2, 0xbbedbbe2, 0x3c8bb62f, 0x3cc7454a,
    0xbd89019d, 0xbd0fb373, 0xbd0d57a4, 0x3d7d38a2, 0x3d488dea, 0x3c1c5475, 0xbcbc9fea, 0xbc369ae9,
    0x3d01ac18, 0x3c7fe3b7, 0x3cb5f798, 0xbcb7bad5, 0xba9bcb0c, 0xbbf09701, 0x3b7536b5, 0xb98aa7e0,
    0x3cebdb62, 0x3d0ca513, 0xbc021f6f, 0x3d1699b4, 0x3d0000bc, 0x3c9f3fb6, 0xbc9cdfa2, 0xbd2c6209,
    0xbc125186, 0x3c046434, 0x3b0d2266, 0xbc19a7c5, 0x3b8f2633, 0x3c002c6b, 0x3b6c6a95, 0xbc34735b,
    0x3bfcb8e5, 0xbc0f114e, 0xbb8fbe70, 0x3be0e9a9, 0x3b942804, 0xbc0467ff, 0xbc527b5f, 0xbc34fc61,
    0x3d01cf71, 0xbbf71897, 0x3c98b32a, 0xbc78b1c3, 0xbaf36bd4, 0x3c6054bd, 0xb9d1b164, 0xba4966bd,
    0xbcdcb61a, 0xbd5fa0b6, 0x3c4e39ad, 0xbd3a1afa, 0xbd516cb2, 0xbd8f17c9, 0x3cc28cfe, 0x3d81d409,
    0xba2376a1, 0x3bf5202b, 0x3bbbd908, 0x3b96f63e, 0xbb2bcdfe, 0x3c142697, 0xbc4811c0, 0xb9954b3f,
    0xbb8dea77, 0x3c8e8e8e, 0x3c875946, 0xbc735096, 0xbc31bfea, 0x3bd197c5, 0x3ca10393, 0x3c03298f,
    0xba5c026f, 0x3b099169, 0x3bb3bb2c, 0xbca80f83, 0x3c1c1937, 0xbadb5aff, 0xbb72f0e7, 0xbb871324,
    0xbd328beb, 0xbd188444, 0x3d1b32ee, 0xbd0b21c1, 0xbcf6efa2, 0xbcfef27c, 0xbd093ac5, 0x3d2fdefb,
    0xbacd2b8a, 0x3b03ce6a, 0x3986dad5, 0x3ba3469d, 0xbae7b249, 0x3af8fe4b, 0xbb33ea5a, 0x3bb30e2b,
    0xbb2ccbca, 0x3c22bb0d, 0x3c1a850d, 0xbbd94038, 0xbb2728bd, 0xbb527c7c, 0x3be5c8c5, 0xbb2440c8,
    0xbc86d7f8, 0x3cc919fc, 0xbce87532, 0xbca2f0ae, 0xbc9a4a1a, 0xbcea8847, 0x3d08a61c, 0xbc6723a6,
    0x3c4119fa, 0xbb992586, 0xbcefbdbb, 0xbd094cba, 0xbd5b8f92, 0xbd214d90, 0x3c807e22, 0x3cb36a16,
    0xbc7d170d, 0xbbe81f96, 0xbc399432, 0x3caab144, 0xbc8c41d6, 0xbb3a4614, 0xbc683d0e, 0x3cb8b135,
    0xbc8d0e38, 0x3b1da734, 0xbc00a03e, 0x3c09d3bc, 0xbc2aaa23, 0x3aac1909, 0x3ba048aa, 0x3cfb0c44,
    0xba8a054f, 0x3cbd629c, 0x3c844c70, 0xbc63dd64, 0xbbdcd291, 0x3bc56b41, 0xbbf9d841, 0x3b20b514,
    0x3ceca46f, 0x3ccab7bb, 0xbcc58347, 0x3c7af108, 0x3d164509, 0x3d51bf16, 0x3d340f1a, 0x3aa5658e,
    0x3c5e6eab, 0xbcaa4ee3, 0x3b65c325, 0x3c2ae6c2, 0xbcca3f9f, 0xbc6fbb1e, 0xbcde76d2, 0xbc8da512,
    0x3c1cd132, 0xbc4fe142, 0x3c8f923c, 0xbc73faf8, 0x398b2eef, 0x3c708510, 0xbc7ccfe1, 0xbd13b3da,
    0x3d477764, 0x3d4a2f84, 0xbcc5fb1b, 0xbda26356, 0x3c8ce588, 0xbd9f41a8, 0x3d4334c1, 0xbcbf175f,
    0x3a93538f, 0x3d1e85b0, 0x3c8107cd, 0x3cfd5bd9, 0x3d15f7ff, 0xbc67b6dc, 0xbbb7f925, 0x3b592ff9,
    0xbb7ca441, 0x3befab53, 0xbbffa9bb, 0x3ca870f4, 0x3c620a70, 0xbcdfe92b, 0x3bafb5ef, 0xbbdc16f9,
    0x3c1988f1, 0x3c0b82f7, 0x39e67fed, 0x3c5b2ad7, 0x3c2b79ee, 0x3a99cda9, 0x3c254518, 0xbab8bd20,
    0xbbc79f5c, 0xbc4ce28d, 0xbb4d06ec, 0x3c167000, 0xbb01023e, 0x3b724890, 0x3bb9fc1a, 0xbb2136a5,
    0x3d0b584d, 0x3d135e3d, 0xbd2ec49d, 0xbd248872, 0x3d0a00c9, 0xbd115a21, 0xbc36f20f, 0xbc1b3a95,
    0xbbe3e583, 0xbc1f028a, 0x3b0799d4, 0x3bd6ae2f, 0x3c2a1cbd, 0xbbcbde01, 0xbc3f8de0, 0xbc4db559,
    0x3c12d40a, 0x3d2c117b, 0xbbaaca5b, 0x3d0ab18f, 0x3d28166e, 0xbc57edc0, 0xbd829869, 0x3d0d1723,
    0x3a38f658, 0x3b8bc91e, 0x3a98afdf, 0xbb1703a7, 0xbb966667, 0x399cff9d, 0x3b4cfa73, 0xba65bd9a,
    0xb9800b43, 0x3cac6e49, 0xbcc5e80c, 0xbcc2f48a, 0x3cb5041d, 0x3bd346db, 0x3bd6f4cd, 0xbb1b7cc1,
    0xb817dc41, 0xba3de4a2, 0x3b257596, 0xb9e93bcc, 0xbb5dff03, 0x3b2b9599, 0xbb0d1441, 0xba0cc1ea,
    0x384f6917, 0x3cb665e5, 0x3bb583ae, 0x3c90dc1f, 0x3c9a2622, 0xbb4bc9c6, 0xbcf1ffed, 0x3c8ec3a7,
    0x3b1fe55c, 0xbc964c7c, 0xbc591a0e, 0xbcb33525, 0xbc936a26, 0x3bfd4789, 0xbbf82a68, 0xbbccab43,
    0xbc864856, 0x3c13a5d7, 0x3bbd0462, 0x3c8537e0, 0x3bb445a6, 0xbc71c1ab, 0x3c85b7fa, 0xbc8c265f,
    0x3c00be03, 0xbbca525a, 0x3c2db410, 0x3b465fc2, 0x3c592bc4, 0xbc82ffef, 0x3bc2b063, 0xbbdbab49,
    0x3cf70d1f, 0xbd1e6bab, 0x3d1178df, 0xb9cee9e9, 0x3cf775a2, 0x3ca0dd56, 0x3b2f7bdb, 0x3cb9f422,
    0xbc827eb1, 0x3cc5f6ab, 0x3d1b290f, 0x3cd11b18, 0x3d2bb526, 0xbc9d517b, 0x3c8de778, 0x3c99b384,
    0x3be9a9ee, 0xbbb31a46, 0x3a49f200, 0x3c1b260a, 0xbc0dce5e, 0xbc89c5ac, 0xbacb1754, 0xbbac855a,
    0x3b8b3fca, 0xbcdf87da, 0xbd3a728c, 0xbcbb0958, 0x3b441d03, 0xbc9523e7, 0xbd1b9b20, 0x3bda77f8,
    0x3be3d21f, 0xbb8e6359, 0xb9ca5c96, 0x3bcb0823, 0x3baa1b8c, 0x3bb110ed, 0xba9d4050, 0x3b8b0f01,
    0xbb8a5b95, 0x3b9ec313, 0x3c1a82c0, 0x3b2fd620, 0x3b815aa9, 0xbc03055f, 0xbb7c677a, 0x3b485200,
    0xbd4ad563, 0x3d0c7016, 0xbc3f9e3f, 0xbca2e29d, 0x3b8ba12b, 0xbd64db25, 0xbcda930a, 0x3d3731de,
    0xb8b7ca41, 0xbcfe1d1f, 0xbb63c81d, 0xbc985c24, 0x3bc1e614, 0xbb77a408, 0xbd2d36ba, 0xbc11a56d,
    0xbd3e667a, 0x3cf6e5f8, 0x3cb43bb4, 0x39a38e9e, 0xbd29b3a0, 0x3c409ff5, 0x3ce0be81, 0x3cfe0d41,
    0xbb8b9358, 0xbd6975f4, 0x3cd4dd54, 0xbcc2eda1, 0x3c98afcf, 0x3d532bf8, 0xbb4db0b0, 0xbca71cbd,
    0xbc92869f, 0x3c7ae093, 0xbc3ca9e1, 0xbc74f2d3, 0x3bca4cd2, 0x3b12bca6, 0xbc66531c, 0xbaf89f1c,
    0xbaa17ffa, 0xbb33c021, 0x3cc7ecf4, 0x3b1e0c06, 0x3c11756b, 0xbce1a54b, 0x3c958627, 0xbc2c0817,
    0xbcb467d9, 0x3ccfd58b, 0xbbc587dd, 0xbb9744f7, 0xbc48f57e, 0xba85efbf, 0x3ae39544, 0x3b151d3f,
    0x3da97543, 0x3d1d6329, 0xbdc9cea2, 0x3d2a65c0, 0xbdbc5c02, 0xbdb6ed64, 0x3d5e7adb, 0xbd1e69b0,
    0x3d668450, 0xbd02c13c, 0xbd052672, 0x3d8f52c9, 0xbcd1178d, 0xbd360d1d, 0xbd2dc570, 0xbd64bff8,
    0x3cd356a2, 0x3d238109, 0x3d458999, 0x3cacfcc5, 0x3bf7c080, 0xbd526d8e, 0x3d27b902, 0xbd45f799,
    0x3d6596aa, 0x3d9b7d3f, 0x3d960d84, 0xbd78f046, 0x3cc3e64f, 0xbd7a160f, 0xbdcb6f86, 0x3d32e0c0,
    0xbc4b1aa0, 0xbd18c3e3, 0x3d5cfc72, 0x3b08563e, 0x3d3b0281, 0x3ae2f885, 0x3d916056, 0xbd82ffcf,
    0xbbfb58bc, 0x3bf82980, 0x3c0a99d0, 0xbc466157, 0x3c4529ac, 0x3bf36e18, 0x3d05615e, 0x3cad743d,
    0xbc5662ad, 0x3bcae6dd, 0x3d26631d, 0xbc9e221c, 0x3a2bb06e, 0xbc5dec66, 0xbc181f71, 0xbc709f33,
    0xbcad65ba, 0xbcc38040, 0xbc8bc017, 0x3c47bcee, 0xbbc77ccb, 0x3c1902de, 0x3cf48ded, 0xbc375beb,
    0xbd067408, 0x3be29b93, 0x3d6b689a, 0x3c60605e, 0xbb653e26, 0x3bceaa30, 0x3c8cb036, 0xbd0b9c74,
    0xbd3e71ad, 0x3db80b90, 0xbcd466cb, 0x3d9f0f26, 0xbc774c8a, 0xbce0822a, 0x3d1537a9, 0x3d176ff7,
    0x3bab21f1, 0x3c808a90, 0x3c357b97, 0xbc6365bb, 0xbace7e93, 0xbc4aacac, 0xbc382b33, 0xbc531d9e,
    0xbcf95e39, 0xbcf8758c, 0x3d23e378, 0xbc9c77ad, 0x3cd4b9b8, 0x3cf2d108, 0x3cf70f3f, 0xbd2670af,
    0xbc5974c7, 0xbcb3e30d, 0x3d120357, 0xbb9674a4, 0x3d14ca99, 0x3c57a255, 0x3d09ab99, 0xbcd043da,
    0xbcaa0b69, 0x3d104671, 0xbc0ebb0a, 0x3cbe15e6, 0xbb54ce13, 0xbba80cf6, 0x3c455af0, 0x3c5460ad,
    0xbbf48d73, 0xbb83c0e8, 0x3bfce647, 0xbc3bd786, 0xbb16d726, 0xba6cdad9, 0xbb9f8556, 0x3b85f23a,
    0xbb0eb209, 0xbb0bfd1a, 0x3a3ae171, 0xb82d811f, 0x3a99f6a3, 0x3b20ea00, 0x3b30b2c1, 0xbb19c440,
    0xbbd91b63, 0xbd33e962, 0xb9b7bf24, 0xbc58c54f, 0x3d561f4b, 0xbcbc0d6d, 0x3bc1cce0, 0x3c84369e,
    0x3b1866c0, 0xbaa28d79, 0x3ba6f0f4, 0x3c283389, 0xbb89a400, 0x3bb6edd0, 0x3b5d7710, 0x3bceb904,
    0xbc2b8d93, 0x3abf1ed7, 0x3babb857, 0xbb85ece7, 0xbb5aa556, 0x3b061d2f, 0xbbb5783c, 0x3c18d1fb,
    0x3ca4273f, 0x3bb266dc, 0x3b147c95, 0xbba08eff, 0x3b517da4, 0xbc0a8c56, 0x3b0b9902, 0x3c59cef0,
    0xbd5cf8ec, 0xbc6be382, 0xbc384454, 0xba2ca3ae, 0x3c822cae, 0xbc160d9e, 0xbadf04b6, 0x3be551ef,
    0xbd083c8e, 0xbc3b3dc3, 0x3c984502, 0xbb89b104, 0x3c9fd77e, 0x3bf66d6d, 0xbc4929f5, 0x3c3ebf83,
    0xbc006f22, 0x3c2e3c6b, 0xbacb4e27, 0x3c1f034b, 0x3c205c37, 0xbc9e4fa7, 0xbbd58d86, 0xbb93a406,
    0x3cd4a9a7, 0x3c343332, 0xbb467968, 0xba2fae30, 0x3c431606, 0xbc029e11, 0xbbccf79c, 0x3c59bdd6,
    0xb98c84e7, 0xbbeae16f, 0xbc30b9c0, 0xbc526fdb, 0x3bedbb1e, 0xbca26cec, 0xbc716715, 0x3c737b39,
    0x3d02ee39, 0x3b06d823, 0x3b9bf790, 0x3c97ed43, 0xbc52da81, 0x3c966777, 0x3ccfb613, 0x3c92a16f,
    0x3bf5eb98, 0x3bfe6cff, 0xbc2b566f, 0xba7746db, 0x3a6d4d17, 0x39a28190, 0x3c483d09, 0xbc299e41,
    0x3d0d54ae, 0xbc38a4f3, 0x3d12d32b, 0xbd3e735f, 0xbc4f1301, 0xbd09c95f, 0xbba2cb09, 0x3d1971bc,
    0xbd4b02f3, 0x3d9171ba, 0x3dd029b1, 0x3d2d6aa0, 0x3cd39097, 0xbb8092e4, 0xbd85122b, 0x3cf571a1,
    0xbd055e1f, 0x3cf7bca2, 0x3ba61e48, 0x3b8a09a9, 0xbbc4c94b, 0x3b01dd8b, 0xbc9904e8, 0x3cb4d1f9,
    0xbc1443bd, 0x3c922f39, 0xbd0776f7, 0xbaca4764, 0x3ab1bd08, 0x3c802a73, 0xb939d321, 0xbcb856b1,
    0xbd3087dd, 0x3ca81fa9, 0xbd442688, 0x3d8e0b80, 0x3ca27834, 0x3d4a84b1, 0x3bbdbb2a, 0xbd490c4c,
    0x3d12dd70, 0xbd0ea455, 0xbd44b3c5, 0xbd154718, 0x3cda39c0, 0xbc917f2a, 0xbb1f62e5, 0xbc6a8d90,
    0xbd05126a, 0x3cae3afc, 0x3d752fd4, 0x3ce3275f, 0x3cc4de2f, 0xbc6aa0f8, 0x3d64ebcd, 0xbce14207,
    0xb62d46c1, 0xbbc49412, 0x3c1be2e8, 0x3a3c7a92, 0xbb7b96c7, 0xb9cda32a, 0x39377224, 0x3bd81c33,
    0xbc398834, 0x3bc3b993, 0x3c8e3575, 0xbc7f3f52, 0xbcea74f5, 0xbad61a1f, 0x3cf391a6, 0xbc5428df,
    0xbc2db18b, 0x3b92978b, 0x3cfd9692, 0x3cafe831, 0x3d49c667, 0x3b39e1a2, 0xbcf66960, 0x3c0c08ee,
    0xbcc03369, 0x3ca33420, 0xbbb25959, 0xbc1030fd, 0xbc0c4094, 0x3b9def5d, 0x3c2d590a, 0x389d9757,
    0xbc96b6ce, 0xbb8525b4, 0x3c3fa591, 0x3c892d52, 0xbb040acb, 0x3bcc8ab8, 0x3bf46582, 0x3cad7e9a,
    0xbd0428c8, 0x3b9974dd, 0x3c2c0fa5, 0xbcb16b00, 0x3c1d7c49, 0x3c8575ca, 0x3d0160f1, 0x3c6ffa5d,
    0x3c124d9f, 0x3a4bb1ad, 0x3a8b50c6, 0x3ce6db7f, 0xbbab1a0d, 0xbc2692f2, 0x3c1e65a0, 0x3c52db67,
    0x3c260b47, 0x3be2d1af, 0x3cfd89a1, 0xbcc750ed, 0x3b131649, 0xbd3219a7, 0xbb0ddc23, 0x3a93ce16,
    0x3c0886d6, 0x3c052c58, 0x38165493, 0xbc6e0d8b, 0x3b0a73da, 0xbba20c2b, 0xbc2ecfc2, 0xbbe9f393,
    0x3cb381e6, 0xbbcba07c, 0x3c303ab2, 0x3c9d4e35, 0x3c9e4f07, 0xbcb4f343, 0xbcdb0621, 0xbcaa384b,
    0x3d187b59, 0xbd11a9ef, 0x3d0b2af1, 0x3c7fe1ba, 0xbc16b42a, 0xbd31e285, 0xbce45e5f, 0xbaede44d,
    0x3bcbea2b, 0x3b91d2a7, 0xbb586354, 0xbbd0b8bc, 0xbc2973d8, 0xbc4aca1f, 0x3b93e101, 0xbb5a1cfc,
    0x39fc6595, 0x3c74966c, 0xb9e7620c, 0xbb38660d, 0x3c4c58eb, 0xbc4575ea, 0xbbb67816, 0xbc0a20a5,
    0x3c458d7e, 0xbc47d6f6, 0x3c64e415, 0x3b1b7b98, 0xbcec05b6, 0xbc2baca2, 0xbcade380, 0x3c910a6d,
    0x3b9cdcdd, 0x3a2606c7, 0x3bf7f1f5, 0xbb173fef, 0xbc27c582, 0xbbf60692, 0xbb52dd22, 0x3c1b220f,
    0xbb006455, 0x3bea6c8d, 0xbb120721, 0x3b6c41c1, 0x3c9011ab, 0x3b0d3073, 0xbab8b219, 0x3ade424a,
    0xbba8aca5, 0xbb9f7fe1, 0xbbee45e8, 0xbac5ab36, 0xbc43fcfe, 0x3a987332, 0x3812430a, 0xbbf7b67f,
    0xbb1f599e, 0x3c214722, 0xbca6117e, 0x3b1d5d23, 0x3c957343, 0x3c4b537a, 0x3c3ffec1, 0xbc870071,
    0xba3ef8c8, 0xbbf366eb, 0x3bee663b, 0x3b7de1ad, 0x3bdba46f, 0xbbbdd9cc, 0xbb7deffb, 0xbb508290,
    0xbc2a9408, 0xbc90152f, 0xbcc3d336, 0xbc6d1efd, 0x3a5e26ba, 0xbc85fc8d, 0x3b078512, 0xbcfa3499,
    0x39b1765e, 0xbc4c39df, 0x3b9c1891, 0x3b3d8449, 0x3c85f617, 0xbbeb09d5, 0xbc90c2e3, 0xbbdf2cdb,
    0xbbfe2d10, 0xb947e918, 0x3c062b7f, 0x3a979c7f, 0x3b922562, 0xbb41c92d, 0x3b96c40e, 0x3c639a39,
    0xba537506, 0x39bf884f, 0xbb1367e3, 0xbb3bffb4, 0xbc09c931, 0x3bde37f7, 0xbb8a5a1e, 0xbb9cb7e4,
    0xbc90f061, 0x3bb29d76, 0x3b02e2ec, 0xbaafae3c, 0xbcb3d587, 0x3bf5778e, 0x3aa3c662, 0x3bd56792,
    0x3a7fddf4, 0xbb82f088, 0xbcbcec4b, 0x3c3c44c4, 0x3b019465, 0x3be1abc7, 0x3c3b69ab, 0x3aae4bd9,
    0xbced0fcf, 0xbc2f5097, 0x3cf4282e, 0xbab46519, 0xbc386700, 0x3c438ea4, 0x3b3c7b7e, 0x3cf42414,
    0x3bafa40d, 0x3bcf46a8, 0xb8c000ce, 0xbbfd7006, 0x3bc33a39, 0xbc3f07ad, 0xbb5215cd, 0x3c602636,
    0xbc9d309e, 0x3c661b3d, 0x3c23ec90, 0x3abe585b, 0xbceb545c, 0x3c73a2c9, 0x3b3d9aad, 0x3ca5ae05,
    0x3c21bbe8, 0x3bcac9af, 0x3ca43190, 0xbc976750, 0xbcaecf5d, 0xbbf20990, 0xbb11ef22, 0xbc3806cd,
    0xbd19984c, 0xbc78c5b1, 0x3d25630f, 0xbb23246f, 0xbc8e4100, 0x3c83fcf1, 0x3b76509c, 0x3d22567a,
    0xbb86d51d, 0x3c08a49b, 0xbc5e1936, 0x3b366eed, 0x3b8bfd31, 0xbb4d2a62, 0x3b2fdb47, 0xba43929b,
    0x3bc92814, 0xbb47f471, 0xbbe207ec, 0xbc11a38e, 0x3c2c9437, 0xbc228067, 0x3bde8674, 0xbbd0ff39,
    0x3c3c6843, 0xbba964d0, 0xbc766241, 0x3c37ee8a, 0x3bc3e934, 0x3bdcc651, 0x3bce00fb, 0x3b956504,
    0x3c9f72d2, 0x3a93790d, 0xbc030316, 0xbb8d22f7, 0xb975e7ca, 0xbc2afb67, 0x3bd50265, 0xbc2b430b,
    0x3c29393f, 0xbb5bf0de, 0xbc0eea75, 0xbb3e0487, 0xb879d99c, 0x3a80da89, 0xbc10270b, 0x3b837fa3,
    0xbba43dae, 0xbc801cea, 0xba6485c3, 0xbab82e47, 0xbc2685b5, 0x3c755911, 0xbb16dc4f, 0xbc302f7a,
    0x3d0af94d, 0xbccc0a04, 0xbcaeeafd, 0x3c6f8da7, 0xbcb93459, 0x3b105fa4, 0x3cd7b959, 0xbc925908,
    0x3ce26578, 0xbbf7a826, 0xbd19e204, 0xbc0675a7, 0x3a8bd728, 0xbcf915da, 0xbc8fa014, 0xbd469859,
    0x3c135117, 0xbca7afbd, 0x398112e7, 0xbc76168d, 0xbc7bb7c1, 0x3ce9fe2f, 0x3c61d4f2, 0xbc06e916,
    0x3d16bd2d, 0x3cdd538e, 0x3c677314, 0xbd52aa34, 0x3c6fd8d6, 0xbcc862da, 0xbcd2da25, 0xbd269a43,
    0x3bb2f383, 0xbd712174, 0x3d6dc54c, 0xbbc6c1ae, 0x3d6e8f31, 0xbd850b74, 0xbda1bc50, 0x3dab26e6,
    0x3d3e88bf, 0x3c6cf46d, 0x3d038d97, 0xbccbf781, 0x3b7be471, 0x3c99afcc, 0xbca3607d, 0xbcced89a,
];
const GOLDEN_STATE: [u32; 256] = [
    0x3bae605f, 0xbc6ae057, 0x3d5dcb02, 0xbb023078, 0x3c9c0636, 0x3c8bc69e, 0xbc8c5862, 0xbd43e2c6,
    0xbd030a92, 0xbc135960, 0xbe039781, 0xbd2f6a04, 0xbd23383f, 0xbd84beb1, 0x3de6728f, 0x3dd6c596,
    0x3c090d4f, 0xbc1705c0, 0xbdacc02f, 0xbd3c137c, 0x3d21da2e, 0xbd004071, 0x3d38eaf8, 0x3d9ca663,
    0xbbd877f7, 0x3cba622f, 0xbc2b185a, 0x3d45a2da, 0xbd2d63f8, 0x3d127f96, 0x3b8ce958, 0x3cf2851e,
    0xbd11ed31, 0xbbf1da5c, 0xbdbb0992, 0xbce7b206, 0xbd1aac78, 0xbd478ab6, 0x3da66640, 0x3da2923c,
    0x3d1c0f93, 0xbc95ba52, 0x3db7cb4d, 0xba44b640, 0x3d8f2aff, 0x3c7bb6e0, 0xbdaa0e6e, 0xbdb65693,
    0x3c855b23, 0x3ce4b17d, 0x3d90470d, 0x3ce58115, 0x3c4401f6, 0x3c8a1282, 0xbdd19b38, 0xbd76d89e,
    0xbca57efb, 0xbb36aff0, 0xbd119665, 0xbcbfdc8b, 0xbc9c1a5c, 0xbcbb6042, 0x3d3cc6cb, 0x3cf7ad28,
    0xbcd47111, 0x3dfe788c, 0xbde95ca4, 0x3bc05260, 0x3d9e0fa9, 0xbe378aa7, 0xbd31d181, 0xbdf11437,
    0xbd9095d5, 0x3d295266, 0xbdf4af1f, 0xbd21843c, 0x3cc03790, 0xbdf2d06a, 0xbe17fcd4, 0xbe03d900,
    0x3bc13629, 0x3d8b94f3, 0xbd7c8a58, 0x3ce0cbc0, 0x3cba34c7, 0xbd6c4882, 0xbc1c9f74, 0xbce271fa,
    0xbda804a5, 0xbe6b0c3b, 0x3dfd06e4, 0xbe07e44d, 0xbdf8ed07, 0x3e60f910, 0xbe087b11, 0x3cabc7e2,
    0xbda307f6, 0x39156000, 0xbd46662b, 0xbdb6ef71, 0x3cc070d4, 0xbdcda5c2, 0xbdfc19f7, 0xbe0790c2,
    0x3da097cc, 0xbc872f2d, 0x3d9ce98a, 0x3d812d4e, 0xbd0f44e5, 0x3e030441, 0x3e03cfc9, 0x3e0d1b1c,
    0xbd42e194, 0xbe1240d9, 0x3d887c60, 0xbdc7bc67, 0xbd88d4af, 0x3de96a0d, 0xbd95b8e6, 0x3c843494,
    0xbda380dc, 0xbe4dd133, 0x3da029c4, 0xbe3839f4, 0xbdf9c329, 0x3e17fa9f, 0xbe2a3c06, 0xbca311b8,
    0xbd2f6149, 0xbbaa4150, 0x3e71ef31, 0x3d92e25e, 0xbdba14e0, 0x3d637d88, 0xbdb60d41, 0xbe6c1dcc,
    0x3ccb209f, 0x3e0a4ac5, 0xbdeda83f, 0x3de5c392, 0xbdc884fc, 0x3e0e73ba, 0x3cb9ea20, 0x3e55e903,
    0x3ca3ed62, 0xbd01c1b3, 0xbdf02698, 0xbe7284d7, 0x3e4c50b4, 0xbe4944bc, 0xbd02f43e, 0x3da40792,
    0xbcb4a6fc, 0xbd1702ba, 0x3da26fd7, 0x3da4f09b, 0xbd86bb8d, 0x3cfcab52, 0x3b60d8ba, 0xbdcf65a6,
    0x3bf9dc5e, 0xbdcd392a, 0xbdd99d2f, 0xbe2209e6, 0x3e158095, 0xbe6847e7, 0xbc942b24, 0xbbb8c0e0,
    0xbcbba301, 0x3d9535ac, 0xbc8ee6fb, 0x3e318ca2, 0xbe351274, 0x3e1576d6, 0x3d816de3, 0x3d98bf98,
    0xbd40d187, 0x3b769c60, 0xbdb26337, 0x3def42d0, 0xbe3c2dd2, 0x3df7e049, 0x3e6d1994, 0x3e067480,
    0x3d7817a4, 0xbc846852, 0x3e4bef3e, 0xbd2ba9f6, 0x3e1aa66a, 0xbd94b754, 0xbe930f70, 0xbe7c9365,
    0xbd54772d, 0x3e5b002e, 0xbdac6079, 0x3d0c7738, 0x3e41772d, 0xbe81ddcd, 0x3d24560a, 0xbe3836b4,
    0xbdbec8f6, 0x3d0caa2a, 0xbd225dad, 0xbd9aac61, 0x3d3b796f, 0xbe04124b, 0xbe01dc2d, 0xbe35c6ab,
    0x3deae573, 0x3da08e92, 0xbe09567a, 0x3d501b7e, 0xbd0c9727, 0xbdd03e61, 0x3d03be92, 0x3d90aad7,
    0xbd7c6880, 0xbdfadc87, 0x3d89d150, 0xbdb88ea8, 0xbd771248, 0x3da4d566, 0xbdd35d37, 0xbd08013f,
    0xbda25e37, 0x3c986b90, 0xbaaee350, 0xbd32b281, 0x3d1f4e9e, 0xbd8f6b73, 0xbdab1b74, 0xbe05ebe9,
    0xbd151dc8, 0x3c06d2ad, 0xbdd2c38a, 0xbcb1a881, 0xbcb947c1, 0xbd9862e2, 0xbe0ee15c, 0xbd9b9b2c,
    0xbdd7c34a, 0xbe344f94, 0x3cda9b70, 0xbe353325, 0xbda8090c, 0x3d9935f5, 0xbe402242, 0xbd7654e0,
    0xbd0f9866, 0xbe7885dc, 0x3b825500, 0xbe0e0f8d, 0xbe391a30, 0x3e1c16b9, 0xbe4ec7b7, 0x3d57e7ba,
];

// ─── Deterministic inputs (mirror of the oracle's construction) ──────

const G_NK: usize = 2;
const G_NV: usize = 4;
const G_HK: usize = 8;
const G_HV: usize = 8;
const G_T: usize = 32;
const GOLDEN_SEED: u64 = 0x0DDB_A11C_0FFE_E123;

/// splitmix64 — integer-only, so C++ and Rust agree bit for bit.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-0.5, 0.5)`; 24 bits over 2²⁴ is exact in binary32.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 / 16_777_216.0 - 0.5
    }
}

fn l2_normalize(x: &mut [f32]) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0f32 / (ss + 1e-6f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// One layer's worth of activations and gates for `t_len` tokens.
struct Inputs {
    state: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    dt_bias: Vec<f32>,
    a_neg: Vec<f32>,
}

impl Inputs {
    fn gates(&self) -> GdnGates<'_> {
        GdnGates::bonsai2(&self.alpha, &self.beta, &self.dt_bias, &self.a_neg)
    }
}

/// The exact input construction of `scratchpad/gdn_golden_gen.cpp`.
fn golden_inputs() -> Inputs {
    let mut rng = SplitMix64::new(GOLDEN_SEED);
    let mut state = vec![0.0f32; G_NV * G_HV * G_HK];
    for s in state.iter_mut() {
        *s = rng.next_f32() * 0.25;
    }
    let mut dt_bias = vec![0.0f32; G_NV];
    let mut a_neg = vec![0.0f32; G_NV];
    for h in 0..G_NV {
        dt_bias[h] = 0.35f32 - 0.20f32 * h as f32;
        a_neg[h] = -(0.5f32 + 0.25f32 * h as f32);
    }

    let mut q = vec![0.0f32; G_T * G_NK * G_HK];
    let mut k = vec![0.0f32; G_T * G_NK * G_HK];
    let mut v = vec![0.0f32; G_T * G_NV * G_HV];
    let mut alpha = vec![0.0f32; G_T * G_NV];
    let mut beta = vec![0.0f32; G_T * G_NV];
    for t in 0..G_T {
        for h in 0..G_NK {
            let off = (t * G_NK + h) * G_HK;
            for i in 0..G_HK {
                q[off + i] = rng.next_f32();
            }
            for i in 0..G_HK {
                k[off + i] = rng.next_f32();
            }
            l2_normalize(&mut q[off..off + G_HK]);
            l2_normalize(&mut k[off..off + G_HK]);
        }
        for h in 0..G_NV {
            let off = (t * G_NV + h) * G_HV;
            for i in 0..G_HV {
                v[off + i] = rng.next_f32();
            }
        }
        for h in 0..G_NV {
            alpha[t * G_NV + h] = rng.next_f32() * 4.0;
            beta[t * G_NV + h] = rng.next_f32() * 4.0;
        }
    }

    Inputs {
        state,
        q,
        k,
        v,
        alpha,
        beta,
        dt_bias,
        a_neg,
    }
}

/// Deterministic pseudo-random inputs for an arbitrary geometry.
fn random_inputs(dims: &GdnDims, t_len: usize, seed: u64) -> Inputs {
    let mut rng = SplitMix64::new(seed);
    let mut state = vec![0.0f32; dims.state_len()];
    for s in state.iter_mut() {
        *s = rng.next_f32() * 0.25;
    }
    let mut dt_bias = vec![0.0f32; dims.n_v_heads];
    let mut a_neg = vec![0.0f32; dims.n_v_heads];
    for h in 0..dims.n_v_heads {
        // Spans the softplus cutoff: head 0 sits far below it, the last head
        // far above once `alpha` is added.
        dt_bias[h] = 0.25f32 + 1.5f32 * h as f32;
        a_neg[h] = -(0.125f32 + 0.0625f32 * (h % 5) as f32);
    }

    let mut q = vec![0.0f32; t_len * dims.qk_len()];
    let mut k = vec![0.0f32; t_len * dims.qk_len()];
    let mut v = vec![0.0f32; t_len * dims.v_len()];
    let mut alpha = vec![0.0f32; t_len * dims.n_v_heads];
    let mut beta = vec![0.0f32; t_len * dims.n_v_heads];
    for t in 0..t_len {
        for h in 0..dims.n_k_heads {
            let off = (t * dims.n_k_heads + h) * dims.head_k_dim;
            for i in 0..dims.head_k_dim {
                q[off + i] = rng.next_f32();
            }
            for i in 0..dims.head_k_dim {
                k[off + i] = rng.next_f32();
            }
            l2_normalize(&mut q[off..off + dims.head_k_dim]);
            l2_normalize(&mut k[off..off + dims.head_k_dim]);
        }
        for i in 0..dims.v_len() {
            v[t * dims.v_len() + i] = rng.next_f32();
        }
        for h in 0..dims.n_v_heads {
            alpha[t * dims.n_v_heads + h] = rng.next_f32() * 6.0;
            beta[t * dims.n_v_heads + h] = rng.next_f32() * 4.0;
        }
    }

    Inputs {
        state,
        q,
        k,
        v,
        alpha,
        beta,
        dt_bias,
        a_neg,
    }
}

// ─── Comparison helpers ──────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine over different lengths");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "diff over different lengths");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths differ");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i} ({x} vs {y})");
    }
}

fn bits(words: &[u32]) -> Vec<f32> {
    words.iter().copied().map(f32::from_bits).collect()
}

// ─── Independent dense f64 reference ─────────────────────────────────

/// Replays the gated delta rule from `s0` for `t_end + 1` tokens with dense,
/// **non-transposed** `[head_k_dim][head_v_dim]` matrices in `f64`.
///
/// `S_t = decay_t · (I − β_t k_t k_tᵀ) · S_{t−1} + β_t k_t v_tᵀ`,
/// `o_t = S_tᵀ q_t / sqrt(head_v_dim)`.
///
/// Every index, loop order and accumulator type differs from the kernel, so
/// agreement is evidence about the recurrence rather than about a shared
/// implementation. Returns `(output at t_end, final state in kernel layout)`.
#[allow(clippy::too_many_arguments)]
fn naive_dense(
    inputs: &Inputs,
    dims: &GdnDims,
    order: GdnHeadOrder,
    t_end: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (hk, hv, nk, nv) = (
        dims.head_k_dim,
        dims.head_v_dim,
        dims.n_k_heads,
        dims.n_v_heads,
    );
    let scale = 1.0f64 / (hv as f64).sqrt();
    let mut out = vec![0.0f32; nv * hv];
    let mut final_state = vec![0.0f32; dims.state_len()];

    for h in 0..nv {
        let k_head = order.k_head(h, dims);
        // s[i * hv + j] == S[i][j]
        let mut s = vec![0.0f64; hk * hv];
        for i in 0..hk {
            for j in 0..hv {
                s[i * hv + j] = f64::from(inputs.state[h * hv * hk + j * hk + i]);
            }
        }

        for t in 0..=t_end {
            let idx = t * nv + h;
            let beta = 1.0 / (1.0 + (-f64::from(inputs.beta[idx])).exp());
            let x = f64::from(inputs.alpha[idx]) + f64::from(inputs.dt_bias[h]);
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            let decay = (f64::from(inputs.a_neg[h]) * sp).exp();

            let qk_off = (t * nk + k_head) * hk;
            let q_t = &inputs.q[qk_off..qk_off + hk];
            let k_t = &inputs.k[qk_off..qk_off + hk];
            let v_t = &inputs.v[idx * hv..idx * hv + hv];

            // u[j] = Σ_i S[i][j] k[i], on the state *before* the decay.
            let mut u = vec![0.0f64; hv];
            for (j, u_j) in u.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for i in 0..hk {
                    acc += s[i * hv + j] * f64::from(k_t[i]);
                }
                *u_j = acc;
            }
            for i in 0..hk {
                let k_i = f64::from(k_t[i]);
                for j in 0..hv {
                    let delta = beta * (f64::from(v_t[j]) - decay * u[j]);
                    s[i * hv + j] = decay * s[i * hv + j] + k_i * delta;
                }
            }
            if t == t_end {
                for j in 0..hv {
                    let mut acc = 0.0f64;
                    for i in 0..hk {
                        acc += s[i * hv + j] * f64::from(q_t[i]);
                    }
                    out[h * hv + j] = (acc * scale) as f32;
                }
            }
        }

        for i in 0..hk {
            for j in 0..hv {
                final_state[h * hv * hk + j * hk + i] = s[i * hv + j] as f32;
            }
        }
    }
    (out, final_state)
}

/// kda reference: the decay is per key-channel, `S[i][:] *= exp(g[i])`.
fn naive_dense_per_channel(
    inputs: &Inputs,
    g: &[f32],
    dims: &GdnDims,
    t_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (hk, hv, nk, nv) = (
        dims.head_k_dim,
        dims.head_v_dim,
        dims.n_k_heads,
        dims.n_v_heads,
    );
    let scale = 1.0f64 / (hv as f64).sqrt();
    let mut out = vec![0.0f32; t_len * nv * hv];
    let mut final_state = vec![0.0f32; dims.state_len()];

    for h in 0..nv {
        let k_head = GdnHeadOrder::Grouped.k_head(h, dims);
        let mut s = vec![0.0f64; hk * hv];
        for i in 0..hk {
            for j in 0..hv {
                s[i * hv + j] = f64::from(inputs.state[h * hv * hk + j * hk + i]);
            }
        }
        for t in 0..t_len {
            let idx = t * nv + h;
            let beta = 1.0 / (1.0 + (-f64::from(inputs.beta[idx])).exp());
            let qk_off = (t * nk + k_head) * hk;
            let q_t = &inputs.q[qk_off..qk_off + hk];
            let k_t = &inputs.k[qk_off..qk_off + hk];
            let v_t = &inputs.v[idx * hv..idx * hv + hv];
            let g_t = &g[idx * hk..idx * hk + hk];

            for i in 0..hk {
                let decay_i = f64::from(g_t[i]).exp();
                for j in 0..hv {
                    s[i * hv + j] *= decay_i;
                }
            }
            let mut u = vec![0.0f64; hv];
            for (j, u_j) in u.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for i in 0..hk {
                    acc += s[i * hv + j] * f64::from(k_t[i]);
                }
                *u_j = acc;
            }
            for i in 0..hk {
                let k_i = f64::from(k_t[i]);
                for j in 0..hv {
                    s[i * hv + j] += k_i * beta * (f64::from(v_t[j]) - u[j]);
                }
            }
            for j in 0..hv {
                let mut acc = 0.0f64;
                for i in 0..hk {
                    acc += s[i * hv + j] * f64::from(q_t[i]);
                }
                out[idx * hv + j] = (acc * scale) as f32;
            }
        }
        for i in 0..hk {
            for j in 0..hv {
                final_state[h * hv * hk + j * hk + i] = s[i * hv + j] as f32;
            }
        }
    }
    (out, final_state)
}

/// Run `t_len` decode steps, one token at a time, into `state`.
fn run_steps(
    inputs: &Inputs,
    state: &mut [f32],
    dims: &GdnDims,
    t_len: usize,
    order: GdnHeadOrder,
    path: GdnPath,
) -> Vec<f32> {
    let mut out = vec![0.0f32; t_len * dims.v_len()];
    for t in 0..t_len {
        let qk = dims.qk_len();
        let vl = dims.v_len();
        let nv = dims.n_v_heads;
        let gates = GdnGates {
            beta: GdnBeta::Raw(&inputs.beta[t * nv..(t + 1) * nv]),
            decay: GdnDecay::ScalarRaw {
                alpha_raw: &inputs.alpha[t * nv..(t + 1) * nv],
                dt_bias: &inputs.dt_bias,
                a_neg: &inputs.a_neg,
            },
        };
        gdn_step_with(
            state,
            &inputs.q[t * qk..(t + 1) * qk],
            &inputs.k[t * qk..(t + 1) * qk],
            &inputs.v[t * vl..(t + 1) * vl],
            &gates,
            &mut out[t * vl..(t + 1) * vl],
            dims,
            order,
            path,
        )
        .expect("step must succeed");
    }
    out
}

// ─── 1. Golden parity against the fork transliteration ───────────────

#[test]
fn gated_delta_golden_matches_fork_transliteration_over_32_steps() {
    let dims = GdnDims::new(G_NK, G_NV, G_HK, G_HV);
    let inputs = golden_inputs();
    let mut state = inputs.state.clone();
    let out = run_steps(
        &inputs,
        &mut state,
        &dims,
        G_T,
        GdnHeadOrder::Tiled,
        GdnPath::Fused,
    );

    let golden_out = bits(&GOLDEN_OUT);
    let golden_state = bits(&GOLDEN_STATE);

    let cos_out = cosine(&out, &golden_out);
    let cos_state = cosine(&state, &golden_state);
    assert!(
        cos_out >= 0.9999,
        "output cosine {cos_out} < 0.9999 (max |Δ| = {})",
        max_abs_diff(&out, &golden_out)
    );
    assert!(
        cos_state >= 0.9999,
        "state cosine {cos_state} < 0.9999 (max |Δ| = {})",
        max_abs_diff(&state, &golden_state)
    );
    // The oracle only differs by its f64 dot accumulation, so the agreement is
    // far tighter than the acceptance threshold; pin that too.
    assert!(
        max_abs_diff(&out, &golden_out) <= 1e-4,
        "output max |Δ| = {}",
        max_abs_diff(&out, &golden_out)
    );
    assert!(
        max_abs_diff(&state, &golden_state) <= 1e-4,
        "state max |Δ| = {}",
        max_abs_diff(&state, &golden_state)
    );
}

#[test]
fn gated_delta_golden_agrees_with_the_dense_reference() {
    // Cross-check of the oracle itself: the C++ dump must also match the
    // independent dense f64 replay, otherwise a shared misreading of the fork
    // could hide in both.
    let dims = GdnDims::new(G_NK, G_NV, G_HK, G_HV);
    let inputs = golden_inputs();
    let (naive_out, naive_state) = naive_dense(&inputs, &dims, GdnHeadOrder::Tiled, G_T - 1);
    let golden_out = bits(&GOLDEN_OUT);
    let golden_state = bits(&GOLDEN_STATE);
    let last = &golden_out[(G_T - 1) * dims.v_len()..];
    assert!(
        max_abs_diff(last, &naive_out) <= 1e-4,
        "oracle vs dense reference, final output: {}",
        max_abs_diff(last, &naive_out)
    );
    assert!(
        max_abs_diff(&golden_state, &naive_state) <= 1e-4,
        "oracle vs dense reference, final state: {}",
        max_abs_diff(&golden_state, &naive_state)
    );
}

// ─── 2. Fused single pass == fork three-pass ─────────────────────────

#[test]
fn gated_delta_fused_and_three_pass_agree_bitwise() {
    let dims = GdnDims::new(4, 12, 128, 128);
    let inputs = random_inputs(&dims, 6, 0x5EED_0002);
    let mut fused_state = inputs.state.clone();
    let mut split_state = inputs.state.clone();

    let fused = run_steps(
        &inputs,
        &mut fused_state,
        &dims,
        6,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );
    let split = run_steps(
        &inputs,
        &mut split_state,
        &dims,
        6,
        GdnHeadOrder::Grouped,
        GdnPath::ThreePass,
    );

    assert_bitwise(&fused, &split, "fused vs three-pass output");
    assert_bitwise(&fused_state, &split_state, "fused vs three-pass state");
    // The acceptance threshold, asserted explicitly as well.
    assert!(max_abs_diff(&fused, &split) <= 1e-6);
    assert!(max_abs_diff(&fused_state, &split_state) <= 1e-6);
}

// ─── 3. Prefill == T × step, bitwise ─────────────────────────────────

#[test]
fn gated_delta_prefill_equals_repeated_steps_bitwise() {
    let dims = GdnDims::new(4, 12, 128, 128);
    let t_len = 5;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0003);

    let mut step_state = inputs.state.clone();
    let step_out = run_steps(
        &inputs,
        &mut step_state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );

    let mut prefill_state = inputs.state.clone();
    let mut prefill_out = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_f32(
        &mut prefill_state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.alpha,
        &inputs.beta,
        &inputs.dt_bias,
        &inputs.a_neg,
        &mut prefill_out,
        t_len,
        dims.n_k_heads,
        dims.n_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
    )
    .expect("prefill must succeed");

    assert_bitwise(&step_out, &prefill_out, "prefill output");
    assert_bitwise(&step_state, &prefill_state, "prefill state");
}

#[test]
fn gated_delta_prefill_equals_repeated_steps_on_the_real_bonsai2_geometry() {
    let dims = GdnDims::bonsai2();
    assert_eq!(dims.state_len(), 48 * 128 * 128);
    let t_len = 3;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0004);

    let mut step_state = inputs.state.clone();
    let step_out = run_steps(
        &inputs,
        &mut step_state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );

    let mut prefill_state = inputs.state.clone();
    let mut prefill_out = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with(
        &mut prefill_state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.gates(),
        &mut prefill_out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("prefill must succeed");

    assert_bitwise(&step_out, &prefill_out, "27B prefill output");
    assert_bitwise(&step_state, &prefill_state, "27B prefill state");
}

// ─── 4. Versus the O(T²) naive reference ─────────────────────────────

#[test]
fn gated_delta_matches_the_naive_quadratic_reference_at_every_step() {
    let dims = GdnDims::new(2, 6, 16, 16);
    let t_len = 16;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0005);

    let mut state = inputs.state.clone();
    let out = run_steps(
        &inputs,
        &mut state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );

    // O(T²): every step is replayed from the *snapshot* of S₀, independently.
    for t in 0..t_len {
        let (expected, _) = naive_dense(&inputs, &dims, GdnHeadOrder::Grouped, t);
        let got = &out[t * dims.v_len()..(t + 1) * dims.v_len()];
        let diff = max_abs_diff(got, &expected);
        assert!(diff <= 1e-4, "token {t}: max |Δ| = {diff}");
        assert!(cosine(got, &expected) >= 0.9999, "token {t} cosine");
    }
    let (_, expected_state) = naive_dense(&inputs, &dims, GdnHeadOrder::Grouped, t_len - 1);
    assert!(
        max_abs_diff(&state, &expected_state) <= 1e-4,
        "final state max |Δ| = {}",
        max_abs_diff(&state, &expected_state)
    );
}

#[test]
fn gated_delta_state_does_not_drift_over_64_steps() {
    let dims = GdnDims::new(2, 6, 32, 32);
    let t_len = 64;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0006);

    let mut state = inputs.state.clone();
    let out = run_steps(
        &inputs,
        &mut state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );

    let (expected_out, expected_state) =
        naive_dense(&inputs, &dims, GdnHeadOrder::Grouped, t_len - 1);
    let last = &out[(t_len - 1) * dims.v_len()..];
    let out_diff = max_abs_diff(last, &expected_out);
    let state_diff = max_abs_diff(&state, &expected_state);
    assert!(out_diff <= 1e-4, "64-step output drift {out_diff}");
    assert!(state_diff <= 1e-4, "64-step state drift {state_diff}");
    assert!(cosine(&state, &expected_state) >= 0.9999, "64-step cosine");
    assert!(
        state.iter().all(|x| x.is_finite()),
        "state must stay finite over 64 steps"
    );

    // The three-pass form must not drift differently from the fused one.
    let mut three_pass_state = inputs.state.clone();
    let three_pass = run_steps(
        &inputs,
        &mut three_pass_state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::ThreePass,
    );
    assert_bitwise(&out, &three_pass, "64-step fused vs three-pass output");
    assert_bitwise(
        &state,
        &three_pass_state,
        "64-step fused vs three-pass state",
    );
}

// ─── 5. Softplus cutoff, end to end through the kernel ───────────────

#[test]
fn gated_delta_softplus_cutoff_of_20_is_observable_through_the_kernel() {
    // With β pinned to 0 the delta term vanishes and the step degenerates to
    // `S ← S · exp(g)`, so the decay — and with it the cutoff branch — can be
    // read straight off the updated state.
    let dims = GdnDims::new(1, 3, 8, 8);
    let mut state = vec![0.0f32; dims.state_len()];
    for (i, s) in state.iter_mut().enumerate() {
        *s = 0.5 + (i % 5) as f32 * 0.125;
    }
    let reference_state = state.clone();

    let just_above = f32::from_bits(20.0f32.to_bits() + 1);
    let just_below = f32::from_bits(20.0f32.to_bits() - 1);
    let alpha = [20.0f32, just_above, just_below];
    let dt_bias = [0.0f32; 3];
    let a_neg = [-1.0f32, -1.0, -1.0];
    let beta_zero = [0.0f32; 3];
    let q = vec![0.0f32; dims.qk_len()];
    let k = vec![0.0f32; dims.qk_len()];
    let v = vec![0.0f32; dims.v_len()];
    let mut out = vec![0.0f32; dims.v_len()];

    let gates = GdnGates {
        beta: GdnBeta::Activated(&beta_zero),
        decay: GdnDecay::ScalarRaw {
            alpha_raw: &alpha,
            dt_bias: &dt_bias,
            a_neg: &a_neg,
        },
    };
    gdn_step_with(
        &mut state,
        &q,
        &k,
        &v,
        &gates,
        &mut out,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("step must succeed");

    let per_head = dims.head_v_dim * dims.head_k_dim;
    for (h, x) in alpha.iter().enumerate() {
        let expected_decay = (-softplus(*x)).exp();
        let got = state[h * per_head] / reference_state[h * per_head];
        assert!(
            (got - expected_decay).abs() <= 1e-6,
            "head {h}: decay {got} != exp(-softplus({x})) = {expected_decay}"
        );
    }

    // `x == 20.0` must take the ln(1+exp(x)) branch, `x` just above it the
    // identity branch — the two differ, so the `>` is pinned, not merely `>=`.
    let at_cutoff = (-softplus(20.0f32)).exp();
    let above_cutoff = (-just_above).exp();
    assert_ne!(
        state[0].to_bits(),
        (reference_state[0] * above_cutoff).to_bits(),
        "x == 20.0 must not use the identity branch"
    );
    assert!((state[0] - reference_state[0] * at_cutoff).abs() <= 1e-6);
}

// ─── 6. `ssm_a` is consumed as-is; a positive value is rejected ──────

#[test]
fn gated_delta_a_positive_ssm_a_is_rejected_by_the_kernel_entry_points() {
    let dims = GdnDims::new(1, 2, 8, 8);
    let mut state = vec![0.0f32; dims.state_len()];
    let q = vec![0.1f32; dims.qk_len()];
    let k = vec![0.1f32; dims.qk_len()];
    let v = vec![0.1f32; dims.v_len()];
    let alpha = vec![0.5f32; dims.n_v_heads];
    let beta = vec![0.5f32; dims.n_v_heads];
    let dt_bias = vec![0.0f32; dims.n_v_heads];
    let mut out = vec![0.0f32; dims.v_len()];

    // `A = -exp(A_log)` is strictly negative; 0 is allowed (underflowed exp).
    let good = vec![-0.5f32, 0.0];
    assert!(validate_a_neg(&good).is_ok());
    assert!(gdn_step_f32(
        &mut state,
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        &dt_bias,
        &good,
        &mut out,
        dims.n_k_heads,
        dims.n_v_heads,
        dims.head_k_dim,
        dims.head_v_dim,
    )
    .is_ok());

    for bad in [vec![-0.5f32, 0.25], vec![f32::NAN, -0.5]] {
        let err = gdn_step_f32(
            &mut state,
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &dt_bias,
            &bad,
            &mut out,
            dims.n_k_heads,
            dims.n_v_heads,
            dims.head_k_dim,
            dims.head_v_dim,
        )
        .expect_err("a positive or NaN ssm_a must be rejected");
        assert!(
            err.to_string().contains("ssm_a"),
            "error must name ssm_a: {err}"
        );
        assert!(validate_a_neg(&bad).is_err());
    }
}

#[test]
fn gated_delta_ssm_a_is_used_without_a_sign_flip() {
    // A more negative `a` must decay the state harder. If the loader or the
    // kernel "helpfully" took |a| or -a, the ordering below would invert.
    let dims = GdnDims::new(1, 2, 8, 8);
    let mut state = vec![1.0f32; dims.state_len()];
    let q = vec![0.0f32; dims.qk_len()];
    let k = vec![0.0f32; dims.qk_len()];
    let v = vec![0.0f32; dims.v_len()];
    let alpha = vec![0.0f32; dims.n_v_heads];
    let dt_bias = vec![0.0f32; dims.n_v_heads];
    let a_neg = vec![-0.25f32, -4.0];
    let beta_zero = [0.0f32; 2];
    let mut out = vec![0.0f32; dims.v_len()];
    let gates = GdnGates {
        beta: GdnBeta::Activated(&beta_zero),
        decay: GdnDecay::ScalarRaw {
            alpha_raw: &alpha,
            dt_bias: &dt_bias,
            a_neg: &a_neg,
        },
    };
    gdn_step_with(
        &mut state,
        &q,
        &k,
        &v,
        &gates,
        &mut out,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("step must succeed");

    let per_head = dims.head_v_dim * dims.head_k_dim;
    let soft = softplus(0.0);
    assert!((state[0] - (-0.25f32 * soft).exp()).abs() <= 1e-6);
    assert!((state[per_head] - (-4.0f32 * soft).exp()).abs() <= 1e-6);
    assert!(
        state[per_head] < state[0],
        "a more negative ssm_a must decay harder"
    );
}

// ─── 7. kda per-channel gate ─────────────────────────────────────────

#[test]
fn gated_delta_per_channel_kda_gate_matches_the_dense_reference() {
    let dims = GdnDims::new(1, 2, 8, 8);
    let t_len = 6;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0007);
    let mut rng = SplitMix64::new(0x5EED_0077);
    let g: Vec<f32> = (0..t_len * dims.n_v_heads * dims.head_k_dim)
        .map(|_| rng.next_f32() - 0.5)
        .collect();

    let gates = GdnGates {
        beta: GdnBeta::Raw(&inputs.beta),
        decay: GdnDecay::PerChannelLog(&g),
    };
    assert!(gates.is_per_channel());
    // The fusion is invalid per column, so a Fused request must still resolve
    // to the three-pass form.
    assert_eq!(gates.effective_path(GdnPath::Fused), GdnPath::ThreePass);

    let mut state = inputs.state.clone();
    let mut out = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with(
        &mut state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &gates,
        &mut out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("kda prefill must succeed");

    let (expected_out, expected_state) = naive_dense_per_channel(&inputs, &g, &dims, t_len);
    assert!(
        max_abs_diff(&out, &expected_out) <= 1e-4,
        "kda output max |Δ| = {}",
        max_abs_diff(&out, &expected_out)
    );
    assert!(
        max_abs_diff(&state, &expected_state) <= 1e-4,
        "kda state max |Δ| = {}",
        max_abs_diff(&state, &expected_state)
    );

    // Asking for ThreePass explicitly must produce the very same bits.
    let mut state_tp = inputs.state.clone();
    let mut out_tp = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with(
        &mut state_tp,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &gates,
        &mut out_tp,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::ThreePass,
    )
    .expect("kda prefill must succeed");
    assert_bitwise(&out, &out_tp, "kda fused-request vs three-pass");
    assert_bitwise(&state, &state_tp, "kda fused-request vs three-pass state");
}

#[test]
fn gated_delta_a_constant_per_channel_gate_equals_the_scalar_gate() {
    // A kda gate whose channels all carry the same value must reproduce the
    // scalar-decay path exactly (up to the different traversal), which ties the
    // two branches together.
    let dims = GdnDims::new(1, 2, 8, 8);
    let t_len = 4;
    let inputs = random_inputs(&dims, t_len, 0x5EED_0008);
    let log_decay: Vec<f32> = (0..t_len * dims.n_v_heads)
        .map(|i| -0.125 * (i % 3 + 1) as f32)
        .collect();
    let mut per_channel = vec![0.0f32; t_len * dims.n_v_heads * dims.head_k_dim];
    for (idx, g) in log_decay.iter().enumerate() {
        for i in 0..dims.head_k_dim {
            per_channel[idx * dims.head_k_dim + i] = *g;
        }
    }

    let scalar_gates = GdnGates {
        beta: GdnBeta::Raw(&inputs.beta),
        decay: GdnDecay::ScalarLog(&log_decay),
    };
    let channel_gates = GdnGates {
        beta: GdnBeta::Raw(&inputs.beta),
        decay: GdnDecay::PerChannelLog(&per_channel),
    };

    let mut state_a = inputs.state.clone();
    let mut out_a = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with(
        &mut state_a,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &scalar_gates,
        &mut out_a,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("scalar-log prefill");

    let mut state_b = inputs.state.clone();
    let mut out_b = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with(
        &mut state_b,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &channel_gates,
        &mut out_b,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("per-channel prefill");

    assert!(max_abs_diff(&out_a, &out_b) <= 1e-6);
    assert!(max_abs_diff(&state_a, &state_b) <= 1e-6);
}

// ─── 8. Head orders ──────────────────────────────────────────────────

#[test]
fn gated_delta_grouped_and_tiled_orders_agree_under_the_vhead_map() {
    // Design §3.3: grouped v-head `m` is tiled v-head `(m % rep) * nk + m / rep`
    // (`runtime.py::vperm` with unit = 1). Permuting every v-indexed buffer with
    // that map must turn a tiled-order run into a grouped-order one, bitwise.
    let dims = GdnDims::new(2, 6, 16, 16);
    let rep = dims.v_per_k();
    let t_len = 4;
    let tiled = random_inputs(&dims, t_len, 0x5EED_0009);

    let map: Vec<usize> = (0..dims.n_v_heads)
        .map(|m| (m % rep) * dims.n_k_heads + m / rep)
        .collect();
    for (m, j) in map.iter().enumerate() {
        assert_eq!(
            GdnHeadOrder::Grouped.k_head(m, &dims),
            GdnHeadOrder::Tiled.k_head(*j, &dims),
            "grouped {m} and tiled {j} must read the same k-head"
        );
    }

    let (hv, hk, nv) = (dims.head_v_dim, dims.head_k_dim, dims.n_v_heads);
    let mut grouped = Inputs {
        state: vec![0.0; dims.state_len()],
        q: tiled.q.clone(),
        k: tiled.k.clone(),
        v: vec![0.0; t_len * dims.v_len()],
        alpha: vec![0.0; t_len * nv],
        beta: vec![0.0; t_len * nv],
        dt_bias: vec![0.0; nv],
        a_neg: vec![0.0; nv],
    };
    for (m, j) in map.iter().enumerate() {
        grouped.state[m * hv * hk..(m + 1) * hv * hk]
            .copy_from_slice(&tiled.state[j * hv * hk..(j + 1) * hv * hk]);
        grouped.dt_bias[m] = tiled.dt_bias[*j];
        grouped.a_neg[m] = tiled.a_neg[*j];
        for t in 0..t_len {
            grouped.v[(t * nv + m) * hv..(t * nv + m + 1) * hv]
                .copy_from_slice(&tiled.v[(t * nv + j) * hv..(t * nv + j + 1) * hv]);
            grouped.alpha[t * nv + m] = tiled.alpha[t * nv + j];
            grouped.beta[t * nv + m] = tiled.beta[t * nv + j];
        }
    }

    let mut tiled_state = tiled.state.clone();
    let tiled_out = run_steps(
        &tiled,
        &mut tiled_state,
        &dims,
        t_len,
        GdnHeadOrder::Tiled,
        GdnPath::Fused,
    );
    let mut grouped_state = grouped.state.clone();
    let grouped_out = run_steps(
        &grouped,
        &mut grouped_state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );

    for (m, j) in map.iter().enumerate() {
        for t in 0..t_len {
            let g = &grouped_out[(t * nv + m) * hv..(t * nv + m + 1) * hv];
            let s = &tiled_out[(t * nv + j) * hv..(t * nv + j + 1) * hv];
            assert_bitwise(g, s, "grouped vs tiled output");
        }
        assert_bitwise(
            &grouped_state[m * hv * hk..(m + 1) * hv * hk],
            &tiled_state[j * hv * hk..(j + 1) * hv * hk],
            "grouped vs tiled state",
        );
    }
}

#[test]
fn gated_delta_head_orders_coincide_when_every_v_head_has_its_own_k_head() {
    let dims = GdnDims::new(4, 4, 16, 16);
    let inputs = random_inputs(&dims, 3, 0x5EED_000A);
    let mut grouped_state = inputs.state.clone();
    let mut tiled_state = inputs.state.clone();
    let grouped = run_steps(
        &inputs,
        &mut grouped_state,
        &dims,
        3,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );
    let tiled = run_steps(
        &inputs,
        &mut tiled_state,
        &dims,
        3,
        GdnHeadOrder::Tiled,
        GdnPath::Fused,
    );
    assert_bitwise(&grouped, &tiled, "rep == 1 output");
    assert_bitwise(&grouped_state, &tiled_state, "rep == 1 state");
}

// ─── 9. Cross-tier parity at the real head width ─────────────────────

#[test]
fn gated_delta_simd_tier_matches_the_scalar_tier_at_128_channels() {
    let dims = GdnDims::new(4, 12, 128, 128);
    let t_len = 4;
    let inputs = random_inputs(&dims, t_len, 0x5EED_000B);

    let mut scalar_state = inputs.state.clone();
    let mut scalar_out = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with_tier(
        &mut scalar_state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.gates(),
        &mut scalar_out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
        GdnTier::Scalar,
    )
    .expect("scalar prefill");

    let mut simd_state = inputs.state.clone();
    let mut simd_out = vec![0.0f32; t_len * dims.v_len()];
    gdn_prefill_with_tier(
        &mut simd_state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.gates(),
        &mut simd_out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
        GdnTier::detect(),
    )
    .expect("simd prefill");

    let out_diff = max_abs_diff(&scalar_out, &simd_out);
    let state_diff = max_abs_diff(&scalar_state, &simd_state);
    assert!(out_diff <= 1e-6, "cross-tier output max |Δ| = {out_diff}");
    assert!(
        state_diff <= 1e-6,
        "cross-tier state max |Δ| = {state_diff}"
    );
    assert!(cosine(&scalar_out, &simd_out) >= 0.999_999);
}

// ─── 10. Non-square geometry ─────────────────────────────────────────

#[test]
fn gated_delta_non_square_head_dims_follow_the_documented_semantics() {
    // Row length is head_k_dim, row count is head_v_dim, and the output scale
    // is 1/sqrt(head_v_dim) — the distinction the square Bonsai 2 geometry
    // cannot expose. Validated against the dense reference, not the fork.
    let dims = GdnDims::new(1, 2, 8, 4);
    assert!((dims.out_scale() - 0.5).abs() <= f32::EPSILON);
    let t_len = 5;
    let inputs = random_inputs(&dims, t_len, 0x5EED_000C);

    let mut state = inputs.state.clone();
    let out = run_steps(
        &inputs,
        &mut state,
        &dims,
        t_len,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    );
    let (expected_out, expected_state) =
        naive_dense(&inputs, &dims, GdnHeadOrder::Grouped, t_len - 1);
    let last = &out[(t_len - 1) * dims.v_len()..];
    assert!(max_abs_diff(last, &expected_out) <= 1e-4);
    assert!(max_abs_diff(&state, &expected_state) <= 1e-4);
}

// ─── 11. State object and the chunk/step entry points ────────────────

#[test]
fn gated_delta_gdn_state_carries_layers_independently_and_chunk_equals_step() {
    let dims = GdnDims::new(2, 4, 16, 16);
    let t_len = 3;
    let inputs = random_inputs(&dims, t_len, 0x5EED_000D);

    let mut state = GdnState::with_layers(dims, 3).expect("state");
    assert_eq!(state.n_layers(), 3);
    assert_eq!(state.bytes(), 3 * dims.state_len() * 4);
    state
        .layer_mut(1)
        .expect("layer 1")
        .copy_from_slice(&inputs.state);

    // Layer 1 through the step API, one token at a time.
    let mut step_out = vec![0.0f32; t_len * dims.v_len()];
    for t in 0..t_len {
        let qk = dims.qk_len();
        let vl = dims.v_len();
        let nv = dims.n_v_heads;
        gdn_step(
            &inputs.q[t * qk..(t + 1) * qk],
            &inputs.k[t * qk..(t + 1) * qk],
            &inputs.v[t * vl..(t + 1) * vl],
            &inputs.alpha[t * nv..(t + 1) * nv],
            &inputs.beta[t * nv..(t + 1) * nv],
            &inputs.a_neg,
            &inputs.dt_bias,
            &mut state,
            1,
            &mut step_out[t * vl..(t + 1) * vl],
        )
        .expect("step");
    }
    // Layers 0 and 2 must be untouched.
    assert!(state.layer(0).expect("layer 0").iter().all(|x| *x == 0.0));
    assert!(state.layer(2).expect("layer 2").iter().all(|x| *x == 0.0));

    let mut chunk_state = GdnState::with_layers(dims, 3).expect("state");
    chunk_state
        .layer_mut(1)
        .expect("layer 1")
        .copy_from_slice(&inputs.state);
    let mut chunk_out = vec![0.0f32; t_len * dims.v_len()];
    gdn_chunk(
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.alpha,
        &inputs.beta,
        &inputs.a_neg,
        &inputs.dt_bias,
        &mut chunk_state,
        1,
        &mut chunk_out,
        t_len,
    )
    .expect("chunk");

    assert_bitwise(&step_out, &chunk_out, "chunk vs step output");
    assert_bitwise(
        state.layer(1).expect("layer 1"),
        chunk_state.layer(1).expect("layer 1"),
        "chunk vs step state",
    );

    assert!(gdn_chunk(
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.alpha,
        &inputs.beta,
        &inputs.a_neg,
        &inputs.dt_bias,
        &mut chunk_state,
        3,
        &mut chunk_out,
        t_len,
    )
    .is_err());

    state.reset();
    assert!(state.as_slice().iter().all(|x| *x == 0.0));
}

#[test]
fn gated_delta_an_empty_chunk_is_a_no_op() {
    let dims = GdnDims::new(2, 4, 8, 8);
    let inputs = random_inputs(&dims, 1, 0x5EED_000E);
    let mut state = inputs.state.clone();
    let mut out: Vec<f32> = Vec::new();
    gdn_prefill_with(
        &mut state,
        &[],
        &[],
        &[],
        &inputs.gates(),
        &mut out,
        0,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect("an empty chunk must be accepted");
    assert_bitwise(&state, &inputs.state, "empty chunk state");
}

// ─── 12. Length contract (K-02) ──────────────────────────────────────

#[test]
fn gated_delta_every_short_buffer_is_named_in_the_error() {
    let dims = GdnDims::new(2, 4, 8, 8);
    let t_len = 2;
    let inputs = random_inputs(&dims, t_len, 0x5EED_000F);
    let mut state = inputs.state.clone();
    let mut out = vec![0.0f32; t_len * dims.v_len()];

    let short = |name: &str, err: KernelError| {
        assert_eq!(err.buffer_name(), Some(name), "expected '{name}': {err}");
        assert_eq!(err.error_code(), "BUFFER_TOO_SMALL");
    };

    short(
        "q",
        gdn_prefill_with(
            &mut state,
            &inputs.q[..dims.qk_len()],
            &inputs.k,
            &inputs.v,
            &inputs.gates(),
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short q"),
    );
    short(
        "k",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k[..dims.qk_len()],
            &inputs.v,
            &inputs.gates(),
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short k"),
    );
    short(
        "v",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k,
            &inputs.v[..dims.v_len()],
            &inputs.gates(),
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short v"),
    );
    short(
        "out",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &inputs.gates(),
            &mut out[..dims.v_len()],
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short out"),
    );
    short(
        "beta",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &GdnGates {
                beta: GdnBeta::Raw(&inputs.beta[..dims.n_v_heads]),
                decay: GdnDecay::ScalarRaw {
                    alpha_raw: &inputs.alpha,
                    dt_bias: &inputs.dt_bias,
                    a_neg: &inputs.a_neg,
                },
            },
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short beta"),
    );
    short(
        "alpha_raw",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &GdnGates {
                beta: GdnBeta::Raw(&inputs.beta),
                decay: GdnDecay::ScalarRaw {
                    alpha_raw: &inputs.alpha[..dims.n_v_heads],
                    dt_bias: &inputs.dt_bias,
                    a_neg: &inputs.a_neg,
                },
            },
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short alpha_raw"),
    );
    short(
        "state",
        gdn_prefill_with(
            &mut inputs.state.clone()[..dims.state_len() - 1],
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &inputs.gates(),
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short state"),
    );
    short(
        "g",
        gdn_prefill_with(
            &mut state,
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &GdnGates {
                beta: GdnBeta::Raw(&inputs.beta),
                decay: GdnDecay::PerChannelLog(&inputs.v[..dims.head_k_dim]),
            },
            &mut out,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect_err("short per-channel g"),
    );

    // A geometry whose v-heads are not a whole multiple of its k-heads.
    let bad = GdnDims::new(5, 12, 8, 8);
    let err = gdn_prefill_with(
        &mut state,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.gates(),
        &mut out,
        t_len,
        &bad,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
    .expect_err("non-multiple head counts");
    assert_eq!(err.error_code(), "DIMENSION_MISMATCH");
}
