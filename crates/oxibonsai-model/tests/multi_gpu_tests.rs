#![cfg(feature = "multi-device")]
//! Tests for the multi-GPU / multi-device utilities (`oxibonsai_model::multi_gpu`).
//!
//! All data is deterministic — no rand crate is used.

use oxibonsai_model::multi_gpu::{
    merge_column_shards, partition_weights_column, partition_weights_row, SimulatedCollectives,
    SimulatedDeviceMesh,
};

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedDeviceMesh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn device_mesh_tp_only() {
    let mesh = SimulatedDeviceMesh::tensor_parallel(4);
    assert_eq!(mesh.size(), 4, "tensor_parallel(4) should have 4 devices");
}

#[test]
fn device_mesh_2d() {
    let mesh = SimulatedDeviceMesh::new(2, 2);
    assert_eq!(mesh.size(), 4, "new(2,2) should have 4 devices");
}

#[test]
fn device_mesh_get_valid() {
    let mesh = SimulatedDeviceMesh::new(2, 3);
    assert!(mesh.get(0, 0).is_some(), "get(0,0) should return Some");
    assert!(
        mesh.get(1, 2).is_some(),
        "get(1,2) should return Some for a 2×3 mesh"
    );
}

#[test]
fn device_mesh_get_oob() {
    let mesh = SimulatedDeviceMesh::new(2, 2);
    assert!(
        mesh.get(99, 0).is_none(),
        "out-of-bounds tp_rank should return None"
    );
    assert!(
        mesh.get(0, 99).is_none(),
        "out-of-bounds pp_rank should return None"
    );
    assert!(
        mesh.get(99, 99).is_none(),
        "both out-of-bounds should return None"
    );
}

#[test]
fn device_mesh_tp_group_size() {
    let mesh = SimulatedDeviceMesh::new(4, 2);
    let grp = mesh.tp_group(0);
    assert_eq!(grp.len(), 4, "tp_group should contain tp_size devices");
}

#[test]
fn device_mesh_pp_group_size() {
    let mesh = SimulatedDeviceMesh::new(4, 3);
    let grp = mesh.pp_group(0);
    assert_eq!(grp.len(), 3, "pp_group should contain pp_size devices");
}

#[test]
fn device_mesh_tp_group_oob() {
    let mesh = SimulatedDeviceMesh::new(2, 2);
    let grp = mesh.tp_group(99);
    assert!(
        grp.is_empty(),
        "out-of-bounds pp_rank should return empty tp_group"
    );
}

#[test]
fn device_mesh_pp_group_oob() {
    let mesh = SimulatedDeviceMesh::new(2, 2);
    let grp = mesh.pp_group(99);
    assert!(
        grp.is_empty(),
        "out-of-bounds tp_rank should return empty pp_group"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceInfo
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn device_info_has_name() {
    let mesh = SimulatedDeviceMesh::tensor_parallel(1);
    let dev = mesh.get(0, 0).expect("device 0 should exist");
    assert!(!dev.name.is_empty(), "device name should not be empty");
}

#[test]
fn device_info_memory_is_honestly_unknown() {
    // `memory_bytes` used to be a fabricated 24 GiB constant. This
    // simulation has no real hardware to probe, so it must honestly
    // report `None` rather than hand out a made-up number.
    let mesh = SimulatedDeviceMesh::tensor_parallel(2);
    for tp in 0..2 {
        let dev = mesh.get(tp, 0).expect("device should exist");
        assert!(
            dev.memory_bytes.is_none(),
            "simulated device memory must be honestly None, not a fabricated number"
        );
        assert!(
            dev.compute_units > 0,
            "simulated compute units should be positive"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives — all_reduce_sum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nccl_all_reduce_sum_single() {
    let shard = vec![1.0f32, 2.0, 3.0];
    let result = SimulatedCollectives::all_reduce_sum(std::slice::from_ref(&shard));
    assert_eq!(
        result.data, shard,
        "single-shard all-reduce should be identity"
    );
    assert_eq!(result.participating_devices, 1);
}

#[test]
fn nccl_all_reduce_sum_two() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let result = SimulatedCollectives::all_reduce_sum(&[a, b]);
    assert_eq!(
        result.data,
        vec![5.0f32, 7.0, 9.0],
        "[1,2,3]+[4,5,6] should be [5,7,9]"
    );
    assert_eq!(result.participating_devices, 2);
}

#[test]
fn nccl_all_reduce_sum_three() {
    let shards = vec![vec![1.0f32, 0.0], vec![2.0f32, 0.0], vec![3.0f32, 0.0]];
    let result = SimulatedCollectives::all_reduce_sum(&shards);
    assert!((result.data[0] - 6.0).abs() < 1e-6, "sum of 1+2+3=6");
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives — all_reduce_max
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nccl_all_reduce_max() {
    let a = vec![1.0f32, 5.0, 3.0];
    let b = vec![4.0f32, 2.0, 6.0];
    let result = SimulatedCollectives::all_reduce_max(&[a, b]);
    assert_eq!(
        result.data,
        vec![4.0f32, 5.0, 6.0],
        "element-wise max should be [4,5,6]"
    );
    assert_eq!(result.op_name, "all_reduce_max");
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives — all_gather
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nccl_all_gather_concatenates() {
    let shards = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0], vec![5.0f32, 6.0]];
    let result = SimulatedCollectives::all_gather(&shards);
    assert_eq!(
        result.data,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        "all_gather should concatenate shards in rank order"
    );
    assert_eq!(result.participating_devices, 3);
    assert_eq!(result.op_name, "all_gather");
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives — scatter_equal_shards
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nccl_scatter_equal_shards_correct_shard_count() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let shards = SimulatedCollectives::scatter_equal_shards(&data, 4);
    assert_eq!(
        shards.len(),
        4,
        "scatter_equal_shards should produce world_size shards"
    );
}

#[test]
fn nccl_scatter_equal_shards_covers_all_data() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let shards = SimulatedCollectives::scatter_equal_shards(&data, 3);
    let total_elements: usize = shards.iter().map(|s| s.len()).sum();
    assert_eq!(total_elements, data.len(), "all elements should be covered");
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedCollectives — broadcast
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nccl_broadcast_replicates() {
    let data = vec![1.0f32, 2.0, 3.0];
    let replicas = SimulatedCollectives::broadcast(&data, 4);
    assert_eq!(
        replicas.len(),
        4,
        "broadcast should return world_size copies"
    );
    for replica in &replicas {
        assert_eq!(
            replica, &data,
            "every replica should equal the original data"
        );
    }
}

#[test]
fn nccl_broadcast_single() {
    let data = vec![42.0f32];
    let replicas = SimulatedCollectives::broadcast(&data, 1);
    assert_eq!(replicas.len(), 1);
    assert_eq!(replicas[0], data);
}

// ─────────────────────────────────────────────────────────────────────────────
// CollectiveResult — op_name
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn collective_result_op_name_all_reduce_sum() {
    let shards = vec![vec![1.0f32]];
    let result = SimulatedCollectives::all_reduce_sum(&shards);
    assert_eq!(result.op_name, "all_reduce_sum");
}

#[test]
fn collective_result_op_name_all_gather() {
    let shards = vec![vec![1.0f32]];
    let result = SimulatedCollectives::all_gather(&shards);
    assert_eq!(result.op_name, "all_gather");
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight partition — column parallel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn partition_weights_column_count() {
    // 2 rows × 8 cols split into 4 shards
    let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let shards = partition_weights_column(&weights, 2, 8, 4);
    assert_eq!(shards.len(), 4, "should produce 4 column-parallel shards");
}

#[test]
fn partition_weights_column_total_elements() {
    let weights: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let shards = partition_weights_column(&weights, 3, 8, 4);
    let total: usize = shards.iter().map(|s| s.len()).sum();
    assert_eq!(total, 24, "partitioned shards should cover all elements");
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight partition — row parallel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn partition_weights_row_count() {
    let weights: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let shards = partition_weights_row(&weights, 8, 4, 4);
    assert_eq!(shards.len(), 4, "should produce 4 row-parallel shards");
}

#[test]
fn partition_weights_row_total_elements() {
    let weights: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let shards = partition_weights_row(&weights, 8, 4, 4);
    let total: usize = shards.iter().map(|s| s.len()).sum();
    assert_eq!(
        total, 32,
        "row-partitioned shards should cover all elements"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// merge_column_shards — roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn merge_column_shards_reconstructs() {
    // 3 rows × 8 cols, split into 4 column-parallel shards, then merged back.
    let rows = 3;
    let cols = 8;
    let original: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let shards = partition_weights_column(&original, rows, cols, 4);
    let merged = merge_column_shards(&shards, rows);
    assert_eq!(
        merged, original,
        "partition then merge should reconstruct the original weight matrix"
    );
}

#[test]
fn merge_column_shards_single_shard() {
    let weights: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let shards = partition_weights_column(&weights, 3, 4, 1);
    let merged = merge_column_shards(&shards, 3);
    assert_eq!(merged, weights, "single shard merge should be identity");
}
