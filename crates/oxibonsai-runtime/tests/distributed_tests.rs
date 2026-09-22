//! Integration tests for the distributed serving module.
//!
//! Covers [`ConsistentHashRing`], [`NodeRegistry`], [`DistributedCoordinator`],
//! and the [`fnv1a_hash`] helper.

// The `distributed` module is only compiled with the `server` feature, so this
// integration test must be gated the same way to keep `--no-default-features` green.
#![cfg(feature = "server")]

use oxibonsai_runtime::distributed::{
    fnv1a_hash, ConsistentHashRing, CoordinatorConfig, DistributedCoordinator, NodeInfo,
    NodeRegistry,
};

// ─── fnv1a_hash ───────────────────────────────────────────────────────────────

#[test]
fn fnv1a_deterministic() {
    assert_eq!(
        fnv1a_hash("oxibonsai"),
        fnv1a_hash("oxibonsai"),
        "same input must always produce same hash"
    );
}

#[test]
fn fnv1a_different_inputs() {
    assert_ne!(
        fnv1a_hash("node-0"),
        fnv1a_hash("node-1"),
        "distinct inputs should give distinct hashes"
    );
    assert_ne!(
        fnv1a_hash("alpha"),
        fnv1a_hash("beta"),
        "spot-check: alpha vs beta"
    );
    assert_ne!(
        fnv1a_hash(""),
        fnv1a_hash(" "),
        "empty string vs space differ"
    );
}

// ─── ConsistentHashRing ───────────────────────────────────────────────────────

#[test]
fn hash_ring_empty() {
    let ring = ConsistentHashRing::new(10);
    assert!(
        ring.get_node("any-key").is_none(),
        "empty ring must return None"
    );
}

#[test]
fn hash_ring_single_node() {
    let mut ring = ConsistentHashRing::new(50);
    ring.add_node("only-node");
    for key in &["a", "b", "c", "hello", "world", "request-42"] {
        assert_eq!(
            ring.get_node(key),
            Some("only-node"),
            "all keys should route to the single node"
        );
    }
}

#[test]
fn hash_ring_two_nodes_both_receive() {
    let mut ring = ConsistentHashRing::new(150);
    ring.add_node("node-a");
    ring.add_node("node-b");

    // With 150 replicas per node the distribution should be roughly even.
    // We generate diverse keys across a wide hash-space range and verify
    // both nodes receive traffic.  Using varied prefixes ensures the keys
    // are spread across the u64 ring.
    let prefixes = [
        "alpha:", "beta:", "gamma:", "delta:", "epsilon:", "zeta:", "eta:", "theta:", "iota:",
        "kappa:",
    ];
    let mut saw_a = false;
    let mut saw_b = false;
    'outer: for prefix in &prefixes {
        for i in 0..100u64 {
            let key = format!("{}{}", prefix, i);
            match ring.get_node(&key) {
                Some("node-a") => saw_a = true,
                Some("node-b") => saw_b = true,
                other => panic!("unexpected node: {:?}", other),
            }
            if saw_a && saw_b {
                break 'outer;
            }
        }
    }
    assert!(saw_a, "node-a should receive at least some keys");
    assert!(saw_b, "node-b should receive at least some keys");
}

#[test]
fn hash_ring_add_remove_node() {
    let mut ring = ConsistentHashRing::new(100);
    ring.add_node("node-a");
    ring.add_node("node-b");

    // Remove node-b — all keys must now route to node-a.
    ring.remove_node("node-b");
    for key in &["x", "y", "z", "request-1"] {
        assert_eq!(
            ring.get_node(key),
            Some("node-a"),
            "after removing node-b everything routes to node-a"
        );
    }
}

#[test]
fn hash_ring_consistent() {
    let mut ring = ConsistentHashRing::new(100);
    ring.add_node("n1");
    ring.add_node("n2");
    ring.add_node("n3");

    // Route the same key many times — must always return the same result.
    let key = "stable-request";
    let first = ring.get_node(key).expect("ring is non-empty");
    for _ in 0..50 {
        assert_eq!(
            ring.get_node(key),
            Some(first),
            "routing must be deterministic"
        );
    }
}

#[test]
fn hash_ring_get_multiple() {
    let mut ring = ConsistentHashRing::new(100);
    ring.add_node("n1");
    ring.add_node("n2");
    ring.add_node("n3");

    let nodes = ring.get_nodes("some-key", 2);
    assert_eq!(nodes.len(), 2, "get_nodes(key, 2) must return exactly 2");
    assert_ne!(nodes[0], nodes[1], "returned nodes must be distinct");

    // Requesting more than available nodes returns all unique nodes.
    let all = ring.get_nodes("some-key", 10);
    assert_eq!(all.len(), 3, "cannot return more distinct nodes than exist");
}

#[test]
fn hash_ring_vnode_count() {
    let replicas = 75_usize;
    let mut ring = ConsistentHashRing::new(replicas);
    ring.add_node("a");
    ring.add_node("b");
    assert_eq!(
        ring.vnode_count(),
        replicas * 2,
        "vnode_count must equal replicas * node_count"
    );
    ring.add_node("c");
    assert_eq!(ring.vnode_count(), replicas * 3);
}

// ─── NodeRegistry ─────────────────────────────────────────────────────────────

#[test]
fn node_registry_register() {
    let mut reg = NodeRegistry::new();
    reg.register(NodeInfo::new("n1", "127.0.0.1:8001"));
    let all = reg.all_nodes();
    assert_eq!(all.len(), 1, "registry should contain exactly one node");
    assert_eq!(all[0].id, "n1");
}

#[test]
fn node_registry_deregister() {
    let mut reg = NodeRegistry::new();
    reg.register(NodeInfo::new("n1", "127.0.0.1:8001"));
    reg.register(NodeInfo::new("n2", "127.0.0.1:8002"));
    reg.deregister("n1");
    let all = reg.all_nodes();
    assert_eq!(all.len(), 1, "one node should remain after deregistration");
    assert_eq!(all[0].id, "n2");
}

#[test]
fn node_registry_mark_healthy() {
    let mut reg = NodeRegistry::new();
    reg.register(NodeInfo::new("n1", "127.0.0.1:8001"));
    reg.register(NodeInfo::new("n2", "127.0.0.1:8002"));

    reg.mark_healthy("n1", false);

    let healthy: Vec<_> = reg.healthy_nodes();
    assert_eq!(healthy.len(), 1, "only one healthy node should remain");
    assert_eq!(
        healthy[0].id, "n2",
        "n2 should be the surviving healthy node"
    );
}

#[test]
fn node_registry_update_load() {
    let mut reg = NodeRegistry::new();
    reg.register(NodeInfo::new("n1", "127.0.0.1:8001"));
    reg.update_load("n1", 0.75);

    let node = reg
        .all_nodes()
        .into_iter()
        .find(|n| n.id == "n1")
        .expect("n1 must still be registered");
    assert!(
        (node.load - 0.75).abs() < 1e-6,
        "load should be updated to 0.75"
    );
}

#[test]
fn node_registry_route_request() {
    let mut reg = NodeRegistry::new();
    reg.register(NodeInfo::new("n1", "127.0.0.1:8001"));
    reg.register(NodeInfo::new("n2", "127.0.0.1:8002"));

    let routed = reg.route_request("some-request-key");
    assert!(routed.is_some(), "should route to a healthy node");
    let id = routed.expect("checked above").id.as_str();
    assert!(
        id == "n1" || id == "n2",
        "routed node must be one of the registered nodes"
    );
}

// ─── DistributedCoordinator ───────────────────────────────────────────────────

fn make_coordinator(node_id: &str) -> DistributedCoordinator {
    DistributedCoordinator::new(CoordinatorConfig {
        node_id: node_id.to_string(),
        bind_addr: "127.0.0.1:9000".to_string(),
        peers: Vec::new(),
        heartbeat_interval_ms: 1_000,
        health_timeout_ms: 5_000,
    })
}

#[test]
fn coordinator_register_self() {
    let mut coord = make_coordinator("self-node");
    coord.register_self();
    assert_eq!(coord.cluster_size(), 1, "coordinator should list itself");
    let nodes = coord.registry().all_nodes();
    assert_eq!(nodes[0].id, "self-node");
}

#[test]
fn coordinator_healthy_count() {
    let mut coord = make_coordinator("coord-0");
    coord.register_self();
    coord.add_peer("127.0.0.1:9001", "peer-1");
    coord.add_peer("127.0.0.1:9002", "peer-2");
    assert_eq!(coord.cluster_size(), 3);
    assert_eq!(coord.healthy_count(), 3, "all nodes start healthy");

    coord.set_peer_health("peer-1", false);
    assert_eq!(coord.healthy_count(), 2, "one node marked unhealthy");
}

#[test]
fn coordinator_topology_summary_nonempty() {
    let mut coord = make_coordinator("summary-node");
    coord.register_self();
    let summary = coord.topology_summary();
    assert!(!summary.is_empty(), "topology summary must not be empty");
    assert!(
        summary.contains("summary-node"),
        "summary should contain the node's own ID"
    );
}
