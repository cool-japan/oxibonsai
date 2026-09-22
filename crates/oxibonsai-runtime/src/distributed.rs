//! Distributed serving infrastructure for OxiBonsai.
//!
//! Provides a consistent hash ring for request routing across multiple
//! inference nodes, a node registry with health tracking, and a multi-node
//! coordinator that manages cluster topology.
//!
//! # Architecture
//!
//! ```text
//!  ┌────────────────────────────────────────────┐
//!  │  DistributedCoordinator                    │
//!  │  ┌──────────────────────────────────────┐  │
//!  │  │  NodeRegistry                        │  │
//!  │  │  ┌────────────────────────────────┐  │  │
//!  │  │  │  ConsistentHashRing            │  │  │
//!  │  │  │  [VNode, VNode, ..., VNode]    │  │  │
//!  │  │  └────────────────────────────────┘  │  │
//!  │  │  HashMap<node_id, NodeInfo>          │  │
//!  │  └──────────────────────────────────────┘  │
//!  └────────────────────────────────────────────┘
//! ```
//!
//! The hash ring uses FNV-1a (64-bit) with virtual nodes for even distribution.
//! All state is in-memory — no actual TCP connections are made.
//!
//! ## Reachability
//!
//! [`ConsistentHashRing`] and [`NodeRegistry`]'s hashing/bookkeeping logic
//! is correct and exercised by this module's own tests, but as of this
//! writing nothing else in `oxibonsai-runtime` (the server request path,
//! the engine, the CLI) constructs a [`DistributedCoordinator`] or routes a
//! real request through it — there is no wired multi-node serving feature
//! today. If a README or release note describes "distributed serving", that
//! claim must be qualified against this module being a primitive rather
//! than a running feature.

use std::collections::HashMap;

// ─── FNV-1a hash ──────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — fast, good distribution, no external deps.
///
/// Reference: <http://www.isthe.com/chongo/tech/comp/fnv/>
pub fn fnv1a_hash(input: &str) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash: u64 = OFFSET_BASIS;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ─── Consistent Hash Ring ─────────────────────────────────────────────────────

/// A virtual node on the ring — maps a hash position to a backend node.
#[derive(Debug, Clone)]
pub struct VNode {
    /// Position on the ring (FNV-1a hash of `"<node_id>#<replica_index>"`).
    pub hash: u64,
    /// The real node this virtual node represents.
    pub node_id: String,
}

/// Consistent hash ring with virtual nodes for even load distribution.
///
/// Virtual nodes (`replicas` per real node) help achieve more uniform key
/// distribution even with small cluster sizes.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::distributed::ConsistentHashRing;
///
/// let mut ring = ConsistentHashRing::new(150);
/// ring.add_node("node-a");
/// ring.add_node("node-b");
/// let target = ring.get_node("my-request-key");
/// assert!(target.is_some());
/// ```
pub struct ConsistentHashRing {
    /// Virtual nodes sorted by hash — the ring's backbone.
    vnodes: Vec<VNode>,
    /// Number of virtual nodes created per real node.
    replicas: usize,
}

impl ConsistentHashRing {
    /// Create a new empty ring.
    ///
    /// `replicas` controls how many virtual nodes are placed on the ring per
    /// real node. Higher values give better distribution at the cost of memory.
    /// A value of 100–200 is typical.
    pub fn new(replicas: usize) -> Self {
        Self {
            vnodes: Vec::new(),
            replicas: replicas.max(1),
        }
    }

    /// Add a node to the ring by inserting `replicas` virtual nodes.
    ///
    /// Virtual node keys are `"<node_id>#<i>"` for `i` in `0..replicas`.
    /// After insertion the internal slice is re-sorted.
    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.replicas {
            let key = format!("{}#{}", node_id, i);
            let hash = fnv1a_hash(&key);
            self.vnodes.push(VNode {
                hash,
                node_id: node_id.to_string(),
            });
        }
        self.vnodes.sort_unstable_by_key(|v| v.hash);
    }

    /// Remove all virtual nodes belonging to `node_id` from the ring.
    pub fn remove_node(&mut self, node_id: &str) {
        self.vnodes.retain(|v| v.node_id != node_id);
    }

    /// Route a key to the first virtual node at or after its hash position.
    ///
    /// Returns `None` if the ring is empty.
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.vnodes.is_empty() {
            return None;
        }
        let hash = fnv1a_hash(key);
        // Binary search for the first vnode with hash >= key hash.
        let idx = self.vnodes.partition_point(|v| v.hash < hash) % self.vnodes.len();
        Some(&self.vnodes[idx].node_id)
    }

    /// Route a key and return up to `count` distinct real nodes in ring order.
    ///
    /// Useful for replication — returns the first `count` unique node IDs
    /// encountered walking clockwise from the key's position.
    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<&str> {
        if self.vnodes.is_empty() || count == 0 {
            return Vec::new();
        }
        let hash = fnv1a_hash(key);
        let start = self.vnodes.partition_point(|v| v.hash < hash) % self.vnodes.len();

        let mut result: Vec<&str> = Vec::with_capacity(count);
        let total = self.vnodes.len();

        for offset in 0..total {
            let idx = (start + offset) % total;
            let node_id = self.vnodes[idx].node_id.as_str();
            if !result.contains(&node_id) {
                result.push(node_id);
            }
            if result.len() >= count {
                break;
            }
        }
        result
    }

    /// Number of distinct real nodes currently on the ring.
    pub fn node_count(&self) -> usize {
        let mut seen: Vec<&str> = Vec::new();
        for v in &self.vnodes {
            let s = v.node_id.as_str();
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen.len()
    }

    /// Total number of virtual nodes on the ring (`replicas × node_count`).
    pub fn vnode_count(&self) -> usize {
        self.vnodes.len()
    }
}

// ─── Node Registry ────────────────────────────────────────────────────────────

/// Runtime information about a single serving node.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Unique node identifier (e.g. `"node-0"`, `"gpu-west-1"`).
    pub id: String,
    /// Network address, `"host:port"` format (informational only).
    pub addr: String,
    /// Whether the node passed its most recent health check.
    pub healthy: bool,
    /// Normalised load factor in `[0.0, 1.0]` — 0 is idle, 1 is saturated.
    pub load: f32,
    /// Epoch-milliseconds timestamp of the last heartbeat/update.
    pub last_seen_ms: u64,
}

impl NodeInfo {
    /// Construct a healthy node with zero load.
    pub fn new(id: impl Into<String>, addr: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            addr: addr.into(),
            healthy: true,
            load: 0.0,
            last_seen_ms: current_time_ms(),
        }
    }
}

/// Cluster membership store backed by a consistent hash ring.
///
/// Maintains a live map of [`NodeInfo`] and mirrors add/remove operations
/// into a [`ConsistentHashRing`] so routing decisions stay in sync.
pub struct NodeRegistry {
    nodes: HashMap<String, NodeInfo>,
    ring: ConsistentHashRing,
}

impl NodeRegistry {
    /// Create an empty registry with 150 virtual nodes per real node.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            ring: ConsistentHashRing::new(150),
        }
    }

    /// Register a node (or overwrite an existing entry with the same ID).
    pub fn register(&mut self, info: NodeInfo) {
        let id = info.id.clone();
        // If already present, remove its old vnodes before re-adding.
        if self.nodes.contains_key(&id) {
            self.ring.remove_node(&id);
        }
        self.ring.add_node(&id);
        self.nodes.insert(id, info);
    }

    /// Remove a node from the registry and the hash ring.
    pub fn deregister(&mut self, node_id: &str) {
        self.ring.remove_node(node_id);
        self.nodes.remove(node_id);
    }

    /// Update the health status of a node.
    ///
    /// If `healthy` is `false` the node is kept in the registry but excluded
    /// from routing via `route_request` and `healthy_nodes`.
    pub fn mark_healthy(&mut self, node_id: &str, healthy: bool) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.healthy = healthy;
            node.last_seen_ms = current_time_ms();
        }
    }

    /// Update the load factor of a node.  Clamped to `[0.0, 1.0]`.
    pub fn update_load(&mut self, node_id: &str, load: f32) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.load = load.clamp(0.0, 1.0);
            node.last_seen_ms = current_time_ms();
        }
    }

    /// Route a request to a healthy node using consistent hashing.
    ///
    /// Walks the ring starting at `request_key`'s hash position and returns
    /// the first node that exists in the registry **and** is healthy.
    /// Returns `None` if there are no healthy nodes.
    pub fn route_request(&self, request_key: &str) -> Option<&NodeInfo> {
        // Ask the ring for up to all nodes in order, then pick the first
        // healthy one.
        let candidates = self.ring.get_nodes(request_key, self.nodes.len().max(1));
        for node_id in candidates {
            if let Some(info) = self.nodes.get(node_id) {
                if info.healthy {
                    return Some(info);
                }
            }
        }
        None
    }

    /// Returns references to all healthy nodes (arbitrary order).
    pub fn healthy_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes.values().filter(|n| n.healthy).collect()
    }

    /// Returns references to all registered nodes (arbitrary order).
    pub fn all_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes.values().collect()
    }

    /// Expose a reference to the underlying ring (read-only).
    pub fn ring(&self) -> &ConsistentHashRing {
        &self.ring
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Multi-node Coordinator ───────────────────────────────────────────────────

/// Configuration for a [`DistributedCoordinator`] instance.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Identifier for *this* node.
    pub node_id: String,
    /// Address this node listens on (`"host:port"`).
    pub bind_addr: String,
    /// Peer addresses to seed the cluster with.
    pub peers: Vec<String>,
    /// How often to send heartbeats (milliseconds).
    pub heartbeat_interval_ms: u64,
    /// Age after which a node is considered unhealthy (milliseconds).
    pub health_timeout_ms: u64,
}

impl CoordinatorConfig {
    /// Sensible defaults for a single-node development setup.
    pub fn local_default(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            bind_addr: "127.0.0.1:8080".to_string(),
            peers: Vec::new(),
            heartbeat_interval_ms: 1_000,
            health_timeout_ms: 5_000,
        }
    }
}

/// In-memory multi-node coordinator.
///
/// Manages cluster topology, routes incoming requests to healthy nodes via
/// consistent hashing, and exposes cluster health information.
///
/// **Note:** This implementation is intentionally in-memory only — no actual
/// TCP connections are established. It is designed for unit testing and
/// single-process simulation. Production deployments would wrap this with a
/// gRPC/HTTP gossip layer.
pub struct DistributedCoordinator {
    config: CoordinatorConfig,
    registry: NodeRegistry,
}

impl DistributedCoordinator {
    /// Create a new coordinator with the given configuration.
    ///
    /// Does not automatically register `self` — call `register_self` to
    /// add this node to the ring.
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            registry: NodeRegistry::new(),
        }
    }

    /// Register this node in the local registry so it participates in routing.
    pub fn register_self(&mut self) {
        let info = NodeInfo::new(self.config.node_id.clone(), self.config.bind_addr.clone());
        self.registry.register(info);
    }

    /// Add a peer to the registry as a healthy node with zero load.
    ///
    /// `addr` is the peer's `"host:port"` bind address.
    /// `node_id` is the peer's unique identifier.
    pub fn add_peer(&mut self, addr: &str, node_id: &str) {
        let info = NodeInfo::new(node_id, addr);
        self.registry.register(info);
    }

    /// Route `request_key` to a healthy node and return its address.
    ///
    /// Returns `None` if no healthy nodes are available.
    pub fn route(&self, request_key: &str) -> Option<String> {
        self.registry
            .route_request(request_key)
            .map(|n| n.addr.clone())
    }

    /// Total number of nodes registered in the cluster (healthy + unhealthy).
    pub fn cluster_size(&self) -> usize {
        self.registry.all_nodes().len()
    }

    /// Number of nodes currently marked as healthy.
    pub fn healthy_count(&self) -> usize {
        self.registry.healthy_nodes().len()
    }

    /// A human-readable summary of current cluster topology.
    ///
    /// Format (not stable across versions):
    /// ```text
    /// cluster[nodes=3 healthy=2 vnodes=450 self=node-0]
    /// ```
    pub fn topology_summary(&self) -> String {
        let total = self.cluster_size();
        let healthy = self.healthy_count();
        let vnodes = self.registry.ring().vnode_count();
        let self_id = &self.config.node_id;
        format!("cluster[nodes={total} healthy={healthy} vnodes={vnodes} self={self_id}]")
    }

    /// Access the underlying registry (read-only).
    pub fn registry(&self) -> &NodeRegistry {
        &self.registry
    }

    /// Access the coordinator config.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Mark a peer node as healthy or unhealthy.
    pub fn set_peer_health(&mut self, node_id: &str, healthy: bool) {
        self.registry.mark_healthy(node_id, healthy);
    }

    /// Update a peer's reported load factor.
    pub fn update_peer_load(&mut self, node_id: &str, load: f32) {
        self.registry.update_load(node_id, load);
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return current wall-clock time in milliseconds since the Unix epoch.
///
/// Falls back to `0` if the system clock is before the epoch (unlikely).
fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_deterministic() {
        assert_eq!(fnv1a_hash("hello"), fnv1a_hash("hello"));
        assert_eq!(fnv1a_hash(""), fnv1a_hash(""));
    }

    #[test]
    fn fnv1a_different_inputs() {
        assert_ne!(fnv1a_hash("foo"), fnv1a_hash("bar"));
        assert_ne!(fnv1a_hash("node-0"), fnv1a_hash("node-1"));
    }

    #[test]
    fn hash_ring_empty_returns_none() {
        let ring = ConsistentHashRing::new(10);
        assert!(ring.get_node("any-key").is_none());
    }

    #[test]
    fn hash_ring_single_node_always_routes_there() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node("solo");
        for key in &["a", "b", "c", "hello", "world", "12345"] {
            assert_eq!(ring.get_node(key), Some("solo"));
        }
    }

    #[test]
    fn hash_ring_vnode_count_equals_replicas_times_nodes() {
        let mut ring = ConsistentHashRing::new(50);
        ring.add_node("n1");
        assert_eq!(ring.vnode_count(), 50);
        ring.add_node("n2");
        assert_eq!(ring.vnode_count(), 100);
        ring.add_node("n3");
        assert_eq!(ring.vnode_count(), 150);
    }

    #[test]
    fn hash_ring_node_count() {
        let mut ring = ConsistentHashRing::new(10);
        assert_eq!(ring.node_count(), 0);
        ring.add_node("a");
        assert_eq!(ring.node_count(), 1);
        ring.add_node("b");
        assert_eq!(ring.node_count(), 2);
    }
}
