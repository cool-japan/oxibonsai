//! Multi-model serving: manage base models + LoRA adapters with smart routing.
//!
//! Supports:
//! - Multiple base model configurations
//! - Hot-swappable LoRA adapter registry
//! - Request routing by model ID
//! - Model alias resolution
//! - Adapter composition (stacking multiple LoRAs)

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// ModelId
// ─────────────────────────────────────────────────────────────────────────────

/// A model endpoint identifier.
///
/// Uses a convention where `"base_name"` denotes a base model and
/// `"base_name:adapter_name"` denotes a base model with a LoRA adapter applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl ModelId {
    /// Create a new model identifier from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` if this is a base model (no `":"` separator).
    pub fn is_base(&self) -> bool {
        !self.0.contains(':')
    }

    /// If the identifier has the form `"base:adapter"`, return `Some("adapter")`.
    /// Otherwise return `None`.
    pub fn adapter_name(&self) -> Option<&str> {
        self.0.split_once(':').map(|(_, adapter)| adapter)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The id a served model is listed under
// ─────────────────────────────────────────────────────────────────────────────

/// `general.name` values that name no model at all. Some converters leave the
/// hub's placeholder (`Hf`) or a generic word in the field — the Bonsai 2 27B
/// GGUFs carry `general.name = "Hf"` — and listing the model under one of
/// them would make `/v1/models` useless.
pub const PLACEHOLDER_MODEL_NAMES: [&str; 4] = ["hf", "model", "gguf", "llama"];

/// The fewest characters a `general.name` needs to serve as a model id.
pub const MIN_MODEL_NAME_CHARS: usize = 4;

/// Whether `name` (a GGUF's `general.name`) is fit to be the id a model is
/// served under: at least [`MIN_MODEL_NAME_CHARS`] characters once trimmed
/// and not, case-insensitively, one of [`PLACEHOLDER_MODEL_NAMES`].
#[must_use]
pub fn is_usable_model_name(name: &str) -> bool {
    let name = name.trim();
    name.chars().count() >= MIN_MODEL_NAME_CHARS
        && !PLACEHOLDER_MODEL_NAMES
            .iter()
            .any(|placeholder| name.eq_ignore_ascii_case(placeholder))
}

/// The id a server lists its model under (`GET /v1/models`, and the `model`
/// member of every completion it returns).
///
/// The rule is deterministic:
///
/// 1. `general_name` — the GGUF's own `general.name`, exactly as the file
///    spells it — when [`is_usable_model_name`] accepts it;
/// 2. otherwise `file_stem` — the model file's name without its extension,
///    e.g. `Ternary-Bonsai-2-27B-PQ2_0` for
///    `models/Ternary-Bonsai-2-27B-PQ2_0.gguf` — when it is not blank;
/// 3. otherwise `None`: the file names nothing usable and the caller keeps
///    whatever name it already has.
///
/// `general_name` must be the raw metadata value (`None` when the key is
/// absent), not a name a loader already substituted for a missing one — the
/// dense loader reports the architecture tag when `general.name` is absent,
/// and that tag would pass rule 1.
#[must_use]
pub fn served_model_id(general_name: Option<&str>, file_stem: Option<&str>) -> Option<String> {
    general_name
        .filter(|name| is_usable_model_name(name))
        .map(str::trim)
        .or_else(|| file_stem.map(str::trim).filter(|stem| !stem.is_empty()))
        .map(str::to_string)
}

// ─────────────────────────────────────────────────────────────────────────────
// EndpointStatus
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a model endpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndpointStatus {
    /// The model is loaded and ready to serve requests.
    Ready,
    /// The model is currently being loaded.
    Loading,
    /// The model encountered an error and is unavailable.
    Error,
    /// The model has been explicitly disabled by an administrator.
    Disabled,
}

impl EndpointStatus {
    /// Returns `true` if the endpoint is available for serving requests.
    pub fn is_available(&self) -> bool {
        *self == Self::Ready
    }

    /// Human-readable name for this status.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Loading => "loading",
            Self::Error => "error",
            Self::Disabled => "disabled",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelEndpoint
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a served model variant.
///
/// Each endpoint represents a unique model configuration that can receive
/// inference requests. A base model may have multiple endpoints, each with
/// a different LoRA adapter applied.
#[derive(Debug, Clone)]
pub struct ModelEndpoint {
    /// Unique identifier for this endpoint.
    pub id: ModelId,
    /// Human-readable display name.
    pub display_name: String,
    /// Longer description of what this endpoint provides.
    pub description: String,
    /// Name of the underlying base model.
    pub base_model: String,
    /// Optional LoRA adapter name applied on top of the base model.
    pub adapter: Option<String>,
    /// Maximum context length (in tokens) this endpoint supports.
    pub max_context_length: usize,
    /// Whether this endpoint is the default when no model is specified.
    pub is_default: bool,
    /// Current operational status.
    pub status: EndpointStatus,
}

impl ModelEndpoint {
    /// Create a new endpoint with sensible defaults.
    ///
    /// Status is set to `Ready`, no adapter, default context length of 4096.
    pub fn new(id: impl Into<String>, base_model: impl Into<String>) -> Self {
        let id_str: String = id.into();
        let base: String = base_model.into();
        Self {
            display_name: id_str.clone(),
            id: ModelId::new(id_str),
            description: String::new(),
            base_model: base,
            adapter: None,
            max_context_length: 4096,
            is_default: false,
            status: EndpointStatus::Ready,
        }
    }

    /// The endpoint of a model file: listed under the id [`served_model_id`]
    /// derives from the file's `general.name` and its stem, or under
    /// `fallback_id` when the file names nothing usable. The base model is
    /// the file's own `general.name` (the served id when the file has none).
    pub fn from_model_file(
        general_name: Option<&str>,
        file_stem: Option<&str>,
        fallback_id: impl Into<String>,
    ) -> Self {
        let id = served_model_id(general_name, file_stem).unwrap_or_else(|| fallback_id.into());
        let base = general_name
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map_or_else(|| id.clone(), str::to_string);
        Self::new(id, base)
    }

    /// Attach a LoRA adapter to this endpoint.
    pub fn with_adapter(mut self, adapter: impl Into<String>) -> Self {
        self.adapter = Some(adapter.into());
        self
    }

    /// Set a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the maximum context length.
    pub fn with_context_length(mut self, ctx: usize) -> Self {
        self.max_context_length = ctx;
        self
    }

    /// Mark this endpoint as the default.
    pub fn set_default(mut self) -> Self {
        self.is_default = true;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// The multi-model registry.
///
/// Manages a collection of [`ModelEndpoint`] instances and supports alias
/// resolution so that clients can refer to models by friendly names.
pub struct ModelRegistry {
    endpoints: HashMap<ModelId, ModelEndpoint>,
    aliases: HashMap<String, ModelId>,
    default_model: Option<ModelId>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            aliases: HashMap::new(),
            default_model: None,
        }
    }

    /// Register a model endpoint.
    ///
    /// If the endpoint has `is_default` set, it becomes the default model.
    /// Replaces any existing endpoint with the same ID.
    pub fn register(&mut self, endpoint: ModelEndpoint) {
        if endpoint.is_default {
            self.default_model = Some(endpoint.id.clone());
        }
        self.endpoints.insert(endpoint.id.clone(), endpoint);
    }

    /// Remove an endpoint from the registry.
    ///
    /// Also clears the default-model pointer if it pointed to the removed
    /// endpoint, and removes any aliases that targeted this ID.
    pub fn unregister(&mut self, id: &ModelId) -> Option<ModelEndpoint> {
        let removed = self.endpoints.remove(id);
        if removed.is_some() {
            // Clear default if it was this model.
            if self.default_model.as_ref() == Some(id) {
                self.default_model = None;
            }
            // Remove aliases pointing to this model.
            self.aliases.retain(|_, target| target != id);
        }
        removed
    }

    /// Add an alias: e.g. `"gpt-4"` maps to `ModelId("bonsai-8b")`.
    pub fn add_alias(&mut self, alias: impl Into<String>, target: ModelId) {
        self.aliases.insert(alias.into(), target);
    }

    /// Resolve a model identifier (checks ID first, then aliases).
    ///
    /// Returns `None` if neither a direct ID nor an alias matches.
    pub fn resolve(&self, id_or_alias: &str) -> Option<&ModelEndpoint> {
        let model_id = ModelId::new(id_or_alias);
        if let Some(ep) = self.endpoints.get(&model_id) {
            return Some(ep);
        }
        // Try alias resolution.
        if let Some(target_id) = self.aliases.get(id_or_alias) {
            return self.endpoints.get(target_id);
        }
        None
    }

    /// Get the default model endpoint.
    pub fn default_endpoint(&self) -> Option<&ModelEndpoint> {
        self.default_model
            .as_ref()
            .and_then(|id| self.endpoints.get(id))
    }

    /// List all available (Ready) endpoints.
    pub fn available_endpoints(&self) -> Vec<&ModelEndpoint> {
        self.endpoints
            .values()
            .filter(|ep| ep.status.is_available())
            .collect()
    }

    /// List all registered endpoints (including non-ready ones).
    pub fn all_endpoints(&self) -> Vec<&ModelEndpoint> {
        self.endpoints.values().collect()
    }

    /// Update an endpoint's status.
    ///
    /// Returns `true` if the endpoint was found and updated.
    pub fn set_status(&mut self, id: &ModelId, status: EndpointStatus) -> bool {
        if let Some(ep) = self.endpoints.get_mut(id) {
            ep.status = status;
            true
        } else {
            false
        }
    }

    /// Number of registered endpoints.
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// Is the registry empty?
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when routing a request to a model endpoint.
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    /// The requested model was not found in the registry.
    #[error("model '{0}' not found")]
    ModelNotFound(String),

    /// The requested model cannot accommodate the required context length.
    #[error("model '{model}' cannot handle context length {required} (max: {available})")]
    ContextTooLong {
        model: String,
        required: usize,
        available: usize,
    },

    /// No models are currently available in the registry.
    #[error("no models are currently available")]
    NoModelsAvailable,

    /// The model was found but is not in a ready state.
    #[error("model '{0}' is not ready (status: {1})")]
    ModelNotReady(String, String),
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Smart request router: selects the best model endpoint for a request.
///
/// Wraps a [`ModelRegistry`] and adds routing logic including fallback to
/// the default model, context-length awareness, and OpenAI-compatible
/// model listing.
pub struct ModelRouter {
    registry: ModelRegistry,
}

impl ModelRouter {
    /// Create a new router backed by the given registry.
    pub fn new(registry: ModelRegistry) -> Self {
        Self { registry }
    }

    /// Route a request: resolve `model_id` from the request.
    ///
    /// Falls back to the default model if `requested_model` is `None`.
    /// Returns an error if the resolved endpoint is not in a `Ready` state.
    pub fn route(&self, requested_model: Option<&str>) -> Result<&ModelEndpoint, RoutingError> {
        let endpoint = match requested_model {
            Some(model_name) => self
                .registry
                .resolve(model_name)
                .ok_or_else(|| RoutingError::ModelNotFound(model_name.to_string()))?,
            None => self
                .registry
                .default_endpoint()
                .ok_or(RoutingError::NoModelsAvailable)?,
        };

        if !endpoint.status.is_available() {
            return Err(RoutingError::ModelNotReady(
                endpoint.id.to_string(),
                endpoint.status.name().to_string(),
            ));
        }

        Ok(endpoint)
    }

    /// Route with context-length awareness: pick a model that can accommodate
    /// the required context length.
    ///
    /// If a specific model is requested, validates it has sufficient context.
    /// If no model is specified, finds the default model that fits, or falls
    /// back to any available model with sufficient context capacity.
    pub fn route_for_context(
        &self,
        requested_model: Option<&str>,
        required_context: usize,
    ) -> Result<&ModelEndpoint, RoutingError> {
        let endpoint = self.route(requested_model)?;

        if endpoint.max_context_length < required_context {
            // If a specific model was requested but is too small, error out.
            if requested_model.is_some() {
                return Err(RoutingError::ContextTooLong {
                    model: endpoint.id.to_string(),
                    required: required_context,
                    available: endpoint.max_context_length,
                });
            }

            // No specific model requested — try to find any available endpoint
            // with sufficient context capacity.
            let fallback = self
                .registry
                .available_endpoints()
                .into_iter()
                .filter(|ep| ep.max_context_length >= required_context)
                .max_by_key(|ep| ep.max_context_length);

            return fallback.ok_or(RoutingError::ContextTooLong {
                model: endpoint.id.to_string(),
                required: required_context,
                available: endpoint.max_context_length,
            });
        }

        Ok(endpoint)
    }

    /// OpenAI-compatible `/v1/models` list.
    ///
    /// Returns an entry for every available endpoint in the registry.
    pub fn models_list(&self) -> Vec<ModelListEntry> {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.registry
            .available_endpoints()
            .into_iter()
            .map(|ep| ModelListEntry {
                id: ep.id.to_string(),
                object: "model".to_string(),
                owned_by: "oxibonsai".to_string(),
                created,
            })
            .collect()
    }

    /// Immutable access to the underlying registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// Mutable access to the underlying registry.
    pub fn registry_mut(&mut self) -> &mut ModelRegistry {
        &mut self.registry
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelListEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Entry for an OpenAI-compatible `/v1/models` response.
#[derive(Debug, Clone)]
pub struct ModelListEntry {
    /// Model identifier string.
    pub id: String,
    /// Object type — always `"model"`.
    pub object: String,
    /// Organisation that owns the model.
    pub owned_by: String,
    /// Unix timestamp when the model was created/registered.
    pub created: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// AdapterRef / AdapterStack
// ─────────────────────────────────────────────────────────────────────────────

/// A reference to a single LoRA adapter with a blending weight.
#[derive(Debug, Clone)]
pub struct AdapterRef {
    /// Name of the LoRA adapter.
    pub name: String,
    /// Blending weight in the range `[0.0, 1.0]`.
    pub weight: f32,
}

/// Adapter composition: apply multiple LoRA adapters in sequence.
///
/// Allows stacking several adapters with independent blending weights.
/// Weights can be normalized so they sum to 1.0, which is useful for
/// even blending across adapters.
#[derive(Debug, Clone)]
pub struct AdapterStack {
    /// The ordered list of adapters to apply.
    pub adapters: Vec<AdapterRef>,
}

impl AdapterStack {
    /// Create an empty adapter stack.
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Add an adapter with the given blending weight.
    pub fn add(mut self, name: impl Into<String>, weight: f32) -> Self {
        self.adapters.push(AdapterRef {
            name: name.into(),
            weight,
        });
        self
    }

    /// Number of adapters in the stack.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Whether the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Sum of all adapter weights.
    pub fn total_weight(&self) -> f32 {
        self.adapters.iter().map(|a| a.weight).sum()
    }

    /// Normalize weights so they sum to 1.0.
    ///
    /// If the total weight is zero (or very close to it), weights are left
    /// unchanged to avoid division by zero.
    pub fn normalize_weights(&mut self) {
        let total = self.total_weight();
        if total.abs() < f32::EPSILON {
            return;
        }
        for adapter in &mut self.adapters {
            adapter.weight /= total;
        }
    }
}

impl Default for AdapterStack {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_display() {
        let id = ModelId::new("bonsai-8b");
        assert_eq!(format!("{id}"), "bonsai-8b");
    }

    #[test]
    fn endpoint_status_name() {
        assert_eq!(EndpointStatus::Ready.name(), "ready");
        assert_eq!(EndpointStatus::Loading.name(), "loading");
        assert_eq!(EndpointStatus::Error.name(), "error");
        assert_eq!(EndpointStatus::Disabled.name(), "disabled");
    }

    #[test]
    fn endpoint_display_name_defaults_to_id() {
        let ep = ModelEndpoint::new("bonsai-8b", "qwen3-8b");
        assert_eq!(ep.display_name, "bonsai-8b");
    }

    // ── The id a served model is listed under ───────────────────────────────

    const STEM: &str = "Ternary-Bonsai-2-27B-PQ2_0";

    #[test]
    fn a_real_general_name_is_the_served_id() {
        assert_eq!(
            served_model_id(Some("Ternary-Bonsai-2-27B"), Some(STEM)).as_deref(),
            Some("Ternary-Bonsai-2-27B"),
            "a usable name wins over the file stem"
        );
        // Exactly the minimum length is enough, and surrounding whitespace
        // is not part of the name.
        assert_eq!(
            served_model_id(Some("  Qwen "), Some(STEM)).as_deref(),
            Some("Qwen")
        );
    }

    #[test]
    fn the_27b_placeholder_name_falls_back_to_the_file_stem() {
        assert_eq!(
            served_model_id(Some("Hf"), Some(STEM)).as_deref(),
            Some(STEM),
            "the Bonsai 2 27B files carry general.name = \"Hf\""
        );
    }

    #[test]
    fn every_placeholder_and_short_name_is_refused_case_insensitively() {
        for name in [
            "hf", "HF", "Hf", "model", "MODEL", "Model", "gguf", "GGUF", "GgUf", "llama", "LLaMA",
            "LLAMA",
        ] {
            assert!(!is_usable_model_name(name), "{name:?} names no model");
            assert_eq!(
                served_model_id(Some(name), Some(STEM)).as_deref(),
                Some(STEM),
                "{name:?}"
            );
        }
        for short in ["", " ", "a", "ab", "abc", " abc ", "日本語"] {
            assert!(!is_usable_model_name(short), "{short:?} is too short");
        }
        // Length counts characters, not bytes: four CJK characters are a name.
        assert!(is_usable_model_name("日本語処"));
        // A placeholder is only a placeholder as a whole name.
        for name in ["llama-3", "hf-model", "my-model", "gguf.v2", "Llama2"] {
            assert!(is_usable_model_name(name), "{name:?} is a real name");
        }
    }

    #[test]
    fn a_missing_name_and_a_missing_stem_leave_the_caller_its_own_fallback() {
        assert_eq!(served_model_id(None, Some(STEM)).as_deref(), Some(STEM));
        assert_eq!(served_model_id(Some("Hf"), None), None);
        assert_eq!(served_model_id(Some("Hf"), Some("  ")), None);
        assert_eq!(served_model_id(None, None), None);
        assert_eq!(
            served_model_id(Some("Hf"), Some(" spaced-stem ")).as_deref(),
            Some("spaced-stem")
        );
    }

    #[test]
    fn an_endpoint_built_from_a_model_file_uses_the_rule() {
        let placeholder = ModelEndpoint::from_model_file(Some("Hf"), Some(STEM), "fallback");
        assert_eq!(placeholder.id.as_str(), STEM);
        assert_eq!(placeholder.display_name, STEM);
        assert_eq!(placeholder.base_model, "Hf", "the file's own name is kept");

        let named = ModelEndpoint::from_model_file(Some("Bonsai-8B"), Some("other"), "fallback");
        assert_eq!(named.id.as_str(), "Bonsai-8B");

        let nameless = ModelEndpoint::from_model_file(None, None, "fallback");
        assert_eq!(nameless.id.as_str(), "fallback");
        assert_eq!(nameless.base_model, "fallback");
    }
}
