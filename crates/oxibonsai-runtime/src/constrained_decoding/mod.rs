//! Grammar-constrained decoding for token-by-token generation.
//!
//! This module provides the [`TokenConstraint`] trait and concrete implementations
//! that restrict which tokens the model can emit at each decoding step:
//!
//! - [`NoConstraint`] — passthrough, all tokens allowed
//! - [`RegexConstraint`] — restricts output to strings matching a regex pattern
//! - [`JsonConstraint`] — restricts output to syntactically valid JSON
//! - [`AllowListConstraint`] — restricts output to one of a finite set of token sequences
//! - [`SequenceConstraint`] — forces output to reproduce a specific token sequence
//! - [`LengthConstraint`] — enforces hard minimum and maximum generation lengths
//!
//! The [`ConstrainedSampler`] wraps a [`crate::sampling_advanced::SamplerChain`] and
//! applies a mask to logits before sampling so that only valid continuations are drawn.
//!
//! ## Example
//! ```rust
//! use oxibonsai_runtime::constrained_decoding::{ConstrainedSamplerBuilder, TokenConstraint};
//!
//! let mut sampler = ConstrainedSamplerBuilder::new(128, 42)
//!     .with_json_constraint();
//! assert!(!sampler.is_complete());
//! ```
//!
//! # Module structure
//!
//! Phase 30B split the monolithic `constrained_decoding.rs` (1966 lines) into
//! focused sub-modules; all external `crate::constrained_decoding::*` access
//! paths are preserved through the re-exports below.
//!
//!   - [`error_trait`][] — [`ConstraintError`], the [`TokenConstraint`] trait,
//!     and the passthrough [`NoConstraint`].
//!   - [`regex`][] — NFA-based [`RegexConstraint`].
//!   - [`json`][] — JSON-grammar [`JsonConstraint`] and its [`JsonParseState`].
//!   - [`sampler`][] — [`ConstrainedSampler`] and [`ConstrainedSamplerBuilder`].
//!   - [`allow_list`][] — [`AllowListConstraint`].
//!   - [`sequence`][] — [`SequenceConstraint`].
//!   - [`length`][] — [`LengthConstraint`].

mod allow_list;
mod error_trait;
mod json;
mod length;
mod regex;
mod sampler;
mod sequence;

pub use allow_list::AllowListConstraint;
pub use error_trait::{ConstraintError, NoConstraint, TokenConstraint};
pub use json::{JsonConstraint, JsonParseState};
pub use length::LengthConstraint;
pub use regex::RegexConstraint;
pub use sampler::{ConstrainedSampler, ConstrainedSamplerBuilder};
pub use sequence::SequenceConstraint;
