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
//! ## Real tokenizers vs. the demonstration path
//!
//! [`RegexConstraint`] and [`JsonConstraint`] restrict generation based on the
//! **text** a token decodes to, so they need a decode function
//! (`Fn(u32) -> Option<String>`) that maps each token id to the text it emits.
//! Use the `*_with_decoder` builder methods (or the `with_decoder` constructors)
//! for any real tokenizer.  The no-argument
//! [`with_json_constraint`](ConstrainedSamplerBuilder::with_json_constraint) /
//! [`with_regex_constraint`](ConstrainedSamplerBuilder::with_regex_constraint) and
//! the [`JsonConstraint::new`] / [`RegexConstraint::new`] constructors are
//! **demonstration-only**: they treat each raw token id as a Unicode code point,
//! which is meaningful *only* for a synthetic vocabulary where
//! `token_id == codepoint`.  For byte-exact CFG constraints see
//! [`crate::grammar::GrammarConstraint`].
//!
//! ## Example
//! ```rust
//! use oxibonsai_runtime::constrained_decoding::{ConstrainedSamplerBuilder, TokenConstraint};
//!
//! // A real tokenizer supplies `decode_fn`; here a tiny toy vocab stands in.
//! let decode = |id: u32| match id {
//!     0 => Some("{".to_string()),
//!     1 => Some("}".to_string()),
//!     _ => None,
//! };
//! let mut sampler = ConstrainedSamplerBuilder::new(2, 42)
//!     .with_json_constraint_decoder(decode);
//! assert!(!sampler.is_complete());
//! ```
//!
//! # Module structure
//!
//! Phase 30B split the monolithic `constrained_decoding.rs` (1966 lines) into
//! focused sub-modules; all external `crate::constrained_decoding::*` access
//! paths are preserved through the re-exports below.
//!
//!   - `error_trait` — [`ConstraintError`], the [`TokenConstraint`] trait,
//!     and the passthrough [`NoConstraint`].
//!   - `regex` — NFA-based [`RegexConstraint`].
//!   - `json` — JSON-grammar [`JsonConstraint`] and its [`JsonParseState`].
//!   - `sampler` — [`ConstrainedSampler`] and [`ConstrainedSamplerBuilder`].
//!   - `allow_list` — [`AllowListConstraint`].
//!   - `sequence` — [`SequenceConstraint`].
//!   - `length` — [`LengthConstraint`].

mod allow_list;
mod decoder;
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
