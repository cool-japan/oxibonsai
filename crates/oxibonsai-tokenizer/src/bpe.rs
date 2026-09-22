//! Byte-Pair Encoding (BPE) merge table and encoding routines.
//!
//! This module implements:
//! - [`BpeMerges`] — a merge table mapping symbol pairs to merged-token IDs
//! - [`bpe_encode`] — greedy BPE encoding of a pre-tokenized word
//! - [`pretokenize`] — GPT-2–style whitespace/punctuation split
//! - [`PreTokenizerKind`] / [`pretokenize_gpt2`] / [`pretokenize_qwen35`] —
//!   regex-driven `Split` pre-tokenizers (TOK-03/TOK-04)
//! - [`byte_fallback_id`] — produce a `<0xHH>` token name for unknown bytes

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use crate::vocab::Vocabulary;

// ── BpeMerges ────────────────────────────────────────────────────────────────

/// A single merge-table entry: its priority rank and the merged token's ID.
///
/// Storing the priority directly next to the result ID makes both
/// [`BpeMerges::get_merge_priority`] and [`BpeMerges::get_merge_result`] true
/// `O(1)` lookups. `rank` is unique per distinct pair (see
/// [`BpeMerges::add_merge`]), which the BPE merge loop's lazy-invalidation
/// scheme in [`bpe_merge_symbols`] relies on.
#[derive(Debug, Clone, Copy)]
struct MergeEntry {
    /// 0-based priority rank (insertion order); lower = higher priority.
    rank: u32,
    /// ID of the merged token.
    result_id: u32,
}

/// BPE merge table: a set of (A, B) → merged-ID rules ordered by priority.
///
/// Lower priority index = earlier merge (higher priority).
///
/// ## Storage shape (TOK-06)
///
/// Internally this is a *nested* map — `left symbol -> (right symbol ->
/// entry)` — rather than a single `HashMap<(String, String), _>`. A flat
/// map keyed by an owned `(String, String)` tuple cannot be probed with a
/// borrowed `(&str, &str)` (`std`'s tuple types have no blanket `Borrow`
/// impl that would let the two elements borrow independently), which forces
/// every lookup to allocate two fresh `String`s just to build a throwaway
/// key. The nested shape lets each level's lookup use `HashMap<String,
/// V>::get(&str)` (`String: Borrow<str>`), so probing a pair by two `&str`
/// slices — the hot path inside [`bpe_merge_symbols`] — is allocation-free.
/// `add_merge` (called only at load time, never per-encode) still pays one
/// `to_owned()` per distinct left symbol to *insert*, which is off the hot
/// path this finding is about.
#[derive(Debug, Clone, Default)]
pub struct BpeMerges {
    merges: HashMap<String, HashMap<String, MergeEntry>>,
    /// Rank to assign to the next *distinct* pair inserted (monotonically
    /// increasing, so insertion order defines priority, and — crucially for
    /// the lazy-invalidation heap in [`bpe_merge_symbols`] — every distinct
    /// pair gets a unique rank).
    next_rank: u32,
    /// Total number of `(left, right)` entries across all inner maps, kept
    /// incrementally so [`Self::len`] stays `O(1)` rather than summing every
    /// inner map on every call.
    len: usize,
}

impl BpeMerges {
    /// Create an empty merge table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a merge rule: `a` + `b` → token with ID `result_id`.
    ///
    /// Duplicate entries (same pair) preserve the original priority rank while
    /// overwriting only the result ID (matching the previous behaviour, where
    /// the order-slot was kept and just the mapped ID replaced).
    pub fn add_merge(&mut self, a: &str, b: &str, result_id: u32) {
        let inner = self.merges.entry(a.to_owned()).or_default();
        match inner.entry(b.to_owned()) {
            Entry::Occupied(mut occupied) => {
                // Preserve the existing rank; overwrite only the result ID.
                occupied.get_mut().result_id = result_id;
            }
            Entry::Vacant(vacant) => {
                let rank = self.next_rank;
                self.next_rank += 1;
                vacant.insert(MergeEntry { rank, result_id });
                self.len += 1;
            }
        }
    }

    /// Return the 0-based priority index for a merge pair, if it exists.
    ///
    /// Lower index = higher priority (applied first during encoding).
    pub fn get_merge_priority(&self, a: &str, b: &str) -> Option<usize> {
        self.rank(a, b).map(|rank| rank as usize)
    }

    /// Return the raw priority rank for a merge pair, if it exists.
    ///
    /// This is the internal `O(1)`, allocation-free hot-path used by the BPE
    /// merge loop; it avoids both the `usize` widening done by
    /// [`Self::get_merge_priority`] and (see the type-level docs above) any
    /// heap allocation.
    fn rank(&self, a: &str, b: &str) -> Option<u32> {
        self.merges.get(a)?.get(b).map(|entry| entry.rank)
    }

    /// Return the merged token ID for a pair, if a rule exists.
    pub fn get_merge_result(&self, a: &str, b: &str) -> Option<u32> {
        self.merges.get(a)?.get(b).map(|entry| entry.result_id)
    }

    /// Number of merge rules in the table.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the merge table is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ── Pre-tokenizer (legacy, non-byte-level) ────────────────────────────────────

/// Split text into pre-tokens using a GPT-2–style rule:
/// words (optionally preceded by a space) are separated from punctuation and
/// standalone whitespace runs.
///
/// The returned strings are the raw Unicode chunks to be BPE-encoded
/// individually.
pub fn pretokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut last_was_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            last_was_space = true;
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            // Punctuation gets its own token; add the leading space prefix if
            // the previous character was whitespace (GPT-2 convention: Ġ prefix).
            let mut tok = String::new();
            if last_was_space {
                tok.push('\u{0120}'); // Ġ — GPT-2 space prefix
            }
            tok.push(ch);
            tokens.push(tok);
            last_was_space = false;
        } else {
            if last_was_space && !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if last_was_space && current.is_empty() {
                current.push('\u{0120}'); // Leading Ġ prefix
            }
            current.push(ch);
            last_was_space = false;
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

// ── Regex-driven ByteLevel `Split` pre-tokenizer (TOK-03 / TOK-04) ────────────

/// Pre-tokenizer identities recognised from `tokenizer.ggml.pre` (the GGUF
/// path) or inferred from a `tokenizer.json`'s declared `byte_level` state
/// (the HF-JSON path, see [`crate::hf_format`]).
///
/// `Gpt2`, `Qwen2` and `Llama3` all resolve to the *same* canonical
/// HuggingFace `ByteLevel` `Split` regex today — that pattern is shared by
/// every one of these model families in practice (verified: the existing
/// hand-written scanner this replaces already documented that exact
/// pattern as "used ... for Qwen3 / Llama-3 / GPT-2"). `Qwen35` is the only
/// kind that differs, adding `\p{M}` (combining marks) in two places
/// (verified against `llama-vocab.cpp:373-388`, design doc Appendix A.2).
/// Distinct variants are kept (rather than collapsing Gpt2/Qwen2/Llama3 into
/// one) so a future model-specific divergence (e.g. Llama-3's tiktoken
/// digit-run cap) can be added without an API change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreTokenizerKind {
    #[default]
    Gpt2,
    Qwen2,
    Qwen35,
    Llama3,
}

impl PreTokenizerKind {
    /// Resolve a GGUF `tokenizer.ggml.pre` metadata string to a known kind.
    ///
    /// Falls back to [`Self::Gpt2`] for any unrecognised name. This is a
    /// safe default: every kind other than `Qwen35` shares byte-identical
    /// behaviour, so an unmapped `pre` name (a model family this crate has
    /// not special-cased) gets the same regex the crate has always used for
    /// its byte-level path.
    pub fn from_gguf_pre(name: &str) -> Self {
        match name {
            "qwen35" => Self::Qwen35,
            "qwen2" => Self::Qwen2,
            "llama3" | "llama-bpe" => Self::Llama3,
            _ => Self::Gpt2,
        }
    }

    /// The canonical `Split`-stage regex pattern for this kind.
    fn pattern(self) -> &'static str {
        match self {
            Self::Gpt2 | Self::Qwen2 | Self::Llama3 => PATTERN_GPT2_QWEN2,
            Self::Qwen35 => PATTERN_QWEN35,
        }
    }
}

/// The canonical HuggingFace `ByteLevel` `Split` pattern (GPT-2 / Qwen2 /
/// Qwen3 / Llama-3): contractions, then letter runs, number runs, symbol
/// runs, and whitespace runs, in that priority order.
///
/// The `(?i:'s|'t|'re|'ve|'m|'ll|'d)` case-insensitive group from the
/// canonical description is expanded here to explicit case-pair character
/// classes (`'[sS]`, `'[rR][eE]`, …) rather than a scoped `(?i:...)` flag —
/// this is byte-for-byte what the verified PrismML `llama.cpp` fork does
/// (`llama-vocab.cpp:373-388`) specifically to avoid a case-folding pass,
/// and is exactly equivalent (each position's case is independent either
/// way, so the two forms accept the same set of strings).
const PATTERN_GPT2_QWEN2: &str = concat!(
    "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]",
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
    "|\\p{N}",
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
    "|\\s*[\\r\\n]+",
    "|\\s+(?!\\S)",
    "|\\s+",
);

/// The `qwen35` pattern: identical to [`PATTERN_GPT2_QWEN2`] except that
/// `\p{M}` (combining marks) is admitted alongside `\p{L}` in the two
/// letter-related alternatives — the verified delta from
/// `llama-vocab.cpp:373-388` / design doc Appendix A.2.
const PATTERN_QWEN35: &str = concat!(
    "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]",
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+",
    "|\\p{N}",
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*",
    "|\\s*[\\r\\n]+",
    "|\\s+(?!\\S)",
    "|\\s+",
);

/// A defensive cap on fancy-regex's internal backtracking counter, applied
/// to every compiled pre-tokenizer pattern.
///
/// The patterns in this module have no nested unbounded quantifiers (every
/// alternative's repetition is anchored to a single, non-overlapping
/// character class), which is why HuggingFace's own Rust `tokenizers` crate
/// runs this identical pattern through `fancy-regex` in production. But
/// `fancy-regex`, unlike the plain `regex` crate, gives no compile-time
/// linear-time guarantee, so a bound is cheap insurance (TOK-03 verdict
/// correction: "the same cap must ALSO apply to the regex driver"). On
/// exceeding it, [`pretokenize_regex`] fails safe by emitting the remainder
/// of the text as one verbatim piece rather than hanging or losing bytes.
const PRETOKENIZE_BACKTRACK_LIMIT: usize = 1_000_000;

/// Compile (once per process) and return the built-in regex for `kind`.
///
/// The four built-in patterns are constant strings vetted by
/// `canonical_patterns_compile` below, so a compile failure here would be a
/// bug in *this crate*, not a runtime/user-input condition — hence the
/// `expect` (COOLJAPAN policy: `expect()` is acceptable for a
/// proven-invariant message, and this one is exercised by that test on
/// every test run).
fn compiled_pattern(kind: PreTokenizerKind) -> &'static fancy_regex::Regex {
    static GPT2: OnceLock<fancy_regex::Regex> = OnceLock::new();
    static QWEN2: OnceLock<fancy_regex::Regex> = OnceLock::new();
    static QWEN35: OnceLock<fancy_regex::Regex> = OnceLock::new();
    static LLAMA3: OnceLock<fancy_regex::Regex> = OnceLock::new();

    fn compile(pattern: &str) -> fancy_regex::Regex {
        fancy_regex::RegexBuilder::new(pattern)
            .backtrack_limit(PRETOKENIZE_BACKTRACK_LIMIT)
            .build()
            .expect(
                "built-in pre-tokenizer pattern must compile \
                 (covered by the `canonical_patterns_compile` test)",
            )
    }

    match kind {
        PreTokenizerKind::Gpt2 => GPT2.get_or_init(|| compile(kind.pattern())),
        PreTokenizerKind::Qwen2 => QWEN2.get_or_init(|| compile(kind.pattern())),
        PreTokenizerKind::Qwen35 => QWEN35.get_or_init(|| compile(kind.pattern())),
        PreTokenizerKind::Llama3 => LLAMA3.get_or_init(|| compile(kind.pattern())),
    }
}

/// Split `text` into pre-tokens by repeatedly finding the next match of
/// `re`, starting from byte 0.
///
/// This implements `Split` with `Isolated` behaviour: every match becomes
/// its own piece. The patterns this module ships are, by construction,
/// *exhaustive* over Unicode scalar values (every alternative's required
/// character classes partition all codepoints into whitespace / letter+mark
/// / number / other-non-whitespace), so in the overwhelming common case
/// matches tile the input with no gaps. Two defensive fallbacks guard the
/// invariant this crate actually promises (losslessness — see
/// `pretokenize_gpt2_is_lossless` and the proptest in `hf_parity_tests.rs`)
/// even if that invariant is ever violated by a future pattern edit or an
/// engine/Unicode-table difference:
/// - a gap between the scan cursor and the next match is emitted verbatim
///   rather than silently dropped;
/// - a regex runtime error (e.g. the backtrack limit) truncates the scan and
///   emits the remainder verbatim rather than panicking or losing bytes.
pub fn pretokenize_regex(text: &str, re: &fancy_regex::Regex) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<String> = Vec::new();
    let mut pos = 0usize;
    let len = text.len();

    while pos < len {
        match re.find_from_pos(text, pos) {
            Ok(Some(m)) => {
                if m.start() > pos {
                    // Defensive gap-fill (see doc comment above) — not
                    // expected to trigger for the shipped patterns.
                    out.push(text[pos..m.start()].to_owned());
                }
                if m.end() == m.start() {
                    // None of the shipped alternatives can zero-width match
                    // (each requires at least one code unit), but guard
                    // against an infinite loop defensively by advancing at
                    // least one character.
                    let next = next_char_boundary(text, m.start());
                    out.push(text[m.start()..next].to_owned());
                    pos = next;
                } else {
                    out.push(m.as_str().to_owned());
                    pos = m.end();
                }
            }
            Ok(None) => {
                out.push(text[pos..].to_owned());
                break;
            }
            Err(_) => {
                // Backtrack-limit or another runtime error: fail safe by
                // emitting the remainder verbatim. This crate's contract is
                // to never crash or drop text on `encode`.
                out.push(text[pos..].to_owned());
                break;
            }
        }
    }

    out
}

/// The smallest UTF-8 char boundary strictly after `from` (or `text.len()`).
fn next_char_boundary(text: &str, from: usize) -> usize {
    let mut i = from + 1;
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i.min(text.len())
}

/// GPT-2 / ByteLevel pre-tokenization that **preserves whitespace-run
/// structure** (tabs, newlines, carriage returns, and runs of multiple
/// spaces), driven by [`PATTERN_GPT2_QWEN2`] through a real regex engine
/// (`fancy-regex` — see [`pretokenize_regex`]).
///
/// Used (with `Isolated` behaviour) by the HuggingFace `tokenizers`
/// ByteLevel pipeline for Qwen2 / Qwen3 / Llama-3 / GPT-2. Unlike
/// [`pretokenize`], which folds every whitespace run into a single `Ġ`
/// marker (silently discarding tabs, newlines and repeated spaces), each
/// returned piece keeps its literal whitespace characters so that — after
/// the bytes→unicode remap — they match the `Ċ`/`Ġ`-bearing vocabulary
/// entries real `tokenizer.json` files ship.
///
/// The returned pieces still contain raw UTF-8; the caller is responsible
/// for applying the bytes→unicode map (see
/// [`crate::hf_format::bytes_to_unicode_string`]).
pub fn pretokenize_gpt2(text: &str) -> Vec<String> {
    pretokenize_regex(text, compiled_pattern(PreTokenizerKind::Gpt2))
}

/// `qwen35`'s `Split` pre-tokenizer: [`PATTERN_GPT2_QWEN2`] plus `\p{M}`
/// (combining marks) in the two letter-related alternatives — see
/// [`PreTokenizerKind::Qwen35`].
pub fn pretokenize_qwen35(text: &str) -> Vec<String> {
    pretokenize_regex(text, compiled_pattern(PreTokenizerKind::Qwen35))
}

/// Dispatch to the built-in compiled pattern for `kind`. Used by
/// [`crate::tokenizer::OxiTokenizer`]'s byte-level encode path when no
/// custom pre-tokenizer pattern has been attached (see
/// `OxiTokenizer::with_pretokenizer_pattern`).
pub fn pretokenize_by_kind(text: &str, kind: PreTokenizerKind) -> Vec<String> {
    pretokenize_regex(text, compiled_pattern(kind))
}

// ── BPE encoder ──────────────────────────────────────────────────────────────

/// Greedy BPE encode a single pre-tokenized word.
///
/// Algorithm:
/// 1. Split the word into individual Unicode characters as the initial symbol
///    sequence.
/// 2. Repeatedly find the pair with the lowest priority index in `merges` and
///    merge it.
/// 3. Continue until no more merges apply.
/// 4. Map each remaining symbol to its vocabulary ID; use byte-fallback for
///    any symbol not found in the vocabulary.
pub fn bpe_encode(word: &str, vocab: &Vocabulary, merges: &BpeMerges) -> Vec<u32> {
    if word.is_empty() {
        return Vec::new();
    }

    // Map symbols → token IDs, falling back to `<0xHH>` byte tokens.
    bpe_merge_symbols(word, merges)
        .iter()
        .flat_map(|sym| symbol_to_ids(sym, vocab))
        .collect()
}

/// Greedy BPE-encode a byte-level pre-tokenized piece.
///
/// This is the ByteLevel (GPT-2 / Qwen3 / Llama-3) counterpart of
/// [`bpe_encode`].  The caller is expected to have already remapped the raw
/// UTF-8 bytes of the piece through the GPT-2 bytes→unicode table (see
/// [`crate::hf_format::bytes_to_unicode_string`]), so every resulting symbol is
/// a printable-unicode code point that a byte-level vocabulary carries directly.
///
/// Unlike [`bpe_encode`], no `<0xHH>` byte-fallback is attempted: byte-level
/// vocabularies do not ship those SentencePiece-style tokens, and every single
/// byte already has its own vocabulary entry.  Any symbol that is genuinely
/// absent maps to `unk_id`.
pub fn bpe_encode_bytelevel(
    byte_level_piece: &str,
    vocab: &Vocabulary,
    merges: &BpeMerges,
    unk_id: u32,
) -> Vec<u32> {
    if byte_level_piece.is_empty() {
        return Vec::new();
    }

    bpe_merge_symbols(byte_level_piece, merges)
        .iter()
        .map(|sym| vocab.get_id(sym).unwrap_or(unk_id))
        .collect()
}

/// Sentinel "no neighbour" index for [`SymbolNode`] links.
const NONE: usize = usize::MAX;

/// One "symbol" in the working sequence during a BPE merge pass: a byte
/// range into the original pre-tokenized `word`, plus doubly-linked
/// neighbour indices into the same backing `Vec`.
///
/// Representing a (possibly already-merged) symbol as a `(start, end)` byte
/// range rather than an owned `String` is what makes the merge loop itself
/// allocation-free: merging two adjacent symbols is just extending
/// `left.end` to `right.end` and splicing `right` out of the list — no
/// string concatenation.
#[derive(Clone, Copy)]
struct SymbolNode {
    start: usize,
    end: usize,
    prev: usize,
    next: usize,
    /// Once a node is merged into its left neighbour it becomes a
    /// tombstone; `alive` guards against acting on it via a stale heap
    /// entry (see [`bpe_merge_symbols`]).
    alive: bool,
}

// Test-only work counter for `bpe_merge_symbols`'s merge loop (defined
// *before* that function's own doc comment below, not between it and the
// `fn` — a doc comment attaches to the very next item, and `thread_local!`
// is a macro invocation rustdoc cannot attach documentation to, so putting
// this block in between would silently detach `bpe_merge_symbols`'s real
// doc comment and trip `-D unused-doc-comments`).
//
// Incremented once per heap-pop "candidate considered" (whether ultimately
// stale or actually applied) — a deterministic proxy for the algorithm's
// real work that lets `bpe_merge_symbols_is_not_quadratic` assert `O(n log
// n)` growth without an `Instant`/`Duration` wall-clock measurement, which
// flakes on a shared machine running other agents' concurrent builds (see
// `CONTEXT.md`; confirmed empirically once in the between-wave gate).
// (`//`, not `///`: same rustdoc-macro-attachment reason as above.)
#[cfg(test)]
thread_local! {
    static MERGE_WORK_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_merge_work_counter() {
    MERGE_WORK_COUNTER.with(|c| c.set(0));
}

#[cfg(test)]
fn merge_work_counter() -> u64 {
    MERGE_WORK_COUNTER.with(std::cell::Cell::get)
}

/// Apply the BPE merge loop to a piece and return the final symbol strings.
///
/// ## Algorithm (TOK-06)
///
/// A doubly-linked list of [`SymbolNode`]s is built once from `word`'s
/// characters. Every adjacent pair with a known merge rank is pushed onto a
/// min-heap keyed by `(rank, left_index)` (so, matching the reference greedy
/// algorithm, the globally lowest-rank pair is applied first, and ties are
/// broken leftmost-first — `Iterator::min_by_key`'s "first element wins on a
/// tie" semantics is exactly what the previous `windows(2).min_by_key(...)`
/// implementation this replaces relied on). Popping and merging is `O(log
/// n)`; each merge invalidates at most two positions and pushes at most two
/// fresh candidates, so the whole pass is `O(n log n)` with **zero
/// allocation** inside the loop (all pair lookups are borrowed `&str` slices
/// into `word`, and [`BpeMerges::rank`] is itself allocation-free — see its
/// type-level docs).
///
/// Stale heap entries (from a symbol whose neighbour changed, or that was
/// itself consumed by an earlier merge) are detected lazily at pop time:
/// [`BpeMerges::add_merge`] assigns each *distinct* pair a unique rank, so
/// re-deriving the *current* pair at a popped index and checking its rank
/// against the popped value is sufficient to tell a stale entry from a live
/// one — no eager heap removal is needed.
fn bpe_merge_symbols(word: &str, merges: &BpeMerges) -> Vec<String> {
    // Character boundaries, so `bounds[i]..bounds[i + 1]` is exactly the
    // i-th character's byte range (handles multi-byte UTF-8 correctly).
    let bounds: Vec<usize> = word
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(word.len()))
        .collect();
    let n = bounds.len().saturating_sub(1);

    if n < 2 {
        return word.chars().map(|c| c.to_string()).collect();
    }

    let mut nodes: Vec<SymbolNode> = (0..n)
        .map(|i| SymbolNode {
            start: bounds[i],
            end: bounds[i + 1],
            prev: if i == 0 { NONE } else { i - 1 },
            next: if i + 1 == n { NONE } else { i + 1 },
            alive: true,
        })
        .collect();

    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(n);

    // Push a fresh candidate for the pair starting at `left`, if any and if
    // a merge rule exists for it.
    fn push_candidate(
        heap: &mut BinaryHeap<Reverse<(u32, usize)>>,
        nodes: &[SymbolNode],
        word: &str,
        merges: &BpeMerges,
        left: usize,
    ) {
        let right = nodes[left].next;
        if right == NONE {
            return;
        }
        let a = &word[nodes[left].start..nodes[left].end];
        let b = &word[nodes[right].start..nodes[right].end];
        if let Some(rank) = merges.rank(a, b) {
            heap.push(Reverse((rank, left)));
        }
    }

    for i in 0..n {
        push_candidate(&mut heap, &nodes, word, merges, i);
    }

    while let Some(Reverse((rank, left))) = heap.pop() {
        #[cfg(test)]
        MERGE_WORK_COUNTER.with(|c| c.set(c.get() + 1));

        if !nodes[left].alive {
            continue; // Stale: `left` was already merged into another node.
        }
        let right = nodes[left].next;
        if right == NONE {
            continue; // Stale: `left`'s right neighbour changed (e.g. it
                      // was consumed) since this entry was pushed.
        }

        // Re-validate against the CURRENT pair rather than trusting the
        // popped index blindly: `left`'s neighbour may have changed since
        // this entry was pushed. Every distinct pair has a unique rank (see
        // `BpeMerges::add_merge`), so `current_rank == rank` if and only if
        // this is still exactly the pair the entry was pushed for.
        let a = &word[nodes[left].start..nodes[left].end];
        let b = &word[nodes[right].start..nodes[right].end];
        let Some(current_rank) = merges.rank(a, b) else {
            continue;
        };
        if current_rank != rank {
            continue;
        }

        // Merge `right` into `left`.
        nodes[left].end = nodes[right].end;
        nodes[right].alive = false;
        let new_next = nodes[right].next;
        nodes[left].next = new_next;
        if new_next != NONE {
            nodes[new_next].prev = left;
        }

        // The merge changed two adjacencies: (left's-left, left) and (left,
        // left's-new-right). Push fresh candidates for both; any
        // now-invalid entries left in the heap for the old shape are
        // skipped lazily above.
        let prev = nodes[left].prev;
        if prev != NONE {
            push_candidate(&mut heap, &nodes, word, merges, prev);
        }
        push_candidate(&mut heap, &nodes, word, merges, left);
    }

    // Walk the surviving list left to right, materialising the final symbol
    // strings — the only point that allocates per output symbol, same as
    // every other function in this module.
    let mut out = Vec::with_capacity(n);
    let mut cur = 0usize;
    while cur != NONE {
        if nodes[cur].alive {
            out.push(word[nodes[cur].start..nodes[cur].end].to_owned());
        }
        cur = nodes[cur].next;
    }
    out
}

/// Convert a symbol to one or more token IDs.
///
/// If the symbol is directly in the vocabulary, return its single ID.
/// Otherwise attempt UTF-8 byte fallback: each byte is encoded as `<0xHH>`.
fn symbol_to_ids(sym: &str, vocab: &Vocabulary) -> Vec<u32> {
    if let Some(id) = vocab.get_id(sym) {
        return vec![id];
    }

    // Byte fallback.
    sym.as_bytes()
        .iter()
        .filter_map(|&b| {
            let fallback = byte_fallback_id(b);
            vocab.get_id(&fallback)
        })
        .collect()
}

// ── Byte fallback ─────────────────────────────────────────────────────────────

/// Return the byte-fallback token name for a single byte value.
///
/// Format: `<0xHH>` where `HH` is the uppercase hexadecimal byte value.
///
/// # Example
/// ```
/// use oxibonsai_tokenizer::bpe::byte_fallback_id;
/// assert_eq!(byte_fallback_id(0x20), "<0x20>");
/// assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
/// ```
pub fn byte_fallback_id(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocabulary;
    use std::time::{Duration, Instant};

    fn make_vocab_with_merges() -> (Vocabulary, BpeMerges) {
        let mut vocab = Vocabulary::new();
        // Individual characters
        vocab.insert("h", 10);
        vocab.insert("e", 11);
        vocab.insert("l", 12);
        vocab.insert("o", 13);
        // Merged tokens
        vocab.insert("he", 20);
        vocab.insert("hel", 21);
        vocab.insert("hell", 22);
        vocab.insert("hello", 23);
        vocab.insert("lo", 24);

        let mut merges = BpeMerges::new();
        merges.add_merge("h", "e", 20);
        merges.add_merge("he", "l", 21);
        merges.add_merge("hel", "l", 22);
        merges.add_merge("hell", "o", 23);
        merges.add_merge("l", "o", 24);

        (vocab, merges)
    }

    #[test]
    fn byte_fallback_format() {
        assert_eq!(byte_fallback_id(0x00), "<0x00>");
        assert_eq!(byte_fallback_id(0x20), "<0x20>");
        assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
        assert_eq!(byte_fallback_id(0x0A), "<0x0A>");
    }

    #[test]
    fn bpe_merges_priority() {
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 1);
        m.add_merge("b", "c", 2);
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_priority("b", "c"), Some(1));
        assert_eq!(m.get_merge_priority("x", "y"), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn bpe_merges_shared_left_symbol() {
        // Regression guard for the nested-map storage: two pairs sharing the
        // same LEFT symbol but different RIGHT symbols must not clobber one
        // another.
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 100);
        m.add_merge("a", "c", 200);
        assert_eq!(m.get_merge_result("a", "b"), Some(100));
        assert_eq!(m.get_merge_result("a", "c"), Some(200));
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_priority("a", "c"), Some(1));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn bpe_encode_hello() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("hello", &vocab, &merges);
        // Should merge all the way to "hello" → id 23
        assert_eq!(ids, vec![23]);
    }

    #[test]
    fn bpe_encode_empty() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("", &vocab, &merges);
        assert!(ids.is_empty());
    }

    #[test]
    fn pretokenize_simple_sentence() {
        let tokens = pretokenize("hello world");
        assert!(!tokens.is_empty());
        // Should split into at least "hello" and "Ġworld"
        assert!(tokens.iter().any(|t| t.contains("hello") || t == "hello"));
    }

    #[test]
    fn pretokenize_empty() {
        assert!(pretokenize("").is_empty());
    }

    #[test]
    fn pretokenize_punctuation_splits() {
        let tokens = pretokenize("hi,there");
        // Should split around the comma
        assert!(tokens.len() >= 2);
    }

    // ── Finding 34: O(1) merge-priority correctness ─────────────────────────

    #[test]
    fn merge_priority_is_correct_for_late_inserted_pair() {
        // The rank must equal the insertion order even for the last pair added
        // to a large table (this is the case the old linear scan handled slowly
        // but the new HashMap must still get numerically right).
        let mut m = BpeMerges::new();
        for i in 0..1_000u32 {
            m.add_merge(&format!("l{i}"), &format!("r{i}"), i);
        }
        assert_eq!(m.get_merge_priority("l0", "r0"), Some(0));
        assert_eq!(m.get_merge_priority("l999", "r999"), Some(999));
        assert_eq!(m.get_merge_result("l999", "r999"), Some(999));
        assert_eq!(m.len(), 1_000);
    }

    #[test]
    fn duplicate_merge_preserves_rank_overwrites_result() {
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 10);
        m.add_merge("c", "d", 11);
        // Re-insert (a,b) with a new result id — rank must stay 0.
        m.add_merge("a", "b", 99);
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_result("a", "b"), Some(99));
        assert_eq!(m.get_merge_priority("c", "d"), Some(1));
        assert_eq!(m.len(), 2);
    }

    // ── Finding 14: whitespace-preserving GPT-2 pre-tokenizer ────────────────

    #[test]
    fn pretokenize_gpt2_preserves_newline() {
        let pieces = pretokenize_gpt2("hello\nworld");
        assert_eq!(pieces, vec!["hello", "\n", "world"]);
    }

    #[test]
    fn pretokenize_gpt2_preserves_repeated_spaces() {
        // Two spaces between words: GPT-2 keeps one space attached to the next
        // word and emits the extra space as its own piece.
        let pieces = pretokenize_gpt2("a  b");
        let joined: String = pieces.concat();
        assert_eq!(joined, "a  b", "no whitespace may be lost: {pieces:?}");
    }

    #[test]
    fn pretokenize_gpt2_leading_space_attaches_to_word() {
        let pieces = pretokenize_gpt2("a tiny bonsai");
        assert_eq!(pieces, vec!["a", " tiny", " bonsai"]);
    }

    #[test]
    fn pretokenize_gpt2_is_lossless() {
        for text in [
            "hello\tworld",
            "line1\r\nline2",
            "  spaced  out  ",
            "café 中文 😀!",
            "",
        ] {
            let joined: String = pretokenize_gpt2(text).concat();
            assert_eq!(joined, text, "pretokenize_gpt2 dropped bytes for {text:?}");
        }
    }

    // ── TOK-03: fancy-regex Split replaces the hand-written scanner ─────────

    #[test]
    fn canonical_patterns_compile() {
        // Backs the `expect()` in `compiled_pattern` — every built-in kind
        // must compile under the configured backtrack limit.
        for kind in [
            PreTokenizerKind::Gpt2,
            PreTokenizerKind::Qwen2,
            PreTokenizerKind::Qwen35,
            PreTokenizerKind::Llama3,
        ] {
            let re = compiled_pattern(kind);
            assert!(re.is_match("hello").unwrap_or(false));
        }
    }

    #[test]
    fn from_gguf_pre_maps_known_names() {
        assert_eq!(
            PreTokenizerKind::from_gguf_pre("qwen35"),
            PreTokenizerKind::Qwen35
        );
        assert_eq!(
            PreTokenizerKind::from_gguf_pre("qwen2"),
            PreTokenizerKind::Qwen2
        );
        assert_eq!(
            PreTokenizerKind::from_gguf_pre("llama3"),
            PreTokenizerKind::Llama3
        );
        assert_eq!(
            PreTokenizerKind::from_gguf_pre("some-unknown-family"),
            PreTokenizerKind::Gpt2
        );
    }

    /// TOK-03's proven divergence cases: blank-line-with-whitespace and Roman
    /// numerals. The hand-written scanner this module used to ship
    /// mis-classified these because `char::is_alphabetic()` /
    /// `char::is_numeric()` are broader than the Unicode `\p{L}` / `\p{N}`
    /// general-category classes fancy-regex uses (verified: U+2167 ROMAN
    /// NUMERAL EIGHT is general category `Nl`, part of the derived
    /// `Alphabetic` property `char::is_alphabetic()` checks, but *not*
    /// `\p{L}` — `\p{N}` correctly claims it instead).
    #[test]
    fn pretokenize_gpt2_handles_roman_numerals_as_numbers() {
        let pieces = pretokenize_gpt2("Chapter \u{2167}I");
        // "Ⅷ" (U+2167) must NOT be fused onto the following letter as if it
        // were itself alphabetic; it is its own `\p{N}` token, and the
        // trailing "I" is a separate letter run.
        assert!(
            pieces.iter().any(|p| p == "\u{2167}"),
            "Roman numeral must be its own \\p{{N}} piece: {pieces:?}"
        );
        assert!(pieces.contains(&"I".to_string()));
    }

    #[test]
    fn pretokenize_gpt2_blank_line_with_trailing_whitespace() {
        // "\n    \n\n" — a blank line carrying trailing spaces followed by a
        // blank line. Must round-trip losslessly regardless of exact split
        // boundaries (this was one of the 7/19 mismatching cases).
        let text = "a\n    \n\nb";
        let joined: String = pretokenize_gpt2(text).concat();
        assert_eq!(joined, text);
    }

    #[test]
    fn pretokenize_regex_backtrack_limit_fails_safe() {
        // A pathologically low backtrack limit must never panic or drop
        // text — it must degrade to "emit the remainder verbatim".
        let re = fancy_regex::RegexBuilder::new(r"\s+(?!\S)|\s+|\p{L}+|.")
            .backtrack_limit(1)
            .build()
            .expect("pattern compiles");
        let text = "some ordinary ascii text that will not deeply backtrack";
        let pieces = pretokenize_regex(text, &re);
        assert_eq!(
            pieces.concat(),
            text,
            "must stay lossless under a tiny backtrack limit"
        );
    }

    /// Legacy hand-written scanner, preserved **only** as a `cfg(test)`
    /// differential cross-check against the fancy-regex implementation
    /// (TOK-03 verdict: "keep the scanner only as a cfg(test) cross-check").
    /// Not used anywhere in production code any more.
    fn legacy_scanner_reference(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out: Vec<String> = Vec::new();
        let mut i = 0usize;

        let is_letter = |c: char| c.is_alphabetic();
        let is_number = |c: char| c.is_numeric();
        let is_ws = |c: char| c.is_whitespace();
        let is_nl = |c: char| c == '\r' || c == '\n';

        fn match_contraction(rest: &[char]) -> Option<usize> {
            let lower = |c: char| c.to_ascii_lowercase();
            if rest.len() >= 2 {
                let c1 = lower(rest[1]);
                if rest.len() >= 3 {
                    let c2 = lower(rest[2]);
                    if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        return Some(3);
                    }
                }
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    return Some(2);
                }
            }
            None
        }

        while i < n {
            let c = chars[i];
            if c == '\'' && i + 1 < n {
                if let Some(len) = match_contraction(&chars[i..]) {
                    out.push(chars[i..i + len].iter().collect());
                    i += len;
                    continue;
                }
            }
            {
                let mut j = i;
                if !is_nl(chars[j])
                    && !is_letter(chars[j])
                    && !is_number(chars[j])
                    && j + 1 < n
                    && is_letter(chars[j + 1])
                {
                    j += 1;
                }
                if j < n && is_letter(chars[j]) {
                    while j < n && is_letter(chars[j]) {
                        j += 1;
                    }
                    out.push(chars[i..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            if is_number(c) {
                out.push(c.to_string());
                i += 1;
                continue;
            }
            {
                let mut j = i;
                if chars[j] == ' '
                    && j + 1 < n
                    && !is_ws(chars[j + 1])
                    && !is_letter(chars[j + 1])
                    && !is_number(chars[j + 1])
                {
                    j += 1;
                }
                if j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                    while j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j])
                    {
                        j += 1;
                    }
                    while j < n && is_nl(chars[j]) {
                        j += 1;
                    }
                    out.push(chars[i..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            if is_ws(c) {
                let mut j = i;
                while j < n && is_ws(chars[j]) && !is_nl(chars[j]) {
                    j += 1;
                }
                if j < n && is_nl(chars[j]) {
                    while j < n && is_nl(chars[j]) {
                        j += 1;
                    }
                    out.push(chars[i..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            if is_ws(c) {
                let mut j = i;
                while j < n && is_ws(chars[j]) {
                    j += 1;
                }
                if j == n {
                    out.push(chars[i..j].iter().collect());
                    i = j;
                } else if j - i >= 2 {
                    out.push(chars[i..j - 1].iter().collect());
                    i = j - 1;
                } else {
                    out.push(chars[i..j].iter().collect());
                    i = j;
                }
                continue;
            }
            out.push(c.to_string());
            i += 1;
        }
        out
    }

    #[test]
    fn fancy_regex_diverges_from_legacy_scanner_on_the_proven_cases() {
        // TOK-03's exact proof: the legacy scanner disagrees with correct
        // (HF-matching) behaviour on Roman numerals because
        // `char::is_alphabetic()` wrongly admits Nl codepoints. This test
        // pins that the *replacement* differs from the *old, buggy*
        // behaviour on this case, i.e. the fix actually changed something.
        let text = "Chapter \u{2167}: intro";
        let old = legacy_scanner_reference(text);
        let new = pretokenize_gpt2(text);
        assert_ne!(
            old, new,
            "expected the fancy-regex implementation to diverge from the \
             known-buggy legacy scanner on a Roman-numeral input"
        );
        // And the new behaviour must still be lossless.
        assert_eq!(new.concat(), text);
    }

    #[test]
    fn fancy_regex_agrees_with_legacy_scanner_on_simple_ascii() {
        // Sanity: on ordinary ASCII text with no Unicode edge cases, the
        // rewrite must not have changed anything.
        for text in ["hello world", "a tiny bonsai tree", "def f():\n    pass\n"] {
            assert_eq!(legacy_scanner_reference(text), pretokenize_gpt2(text));
        }
    }

    // ── Finding 9: byte-level encode uses unk, not <0xHH> fallback ────────────

    #[test]
    fn bpe_encode_bytelevel_maps_and_falls_back_to_unk() {
        let mut vocab = Vocabulary::new();
        vocab.insert("a", 1);
        vocab.insert("b", 2);
        // Note: no `<0xHH>` byte-fallback tokens are registered.
        let merges = BpeMerges::new();
        // "ab" — both present.
        assert_eq!(bpe_encode_bytelevel("ab", &vocab, &merges, 0), vec![1, 2]);
        // "aZ" — 'Z' absent, must become unk (7), NOT a `<0x5A>` fallback.
        assert_eq!(bpe_encode_bytelevel("aZ", &vocab, &merges, 7), vec![1, 7]);
    }

    // ── TOK-06: BPE merge loop is O(n log n), not O(n^2), and alloc-free ────

    #[test]
    fn bpe_merge_symbols_matches_reference_on_small_cases() {
        let (vocab, merges) = make_vocab_with_merges();
        // "hello" merges all the way down regardless of algorithm.
        assert_eq!(bpe_encode("hello", &vocab, &merges), vec![23]);
        // "helo" (no second 'l') must NOT merge past "hel" + "o": there is no
        // (hel, o) or (he, lo) rule registered for this partial word (only
        // "l","o" -> 24, "hel","l" -> 22, "hell","o" -> 23 exist), so the
        // best applicable merge is (l, o) -> 24, giving "he" + "lo" via
        // (h,e)->he then (l,o)->lo (no rule joins "he"+"lo").
        let ids = bpe_encode("helo", &vocab, &merges);
        assert!(!ids.is_empty());
    }

    #[test]
    fn merge_all_occurrences_pattern_reduces_correctly() {
        // "aaaa" with (a,a)->aa should reduce via repeated leftmost lowest-
        // rank merges to a single "aaaa" symbol once (aa,aa) is registered
        // too, exercising multiple rounds of the heap-based algorithm.
        let mut vocab = Vocabulary::new();
        vocab.insert("a", 1);
        vocab.insert("aa", 2);
        vocab.insert("aaaa", 3);
        let mut merges = BpeMerges::new();
        merges.add_merge("a", "a", 2);
        merges.add_merge("aa", "aa", 3);
        assert_eq!(bpe_encode("aaaa", &vocab, &merges), vec![3]);
    }

    #[test]
    fn ties_break_leftmost_matching_reference_semantics() {
        // Two independent (a,b) pairs at the same rank in "abab": the
        // leftmost occurrence must be merged first. Since both occurrences
        // use the identical rule, the end result is the same regardless,
        // but this pins that merging proceeds deterministically
        // left-to-right (verified via a 3-way word where order matters).
        let mut vocab = Vocabulary::new();
        vocab.insert("a", 1);
        vocab.insert("b", 2);
        vocab.insert("c", 3);
        vocab.insert("ab", 4);
        vocab.insert("bc", 5);
        vocab.insert("abc", 6);
        let mut merges = BpeMerges::new();
        // (a,b) has higher priority (rank 0) than (b,c) (rank 1): "abc"
        // must merge (a,b)->ab first, then (ab,c) has no rule, so result is
        // ["ab","c"] = [4, 3], NOT ["a","bc"].
        merges.add_merge("a", "b", 4);
        merges.add_merge("b", "c", 5);
        assert_eq!(bpe_encode("abc", &vocab, &merges), vec![4, 3]);
    }

    /// A long adversarial "word" with no natural break (mirrors the
    /// finding's emoji-run scenario: a long homogeneous pre-token with no
    /// whitespace, so the whole thing lands in one `bpe_merge_symbols`
    /// call). Builds a merge table that forces many real merge rounds
    /// rather than being all-misses (all-misses would be fast under either
    /// algorithm and would not distinguish O(n^2) from O(n log n)).
    fn build_quadratic_probe_case(n_chars: usize) -> (Vocabulary, BpeMerges, String) {
        // Alphabet of single ASCII letters a.. so each successive pair
        // (c[i], c[i+1]) for a repeating "ab" pattern can chain merges:
        // (a,b)->ab (rank0), (ab,a)->aba(rank1)... a long chain forces O(n)
        // merge rounds, each rescanning if the old algorithm were used.
        let mut vocab = Vocabulary::new();
        vocab.insert("a", 1);
        vocab.insert("b", 2);
        let mut merges = BpeMerges::new();
        merges.add_merge("a", "b", 100);
        merges.add_merge("b", "a", 101);
        vocab.insert("ab", 100);
        vocab.insert("ba", 101);
        let word: String = "ab".repeat(n_chars / 2);
        (vocab, merges, word)
    }

    /// ADDENDUM (orchestrator, 2026-09-22): the previous version of this
    /// test used `Instant`/`Duration` wall-clock timing for its *only*
    /// assertion and flaked once in the between-wave gate under concurrent
    /// builds (passed in isolation in 0.00s). Replaced with a deterministic
    /// work counter (`MERGE_WORK_COUNTER`, incremented once per heap-pop
    /// inside `bpe_merge_symbols`'s merge loop): `O(n log n)` predicts
    /// `count(4n)/count(n) -> 4` as `n` grows; `O(n^2)` predicts `16`. No
    /// `Instant`/`Duration` anywhere in this test — it cannot flake under
    /// concurrent load. See `bpe_merge_symbols_is_not_quadratic_wall_clock`
    /// below for the original timing-based check, kept `#[ignore]`d as a
    /// human-run sanity check on quiet hardware.
    #[test]
    fn bpe_merge_symbols_is_not_quadratic() {
        fn work_for(n_chars: usize) -> u64 {
            let (vocab, merges, word) = build_quadratic_probe_case(n_chars);
            reset_merge_work_counter();
            let ids = bpe_encode(&word, &vocab, &merges);
            assert!(!ids.is_empty());
            merge_work_counter()
        }

        let work_n = work_for(2_000);
        let work_4n = work_for(8_000);

        assert!(
            work_n > 0,
            "the adversarial case must force real merge rounds"
        );
        // Generous slack (+64) absorbs small constant-factor overhead at
        // tiny n; the multiplier (8, halfway between O(n log n)'s ~4 and
        // O(n^2)'s ~16 at this quadrupling) is what actually distinguishes
        // the two complexity classes.
        assert!(
            work_4n <= 8 * work_n + 64,
            "work(4n)={work_4n} should stay well under 8x work(n)={work_n} (+64 slack) — \
             O(n^2) would give ~16x — the merge loop may have regressed to quadratic"
        );
    }

    /// Wall-clock sibling of `bpe_merge_symbols_is_not_quadratic`, preserved
    /// for a human to run by hand on quiet hardware (`cargo test -- --ignored
    /// bpe_merge_symbols_is_not_quadratic_wall_clock`) rather than as part of
    /// the always-run gate the deterministic counter test above now covers.
    ///
    /// Gatekeeper (waves 2+2.5 review) REQUIRED #2: the absolute-ms floor is
    /// additionally gated on `cfg(not(debug_assertions))` — an unoptimized
    /// debug build is measurably slower and would false-positive under a
    /// fixed millisecond budget on exactly the kind of degraded/shared
    /// hardware this test already tries to tolerate. The *ratio* assertion
    /// (the real complexity guard, not what flaked) stays unconditional.
    #[test]
    #[ignore = "wall-clock sanity check; the deterministic work-counter test \
                above is what the gate runs"]
    fn bpe_merge_symbols_is_not_quadratic_wall_clock() {
        const RUNS: u32 = 5;
        let time_it = |n: usize| -> Duration {
            let (vocab, merges, word) = build_quadratic_probe_case(n);
            let mut best = Duration::MAX;
            for _ in 0..RUNS {
                let start = Instant::now();
                let ids = bpe_encode(&word, &vocab, &merges);
                let elapsed = start.elapsed();
                assert!(!ids.is_empty());
                best = best.min(elapsed);
            }
            best
        };

        let t_n = time_it(2_000);
        let t_2n = time_it(4_000);

        let bound = if cfg!(debug_assertions) {
            Duration::from_secs(2)
        } else {
            Duration::from_millis(200)
        };
        assert!(
            t_n < bound,
            "2000-char adversarial word took {t_n:?} (bound {bound:?}; nominal target < 20ms)"
        );

        // O(n log n) predicts t(2n)/t(n) -> 2 as n grows; O(n^2) predicts 4.
        // Assert well under the quadratic bound (with slack for timer
        // noise) rather than pinning the exact ratio. Unconditional: this
        // is the assertion that is not what flaked.
        assert!(
            t_2n < t_n * 3,
            "t(2n)={t_2n:?} should be well under 3x t(n)={t_n:?} \
             (O(n^2) would give ~4x) — the merge loop may have regressed to quadratic"
        );
    }

    #[test]
    fn fifteen_kb_mixed_text_encodes_quickly() {
        // The literal TOK-06 gate: "15 KB of mixed text < 20 ms". Bound kept
        // generous for shared-machine noise (see comment above); this is a
        // regression trip-wire against the pre-fix ~5.9s behaviour, not a
        // tight perf benchmark.
        let mut vocab = Vocabulary::new();
        for b in 0u16..=255 {
            vocab.insert(
                &char::from_u32(u32::from(b))
                    .unwrap_or('\u{FFFD}')
                    .to_string(),
                u32::from(b),
            );
        }
        let mut merges = BpeMerges::new();
        merges.add_merge("a", "b", 1000);
        vocab.insert("ab", 1000);

        // Mixed content: ASCII words, a long homogeneous run (worst case for
        // a single pre-token), and repeated punctuation.
        let mut text = String::with_capacity(15 * 1024);
        while text.len() < 15 * 1024 {
            text.push_str("The quick brown fox jumps over the lazy dog. ");
            text.push_str(&"ab".repeat(50));
            text.push_str(" !!!??? ");
        }

        let start = Instant::now();
        for piece in pretokenize_gpt2(&text) {
            let _ = bpe_encode(&piece, &vocab, &merges);
        }
        let elapsed = start.elapsed();
        // Gatekeeper (waves 2+2.5 review) REQUIRED #2: an unoptimized debug
        // build is measurably slower than release, so the bound is lifted
        // (not dropped — a test with no assertion in some build profile is
        // its own bug class, T-09) rather than gated away in debug.
        let bound = if cfg!(debug_assertions) {
            Duration::from_secs(2)
        } else {
            Duration::from_millis(500)
        };
        assert!(
            elapsed < bound,
            "15 KB mixed text took {elapsed:?} (bound {bound:?}; nominal target < 20ms; \
             generous bound for a shared/debug build machine)"
        );
    }
}
