//! On-disk model cache for fast model reloading.
//!
//! Caches quantized model weights + metadata in a binary format (`.oxcache`)
//! for faster cold-start loading vs. re-parsing GGUF files.
//!
//! Format:
//!   Header: `"OXCA"` (4 bytes) + version u32 + num\_entries u64 + metadata\_len u32
//!   Metadata: JSON string (hand-serialised, no serde)
//!   Per entry: name\_len u32 + name (UTF-8) + quant\_type\_len u32 + quant\_type + data\_len u64 + data bytes
//!
//! ## Reachability
//!
//! As of this writing, nothing in `gguf_loader.rs` or the CLI constructs or
//! loads a [`DiskCache`] — no shipping code path writes a `.oxcache` file
//! or reads one back during model loading. This module is a tested,
//! documented primitive intended to back a future GGUF-loader fast-path,
//! not a wired feature.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::SystemTime;

/// Magic bytes identifying an OxiBonsai disk cache file.
pub const CACHE_MAGIC: &[u8; 4] = b"OXCA";
/// Current cache format version.
pub const CACHE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by disk-cache operations.
#[derive(Debug, thiserror::Error)]
pub enum DiskCacheError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// File does not start with `OXCA`.
    #[error("invalid cache magic")]
    InvalidMagic,
    /// Cache file was written by a newer/older incompatible version.
    #[error("unsupported cache version: {0}")]
    UnsupportedVersion(u32),
    /// Hand-rolled JSON metadata could not be parsed.
    #[error("metadata parse error: {0}")]
    MetadataParse(String),
    /// Cache is older than its source file.
    #[error("cache is stale")]
    StaleCache,
    /// The header declares more entries (or a metadata blob) than the
    /// source file could possibly contain, given its actual on-disk size
    /// and the minimum number of bytes every entry must occupy. Returned
    /// by [`DiskCache::load`] *before* any entry is read, so a corrupted or
    /// adversarial file is rejected immediately with a clear diagnosis
    /// instead of requiring a full, doomed scan to discover the
    /// truncation.
    #[error(
        "cache header declares {declared_entries} entries plus a {declared_meta_len}-byte \
         metadata blob, but the file is only {file_len} bytes (minimum \
         {min_bytes_per_entry} bytes/entry) — the file is truncated or corrupted"
    )]
    HeaderExceedsFileSize {
        /// Number of entries the header claims.
        declared_entries: u64,
        /// Length of the metadata blob the header claims, in bytes.
        declared_meta_len: u64,
        /// Actual size of the source file, in bytes.
        file_len: u64,
        /// Minimum number of bytes every entry must occupy on disk
        /// (the three length-prefix fields alone, ignoring payloads).
        min_bytes_per_entry: u64,
    },
}

// ---------------------------------------------------------------------------
// Bounded reads (CQ-09): never let a file-supplied length drive an eager
// allocation large enough to abort the process.
// ---------------------------------------------------------------------------

/// Upper bound on how much capacity a single length-prefixed field
/// ([`DiskCache::read_from`]'s metadata blob, an entry's name/quant-type
/// string, or an entry's raw data payload) may eagerly pre-allocate,
/// regardless of what the untrusted stream claims. A crafted or corrupted
/// cache file can declare an arbitrary `u32`/`u64` length; without this
/// bound the eager `vec![0u8; len]` this module used to perform could abort
/// the process (SIGABRT from the global allocator) before a single byte of
/// the field had actually been observed in the stream, which is not a
/// catchable `Result` and takes the whole process down.
///
/// Chosen generously (64 MiB) — large enough that even a sizeable single
/// quantized tensor blob is read efficiently in one shot — while still
/// refusing to blindly reserve gigabytes up front for a hostile length.
/// Legitimate fields larger than this bound are still read correctly (see
/// [`read_bytes_bounded`]); they simply grow the buffer incrementally as
/// real bytes are confirmed present instead of pre-reserving everything.
const MAX_EAGER_ALLOC: usize = 64 * 1024 * 1024;

/// Upper bound on how many [`CacheEntry`] slots [`DiskCache::read_from`]
/// will eagerly reserve capacity for, regardless of the file-supplied
/// `num_entries` count. Prevents `Vec::with_capacity(num_entries)` from
/// aborting when `num_entries` is a hostile value (e.g. close to `u64::MAX`
/// truncated to `usize`). A real cache legitimately holding more entries
/// than this still loads correctly — the vector simply grows via ordinary
/// amortized `push` reallocation as entries are actually read.
const MAX_EAGER_ENTRIES: usize = 1_000_000;

/// Minimum number of bytes every entry occupies on disk: the three
/// length-prefix fields (`name_len: u32`, `quant_type_len: u32`,
/// `data_len: u64`) with all payloads empty. Used to sanity-check a
/// declared `num_entries` against the real file size in
/// [`DiskCache::load`].
const MIN_BYTES_PER_ENTRY: u64 = 4 + 4 + 8;

/// Fixed-size portion of the file header: magic (4) + version (4) +
/// num_entries (8) + meta_len (4).
const HEADER_BYTES: u64 = 4 + 4 + 8 + 4;

/// Read exactly `len` bytes from `reader` into a freshly allocated
/// `Vec<u8>`, without ever pre-reserving more than [`MAX_EAGER_ALLOC`]
/// bytes of capacity ahead of what has actually been confirmed present in
/// the stream.
///
/// Bytes are consumed in bounded chunks, so a `len` far larger than the
/// remaining input (e.g. a corrupted or adversarial length field) fails via
/// the ordinary [`DiskCacheError::Io`] path — `read_exact` hitting EOF —
/// rather than triggering an immediate, attacker-controlled allocation of
/// `len` bytes.
fn read_bytes_bounded<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, DiskCacheError> {
    const CHUNK_LEN: usize = 64 * 1024;
    let mut out = Vec::with_capacity(len.min(MAX_EAGER_ALLOC));
    let mut remaining = len;
    let mut chunk = [0u8; CHUNK_LEN];
    while remaining > 0 {
        let take = remaining.min(CHUNK_LEN);
        reader.read_exact(&mut chunk[..take])?;
        out.extend_from_slice(&chunk[..take]);
        remaining -= take;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// An entry in the disk cache, representing one named tensor blob.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Tensor / weight name (e.g. `"layers.0.attn.q_proj"`).
    pub name: String,
    /// Raw bytes of the (possibly quantized) tensor data.
    pub data: Vec<u8>,
    /// Quantization format identifier (e.g. `"f32"`, `"int8"`, `"q1_0_g128"`).
    pub quant_type: String,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(name: impl Into<String>, data: Vec<u8>, quant_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data,
            quant_type: quant_type.into(),
        }
    }

    /// Total size of the raw data in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// DiskCache
// ---------------------------------------------------------------------------

/// In-memory representation of a `.oxcache` file.
#[derive(Debug)]
pub struct DiskCache {
    entries: Vec<CacheEntry>,
    metadata: HashMap<String, String>,
}

impl Default for DiskCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DiskCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Append an entry.
    pub fn add_entry(&mut self, entry: CacheEntry) {
        self.entries.push(entry);
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Look up a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Find an entry by name.
    pub fn get_entry(&self, name: &str) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Number of entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Sum of all entry data sizes.
    pub fn total_data_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    // ----- persistence -----

    /// Save to a file path.
    pub fn save(&self, path: &Path) -> Result<(), DiskCacheError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Load from a file path.
    ///
    /// Unlike the generic [`DiskCache::read_from`], this variant knows the
    /// source's real size on disk and uses it to reject an impossible
    /// header (declared entry count / metadata length that the file could
    /// not physically contain) immediately, before attempting to read a
    /// single entry — see [`DiskCacheError::HeaderExceedsFileSize`].
    pub fn load(path: &Path) -> Result<Self, DiskCacheError> {
        let file_len = std::fs::metadata(path)?.len();
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from_bounded(&mut reader, Some(file_len))
    }

    /// Serialize to an arbitrary writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), DiskCacheError> {
        // Magic
        writer.write_all(CACHE_MAGIC)?;

        // Version (u32 LE)
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;

        // Number of entries (u64 LE)
        writer.write_all(&(self.entries.len() as u64).to_le_bytes())?;

        // Metadata as JSON string
        let meta_json = metadata_to_json(&self.metadata);
        let meta_bytes = meta_json.as_bytes();
        writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(meta_bytes)?;

        // Entries
        for entry in &self.entries {
            // name
            let name_bytes = entry.name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // quant_type
            let qt_bytes = entry.quant_type.as_bytes();
            writer.write_all(&(qt_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(qt_bytes)?;

            // data
            writer.write_all(&(entry.data.len() as u64).to_le_bytes())?;
            writer.write_all(&entry.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Deserialize from an arbitrary reader.
    ///
    /// Every file-supplied length is bounded before use (internally, via a
    /// chunked bounded reader and a capped eager entry-capacity
    /// reservation): a crafted or corrupted stream can no longer abort the
    /// process via an eager oversized allocation (CQ-09) — it fails with a
    /// normal, catchable [`DiskCacheError`] instead, typically
    /// [`DiskCacheError::Io`] (EOF) once the declared length outruns the
    /// actual bytes available.
    ///
    /// This entry point works on any `Read`, including in-memory buffers
    /// with no independently-knowable total length, so it cannot perform
    /// the stronger file-size cross-check that [`DiskCache::load`] does;
    /// use `load` when reading from a real file for the tighter,
    /// fail-fast [`DiskCacheError::HeaderExceedsFileSize`] diagnosis.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, DiskCacheError> {
        Self::read_from_bounded(reader, None)
    }

    /// Shared implementation behind [`DiskCache::read_from`] and
    /// [`DiskCache::load`]. `file_len_hint`, when known (i.e. the reader is
    /// backed by a real file of known size), lets the header's declared
    /// `num_entries` / metadata length be validated against the file's
    /// actual size *before* any entry is read.
    fn read_from_bounded<R: Read>(
        reader: &mut R,
        file_len_hint: Option<u64>,
    ) -> Result<Self, DiskCacheError> {
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(DiskCacheError::InvalidMagic);
        }

        // Version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != CACHE_VERSION {
            return Err(DiskCacheError::UnsupportedVersion(version));
        }

        // Num entries
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let num_entries_u64 = u64::from_le_bytes(buf8);

        // Metadata length
        reader.read_exact(&mut buf4)?;
        let meta_len_u32 = u32::from_le_bytes(buf4);

        // Fast, specific rejection when we know the real file size: a
        // header that claims more data than the file could possibly hold
        // is corrupted or adversarial, and there is no point reading
        // (potentially many) entries just to discover that via EOF.
        if let Some(file_len) = file_len_hint {
            let min_required = HEADER_BYTES
                .saturating_add(meta_len_u32 as u64)
                .saturating_add(num_entries_u64.saturating_mul(MIN_BYTES_PER_ENTRY));
            if min_required > file_len {
                return Err(DiskCacheError::HeaderExceedsFileSize {
                    declared_entries: num_entries_u64,
                    declared_meta_len: meta_len_u32 as u64,
                    file_len,
                    min_bytes_per_entry: MIN_BYTES_PER_ENTRY,
                });
            }
        }

        let meta_len = meta_len_u32 as usize;
        let meta_buf = read_bytes_bounded(reader, meta_len)?;
        let meta_str = String::from_utf8(meta_buf)
            .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;
        let metadata = metadata_from_json(&meta_str)?;

        // Entries. `num_entries_u64` is untrusted and only used as a loop
        // bound (each iteration still consumes real bytes from `reader` and
        // fails fast via `Io`/EOF once the stream is exhausted); the
        // eager capacity reservation below is separately capped so a
        // hostile count cannot itself trigger an oversized allocation.
        let entry_cap = usize::try_from(num_entries_u64)
            .unwrap_or(MAX_EAGER_ENTRIES)
            .min(MAX_EAGER_ENTRIES);
        let mut entries = Vec::with_capacity(entry_cap);
        for _ in 0..num_entries_u64 {
            // name
            reader.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let name_buf = read_bytes_bounded(reader, name_len)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;

            // quant_type
            reader.read_exact(&mut buf4)?;
            let qt_len = u32::from_le_bytes(buf4) as usize;
            let qt_buf = read_bytes_bounded(reader, qt_len)?;
            let quant_type = String::from_utf8(qt_buf)
                .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;

            // data
            reader.read_exact(&mut buf8)?;
            let data_len = u64::from_le_bytes(buf8) as usize;
            let data = read_bytes_bounded(reader, data_len)?;

            entries.push(CacheEntry {
                name,
                data,
                quant_type,
            });
        }

        Ok(Self { entries, metadata })
    }

    /// Check if a cache file exists and has valid magic + version.
    pub fn is_valid_cache(path: &Path) -> bool {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() {
            return false;
        }
        if &magic != CACHE_MAGIC {
            return false;
        }

        let mut buf4 = [0u8; 4];
        if reader.read_exact(&mut buf4).is_err() {
            return false;
        }
        let version = u32::from_le_bytes(buf4);
        version == CACHE_VERSION
    }

    /// Returns `Ok(true)` if the cache file is newer than the source file.
    pub fn is_fresh(cache_path: &Path, source_path: &Path) -> Result<bool, DiskCacheError> {
        let cache_meta = std::fs::metadata(cache_path)?;
        let source_meta = std::fs::metadata(source_path)?;

        let cache_time = cache_meta.modified().map_err(DiskCacheError::Io)?;
        let source_time = source_meta.modified().map_err(DiskCacheError::Io)?;

        Ok(cache_time >= source_time)
    }
}

// ---------------------------------------------------------------------------
// CacheManager
// ---------------------------------------------------------------------------

/// Manages multiple cached model files with LRU eviction.
#[derive(Debug)]
pub struct CacheManager {
    cache_dir: String,
    max_cache_size_bytes: usize,
    entries: Vec<CacheFileInfo>,
}

/// Information about one cached model file on disk.
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    /// Absolute path to the `.oxcache` file.
    pub path: String,
    /// Size on disk in bytes.
    pub size_bytes: usize,
    /// Last time this cache was accessed / loaded.
    pub last_accessed: SystemTime,
    /// Human-readable model name.
    pub model_name: String,
}

impl CacheManager {
    /// Create a new manager for the given directory with a byte budget.
    pub fn new(cache_dir: impl Into<String>, max_size_bytes: usize) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            max_cache_size_bytes: max_size_bytes,
            entries: Vec::new(),
        }
    }

    /// Register a cached file.
    pub fn register(&mut self, info: CacheFileInfo) {
        self.entries.push(info);
    }

    /// Total bytes used by all registered cache files.
    pub fn total_used_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Whether total usage exceeds the budget.
    pub fn should_evict(&self) -> bool {
        self.total_used_bytes() > self.max_cache_size_bytes
    }

    /// Candidates for eviction, sorted oldest-first (LRU).
    pub fn eviction_candidates(&self) -> Vec<&CacheFileInfo> {
        let mut sorted: Vec<&CacheFileInfo> = self.entries.iter().collect();
        sorted.sort_by_key(|e| e.last_accessed);
        sorted
    }

    /// Fraction of budget used (0.0 – 1.0+).
    pub fn utilization(&self) -> f32 {
        if self.max_cache_size_bytes == 0 {
            return 0.0;
        }
        self.total_used_bytes() as f32 / self.max_cache_size_bytes as f32
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let used_mb = self.total_used_bytes() as f64 / (1024.0 * 1024.0);
        let max_mb = self.max_cache_size_bytes as f64 / (1024.0 * 1024.0);
        let pct = self.utilization() * 100.0;
        format!(
            "Cache dir: {dir}, {n} models, {used:.1}/{max:.1} MB ({pct:.1}%)",
            dir = self.cache_dir,
            n = self.entries.len(),
            used = used_mb,
            max = max_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// Manual JSON helpers (no serde)
// ---------------------------------------------------------------------------

/// Serialize a `HashMap<String, String>` to a JSON object string.
fn metadata_to_json(map: &HashMap<String, String>) -> String {
    let mut out = String::from("{");
    let mut first = true;
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for key in keys {
        let value = &map[key];
        if !first {
            out.push(',');
        }
        first = false;
        out.push('"');
        json_escape_into(&mut out, key);
        out.push_str("\":\"");
        json_escape_into(&mut out, value);
        out.push('"');
    }
    out.push('}');
    out
}

/// Deserialize a JSON object string to `HashMap<String, String>`.
fn metadata_from_json(s: &str) -> Result<HashMap<String, String>, DiskCacheError> {
    let s = s.trim();
    if s == "{}" || s.is_empty() {
        return Ok(HashMap::new());
    }
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') || bytes.last() != Some(&b'}') {
        return Err(DiskCacheError::MetadataParse(format!(
            "expected JSON object, got: {s}"
        )));
    }
    let inner = &s[1..s.len() - 1];
    let mut map = HashMap::new();
    if inner.trim().is_empty() {
        return Ok(map);
    }

    let chars: Vec<char> = inner.chars().collect();
    let mut pos = 0usize;

    loop {
        // Skip whitespace / commas.
        while pos < chars.len() && (chars[pos] == ',' || chars[pos].is_whitespace()) {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }
        if chars[pos] != '"' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected '\"' at position {pos}, got '{}'",
                chars[pos]
            )));
        }
        pos += 1;
        let (key, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        // Skip ws, expect ':'
        skip_ws(&chars, &mut pos);
        if pos >= chars.len() || chars[pos] != ':' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected ':' after key '{key}'"
            )));
        }
        pos += 1;
        skip_ws(&chars, &mut pos);

        if pos >= chars.len() || chars[pos] != '"' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected '\"' for value of key '{key}'"
            )));
        }
        pos += 1;
        let (value, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        map.insert(key, value);
    }

    Ok(map)
}

fn parse_json_string(chars: &[char], mut pos: usize) -> Result<(String, usize), DiskCacheError> {
    let mut s = String::new();
    while pos < chars.len() {
        match chars[pos] {
            '"' => {
                pos += 1;
                return Ok((s, pos));
            }
            '\\' => {
                pos += 1;
                if pos >= chars.len() {
                    return Err(DiskCacheError::MetadataParse(
                        "unexpected end after backslash".into(),
                    ));
                }
                match chars[pos] {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    other => {
                        return Err(DiskCacheError::MetadataParse(format!(
                            "unknown escape '\\{other}'"
                        )));
                    }
                }
                pos += 1;
            }
            ch => {
                s.push(ch);
                pos += 1;
            }
        }
    }
    Err(DiskCacheError::MetadataParse("unterminated string".into()))
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn json_escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (CQ-09: bounded reads must never abort on a crafted length; CQ-11's
// "add in-file tests" note applies to every module in this crate, and this
// file previously had zero)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a syntactically-valid header (magic + version + num_entries +
    /// meta_len) with no metadata bytes and no entries following it, so the
    /// declared lengths are the only hostile part of the buffer.
    fn header_only(num_entries: u64, meta_len: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(CACHE_MAGIC);
        buf.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        buf.extend_from_slice(&num_entries.to_le_bytes());
        buf.extend_from_slice(&meta_len.to_le_bytes());
        buf
    }

    /// CQ-09 regression: before the fix, a crafted file declaring a huge
    /// `meta_len` (or `num_entries`) drove an eager `vec![0u8; N]`
    /// allocation large enough to abort the process (SIGABRT), which no
    /// `Result` could ever catch. `read_from` must now return a normal
    /// `Err` — reproduced here at a scale that would have aborted on any
    /// real machine if the bug were still present.
    #[test]
    fn read_from_huge_declared_meta_len_returns_err_not_abort() {
        // Declares a 4 GiB metadata blob backed by a handful of real bytes.
        let bytes = header_only(0, u32::MAX);
        let mut cursor = Cursor::new(bytes);
        let result = DiskCache::read_from(&mut cursor);
        assert!(
            result.is_err(),
            "a declared 4 GiB metadata length over a tiny buffer must error, not succeed"
        );
    }

    #[test]
    fn read_from_huge_declared_num_entries_returns_err_not_abort() {
        // Declares ~2^63 entries backed by zero real entry bytes.
        let bytes = header_only(1u64 << 63, 0);
        let mut cursor = Cursor::new(bytes);
        let result = DiskCache::read_from(&mut cursor);
        assert!(
            result.is_err(),
            "a declared 2^63 entry count over an empty buffer must error, not succeed"
        );
    }

    #[test]
    fn read_from_valid_empty_header_succeeds() {
        let bytes = header_only(0, 0);
        let mut cursor = Cursor::new(bytes);
        let cache = DiskCache::read_from(&mut cursor).expect("a genuinely empty cache is valid");
        assert_eq!(cache.num_entries(), 0);
    }

    /// `DiskCache::load` has the real file size available and must reject
    /// an impossible header immediately via `HeaderExceedsFileSize`, rather
    /// than falling through to a generic I/O EOF error.
    #[test]
    fn load_rejects_header_exceeding_real_file_size() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxibonsai_disk_cache_hostile_{}.oxcache",
            std::process::id()
        ));

        // A tiny file (just the header) that claims a huge entry count.
        let bytes = header_only(1_000_000_000, 0);
        std::fs::write(&path, &bytes).expect("write crafted file");

        let result = DiskCache::load(&path);
        let _ = std::fs::remove_file(&path);

        match result {
            Err(DiskCacheError::HeaderExceedsFileSize {
                declared_entries, ..
            }) => {
                assert_eq!(declared_entries, 1_000_000_000);
            }
            other => panic!("expected HeaderExceedsFileSize, got {other:?}"),
        }
    }

    /// A legitimately large-ish single entry (bigger than the eager-alloc
    /// cap) must still round-trip correctly — the bound only limits eager
    /// pre-reservation, not the maximum representable size.
    #[test]
    fn read_from_entry_larger_than_eager_cap_round_trips() {
        // Comfortably larger than a tiny cap would allow but still fast to
        // run in a unit test.
        let big_len = 2 * 1024 * 1024; // 2 MiB
        let mut cache = DiskCache::new();
        cache.add_entry(CacheEntry::new("big", vec![7u8; big_len], "f32"));

        let mut buf = Vec::new();
        cache.write_to(&mut buf).expect("write");
        let mut cursor = Cursor::new(buf);
        let loaded = DiskCache::read_from(&mut cursor).expect("read");

        let entry = loaded.get_entry("big").expect("entry present");
        assert_eq!(entry.data.len(), big_len);
        assert!(entry.data.iter().all(|&b| b == 7));
    }

    #[test]
    fn read_from_truncated_stream_returns_io_error_not_panic() {
        // A header claiming one entry, but the stream ends before the
        // entry's name bytes are available.
        let mut bytes = header_only(1, 0);
        bytes.extend_from_slice(&4u32.to_le_bytes()); // name_len = 4
                                                      // ... but no name bytes follow.
        let mut cursor = Cursor::new(bytes);
        let result = DiskCache::read_from(&mut cursor);
        assert!(matches!(result, Err(DiskCacheError::Io(_))));
    }
}
