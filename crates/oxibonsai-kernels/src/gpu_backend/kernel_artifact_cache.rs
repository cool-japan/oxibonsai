//! Split out of `gpu_backend/mod.rs` to keep it under the workspace's
//! 2000-line-per-file limit; the module path (`crate::gpu_backend::
//! kernel_artifact_cache`) is unchanged.
//!
//! User-private, integrity-checked, atomically written on-disk cache for
//! compiled GPU kernel artifacts (finding **F1**).
//!
//! The CUDA PTX cache used to live in the **shared** OS temp directory under a
//! predictable name, written non-atomically — on a multi-tenant host, GPU
//! **code injection**: any local user could plant the PTX the next OxiBonsai
//! process loads and executes on the GPU, and two concurrent processes could
//! hand each other a half-written file. Here instead: a `0o700` directory
//! under the caller's **own** cache root (`$XDG_CACHE_HOME`, else
//! `%LOCALAPPDATA%`, else `$HOME/.cache`, where the Metal metallib cache
//! already lives), refused unless it is a real directory (not a symlink) owned
//! by the current user with no group/other write bit; file names carrying the
//! **content hash** of the source *and* a hash of the build/driver
//! environment; a header carrying both hashes, the body length and the **body
//! hash**, all verified before the body is returned, so a tampered, truncated
//! or foreign file is a cache *miss*, never a load; and publishing through a
//! `create_new`, `0o600` temporary file plus `rename`, so a racing process is
//! never handed a file it did not create.
//!
//! The logic is GPU-free, so the ordinary CPU test suite exercises it on every
//! host. Its CUDA consumer (`cuda_graph::functions::compile_or_load_ptx`) is
//! **UNVALIDATED on hardware**: this project has no CUDA device.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};

/// First token of the one-line header in front of every artifact. It is a
/// line comment in both PTX and MSL, so a cached artifact stays readable.
const CACHE_MAGIC: &str = "//oxibonsai-kernel-artifact-cache";

/// On-disk header format version: an artifact written by another version is
/// a miss and is recompiled.
const CACHE_FORMAT_VERSION: u32 = 1;

/// Makes temporary file names unique within a process.
static NEXT_TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// FNV-1a 64-bit hash — the construction the PTX cache already used for its
/// source key, kept so cache keys stay comparable across versions.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = BASIS;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Why an artifact cache could not be opened or written.
#[derive(Debug)]
pub enum CacheError {
    /// No user-private cache root could be determined from the environment.
    NoCacheRoot,
    /// The directory is not a private directory owned by the current user
    /// (foreign owner, group/other-writable, or a symlink).
    NotPrivate(PathBuf),
    /// A path component or tag contained characters not allowed in a cache
    /// file name.
    InvalidTag(String),
    /// A filesystem operation failed.
    Io(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCacheRoot => {
                write!(f, "no user-private cache dir (set XDG_CACHE_HOME or HOME)")
            }
            Self::NotPrivate(p) => write!(
                f,
                "cache directory {} is not privately owned by this user",
                p.display()
            ),
            Self::InvalidTag(t) => write!(f, "invalid cache tag {t:?}"),
            Self::Io(e) => write!(f, "cache I/O error: {e}"),
        }
    }
}

impl std::error::Error for CacheError {}

/// Tags become file-name components, so they use a safe alphabet — this is
/// what makes path traversal through a tag impossible.
fn is_valid_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.len() <= 64
        && tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Resolve the user-private cache root: `$XDG_CACHE_HOME`, else
/// `%LOCALAPPDATA%` on Windows, else `$HOME/.cache`.
pub fn default_cache_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        let path = PathBuf::from(dir);
        if path.is_absolute() {
            return Some(path);
        }
    }
    #[cfg(windows)]
    {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            let path = PathBuf::from(dir);
            if path.is_absolute() {
                return Some(path);
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let path = PathBuf::from(home);
        if path.is_absolute() {
            return Some(path.join(".cache"));
        }
    }
    None
}

/// Create `dir` and its parents, `0o700` on Unix.
fn create_private_dir(dir: &Path) -> Result<(), CacheError> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder
        .create(dir)
        .map_err(|e| CacheError::Io(format!("create {}: {e}", dir.display())))
}

/// Learn this process' effective uid **without a `libc` dependency**: create
/// a file we are guaranteed to own inside `dir` and read its owner back. A
/// directory we cannot create in is a directory we must not trust.
#[cfg(unix)]
fn probe_owner_uid(dir: &Path) -> Result<u32, CacheError> {
    let seq = NEXT_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!(".uid-probe-{}-{seq}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| CacheError::Io(format!("uid probe in {}: {e}", dir.display())))?;
    let uid = file
        .metadata()
        .map(|m| m.uid())
        .map_err(|e| CacheError::Io(format!("stat uid probe: {e}")));
    drop(file);
    let _ = std::fs::remove_file(&path);
    uid
}

/// Require a real directory owned by the current user with no group/other
/// write permission, and return that uid. `symlink_metadata` does not follow
/// links, so a planted symlink fails even when it points at a directory.
#[cfg(unix)]
fn verify_private_dir(dir: &Path) -> Result<u32, CacheError> {
    let meta = std::fs::symlink_metadata(dir)
        .map_err(|e| CacheError::Io(format!("stat {}: {e}", dir.display())))?;
    if !meta.is_dir() {
        return Err(CacheError::NotPrivate(dir.to_path_buf()));
    }
    let uid = probe_owner_uid(dir)?;
    if meta.uid() != uid || meta.mode() & 0o022 != 0 {
        return Err(CacheError::NotPrivate(dir.to_path_buf()));
    }
    Ok(uid)
}

/// Non-Unix twin: only the "real directory" half is enforceable with `std`.
#[cfg(not(unix))]
fn verify_private_dir(dir: &Path) -> Result<(), CacheError> {
    let meta = std::fs::symlink_metadata(dir)
        .map_err(|e| CacheError::Io(format!("stat {}: {e}", dir.display())))?;
    if !meta.is_dir() {
        return Err(CacheError::NotPrivate(dir.to_path_buf()));
    }
    Ok(())
}

/// Build the one-line artifact header.
fn encode_header(src_hash: u64, env_hash: u64, body: &str) -> String {
    format!(
        "{CACHE_MAGIC} v{CACHE_FORMAT_VERSION} src={src_hash:016x} env={env_hash:016x} \
         len={} hash={:016x}\n",
        body.len(),
        fnv1a_64(body.as_bytes())
    )
}

/// Extract `key=<value>` from a whitespace-split header.
fn header_field<'a>(tokens: &[&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).filter(|_| t.len() > key.len()))
}

/// Verify a stored artifact and return its body, or `None` on any mismatch —
/// a corrupt, truncated, foreign or stale file is a cache miss.
fn decode_artifact(raw: &str, src_hash: u64, env_hash: u64) -> Option<String> {
    let split = raw.find('\n')?;
    let (header, rest) = raw.split_at(split);
    let body = rest.get(1..)?;
    let tokens: Vec<&str> = header.split_whitespace().collect();
    if tokens.first().copied() != Some(CACHE_MAGIC) {
        return None;
    }
    let version_tag = format!("v{CACHE_FORMAT_VERSION}");
    if tokens.get(1).copied() != Some(version_tag.as_str()) {
        return None;
    }
    let parse_hex =
        |key: &str| -> Option<u64> { u64::from_str_radix(header_field(&tokens, key)?, 16).ok() };
    if parse_hex("src=")? != src_hash || parse_hex("env=")? != env_hash {
        return None;
    }
    if body.len() != header_field(&tokens, "len=")?.parse::<usize>().ok()? {
        return None;
    }
    if parse_hex("hash=")? != fnv1a_64(body.as_bytes()) {
        return None;
    }
    Some(body.to_string())
}

/// A prepared, ownership-verified, user-private artifact cache directory.
///
/// Open once per process (the CUDA consumer memoises it in a `OnceLock`):
/// opening performs the directory creation, the ownership probe and the
/// permission check.
pub struct ArtifactCache {
    dir: PathBuf,
    extension: String,
    #[cfg(unix)]
    owner_uid: u32,
}

impl ArtifactCache {
    /// Open (creating when needed) `<root>/<sub_path>` as a private cache
    /// directory storing artifacts with file extension `extension`. Every
    /// `/`-separated component and the extension must pass [`is_valid_tag`],
    /// which is what makes traversal out of `root` impossible.
    pub fn open_in(root: &Path, sub_path: &str, extension: &str) -> Result<Self, CacheError> {
        if !is_valid_tag(extension) {
            return Err(CacheError::InvalidTag(extension.to_string()));
        }
        let mut dir = root.to_path_buf();
        for part in sub_path.split('/').filter(|p| !p.is_empty()) {
            if !is_valid_tag(part) {
                return Err(CacheError::InvalidTag(part.to_string()));
            }
            dir.push(part);
        }
        create_private_dir(&dir)?;
        #[cfg(unix)]
        let owner_uid = verify_private_dir(&dir)?;
        #[cfg(not(unix))]
        verify_private_dir(&dir)?;
        Ok(Self {
            dir,
            extension: extension.to_string(),
            #[cfg(unix)]
            owner_uid,
        })
    }

    /// Open the default user-private cache directory for `sub_path`.
    pub fn open_default(sub_path: &str, extension: &str) -> Result<Self, CacheError> {
        let root = default_cache_root().ok_or(CacheError::NoCacheRoot)?;
        Self::open_in(&root, sub_path, extension)
    }

    /// The verified cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// File name for one artifact: `<tag>-<src hash>-<env hash>.<ext>`.
    pub fn file_name(&self, tag: &str, src_hash: u64, env_hash: u64) -> Result<String, CacheError> {
        if !is_valid_tag(tag) {
            return Err(CacheError::InvalidTag(tag.to_string()));
        }
        Ok(format!(
            "{tag}-{src_hash:016x}-{env_hash:016x}.{}",
            self.extension
        ))
    }

    /// Absolute path of one artifact.
    pub fn path_for(&self, tag: &str, src_hash: u64, env_hash: u64) -> Result<PathBuf, CacheError> {
        Ok(self.dir.join(self.file_name(tag, src_hash, env_hash)?))
    }

    /// Load and **verify** an artifact. Any failure — missing, foreign
    /// owner, group/other-writable, wrong key, wrong length, wrong body
    /// hash — is reported as a miss so the caller recompiles.
    pub fn load(&self, tag: &str, src_hash: u64, env_hash: u64) -> Option<String> {
        let path = self.path_for(tag, src_hash, env_hash).ok()?;
        let mut file = File::open(&path).ok()?;
        let meta = file.metadata().ok()?;
        if !meta.is_file() {
            return None;
        }
        #[cfg(unix)]
        {
            if meta.uid() != self.owner_uid || meta.mode() & 0o022 != 0 {
                tracing::warn!(
                    path = %path.display(),
                    "refusing kernel artifact not privately owned by this user"
                );
                return None;
            }
        }
        let mut raw = String::new();
        file.read_to_string(&mut raw).ok()?;
        decode_artifact(&raw, src_hash, env_hash)
    }

    /// Store an artifact atomically: `create_new` + `0o600` temporary file,
    /// flushed, then published with a single `rename`.
    pub fn store(
        &self,
        tag: &str,
        src_hash: u64,
        env_hash: u64,
        body: &str,
    ) -> Result<(), CacheError> {
        let final_path = self.path_for(tag, src_hash, env_hash)?;
        let seq = NEXT_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self
            .dir
            .join(format!(".tmp-{}-{seq}-{tag}", std::process::id()));
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        opts.mode(0o600);
        // Scope the handle so it closes (drops) at the end of this block,
        // before the write-result check and the publishing `rename` below.
        // On native targets `File`'s `Drop` impl is what releases the OS
        // file descriptor; on wasm32 `File` carries no such `Drop` impl, so
        // an explicit `drop(file)` call here would be a
        // `clippy::drop_non_drop` warning under `--target
        // wasm32-unknown-unknown`. A block-scoped implicit drop keeps the
        // exact same close-before-rename ordering on every target without
        // ever calling `drop()` directly.
        let written = {
            let mut file = opts
                .open(&tmp_path)
                .map_err(|e| CacheError::Io(format!("create {}: {e}", tmp_path.display())))?;
            file.write_all(encode_header(src_hash, env_hash, body).as_bytes())
                .and_then(|()| file.write_all(body.as_bytes()))
                .and_then(|()| file.sync_all())
        };
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(CacheError::Io(format!("write {}: {e}", tmp_path.display())));
        }
        if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(CacheError::Io(format!(
                "publish {}: {e}",
                final_path.display()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, uniquely named scratch root under the OS temp dir.
    fn scratch_root(name: &str) -> PathBuf {
        let seq = NEXT_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_artifact_cache_{name}_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn open_scratch(name: &str) -> (PathBuf, ArtifactCache) {
        let root = scratch_root(name);
        let cache =
            ArtifactCache::open_in(&root, "oxibonsai/ptx", "ptx").expect("scratch cache must open");
        (root, cache)
    }

    /// Overwrite an artifact's body while keeping its header — the planting
    /// attack F1 describes.
    fn replace_body(path: &Path, body: &str) {
        let raw = std::fs::read_to_string(path).unwrap();
        let header_end = raw.find('\n').unwrap();
        std::fs::write(path, format!("{}\n{body}", &raw[..header_end])).unwrap();
    }

    #[test]
    fn store_load_round_trip_and_republish() {
        let (root, cache) = open_scratch("round_trip");
        let body = ".version 8.0\n// kernel body\n";
        assert!(cache.load("v7_kernels", 0x1234, 0xabcd).is_none());
        cache.store("v7_kernels", 0x1234, 0xabcd, body).unwrap();
        let loaded = cache.load("v7_kernels", 0x1234, 0xabcd);
        assert_eq!(loaded.as_deref(), Some(body));
        cache.store("v7_kernels", 0x1234, 0xabcd, "second").unwrap();
        let republished = cache.load("v7_kernels", 0x1234, 0xabcd);
        assert_eq!(republished.as_deref(), Some("second"));
        let tmps = std::fs::read_dir(cache.dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .count();
        assert_eq!(tmps, 0, "publish must leave no temporary files");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_different_source_or_environment_hash_never_loads() {
        let (root, cache) = open_scratch("keys");
        cache.store("tag", 0x11, 0x22, "body").unwrap();
        assert!(cache.load("tag", 0x11, 0x23).is_none(), "env hash ignored");
        assert!(cache.load("tag", 0x12, 0x22).is_none(), "src hash ignored");
        assert_eq!(cache.load("tag", 0x11, 0x22).as_deref(), Some("body"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tampered_truncated_and_unheadered_artifacts_are_rejected() {
        let (root, cache) = open_scratch("tamper");
        cache.store("tag", 7, 9, "original ptx").unwrap();
        let path = cache.path_for("tag", 7, 9).unwrap();
        replace_body(&path, "malicious ptx");
        assert!(
            cache.load("tag", 7, 9).is_none(),
            "a body that does not match its recorded hash must not load"
        );
        replace_body(&path, "orig");
        assert!(
            cache.load("tag", 7, 9).is_none(),
            "truncation must not load"
        );
        std::fs::write(&path, "just some ptx\nmore ptx\n").unwrap();
        assert!(cache.load("tag", 7, 9).is_none(), "no header must not load");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn file_names_carry_both_hashes_and_the_dir_is_private() {
        let (root, cache) = open_scratch("naming");
        let name = cache.file_name("v7_kernels", 0xdead_beef, 0x0123_4567);
        assert_eq!(
            name.unwrap(),
            "v7_kernels-00000000deadbeef-0000000001234567.ptx"
        );
        let meta = std::fs::metadata(cache.dir()).unwrap();
        assert!(meta.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(meta.permissions().mode() & 0o777, 0o700);
        }
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        if let Some(p) = default_cache_root() {
            assert!(p.is_absolute(), "cache root must be absolute: {p:?}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tags_that_could_escape_the_cache_directory_are_refused() {
        let (root, cache) = open_scratch("traversal");
        for bad in ["../evil", "a/b", "", "with space", "with.dot"] {
            assert!(
                matches!(cache.file_name(bad, 1, 1), Err(CacheError::InvalidTag(_))),
                "tag {bad:?} must be refused"
            );
            assert!(cache.store(bad, 1, 1, "x").is_err());
            assert!(cache.load(bad, 1, 1).is_none());
        }
        assert!(matches!(
            ArtifactCache::open_in(&root, "oxibonsai/../../etc", "ptx"),
            Err(CacheError::InvalidTag(_))
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A group/other-writable directory is the precondition for planting, so
    /// opening one must fail rather than be used; likewise a symlink.
    #[cfg(unix)]
    #[test]
    fn world_writable_or_symlinked_cache_directories_are_refused() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch_root("not_private");
        let dir = root.join("oxibonsai").join("ptx");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(
            matches!(
                ArtifactCache::open_in(&root, "oxibonsai/ptx", "ptx"),
                Err(CacheError::NotPrivate(_))
            ),
            "a world-writable cache directory must be refused"
        );
        let link_root = scratch_root("symlinked");
        let real = link_root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::create_dir_all(link_root.join("oxibonsai")).unwrap();
        std::os::unix::fs::symlink(&real, link_root.join("oxibonsai").join("ptx")).unwrap();
        assert!(
            matches!(
                ArtifactCache::open_in(&link_root, "oxibonsai/ptx", "ptx"),
                Err(CacheError::NotPrivate(_))
            ),
            "a symlinked cache directory must be refused"
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&link_root);
    }
}
