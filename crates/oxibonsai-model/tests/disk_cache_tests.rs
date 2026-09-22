#![cfg(feature = "training")]
//! Integration tests for the on-disk tensor cache (`oxibonsai_model::disk_cache`).

use oxibonsai_model::disk_cache::{
    CacheEntry, CacheFileInfo, CacheManager, DiskCache, DiskCacheError,
};
use std::io::Cursor;
use std::time::SystemTime;

#[test]
fn cache_entry_size_bytes() {
    let entry = CacheEntry::new("test", vec![0u8; 100], "f32");
    assert_eq!(entry.size_bytes(), 100);
}

#[test]
fn disk_cache_new_empty() {
    let cache = DiskCache::new();
    assert_eq!(cache.num_entries(), 0);
}

#[test]
fn disk_cache_add_entry() {
    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("w1", vec![1, 2, 3], "int8"));
    assert_eq!(cache.num_entries(), 1);
    cache.add_entry(CacheEntry::new("w2", vec![4, 5], "f32"));
    assert_eq!(cache.num_entries(), 2);
}

#[test]
fn disk_cache_get_entry() {
    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("layer.0.q", vec![10, 20], "q1_0_g128"));
    let found = cache.get_entry("layer.0.q");
    assert!(found.is_some());
    let e = found.expect("entry should exist");
    assert_eq!(e.data, vec![10, 20]);
    assert_eq!(e.quant_type, "q1_0_g128");
}

#[test]
fn disk_cache_get_unknown() {
    let cache = DiskCache::new();
    assert!(cache.get_entry("nonexistent").is_none());
}

#[test]
fn disk_cache_metadata() {
    let mut cache = DiskCache::new();
    cache.set_metadata("model", "qwen3-8b");
    cache.set_metadata("version", "1");
    assert_eq!(cache.get_metadata("model"), Some("qwen3-8b"));
    assert_eq!(cache.get_metadata("version"), Some("1"));
    assert_eq!(cache.get_metadata("missing"), None);
}

#[test]
fn disk_cache_total_data_bytes() {
    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("a", vec![0; 50], "f32"));
    cache.add_entry(CacheEntry::new("b", vec![0; 30], "int8"));
    assert_eq!(cache.total_data_bytes(), 80);
}

#[test]
fn disk_cache_write_read_empty() {
    let cache = DiskCache::new();
    let mut buf = Vec::new();
    cache.write_to(&mut buf).expect("write should succeed");
    let mut cursor = Cursor::new(&buf);
    let loaded = DiskCache::read_from(&mut cursor).expect("read should succeed");
    assert_eq!(loaded.num_entries(), 0);
}

#[test]
fn disk_cache_write_read_one_entry() {
    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("weight", vec![1, 2, 3, 4], "int8"));
    let mut buf = Vec::new();
    cache.write_to(&mut buf).expect("write");
    let mut cursor = Cursor::new(&buf);
    let loaded = DiskCache::read_from(&mut cursor).expect("read");
    assert_eq!(loaded.num_entries(), 1);
    let e = loaded.get_entry("weight").expect("entry");
    assert_eq!(e.data, vec![1, 2, 3, 4]);
    assert_eq!(e.quant_type, "int8");
}

#[test]
fn disk_cache_write_read_multiple() {
    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("a", vec![10; 100], "f32"));
    cache.add_entry(CacheEntry::new("b", vec![20; 200], "int8"));
    cache.add_entry(CacheEntry::new("c", vec![30; 50], "q1_0_g128"));
    let mut buf = Vec::new();
    cache.write_to(&mut buf).expect("write");
    let mut cursor = Cursor::new(&buf);
    let loaded = DiskCache::read_from(&mut cursor).expect("read");
    assert_eq!(loaded.num_entries(), 3);
    assert_eq!(loaded.get_entry("a").expect("a").data.len(), 100);
    assert_eq!(loaded.get_entry("b").expect("b").data.len(), 200);
    assert_eq!(loaded.get_entry("c").expect("c").data.len(), 50);
}

#[test]
fn disk_cache_write_read_metadata() {
    let mut cache = DiskCache::new();
    cache.set_metadata("model", "qwen3-8b");
    cache.set_metadata("format", "q1_0_g128");
    let mut buf = Vec::new();
    cache.write_to(&mut buf).expect("write");
    let mut cursor = Cursor::new(&buf);
    let loaded = DiskCache::read_from(&mut cursor).expect("read");
    assert_eq!(loaded.get_metadata("model"), Some("qwen3-8b"));
    assert_eq!(loaded.get_metadata("format"), Some("q1_0_g128"));
}

#[test]
fn disk_cache_save_load_tempfile() {
    let dir = std::env::temp_dir();
    let path = dir.join("oxibonsai_test_cache.oxcache");

    let mut cache = DiskCache::new();
    cache.add_entry(CacheEntry::new("w", vec![42; 16], "f32"));
    cache.set_metadata("test", "yes");
    cache.save(&path).expect("save");

    let loaded = DiskCache::load(&path).expect("load");
    assert_eq!(loaded.num_entries(), 1);
    assert_eq!(loaded.get_metadata("test"), Some("yes"));

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn disk_cache_invalid_magic() {
    let bad = b"BADDxxxxxxxxxxxxxxxx";
    let mut cursor = Cursor::new(bad.as_slice());
    let result = DiskCache::read_from(&mut cursor);
    assert!(result.is_err());
    match result {
        Err(DiskCacheError::InvalidMagic) => {}
        other => panic!("expected InvalidMagic, got {:?}", other),
    }
}

#[test]
fn cache_manager_register() {
    let cache_dir = std::env::temp_dir().join("cache");
    let mut mgr = CacheManager::new(cache_dir.to_str().expect("path is valid UTF-8"), 1_000_000);
    mgr.register(CacheFileInfo {
        path: std::env::temp_dir()
            .join("cache/m1.oxcache")
            .to_str()
            .expect("path is valid UTF-8")
            .to_string(),
        size_bytes: 500_000,
        last_accessed: SystemTime::now(),
        model_name: "model-a".into(),
    });
    assert_eq!(mgr.total_used_bytes(), 500_000);
}

#[test]
fn cache_manager_should_evict() {
    let cache_dir = std::env::temp_dir().join("cache");
    let mut mgr = CacheManager::new(cache_dir.to_str().expect("path is valid UTF-8"), 100);
    mgr.register(CacheFileInfo {
        path: std::env::temp_dir()
            .join("cache/big.oxcache")
            .to_str()
            .expect("path is valid UTF-8")
            .to_string(),
        size_bytes: 200,
        last_accessed: SystemTime::now(),
        model_name: "big".into(),
    });
    assert!(mgr.should_evict());
}

#[test]
fn cache_manager_utilization() {
    let cache_dir = std::env::temp_dir().join("cache");
    let mut mgr = CacheManager::new(cache_dir.to_str().expect("path is valid UTF-8"), 1000);
    mgr.register(CacheFileInfo {
        path: std::env::temp_dir()
            .join("cache/m.oxcache")
            .to_str()
            .expect("path is valid UTF-8")
            .to_string(),
        size_bytes: 250,
        last_accessed: SystemTime::now(),
        model_name: "m".into(),
    });
    let u = mgr.utilization();
    assert!((u - 0.25).abs() < 0.01);
}
