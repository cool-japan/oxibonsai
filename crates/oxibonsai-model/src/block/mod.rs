//! Auto-generated module structure

pub mod functions;
pub mod types;

// Re-export all types
#[cfg(any(
    feature = "metal",
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(crate) use functions::blocks_as_bytes;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(crate) use functions::blocks_as_bytes_ternary;
pub use types::*;
