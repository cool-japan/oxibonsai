//! Inline terminal image display via the kitty graphics protocol.
//!
//! Used by the `oxibonsai repl` command to show a freshly rendered PNG in the
//! terminal instead of round-tripping through an external viewer. Detection is
//! deliberately narrow: we only claim support for Ghostty for now (the one
//! terminal this has been verified against). Other kitty-capable terminals
//! (kitty, WezTerm) fall through to `false` and the REPL writes a file path
//! instead — adding them is a one-line change to [`kitty_supported`] once tested.

use std::io::{self, Write};

/// Whether the current terminal speaks the kitty graphics protocol.
///
/// Ghostty advertises itself through `GHOSTTY_*` env vars and `TERM`/
/// `TERM_PROGRAM`; any one of them is enough.
pub fn kitty_supported() -> bool {
    if std::env::var_os("GHOSTTY_RESOURCES_DIR").is_some()
        || std::env::var_os("GHOSTTY_BIN_DIR").is_some()
    {
        return true;
    }
    if matches!(std::env::var("TERM").as_deref(), Ok("xterm-ghostty")) {
        return true;
    }
    matches!(
        std::env::var("TERM_PROGRAM").as_deref(),
        Ok("ghostty") | Ok("Ghostty")
    )
}

/// Display a PNG inline using the kitty graphics protocol.
///
/// Emits `f=100` (PNG), `a=T` (transmit + display immediately), with the
/// base64 payload split into 4 KiB chunks (`m=1` on every chunk but the last).
/// Only the first chunk carries the format/action keys, per the protocol. A
/// trailing newline advances the cursor below the image.
pub fn display_png_kitty(out: &mut impl Write, png: &[u8]) -> io::Result<()> {
    let b64 = base64_encode(png);
    let mut chunks = b64.as_bytes().chunks(4096).peekable();
    let mut first = true;
    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        if first {
            write!(out, "\x1b_Gf=100,a=T,m={more};")?;
            first = false;
        } else {
            write!(out, "\x1b_Gm={more};")?;
        }
        out.write_all(chunk)?;
        write!(out, "\x1b\\")?;
    }
    writeln!(out)?;
    out.flush()
}

/// Standard base64 (RFC 4648, `+/` alphabet, `=` padding).
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        s.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        s.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        s.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        s.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::base64_encode;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
