//! `oxibonsai tokenizer` — manage the Qwen3 tokenizer (download / inspect).

use super::args::TokenizerCmd;

pub(crate) fn run(tok_cmd: TokenizerCmd) -> anyhow::Result<()> {
    match tok_cmd {
        TokenizerCmd::Download {
            output,
            repo,
            force,
        } => {
            let out_path = std::path::Path::new(&output);
            if out_path.exists() && !force {
                println!("tokenizer.json already exists at {output}");
                println!("Use --force to overwrite.");
                return Ok(());
            }
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!("failed to create directory {}: {e}", parent.display())
                    })?;
                }
            }
            let url = format!("https://huggingface.co/{repo}/resolve/main/tokenizer.json");
            println!("Downloading tokenizer.json from {url}");
            let response = reqwest::blocking::get(&url)
                .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;
            if !response.status().is_success() {
                anyhow::bail!("server returned {} for {url}", response.status());
            }
            let bytes = response
                .bytes()
                .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;
            std::fs::write(out_path, &bytes)
                .map_err(|e| anyhow::anyhow!("failed to write {output}: {e}"))?;
            println!("Saved to {output} ({} KB)", bytes.len() / 1024);
        }

        TokenizerCmd::Info { path } => {
            let data = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&data)
                .map_err(|e| anyhow::anyhow!("invalid JSON in {path}: {e}"))?;
            let model_type = v
                .get("model")
                .and_then(|m| m.get("type"))
                .and_then(|t| t.as_str())
                .unwrap_or("unknown");
            let vocab_size = v
                .get("model")
                .and_then(|m| m.get("vocab"))
                .map(|vocab| {
                    if let Some(obj) = vocab.as_object() {
                        obj.len()
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let added_tokens = v
                .get("added_tokens")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            println!("Tokenizer: {path}");
            println!("  Type:         {model_type}");
            println!("  Vocab size:   {vocab_size}");
            println!("  Added tokens: {added_tokens}");
        }
    }

    Ok(())
}
