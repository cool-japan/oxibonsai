//! Per-request sampling over HTTP: `seed` together with `logprobs`, and a
//! per-request `min_p` on every OpenAI endpoint.
//!
//! Attached to `server.rs` as a `#[cfg(test)]` child module. The engines are
//! weightless and scripted (`InferenceEngine::script_uniform_choice`): every
//! step is an equal choice among a fixed set of byte-token letters, so which
//! letter each step draws is decided by the sampler alone — its PRNG, its
//! penalties and its `min_p` — and the byte-level fixture tokenizer shows it
//! as text.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

// ── seed + logprobs ───────────────────────────────────────────────────────

/// Letters, per-token logprobs and top alternatives of one seeded
/// `logprobs` request on a uniform-letters engine with ambient seed
/// `ambient_seed`.
async fn seeded_logprobs(path: &str, ambient_seed: u64, request_seed: u64) -> serde_json::Value {
    let app = create_router(
        fx::uniform_letters_engine(ambient_seed),
        Some(fx::byte_tokenizer()),
    );
    let (status, _, text) = fx::post(
        app,
        path,
        serde_json::json!({
            "messages": [{"role": "user", "content": "spell something"}],
            "max_tokens": 12,
            "temperature": 0.9,
            "seed": request_seed,
            "logprobs": true,
            "top_logprobs": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{path}: {text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("a chat.completion");
    let choice = &json["choices"][0];
    serde_json::json!({
        "content": choice["message"]["content"],
        "logprobs": choice["logprobs"]["content"],
    })
}

async fn assert_seed_with_logprobs_reproduces(path: &str) {
    let first = seeded_logprobs(path, 1, 1234).await;
    let again = seeded_logprobs(path, 999, 1234).await;
    let other = seeded_logprobs(path, 1, 4321).await;
    let entries = first["logprobs"].as_array().expect("logprobs.content");
    assert_eq!(entries.len(), 12, "{path}: {first}");
    assert_eq!(
        first["content"].as_str().map(str::len),
        Some(12),
        "{path}: {first}"
    );
    assert_eq!(
        first, again,
        "{path}: the same seed must reproduce the same tokens AND logprobs, whatever the \
         replica's ambient seed"
    );
    assert_ne!(
        first["content"], other["content"],
        "{path}: a different seed must draw different tokens"
    );
}

#[tokio::test]
async fn seed_with_logprobs_is_reproducible_on_the_base_endpoint() {
    assert_seed_with_logprobs_reproduces("/v1/chat/completions").await;
}

#[tokio::test]
async fn seed_with_logprobs_is_reproducible_on_the_extended_endpoint() {
    assert_seed_with_logprobs_reproduces("/v1/chat/completions/extended").await;
}

/// The seeded `logprobs` request runs on its own fresh sampler: the replica's
/// ambient PRNG afterwards draws exactly what an untouched replica draws.
#[tokio::test]
async fn seed_with_logprobs_leaves_the_replicas_prng_untouched() {
    async fn next_unseeded(after_seeded: bool) -> serde_json::Value {
        let pool = EnginePool::new(vec![fx::uniform_letters_engine(77)]);
        let app = create_router_with_pool(
            Arc::clone(&pool),
            Some(fx::byte_tokenizer()),
            Arc::new(InferenceMetrics::new()),
        );
        let body = |seed: Option<u64>| {
            let mut body = serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "temperature": 0.9,
                "logprobs": true,
            });
            if let Some(seed) = seed {
                body["seed"] = serde_json::json!(seed);
            }
            body
        };
        if after_seeded {
            let (status, _, text) =
                fx::post(app.clone(), "/v1/chat/completions", body(Some(5))).await;
            assert_eq!(status, StatusCode::OK, "{text}");
        }
        let (status, _, text) = fx::post(app, "/v1/chat/completions", body(None)).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("JSON");
        json["choices"][0]["message"]["content"].clone()
    }
    assert_eq!(next_unseeded(true).await, next_unseeded(false).await);
}

// ── per-request min_p ─────────────────────────────────────────────────────

/// A router over a weightless engine whose every step is an equal choice
/// between `a` and `b`, with a replica BASELINE `min_p` of `1.0`.
///
/// With `frequency_penalty: 2.0` at temperature 1 the letter drawn more
/// often so far is `e^-2 ≈ 0.14` times as likely as the other, so a
/// `min_p` of `1.0` keeps only the lagging letter: every consecutive pair
/// of the answer holds both letters, whatever the seed. With `min_p: 0`
/// the leading letter keeps its 12% chance and some pair repeats a letter.
fn min_p_router() -> Router {
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    };
    let mut engine = fx::weightless_engine(fx::BYTE_VOCAB, params, 7);
    engine.script_uniform_choice(fx::byte_ids("ab"));
    engine.sampler.set_min_p(1.0);
    create_router(engine, Some(fx::byte_tokenizer()))
}

/// Whether every consecutive pair of `letters` holds two different letters.
fn pairs_alternate(letters: &str) -> bool {
    letters.len().is_multiple_of(2)
        && letters
            .as_bytes()
            .chunks(2)
            .all(|pair| pair.len() == 2 && pair[0] != pair[1])
}

/// The request body of one `min_p` draw on `path`.
fn min_p_body(path: &str, seed: u64, min_p: Option<f32>) -> serde_json::Value {
    let mut body = if path == "/v1/completions" {
        serde_json::json!({"prompt": "letters"})
    } else {
        serde_json::json!({"messages": [{"role": "user", "content": "letters"}]})
    };
    body["max_tokens"] = serde_json::json!(64);
    body["temperature"] = serde_json::json!(1.0);
    body["top_p"] = serde_json::json!(1.0);
    body["frequency_penalty"] = serde_json::json!(2.0);
    body["seed"] = serde_json::json!(seed);
    if let Some(min_p) = min_p {
        body["min_p"] = serde_json::json!(min_p);
    }
    body
}

/// The answer text of one `min_p` draw.
async fn min_p_draw(app: Router, path: &str, seed: u64, min_p: Option<f32>) -> String {
    let (status, _, text) = fx::post(app, path, min_p_body(path, seed, min_p)).await;
    assert_eq!(status, StatusCode::OK, "{path}: {text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    let choice = &json["choices"][0];
    let letters = if path == "/v1/completions" {
        choice["text"].as_str()
    } else {
        choice["message"]["content"].as_str()
    };
    letters.unwrap_or_default().to_string()
}

async fn assert_per_request_min_p(path: &str) {
    let app = min_p_router();
    // The replica baseline (1.0) applies to a request without `min_p`.
    for seed in 1..=4u64 {
        let letters = min_p_draw(app.clone(), path, seed, None).await;
        assert_eq!(letters.len(), 64, "{path}: {letters}");
        assert!(pairs_alternate(&letters), "{path} seed {seed}: {letters}");
    }
    // `min_p: 0` disables it for that request: the draw changes.
    let mut repeated_a_letter = false;
    for seed in 1..=4u64 {
        let letters = min_p_draw(app.clone(), path, seed, Some(0.0)).await;
        repeated_a_letter |= !pairs_alternate(&letters);
    }
    assert!(
        repeated_a_letter,
        "{path}: with min_p 0 some seed must draw the leading letter twice in a pair"
    );
    // …for that request only: the next request sees the baseline again.
    let letters = min_p_draw(app.clone(), path, 9, None).await;
    assert!(
        pairs_alternate(&letters),
        "{path}: baseline restored: {letters}"
    );
    // A request's own `min_p` is honoured too, and a seeded request with a
    // `min_p` is deterministic.
    let first = min_p_draw(app.clone(), path, 3, Some(0.5)).await;
    let again = min_p_draw(app, path, 3, Some(0.5)).await;
    assert!(pairs_alternate(&first), "{path}: min_p 0.5 > e^-2: {first}");
    assert_eq!(first, again, "{path}: seeded + min_p is deterministic");
}

#[tokio::test]
async fn per_request_min_p_on_the_base_endpoint() {
    assert_per_request_min_p("/v1/chat/completions").await;
}

#[tokio::test]
async fn per_request_min_p_on_the_extended_endpoint() {
    assert_per_request_min_p("/v1/chat/completions/extended").await;
}

#[tokio::test]
async fn per_request_min_p_on_the_completions_endpoint() {
    assert_per_request_min_p("/v1/completions").await;
}

/// Streaming honours the request's `min_p` as well (the same sampler
/// scope runs the streamed generation).
#[tokio::test]
async fn per_request_min_p_on_a_stream() {
    let app = min_p_router();
    let mut body = min_p_body("/v1/chat/completions", 2, Some(1.0));
    body["stream"] = serde_json::json!(true);
    let (status, _, sse) = fx::post(app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "{sse}");
    let letters = fx::delta_texts(&sse, "content").concat();
    assert_eq!(letters.len(), 64, "{sse}");
    assert!(pairs_alternate(&letters), "{letters}");
}

#[tokio::test]
async fn per_request_min_p_out_of_range_is_refused_on_every_endpoint() {
    for path in [
        "/v1/chat/completions",
        "/v1/chat/completions/extended",
        "/v1/completions",
    ] {
        let (status, _, text) =
            fx::post(min_p_router(), path, min_p_body(path, 1, Some(1.5))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("JSON error");
        assert_eq!(json["error"]["param"], "min_p", "{path}: {json}");
    }
}
