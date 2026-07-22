//! Shared OpenAI-style HTTP error envelope.
//!
//! Every OxiBonsai HTTP route mounted on the OpenAI-compatible router — the
//! chat/completions family (`server.rs`, `api_extensions.rs`,
//! `completions.rs`), the embeddings endpoint (`embeddings.rs`), and the
//! optional RAG endpoints (`rag_server.rs`) — returns error bodies in the
//! canonical OpenAI shape so a client written against the documented error
//! contract can parse `error.message` / `error.type` uniformly regardless of
//! which route produced the error:
//!
//! ```json
//! { "error": { "message": "...", "type": "...", "param": "...", "code": null } }
//! ```
//!
//! Before this helper existed the surface carried three divergent shapes:
//! the nested envelope on the chat routes, a flat `{"error": "<string>"}` on
//! the RAG routes, and a body-less status code on the embeddings
//! input-validation path (finding `serve-api-10`). Routing every handler
//! through `error_response` / `bad_request` collapses them into one.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};

/// Build an OpenAI-compatible JSON error response with the canonical nested
/// envelope `{"error": {"message", "type", "param", "code"}}`.
///
/// The `type` is derived from the status class: any `5xx` maps to
/// `"server_error"`, everything else to `"invalid_request_error"` (the two
/// `type` values OpenAI itself uses for server-side failures and client input
/// problems respectively). `param` is emitted as a JSON string when a specific
/// offending field is known and `null` otherwise; `code` is always `null`
/// (OxiBonsai does not assign machine-readable error codes yet).
pub fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    param: Option<&str>,
) -> Response {
    let error_type = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    error_response_typed(status, message, error_type, param)
}

/// Like [`error_response`] but with an explicit `type` string, for the rare
/// case where the status-derived default is not the right classification.
pub fn error_response_typed(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &str,
    param: Option<&str>,
) -> Response {
    let param_value = match param {
        Some(p) => serde_json::Value::String(p.to_string()),
        None => serde_json::Value::Null,
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": error_type,
                "param": param_value,
                "code": serde_json::Value::Null,
            }
        })),
    )
        .into_response()
}

/// Convenience constructor for a `400 Bad Request` with the
/// `invalid_request_error` type and the named offending `param`.
pub fn bad_request(message: impl Into<String>, param: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, Some(param))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn body_json(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error body must be valid JSON");
        (status, json)
    }

    #[tokio::test]
    async fn bad_request_uses_nested_envelope() {
        let (status, json) = body_json(bad_request("bad", "field")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["message"], "bad");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "field");
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn server_error_type_is_derived_from_status() {
        let (status, json) = body_json(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "boom",
            None,
        ))
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json["error"]["type"], "server_error");
        assert!(json["error"]["param"].is_null());
    }

    #[tokio::test]
    async fn unprocessable_entity_is_client_error_type() {
        let (status, json) = body_json(error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty",
            Some("input"),
        ))
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "input");
    }
}
