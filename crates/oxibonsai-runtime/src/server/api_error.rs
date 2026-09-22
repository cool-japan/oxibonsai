//! Canonical HTTP error type for the OpenAI-compatible surface.
//!
//! Before this module existed the chat endpoints were declared as
//! `Result<Response, StatusCode>`, and every failure arm returned a bare
//! [`StatusCode`]. axum renders a bare status with a **zero-length body**, so a
//! `503` from the engine pool or a `500` from a failed generation reached the
//! client with nothing to parse (finding `SV-04`), directly contradicting
//! [`crate::http_error`]'s module contract. Deserialization failures were worse
//! still: they surfaced as axum's plain-text rejection (finding `SV-05`), so an
//! OpenAI SDK raised a transport-level exception instead of an
//! `InvalidRequestError`.
//!
//! [`ApiError`] is the single type every handler in this module tree returns.
//! It renders the canonical envelope
//!
//! ```json
//! { "error": { "message": "...", "type": "...", "param": null, "code": null } }
//! ```
//!
//! — byte-identical to [`crate::http_error::error_response_typed`], which a unit
//! test asserts so the two cannot drift — plus any structured extra fields a
//! specific error wants to expose (the prompt-budget error attaches
//! `n_prompt_tokens`, `max_input_tokens` and `context_length`, for example), and
//! echoes `X-Request-ID` when the handler resolved one.
//!
//! [`OpenAiJson`] is the request extractor that produces the same envelope for a
//! malformed body.

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;

use crate::request_id::RequestId;
use crate::server::request_id_header_map;

/// OpenAI `type` value for a client-side input problem.
pub const ERROR_TYPE_INVALID_REQUEST: &str = "invalid_request_error";
/// OpenAI `type` value for a server-side failure.
pub const ERROR_TYPE_SERVER: &str = "server_error";
/// OpenAI `type` value for an authentication failure.
pub const ERROR_TYPE_AUTHENTICATION: &str = "authentication_error";

/// The error payload. Boxed inside [`ApiError`] so that a
/// `Result<T, ApiError>` stays cheap to move (clippy's `result_large_err`).
#[derive(Debug, Clone)]
struct ApiErrorInner {
    status: StatusCode,
    message: String,
    error_type: String,
    param: Option<String>,
    code: Option<String>,
    /// Extra members merged into the `error` object (e.g. the token-budget
    /// numbers). Always rendered after the four canonical members.
    fields: Vec<(String, serde_json::Value)>,
    request_id: Option<RequestId>,
}

/// An HTTP error rendered as the canonical OpenAI error envelope.
///
/// Construct one with the status-named constructors ([`ApiError::bad_request`],
/// [`ApiError::internal`], …) and refine it with the builder methods
/// ([`ApiError::with_param`], [`ApiError::with_code`], [`ApiError::with_field`],
/// [`ApiError::with_request_id`]).
#[derive(Debug, Clone)]
pub struct ApiError(Box<ApiErrorInner>);

impl ApiError {
    /// Build an error with an explicit status; the `type` is derived from the
    /// status class exactly as [`crate::http_error::error_response`] does.
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        let error_type = if status.is_server_error() {
            ERROR_TYPE_SERVER
        } else {
            ERROR_TYPE_INVALID_REQUEST
        };
        Self(Box::new(ApiErrorInner {
            status,
            message: message.into(),
            error_type: error_type.to_string(),
            param: None,
            code: None,
            fields: Vec::new(),
            request_id: None,
        }))
    }

    /// `400 Bad Request` naming the offending request field.
    pub fn bad_request(message: impl Into<String>, param: &str) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message).with_param(param)
    }

    /// `401 Unauthorized` (missing or wrong credentials).
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, message).with_type(ERROR_TYPE_AUTHENTICATION)
    }

    /// `500 Internal Server Error`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    /// `503 Service Unavailable` (no engine replica could be acquired).
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, message)
    }

    /// `504 Gateway Timeout` (the per-request deadline elapsed).
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, message).with_code("request_timeout")
    }

    /// Override the `type` member.
    pub fn with_type(mut self, error_type: &str) -> Self {
        self.0.error_type = error_type.to_string();
        self
    }

    /// Name the offending request field (`param`).
    pub fn with_param(mut self, param: &str) -> Self {
        self.0.param = Some(param.to_string());
        self
    }

    /// Attach a machine-readable `code`.
    pub fn with_code(mut self, code: &str) -> Self {
        self.0.code = Some(code.to_string());
        self
    }

    /// Attach a structured extra member to the `error` object.
    pub fn with_field(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.0.fields.push((key.to_string(), value.into()));
        self
    }

    /// Echo the request's correlation id back in the `X-Request-ID` header.
    pub fn with_request_id(mut self, id: RequestId) -> Self {
        self.0.request_id = Some(id);
        self
    }

    /// The HTTP status this error renders with.
    pub fn status(&self) -> StatusCode {
        self.0.status
    }

    /// The human-readable message.
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// The `type` member.
    pub fn error_type(&self) -> &str {
        &self.0.error_type
    }

    /// Render the JSON body (without the status or headers).
    ///
    /// Kept separate from [`IntoResponse`] so the SSE path can embed the very
    /// same object in a terminal `error` event.
    pub fn to_json(&self) -> serde_json::Value {
        let mut error = serde_json::Map::new();
        error.insert(
            "message".to_string(),
            serde_json::Value::String(self.0.message.clone()),
        );
        error.insert(
            "type".to_string(),
            serde_json::Value::String(self.0.error_type.clone()),
        );
        error.insert(
            "param".to_string(),
            match &self.0.param {
                Some(p) => serde_json::Value::String(p.clone()),
                None => serde_json::Value::Null,
            },
        );
        error.insert(
            "code".to_string(),
            match &self.0.code {
                Some(c) => serde_json::Value::String(c.clone()),
                None => serde_json::Value::Null,
            },
        );
        for (key, value) in &self.0.fields {
            error.insert(key.clone(), value.clone());
        }
        serde_json::json!({ "error": serde_json::Value::Object(error) })
    }

    /// Map an axum JSON extractor rejection onto the canonical envelope.
    ///
    /// The rejection's own status is preserved except for `422 Unprocessable
    /// Entity` (axum's "valid JSON, wrong shape" code), which is reported as
    /// `400` because that is what the OpenAI SDKs classify as an
    /// `InvalidRequestError`; a `422` surfaces as an unexpected-status
    /// exception instead (finding `SV-05`).
    pub fn from_json_rejection(rejection: JsonRejection) -> Self {
        let status = match rejection.status() {
            StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
            other => other,
        };
        let code = match &rejection {
            JsonRejection::JsonDataError(_) => "invalid_request_body",
            JsonRejection::JsonSyntaxError(_) => "invalid_json",
            JsonRejection::MissingJsonContentType(_) => "unsupported_media_type",
            _ => "invalid_request_body",
        };
        Self::new(status, rejection.body_text())
            .with_type(ERROR_TYPE_INVALID_REQUEST)
            .with_code(code)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.0.status.as_u16(), self.0.message)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = self.to_json();
        let status = self.0.status;
        match self.0.request_id {
            Some(id) => (status, request_id_header_map(id), axum::Json(body)).into_response(),
            None => (status, axum::Json(body)).into_response(),
        }
    }
}

/// JSON body extractor that rejects with [`ApiError`] instead of axum's
/// plain-text rejection (finding `SV-05`).
///
/// Drop-in replacement for [`axum::Json`] on handler arguments:
///
/// ```ignore
/// async fn handler(OpenAiJson(body): OpenAiJson<ChatCompletionRequest>) -> ...
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiJson<T>(pub T);

impl<T, S> FromRequest<S> for OpenAiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(ApiError::from_json_rejection(rejection)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

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
    async fn envelope_matches_http_error_helper_exactly() {
        // The canonical shape lives in `http_error`; `ApiError` must render the
        // identical object so a client cannot tell which layer produced it.
        let (status_a, json_a) =
            body_json(ApiError::bad_request("bad", "field").into_response()).await;
        let (status_b, json_b) = body_json(crate::http_error::bad_request("bad", "field")).await;
        assert_eq!(status_a, status_b);
        assert_eq!(json_a, json_b);

        let (status_c, json_c) = body_json(ApiError::internal("boom").into_response()).await;
        let (status_d, json_d) = body_json(crate::http_error::error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "boom",
            None,
        ))
        .await;
        assert_eq!(status_c, status_d);
        assert_eq!(json_c, json_d);
    }

    #[tokio::test]
    async fn service_unavailable_has_a_body() {
        // SV-04: the 503 arm used to return a zero-length body.
        let (status, json) =
            body_json(ApiError::service_unavailable("pool busy").into_response()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["error"]["message"], "pool busy");
        assert_eq!(json["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn extra_fields_are_merged_into_the_error_object() {
        let err = ApiError::bad_request("too long", "messages")
            .with_code("context_length_exceeded")
            .with_field("n_prompt_tokens", 4096)
            .with_field("context_length", 2048);
        let (status, json) = body_json(err.into_response()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "context_length_exceeded");
        assert_eq!(json["error"]["n_prompt_tokens"], 4096);
        assert_eq!(json["error"]["context_length"], 2048);
        assert_eq!(json["error"]["param"], "messages");
    }

    #[tokio::test]
    async fn request_id_is_echoed_on_errors() {
        let id = RequestId::new();
        let resp = ApiError::internal("boom")
            .with_request_id(id)
            .into_response();
        let header = resp
            .headers()
            .get(crate::server::REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        assert_eq!(header, Some(id.as_uuid()));
    }

    #[derive(serde::Deserialize)]
    struct TestBody {
        #[allow(dead_code)]
        value: u32,
    }

    fn json_router() -> Router {
        Router::new().route(
            "/",
            post(|OpenAiJson(_body): OpenAiJson<TestBody>| async { StatusCode::OK }),
        )
    }

    #[tokio::test]
    async fn malformed_json_produces_the_envelope() {
        let resp = json_router()
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{not json"))
                    .expect("request"),
            )
            .await
            .expect("response");
        let (status, json) = body_json(resp).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "invalid_json");
        assert!(
            json["error"]["message"].as_str().is_some(),
            "message must be a string: {json}"
        );
    }

    #[tokio::test]
    async fn wrong_shape_json_is_reported_as_400_not_422() {
        let resp = json_router()
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(r#"{"value":"not-a-number"}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        let (status, json) = body_json(resp).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "invalid_request_body");
    }

    #[tokio::test]
    async fn missing_content_type_keeps_415_but_gains_a_body() {
        let resp = json_router()
            .oneshot(
                Request::post("/")
                    .body(axum::body::Body::from(r#"{"value":1}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        let (status, json) = body_json(resp).await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(json["error"]["code"], "unsupported_media_type");
    }

    #[tokio::test]
    async fn valid_json_still_passes_through() {
        let resp = json_router()
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(r#"{"value":7}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
