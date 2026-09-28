//! Authentication for the operator `/admin/*` surface.
//!
//! Finding `sec-15` (with `SV-07`): `create_router_with_options` merged the
//! admin router — `/admin/status`, `/admin/config`, `/admin/cache-stats`,
//! `/admin/workload-stats` and the state-mutating `POST
//! /admin/reset-metrics` — into the served app **unconditionally and with no
//! opt-out**. Authentication existed only as an outer layer that the two serve
//! binaries added *when a bearer token happened to be configured*, and
//! `oxibonsai-serve` binds `0.0.0.0` by default, so the default deployment
//! published its running configuration to the network.
//!
//! This module makes admin authentication **unconditional and independent of
//! the inference bearer token**: every router constructor now carries an
//! [`AuthConfig`], `/admin/*` is wrapped in [`admin_auth_mw`], and a request
//! without the admin credential is refused before it reaches any admin handler.
//!
//! | server state | admin request | result |
//! |---|---|---|
//! | no admin token configured | any | `403` (`admin_auth_not_configured`) |
//! | admin token configured | no / wrong credential | `401` + `WWW-Authenticate: Bearer` |
//! | admin token configured | matching credential | handler runs |
//!
//! The credential is accepted either as `Authorization: Bearer <token>` or as
//! `X-Admin-Token: <token>`, and is compared in constant time. The token comes
//! from [`AuthConfig::with_admin_token`] (what the serve binaries pass for
//! `--admin-token`) or, for the convenience constructors, from the
//! `OXI_ADMIN_TOKEN` environment variable.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::server::api_error::ApiError;

/// Environment variable read by the convenience router constructors for the
/// admin credential.
pub const ADMIN_TOKEN_ENV: &str = "OXI_ADMIN_TOKEN";

/// Alternative header carrying the admin credential.
pub const ADMIN_TOKEN_HEADER: &str = "x-admin-token";

/// How `/admin/*` is protected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdminAuth {
    /// A credential is configured; requests must present it.
    Token(String),
    /// No credential is configured, so no request can be authorized and the
    /// admin surface is refused outright. This is the *safe* default: it is
    /// never silently open.
    NotConfigured,
}

/// Authentication settings carried by a router.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthConfig {
    admin: AdminAuth,
}

impl Default for AuthConfig {
    /// Admin locked down; use [`AuthConfig::with_admin_token`] to enable it.
    fn default() -> Self {
        Self {
            admin: AdminAuth::NotConfigured,
        }
    }
}

impl AuthConfig {
    /// Require `token` for every `/admin/*` request.
    ///
    /// An empty or whitespace-only token is rejected as "not configured"
    /// rather than silently accepting an empty credential.
    pub fn with_admin_token(token: impl Into<String>) -> Self {
        let token = token.into();
        if token.trim().is_empty() {
            return Self::locked();
        }
        Self {
            admin: AdminAuth::Token(token),
        }
    }

    /// Refuse every `/admin/*` request (no credential configured).
    pub fn locked() -> Self {
        Self {
            admin: AdminAuth::NotConfigured,
        }
    }

    /// Resolve the admin credential from `OXI_ADMIN_TOKEN`, falling back to
    /// [`AuthConfig::locked`] when the variable is unset or empty.
    ///
    /// This is what [`crate::server::create_router`] and its siblings use, so
    /// an embedder that never thinks about auth gets the locked-down admin
    /// surface rather than an open one.
    pub fn from_env() -> Self {
        match std::env::var(ADMIN_TOKEN_ENV) {
            Ok(token) if !token.trim().is_empty() => Self::with_admin_token(token),
            _ => Self::locked(),
        }
    }

    /// The configured admin policy.
    pub fn admin(&self) -> &AdminAuth {
        &self.admin
    }

    /// Whether an admin credential is configured.
    pub fn admin_enabled(&self) -> bool {
        matches!(self.admin, AdminAuth::Token(_))
    }

    /// Authorize an admin request from its headers.
    pub fn authorize_admin(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        let expected = match &self.admin {
            AdminAuth::Token(token) => token,
            AdminAuth::NotConfigured => {
                return Err(ApiError::new(
                    StatusCode::FORBIDDEN,
                    "the /admin API requires an admin token, and none is configured on this \
                     server; start it with --admin-token, set OXI_ADMIN_TOKEN, or build the \
                     router with create_router_with_auth",
                )
                .with_type(crate::server::api_error::ERROR_TYPE_AUTHENTICATION)
                .with_code("admin_auth_not_configured"));
            }
        };

        let presented = presented_credential(headers);
        match presented {
            Some(value) if constant_time_eq(value.as_bytes(), expected.as_bytes()) => Ok(()),
            Some(_) => Err(
                ApiError::unauthorized("invalid admin credential for the /admin API")
                    .with_code("invalid_admin_token"),
            ),
            None => Err(ApiError::unauthorized(
                "the /admin API requires an admin credential: send it as \
                 `Authorization: Bearer <token>` or `X-Admin-Token: <token>`",
            )
            .with_code("missing_admin_token")),
        }
    }
}

/// Extract the presented admin credential from the request headers.
fn presented_credential(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = value.trim();
        if let Some(rest) = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
        {
            return Some(rest.trim().to_string());
        }
    }
    headers
        .get(ADMIN_TOKEN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_string())
}

/// Length-independent-leak-free comparison of two credentials.
///
/// Compares every byte of the longer input so the running time does not depend
/// on the position of the first difference. (The lengths themselves are not
/// secret in a way that helps an attacker guess the token content.)
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8;
    let n = a.len().max(b.len());
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

/// axum middleware gating `/admin/*`.
///
/// Applied to the admin sub-router only, so the inference routes are untouched
/// by it and a serve binary is still free to put its own bearer-auth layer in
/// front of everything.
pub async fn admin_auth_mw(
    State(auth): State<Arc<AuthConfig>>,
    req: Request,
    next: Next,
) -> Response {
    if let Err(err) = auth.authorize_admin(req.headers()) {
        let status = err.status();
        let path = req.uri().path().to_string();
        tracing::warn!(
            path = %path,
            status = status.as_u16(),
            "rejected unauthenticated /admin request"
        );
        let mut response = err.into_response();
        if status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"oxibonsai-admin\""),
            );
        }
        return response;
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            let name: axum::http::HeaderName = name.parse().expect("header name");
            map.insert(name, HeaderValue::from_str(value).expect("header value"));
        }
        map
    }

    #[test]
    fn locked_config_refuses_every_credential() {
        let auth = AuthConfig::locked();
        assert!(!auth.admin_enabled());
        let err = auth
            .authorize_admin(&headers(&[("authorization", "Bearer anything")]))
            .expect_err("must refuse");
        assert_eq!(err.status(), StatusCode::FORBIDDEN);
        assert_eq!(err.to_json()["error"]["code"], "admin_auth_not_configured");
    }

    #[test]
    fn empty_token_is_treated_as_not_configured() {
        assert_eq!(AuthConfig::with_admin_token("   "), AuthConfig::locked());
        assert_eq!(AuthConfig::with_admin_token(""), AuthConfig::locked());
    }

    #[test]
    fn bearer_credential_is_accepted() {
        let auth = AuthConfig::with_admin_token("s3cret");
        assert!(auth.admin_enabled());
        assert!(auth
            .authorize_admin(&headers(&[("authorization", "Bearer s3cret")]))
            .is_ok());
        assert!(auth
            .authorize_admin(&headers(&[("authorization", "bearer s3cret")]))
            .is_ok());
    }

    #[test]
    fn admin_token_header_is_accepted() {
        let auth = AuthConfig::with_admin_token("s3cret");
        assert!(auth
            .authorize_admin(&headers(&[("x-admin-token", "s3cret")]))
            .is_ok());
    }

    #[test]
    fn wrong_and_missing_credentials_are_401() {
        let auth = AuthConfig::with_admin_token("s3cret");
        let wrong = auth
            .authorize_admin(&headers(&[("authorization", "Bearer nope")]))
            .expect_err("must refuse");
        assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(wrong.to_json()["error"]["code"], "invalid_admin_token");

        let missing = auth
            .authorize_admin(&HeaderMap::new())
            .expect_err("must refuse");
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(missing.to_json()["error"]["code"], "missing_admin_token");
        assert_eq!(
            missing.error_type(),
            crate::server::api_error::ERROR_TYPE_AUTHENTICATION
        );
    }

    #[test]
    fn prefix_of_the_token_is_not_accepted() {
        let auth = AuthConfig::with_admin_token("s3cret");
        assert!(auth
            .authorize_admin(&headers(&[("authorization", "Bearer s3cre")]))
            .is_err());
        assert!(auth
            .authorize_admin(&headers(&[("authorization", "Bearer s3crett")]))
            .is_err());
    }

    #[test]
    fn constant_time_eq_matches_normal_equality() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"ab", b"abc"));
        assert!(!constant_time_eq(b"\0", b""));
    }

    #[test]
    fn default_is_locked() {
        assert_eq!(AuthConfig::default(), AuthConfig::locked());
        assert_eq!(AuthConfig::default().admin(), &AdminAuth::NotConfigured);
    }
}
