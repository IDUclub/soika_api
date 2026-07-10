from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import aiohttp
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from idu_service_auth import KeycloakAuthError, KeycloakTokenClient, KeycloakTokenConfig
from jwt import InvalidTokenError, PyJWKSet
from loguru import logger

from app.dependencies import get_config_value


http_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    access_token: str
    user_id: str
    username: str | None
    claims: dict[str, Any]


def _get_token_from_header(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=400,
            detail="Token is missing in the authorization header",
        )

    return token


def _get_bool_config(key: str, default: bool) -> bool:
    value = get_config_value(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_config(key: str, default: int) -> int:
    value = get_config_value(key)
    if value is None:
        return default
    return int(value)


def _keycloak_url() -> str | None:
    value = get_config_value("KEYCLOAK_URL")
    return value.rstrip("/") if value else None


def _keycloak_realm() -> str | None:
    return get_config_value("KEYCLOAK_REALM")


def _keycloak_issuer() -> str | None:
    auth_server_url = _keycloak_url()
    realm = _keycloak_realm()
    if not auth_server_url or not realm:
        return None
    return f"{auth_server_url}/realms/{realm}"


def _build_keycloak_config() -> KeycloakTokenConfig | None:
    auth_server_url = _keycloak_url()
    realm = _keycloak_realm()
    client_id = get_config_value("KEYCLOAK_CLIENT_ID")
    client_secret = get_config_value("KEYCLOAK_CLIENT_SECRET")

    if not all([auth_server_url, realm, client_id, client_secret]):
        return None

    return KeycloakTokenConfig(
        auth_server_url=auth_server_url,
        realm=realm,
        client_id=client_id,
        client_secret=client_secret,
        scope=get_config_value("KEYCLOAK_SCOPE") or None,
        request_timeout_seconds=_get_int_config("KEYCLOAK_REQUEST_TIMEOUT_SECONDS", 10),
        refresh_before_expiry_seconds=_get_int_config(
            "KEYCLOAK_REFRESH_BEFORE_EXPIRY_SECONDS",
            30,
        ),
        background_refresh=_get_bool_config("KEYCLOAK_BACKGROUND_REFRESH", True),
    )


def _extract_user_id(claims: dict[str, Any]) -> str:
    user_id_claim = get_config_value("KEYCLOAK_USER_ID_CLAIM", "sub")
    user_id = claims.get(user_id_claim) or claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User token does not contain user id")
    return str(user_id)


def _extract_username(claims: dict[str, Any]) -> str | None:
    username = claims.get("preferred_username") or claims.get("email")
    return str(username) if username else None


class KeycloakUserTokenValidator:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._jwks_by_kid: dict[str, Any] = {}
        self._jwks_expires_at = 0.0

    async def init(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._jwks_by_kid.clear()
        self._jwks_expires_at = 0.0

    async def verify(self, token: str) -> AuthenticatedUser:
        if not _get_bool_config("KEYCLOAK_VALIDATE_USER_TOKEN", True):
            claims = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            return AuthenticatedUser(
                access_token=token,
                user_id=_extract_user_id(claims),
                username=_extract_username(claims),
                claims=claims,
            )

        issuer = _keycloak_issuer()
        if issuer is None:
            raise HTTPException(status_code=503, detail="Keycloak user-token validation is not configured")

        try:
            header = jwt.get_unverified_header(token)
            allowed_algorithms = self._allowed_algorithms()
            algorithm = header.get("alg")
            if algorithm not in allowed_algorithms:
                raise InvalidTokenError("Unsupported JWT algorithm")

            key = await self._get_key(header)
            audience = get_config_value("KEYCLOAK_AUDIENCE") or None
            claims = jwt.decode(
                token,
                key=key,
                algorithms=allowed_algorithms,
                issuer=issuer,
                audience=audience,
                options={"verify_aud": audience is not None},
            )
        except InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail="Invalid authorization token") from exc

        return AuthenticatedUser(
            access_token=token,
            user_id=_extract_user_id(claims),
            username=_extract_username(claims),
            claims=claims,
        )

    async def _get_key(self, header: dict[str, Any]) -> Any:
        kid = header.get("kid")
        if not kid:
            raise InvalidTokenError("Token has no key id")

        if kid not in self._jwks_by_kid or monotonic() >= self._jwks_expires_at:
            await self._refresh_jwks()

        key = self._jwks_by_kid.get(kid)
        if key is None:
            await self._refresh_jwks()
            key = self._jwks_by_kid.get(kid)

        if key is None:
            raise InvalidTokenError("Token key is not available")

        return key

    async def _refresh_jwks(self) -> None:
        issuer = _keycloak_issuer()
        if issuer is None:
            raise HTTPException(status_code=503, detail="Keycloak user-token validation is not configured")

        if self._session is None or self._session.closed:
            await self.init()

        assert self._session is not None
        jwks_url = f"{issuer}/protocol/openid-connect/certs"
        try:
            async with self._session.get(
                jwks_url,
                timeout=_get_int_config("KEYCLOAK_REQUEST_TIMEOUT_SECONDS", 10),
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=503, detail="Keycloak JWKS endpoint is unavailable")
                jwks_payload = await response.json()
        except aiohttp.ClientError as exc:
            raise HTTPException(status_code=503, detail="Keycloak JWKS endpoint is unavailable") from exc

        jwks = PyJWKSet.from_dict(jwks_payload)
        self._jwks_by_kid = {key.key_id: key.key for key in jwks.keys}
        self._jwks_expires_at = monotonic() + _get_int_config("KEYCLOAK_JWKS_CACHE_SECONDS", 300)

    @staticmethod
    def _allowed_algorithms() -> list[str]:
        value = get_config_value("KEYCLOAK_ALLOWED_ALGORITHMS", "RS256") or "RS256"
        return [algorithm.strip() for algorithm in value.split(",") if algorithm.strip()]


class ServiceTokenProvider:
    def __init__(self) -> None:
        self._client: KeycloakTokenClient | None = None

    async def init(self) -> None:
        keycloak_config = _build_keycloak_config()
        if keycloak_config is None:
            logger.warning("Service authentication is not configured")
            return

        self._client = KeycloakTokenClient(keycloak_config)
        await self._client.__aenter__()

        if _get_bool_config("KEYCLOAK_SERVICE_AUTH_FAIL_FAST", False):
            await self._client.get_access_token()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def get_authorization_headers(self) -> dict[str, str]:
        if self._client is None:
            raise HTTPException(status_code=503, detail="Service authentication is not configured")

        try:
            return await self._client.get_authorization_headers()
        except KeycloakAuthError as exc:
            raise HTTPException(status_code=503, detail="Auth provider unavailable") from exc


user_token_validator = KeycloakUserTokenValidator()
service_token_provider = ServiceTokenProvider()


async def init_auth_clients() -> None:
    try:
        await user_token_validator.init()
        await service_token_provider.init()
    except Exception:
        await close_auth_clients()
        raise


async def close_auth_clients() -> None:
    await service_token_provider.close()
    await user_token_validator.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
) -> AuthenticatedUser:
    token = _get_token_from_header(credentials)
    return await user_token_validator.verify(token)


async def get_outgoing_auth_headers(
    actor_user_id: str | None = None,
    user_token: str | None = None,
) -> dict[str, str]:
    if user_token is not None:
        headers = {"Authorization": f"Bearer {user_token}"}
    else:
        headers = await service_token_provider.get_authorization_headers()

    if actor_user_id:
        headers["X-Actor-User-Id"] = actor_user_id

    return headers
