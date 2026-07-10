# soika_api

This is the documentation for the soika_api package.

## Auth configuration

Protected endpoints validate incoming user JWTs against Keycloak and then use a
service token for downstream service calls.

Required Keycloak settings:

- `KEYCLOAK_URL`
- `KEYCLOAK_REALM`
- `KEYCLOAK_CLIENT_ID`
- `KEYCLOAK_CLIENT_SECRET`

Optional settings:

- `KEYCLOAK_SCOPE`
- `KEYCLOAK_AUDIENCE`
- `KEYCLOAK_USER_ID_CLAIM` (`sub` by default)
- `KEYCLOAK_ALLOWED_ALGORITHMS` (`RS256` by default)
- `KEYCLOAK_VALIDATE_USER_TOKEN` (`true` by default)
- `KEYCLOAK_BACKGROUND_REFRESH` (`true` by default)
- `KEYCLOAK_SERVICE_AUTH_FAIL_FAST` (`false` by default)
- `KEYCLOAK_REQUEST_TIMEOUT_SECONDS` (`10` by default)
- `KEYCLOAK_REFRESH_BEFORE_EXPIRY_SECONDS` (`30` by default)
- `KEYCLOAK_JWKS_CACHE_SECONDS` (`300` by default)
