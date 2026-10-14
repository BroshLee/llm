
import hmac
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def validate_security_config() -> None:
    api_key = os.getenv("APP_API_KEY")

    if not api_key or len(api_key) < 32:
        raise RuntimeError(
            "APP_API_KEY must be configured with "
            "a secret of at least 32 characters."
        )


async def require_api_key(
    api_key: str | None = Security(api_key_header),
) -> None:
    expected_key = os.getenv("APP_API_KEY")

    if not expected_key or len(expected_key) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if api_key is None or not hmac.compare_digest(
        api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
