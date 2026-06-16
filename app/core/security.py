from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
  if x_api_key:
    return x_api_key.strip()
  if authorization:
    value = authorization.strip()
    if value.lower().startswith("bearer "):
      return value[7:].strip()
    return value
  return None


async def verify_api_key(
  authorization: str | None = Security(_api_key_header),
  x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
  settings = get_settings()
  if not settings.is_production and settings.api_key == "change-me-to-a-strong-key":
    return "dev"

  provided = _extract_key(authorization, x_api_key)
  if not provided or provided != settings.api_key:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid or missing API key",
    )
  return provided
