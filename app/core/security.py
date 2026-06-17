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
  valid_keys = settings.api_key_list

  # Dev convenience: skip auth only while the default placeholder key is in use.
  if not settings.is_production and (
    not valid_keys or "change-me-to-a-strong-key" in valid_keys
  ):
    return "dev"

  provided = _extract_key(authorization, x_api_key)
  if not provided or provided not in valid_keys:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid or missing API key",
    )
  return provided
