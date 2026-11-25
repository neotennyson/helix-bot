import json
import os
import re
import time
from datetime import datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_BASE_URL = "https://tokens.example.invalid"
CACHE_PATH = Path.home() / ".broker_tokens.json"
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
ALLOWED_BROKERS = {"dhan", "groww"}
IST = ZoneInfo("Asia/Kolkata")


class TokenError(RuntimeError):
    pass


def _api_key() -> str:
    value = os.environ.get("TOKEN_API_KEY", "").strip()
    if value:
        return value
    env_path = Path(__file__).with_name(".env")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, candidate = stripped.partition("=")
        if not separator or key.strip() != "TOKEN_API_KEY":
            continue
        candidate = candidate.strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
            candidate = candidate[1:-1]
        if candidate:
            return candidate
    raise TokenError("TOKEN_API_KEY is not configured")


def _normalise_broker(broker: str) -> str:
    value = str(broker).strip().lower()
    if value not in ALLOWED_BROKERS:
        raise ValueError("broker must be dhan or groww")
    return value


def _parse_timestamp(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", text):
        now = datetime.now(IST)
        parsed = datetime.combine(now.date(), datetime_time.fromisoformat(text), tzinfo=IST)
        if parsed.timestamp() <= now.timestamp():
            parsed += timedelta(days=1)
        return parsed.timestamp()
    try:
        numeric = float(text)
        if numeric > 10_000_000_000:
            numeric /= 1000.0
        if numeric > 10_000_000:
            return numeric
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise TokenError("token service returned an invalid expiry") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.timestamp()


def _read_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_name(f".{CACHE_PATH.name}.{os.getpid()}.tmp")
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, CACHE_PATH)
        os.chmod(CACHE_PATH, 0o600)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _cache_token(cache: dict[str, Any], broker: str, now: float) -> str | None:
    entry = cache.get(broker)
    if not isinstance(entry, dict):
        return None
    token = entry.get("token")
    fetched_at = _parse_timestamp(entry.get("fetched_at"))
    if not isinstance(token, str) or not token or fetched_at is None:
        return None
    if now - fetched_at >= CACHE_MAX_AGE_SECONDS or now < fetched_at - 300:
        return None
    expiry = _parse_timestamp(entry.get("expiry"))
    if expiry is not None and expiry <= now:
        return None
    try:
        os.chmod(CACHE_PATH, 0o600)
    except OSError:
        pass
    return token


AUTH_HTTP_CODES = {401, 403}
RETRYABLE_HTTP = {502, 503, 504}
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 3.0


def _request_api(path: str, method: str) -> Any:
    last_error: str | None = None
    for attempt in range(RETRY_ATTEMPTS):
        request = Request(
            f"{API_BASE_URL}{path}",
            data=b"" if method == "POST" else None,
            headers={"X-API-Key": _api_key(), "Accept": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read()
            break
        except HTTPError as error:
            if error.code in RETRYABLE_HTTP and attempt < RETRY_ATTEMPTS - 1:
                last_error = f"HTTP {error.code}"
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise TokenError(f"token service returned HTTP {error.code}") from None
        except URLError as error:
            last_error = str(error.reason)
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise TokenError("token service is unavailable") from error
    else:
        raise TokenError(f"token service is unavailable ({last_error})")
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenError("token service returned invalid JSON") from error


def _fetch_token(broker: str) -> tuple[str, dict[str, Any]]:
    payload = _request_api(f"/token/{broker}", "GET")
    if not isinstance(payload, dict):
        raise TokenError("token service returned an invalid response")
    token = payload.get("token")
    if not isinstance(token, str) or not token.strip():
        raise TokenError("token service returned no token")
    return token.strip(), payload


def get_token(broker: str) -> str:
    broker = _normalise_broker(broker)
    now = time.time()
    cache = _read_cache()
    cached = _cache_token(cache, broker, now)
    if cached is not None:
        return cached
    token, payload = _fetch_token(broker)
    cache[broker] = {
        "token": token,
        "expiry": payload.get("expiry"),
        "updated": payload.get("updated"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(cache)
    return token


def invalidate_token(broker: str) -> None:
    broker = _normalise_broker(broker)
    cache = _read_cache()
    if cache.pop(broker, None) is not None:
        _write_cache(cache)


def fresh_token(broker: str) -> str:
    """Fetch a brand-new token, bypassing (and replacing) any cached entry."""
    broker = _normalise_broker(broker)
    token, payload = _fetch_token(broker)
    cache = _read_cache()
    cache[broker] = {
        "token": token,
        "expiry": payload.get("expiry"),
        "updated": payload.get("updated"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(cache)
    return token


def refresh_tokens() -> None:
    _request_api("/refresh", "POST")
    cache = _read_cache()
    for broker in list(cache):
        if broker in ALLOWED_BROKERS:
            cache.pop(broker, None)
    if cache:
        _write_cache(cache)
    time.sleep(5)


def _assert_read_only_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise TokenError("broker data URL must be HTTPS")
    lowered = f"{parsed.path}?{parsed.query}".lower()
    blocked = ("place_order", "cancel_order", "order/create", "order/cancel", "trade", "buy", "sell")
    if any(value in lowered for value in blocked):
        raise TokenError("write-like broker endpoint is disabled")


def get_authorized_get(url: str, broker: str, headers: dict[str, str] | None = None, timeout: int = 20) -> bytes:
    broker = _normalise_broker(broker)
    _assert_read_only_url(url)
    token = get_token(broker)
    for attempt in range(3):
        request_headers = dict(headers or {})
        request_headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=request_headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            if error.code in AUTH_HTTP_CODES and attempt < 2:
                invalidate_token(broker)
                token = fresh_token(broker)
                time.sleep(2)
                continue
            raise TokenError(f"broker API returned HTTP {error.code}") from None
        except URLError as error:
            raise TokenError("broker API is unavailable") from error
    raise TokenError("broker API retry failed")


DHAN_HISTORICAL_URLS = {
    "https://api.dhan.co/v2/charts/intraday",
    "https://api.dhan.co/v2/charts/historical",
}


# Only genuinely auth-related codes trigger a token refresh. DH-905/DH-907 are
# request/data errors and must not cause a refresh loop.
DHAN_INVALID_TOKEN_CODES = {"DH-906"}


def _is_dhan_auth_failure(status: int, body: bytes) -> bool:
    if status in AUTH_HTTP_CODES:
        return True
    if status == 400:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        if str(payload.get("errorCode")) in DHAN_INVALID_TOKEN_CODES:
            return True
        return "invalid token" in str(payload.get("errorMessage", "")).lower()
    return False


def post_dhan_historical(url: str, payload: dict[str, Any], timeout: int = 25) -> bytes:
    if url not in DHAN_HISTORICAL_URLS:
        raise TokenError("only Dhan historical-data endpoints are allowed")
    if any("order" in str(key).lower() for key in payload):
        raise TokenError("order-like payload is disabled")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token = get_token("dhan")
    for attempt in range(3):
        request = Request(
            url,
            data=body,
            headers={
                "access-token": token,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            body_bytes = error.read()
            if _is_dhan_auth_failure(error.code, body_bytes) and attempt < 2:
                invalidate_token("dhan")
                token = fresh_token("dhan")
                time.sleep(2)
                continue
            raise TokenError(f"Dhan API returned HTTP {error.code}") from None
        except URLError as error:
            raise TokenError("Dhan API is unavailable") from error
    raise TokenError("Dhan API retry failed")
