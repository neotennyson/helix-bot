import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import tokens


class _FakeHTTPError(HTTPError):
    def __init__(self, base: HTTPError, body: bytes) -> None:
        super().__init__(base.url, base.code, base.msg, base.hdrs, base.fp)
        self._body = body

    def read(self, *args: object) -> bytes:
        return self._body


class TokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.previous_cache = tokens.CACHE_PATH
        tokens.CACHE_PATH = Path(self.directory.name) / ".broker_tokens.json"
        self.environment = patch.dict(os.environ, {"TOKEN_API_KEY": "unit-test-key"})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        tokens.CACHE_PATH = self.previous_cache
        self.directory.cleanup()

    def test_fetch_is_cached_for_less_than_six_hours(self) -> None:
        expiry = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        payload = {"token": "dhan-token", "expiry": expiry, "updated": "now"}
        with patch.object(tokens, "_request_api", return_value=payload) as request:
            self.assertEqual(tokens.get_token("dhan"), "dhan-token")
            self.assertEqual(tokens.get_token("dhan"), "dhan-token")
        request.assert_called_once_with("/token/dhan", "GET")
        self.assertEqual(stat.S_IMODE(tokens.CACHE_PATH.stat().st_mode), 0o600)
        stored = json.loads(tokens.CACHE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored["dhan"]["token"], "dhan-token")
        self.assertNotIn("TOKEN_API_KEY", stored["dhan"])

    def test_expired_cache_is_refetched(self) -> None:
        expired = {
            "dhan": {
                "token": "old-token",
                "expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "fetched_at": (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat(),
            }
        }
        tokens.CACHE_PATH.write_text(json.dumps(expired), encoding="utf-8")
        payload = {"token": "new-token", "expiry": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()}
        with patch.object(tokens, "_request_api", return_value=payload) as request:
            self.assertEqual(tokens.get_token("dhan"), "new-token")
        request.assert_called_once_with("/token/dhan", "GET")

    def test_token_expiry_overrides_cache_age(self) -> None:
        entry = {
            "dhan": {
                "token": "expired-token",
                "expiry": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        tokens.CACHE_PATH.write_text(json.dumps(entry), encoding="utf-8")
        payload = {"token": "replacement", "expiry": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()}
        with patch.object(tokens, "_request_api", return_value=payload):
            self.assertEqual(tokens.get_token("dhan"), "replacement")

    def test_local_env_key_is_read_without_being_printed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            temporary = Path(self.directory.name)
            (temporary / ".env").write_text("TOKEN_API_KEY=local-unit-key\n", encoding="utf-8")
            with patch.object(tokens, "__file__", str(temporary / "tokens.py")):
                self.assertEqual(tokens._api_key(), "local-unit-key")

    def test_authorized_get_uses_fresh_token_after_unauthorized(self) -> None:
        error = HTTPError("https://broker.example/history", 401, "unauthorized", {}, None)
        response = type("Response", (), {"__enter__": lambda self: self, "__exit__": lambda *args: None, "read": lambda self: b"ok"})()
        with patch.object(tokens, "get_token", return_value="old"), patch.object(tokens, "fresh_token", return_value="new") as fresh, patch.object(tokens, "invalidate_token") as invalidate, patch.object(tokens, "urlopen", side_effect=[error, response]):
            result = tokens.get_authorized_get("https://broker.example/history", "groww")
        self.assertEqual(result, b"ok")
        fresh.assert_called_once_with("groww")
        invalidate.assert_called_once_with("groww")

    def test_dhan_post_rejects_invalid_token_response_with_fresh_token(self) -> None:
        error = HTTPError("https://api.dhan.co/v2/charts/historical", 400, "bad request", {}, None)
        error_body = json.dumps({"errorCode": "DH-906", "errorMessage": "Invalid Token"}).encode("utf-8")
        response = type("Response", (), {"__enter__": lambda self: self, "__exit__": lambda *args: None, "read": lambda self: b'{"open": []}'})()
        with patch.object(tokens, "get_token", return_value="old"), patch.object(tokens, "fresh_token", return_value="new") as fresh, patch.object(tokens, "invalidate_token"), patch.object(tokens, "urlopen", side_effect=[_FakeHTTPError(error, error_body), response]):
            result = tokens.post_dhan_historical("https://api.dhan.co/v2/charts/historical", {"securityId": "1"})
        self.assertEqual(result, b'{"open": []}')
        fresh.assert_called_once_with("dhan")

    def test_write_like_broker_url_is_blocked(self) -> None:
        with self.assertRaises(tokens.TokenError):
            tokens.get_authorized_get("https://broker.example/place_order", "dhan")
    def test_dhan_post_is_limited_to_historical_endpoints(self) -> None:
        with self.assertRaises(tokens.TokenError):
            tokens.post_dhan_historical("https://api.dhan.co/v2/orders", {})
        with self.assertRaises(tokens.TokenError):
            tokens.post_dhan_historical("https://api.dhan.co/v2/charts/intraday", {"orderType": "BUY"})


if __name__ == "__main__":
    unittest.main()
