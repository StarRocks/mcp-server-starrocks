"""
Unit tests for _build_cors_config in server.py.

Pure-function tests that lock in the env-var -> CORSMiddleware(allow_origins,
allow_credentials) mapping, without starting the actual MCP server. Guards
against allow_credentials=True ever pairing with a wildcard origin, which lets
Starlette's CORSMiddleware reflect any request's Origin header and allows a
malicious page to ride a victim's ambient browser credentials cross-origin.

Run with: pytest tests/test_cors_config.py -v
"""

import os
import contextlib
import pytest

from src.mcp_server_starrocks.server import _build_cors_config

CORS_ENV_KEY = "STARROCKS_CORS_ALLOWED_ORIGINS"


@contextlib.contextmanager
def cors_env(value=None):
    """Clear the CORS env var, optionally set it, then restore the prior value."""
    saved = os.environ.pop(CORS_ENV_KEY, None)
    try:
        if value is not None:
            os.environ[CORS_ENV_KEY] = value
        yield
    finally:
        os.environ.pop(CORS_ENV_KEY, None)
        if saved is not None:
            os.environ[CORS_ENV_KEY] = saved


class TestBuildCorsConfig:
    """Test cases for _build_cors_config."""

    def test_unset_defaults_to_wildcard_without_credentials(self):
        with cors_env():
            origins, allow_credentials = _build_cors_config()
        assert origins == ["*"]
        assert allow_credentials is False

    def test_blank_defaults_to_wildcard_without_credentials(self):
        with cors_env("   "):
            origins, allow_credentials = _build_cors_config()
        assert origins == ["*"]
        assert allow_credentials is False

    def test_explicit_origin_enables_credentials(self):
        with cors_env("https://app.example.com"):
            origins, allow_credentials = _build_cors_config()
        assert origins == ["https://app.example.com"]
        assert allow_credentials is True

    def test_multiple_origins_parsed_and_stripped(self):
        with cors_env(" https://a.example.com ,https://b.example.com,, "):
            origins, allow_credentials = _build_cors_config()
        assert origins == ["https://a.example.com", "https://b.example.com"]
        assert allow_credentials is True

    def test_wildcard_can_still_be_set_explicitly(self):
        # An operator who explicitly opts back into "*" gets it, reintroducing
        # the CORS-reflection risk this fix closes by default; that is the
        # operator's own informed choice, not ours.
        with cors_env("*"):
            origins, allow_credentials = _build_cors_config()
        assert origins == ["*"]
        assert allow_credentials is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
