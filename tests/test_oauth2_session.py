"""
Unit tests for the OAuth2 login mode (STARROCKS_AUTH_MODE=oauth2).

These tests do NOT require a running StarRocks cluster or identity provider: the
StarRocks connection is replaced with a fake that refuses statements with error 5207
until it is told the browser sign-in happened.

Run with: pytest tests/test_oauth2_session.py -v
"""

import os
import contextlib
import pytest
from unittest.mock import patch

from mysql.connector import errors as mysql_errors

from src.mcp_server_starrocks import oauth2_session
from src.mcp_server_starrocks.oauth2_session import (
    OAuth2GatedConnection,
    OAuth2SessionManager,
    extract_login_url,
)
from src.mcp_server_starrocks.db_client import DBClient

LOGIN_URL = ("https://idp.example.com/oauth2/authorize?response_type=code&client_id=abc"
             "&redirect_uri=https%3A%2F%2Ffe.example.com%2Fapi%2Foauth2&state=42&scope=openid")
GATE_MSG = f"Access denied; Need to access the URL [{LOGIN_URL}] to complete the OAuth2 authentication operation"

ENV_KEYS = [
    "STARROCKS_AUTH_MODE", "STARROCKS_OAUTH2_SESSION_SCOPE", "STARROCKS_OAUTH2_IDLE_TIMEOUT",
    "STARROCKS_USER", "STARROCKS_URL", "STARROCKS_FE_ARROW_FLIGHT_SQL_PORT", "STARROCKS_DUMMY_TEST",
]


@contextlib.contextmanager
def oauth2_env(**overrides):
    """Clear the env vars this mode reads, apply only the given overrides, then restore."""
    saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def gate_error():
    return mysql_errors.ProgrammingError(msg=GATE_MSG, errno=5207, sqlstate="42000")


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self.rowcount = 0

    def execute(self, statement, params=None):
        if not self.conn.signed_in:
            raise gate_error()
        self.conn.statements.append(statement)
        if statement.lower().startswith("select"):
            self.description = [("current_user()",)]

    def fetchall(self):
        return [(f"'{self.conn.user}'@'%'",)]

    def nextset(self):
        return None

    def close(self):
        pass


class FakeConnection:
    """Stands in for OAuth2GatedConnection: gated until ``signed_in`` is set."""

    def __init__(self, user):
        self.user = user
        self.signed_in = False
        self.login_url = LOGIN_URL
        self.statements = []
        self.closed = False

    def finish_login(self):
        if self.signed_in:
            self.login_url = None
        return self.signed_in

    def cursor(self):
        return FakeCursor(self)

    def is_connected(self):
        return not self.closed

    def close(self):
        self.closed = True


class TestExtractLoginUrl:
    def test_from_error(self):
        assert extract_login_url(gate_error()) == LOGIN_URL

    def test_from_error_message_text(self):
        assert extract_login_url(f"Error executing statement 'select 1': 5207 (42000): {GATE_MSG}") == LOGIN_URL

    def test_other_errors_are_not_login_requests(self):
        other = mysql_errors.ProgrammingError(msg="Access denied for user", errno=1045)
        assert extract_login_url(other) is None
        assert extract_login_url("1045 (28000): Access denied") is None
        assert extract_login_url(None) is None


class TestOAuth2GatedConnection:
    def _bare(self):
        return OAuth2GatedConnection.__new__(OAuth2GatedConnection)

    def test_gate_on_first_statement_is_kept_not_raised(self):
        conn = self._bare()
        with patch("mysql.connector.connection.MySQLConnection._post_connection", side_effect=gate_error()):
            conn._post_connection()
        assert conn.login_url == LOGIN_URL

    def test_other_errors_still_fail_the_connect(self):
        conn = self._bare()
        other = mysql_errors.ProgrammingError(msg="boom", errno=1105)
        with patch("mysql.connector.connection.MySQLConnection._post_connection", side_effect=other):
            with pytest.raises(mysql_errors.ProgrammingError):
                conn._post_connection()

    def test_finish_login_until_signed_in(self):
        conn = self._bare()
        conn.login_url = LOGIN_URL
        with patch("mysql.connector.connection.MySQLConnection._post_connection", side_effect=gate_error()):
            assert conn.finish_login() is False
        assert conn.login_url == LOGIN_URL
        with patch("mysql.connector.connection.MySQLConnection._post_connection") as post:
            assert conn.finish_login() is True
            assert conn.finish_login() is True
        assert conn.login_url is None
        post.assert_called_once()


class TestOAuth2SessionManager:
    def _manager(self, idle_timeout=3600):
        manager = OAuth2SessionManager(connect_params={}, idle_timeout=idle_timeout)
        manager._connect = FakeConnection
        return manager

    def test_sessions_are_isolated(self):
        manager = self._manager()
        a = manager.start("a", "alice")
        b = manager.start("b", "bob")
        assert manager.get("a") is a and manager.get("b") is b
        assert a.conn is not b.conn

    def test_start_again_replaces_and_closes_previous(self):
        manager = self._manager()
        first = manager.start("a", "alice")
        second = manager.start("a", "bob")
        assert first.conn.closed and not second.conn.closed
        assert manager.get("a").user == "bob"

    def test_end_closes(self):
        manager = self._manager()
        session = manager.start("a", "alice")
        assert manager.end("a") is session
        assert session.conn.closed
        assert manager.get("a") is None
        assert manager.end("a") is None

    def test_idle_sessions_are_evicted(self):
        manager = self._manager(idle_timeout=60)
        session = manager.start("a", "alice")
        session.last_used -= 120
        manager.evict_idle()
        assert manager.get("a") is None
        assert session.conn.closed


class TestDBClientOAuth2Mode:
    def _client(self, **env):
        with oauth2_env(STARROCKS_AUTH_MODE="oauth2", **env):
            client = DBClient()
        client._oauth2._connect = FakeConnection
        return client

    @pytest.fixture(autouse=True)
    def no_mcp_request(self):
        # Outside an MCP request: the login is keyed by the session id passed in.
        with patch.object(oauth2_session, "current_mcp_session_key", return_value=None), \
                patch("src.mcp_server_starrocks.db_client.current_mcp_session_key", return_value=None):
            yield

    def test_password_mode_is_the_default(self):
        with oauth2_env():
            assert DBClient().oauth2_enabled is False

    def test_invalid_settings_are_rejected(self):
        with oauth2_env(STARROCKS_AUTH_MODE="kerberos"):
            with pytest.raises(ValueError):
                DBClient()
        with oauth2_env(STARROCKS_AUTH_MODE="oauth2", STARROCKS_OAUTH2_SESSION_SCOPE="global"):
            with pytest.raises(ValueError):
                DBClient()
        with oauth2_env(STARROCKS_AUTH_MODE="oauth2", STARROCKS_FE_ARROW_FLIGHT_SQL_PORT="9408"):
            with pytest.raises(ValueError):
                DBClient()

    def test_not_signed_in(self):
        result = self._client().execute("select 1", session_id="s1")
        assert not result.success
        assert "oauth2_login" in result.error_message

    def test_login_flow(self):
        client = self._client()
        message = client.oauth2_login("alice", "s1")
        assert LOGIN_URL in message

        gated = client.execute("select current_user()", session_id="s1")
        assert not gated.success and LOGIN_URL in gated.error_message

        client._oauth2.get("s1").conn.signed_in = True  # the browser sign-in
        result = client.execute("select current_user()", session_id="s1")
        assert result.success
        assert result.rows == [("'alice'@'%'",)]

    def test_each_session_uses_its_own_login(self):
        client = self._client()
        client.oauth2_login("alice", "s1")
        client.oauth2_login("bob", "s2")
        for key in ("s1", "s2"):
            client._oauth2.get(key).conn.signed_in = True
        assert client.execute("select current_user()", session_id="s1").rows == [("'alice'@'%'",)]
        assert client.execute("select current_user()", session_id="s2").rows == [("'bob'@'%'",)]
        assert not client.execute("select 1", session_id="s3").success

    def test_server_scope_shares_one_login(self):
        client = self._client(STARROCKS_OAUTH2_SESSION_SCOPE="server")
        client.oauth2_login("alice", "s1")
        client._oauth2.get("server").conn.signed_in = True
        assert client.execute("select current_user()", session_id="s2").rows == [("'alice'@'%'",)]

    def test_default_user_starts_login_on_first_call(self):
        client = self._client(STARROCKS_USER="alice")
        result = client.execute("select 1", session_id="s1")
        assert not result.success and LOGIN_URL in result.error_message
        assert client._oauth2.get("s1").user == "alice"

    def test_default_user_connect_failure_is_reported(self):
        client = self._client(STARROCKS_USER="alice")
        client._oauth2._connect = lambda user: (_ for _ in ()).throw(
            mysql_errors.InterfaceError(msg="Can't connect to MySQL server", errno=2003))
        result = client.execute("select 1", session_id="s1")
        assert not result.success and "Can't connect" in result.error_message

    def test_login_without_user(self):
        assert client_message_has(self._client().oauth2_login(None, "s1"), "STARROCKS_USER")

    def test_uses_target_db_on_session_connection(self):
        client = self._client()
        client.oauth2_login("alice", "s1")
        conn = client._oauth2.get("s1").conn
        conn.signed_in = True
        assert client.execute("select 1", db="sales", session_id="s1").success
        assert conn.statements == ["USE `sales`", "select 1"]
        assert not conn.closed  # the session keeps its connection

    def test_logout(self):
        client = self._client()
        client.oauth2_login("alice", "s1")
        conn = client._oauth2.get("s1").conn
        assert "alice" in client.oauth2_logout("s1")
        assert conn.closed
        assert "No StarRocks login" in client.oauth2_logout("s1")

    def test_stateless_client_without_scope_is_refused(self):
        client = self._client()
        with patch("src.mcp_server_starrocks.db_client.current_mcp_session_key",
                   side_effect=oauth2_session.NoStableMCPSession("stateless")):
            assert client.oauth2_login("alice", "s1").startswith("Error: stateless")
            result = client.execute("select 1", session_id="s1")
        assert not result.success and "stateless" in result.error_message


def client_message_has(message, text):
    return message.startswith("Error") and text in message
