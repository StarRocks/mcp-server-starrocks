# Copyright 2021-present StarRocks, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-MCP-session login for StarRocks users created with ``IDENTIFIED WITH authentication_oauth2``.

StarRocks runs the OAuth2 authorization-code flow over the MySQL protocol for clients that
cannot open a browser: the handshake is accepted, and every statement is refused with error
5207 carrying the authorization URL until the user signs in through that URL. The IdP then
redirects to the FE (``/api/oauth2``), which binds the token to that one connection.

The login therefore lives on a single connection, so in this mode each MCP session gets a
dedicated connection instead of one from the shared pool, and keeps it until it ends.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from mysql.connector import Error as MySQLError
from mysql.connector.connection import MySQLConnection

OAUTH2_LOGIN_REQUIRED_ERRNO = 5207
_LOGIN_URL_RE = re.compile(r"URL \[(\S+?)\]")


def extract_login_url(error) -> Optional[str]:
    """Return the authorization URL of a StarRocks 'OAuth2 login required' error, else None.

    Accepts a MySQL error, or the text of one as kept in ``ResultSet.error_message``.
    """
    if error is None:
        return None
    if isinstance(error, str):
        if str(OAUTH2_LOGIN_REQUIRED_ERRNO) not in error:
            return None
        text = error
    else:
        if getattr(error, 'errno', None) != OAUTH2_LOGIN_REQUIRED_ERRNO:
            return None
        text = getattr(error, 'msg', None) or str(error)
    match = _LOGIN_URL_RE.search(text)
    return match.group(1) if match else None


class NoStableMCPSession(Exception):
    """The MCP request carries nothing that identifies its client from one request to the next."""


def current_mcp_session_key() -> Optional[str]:
    """Key that follows the client of the MCP request being served, or None outside of one.

    On handshake-era connections this is the MCP session id. 2026-07-28 connections have no
    session, and every request gets a new ``session_id``; FastMCP scopes them to the
    authenticated user instead, or to the process on stdio. An unauthenticated 2026-07-28
    request over HTTP has no such scope, so ``NoStableMCPSession`` is raised.
    """
    try:
        from fastmcp.server.dependencies import get_context
        ctx = get_context()
    except Exception:
        return None
    scope_key = getattr(ctx, '_session_scope_key', None)  # FastMCP >= 4
    try:
        key = scope_key() if scope_key else ctx.session_id
    except RuntimeError:
        return None
    if key is None:
        raise NoStableMCPSession(
            "This MCP client uses the stateless 2026-07-28 protocol over HTTP without "
            "authentication, so nothing ties its requests to one StarRocks login. Use the stdio "
            "transport, or set STARROCKS_OAUTH2_SESSION_SCOPE=server for a single-user server.")
    return key


class OAuth2GatedConnection(MySQLConnection):
    """A connection that survives the OAuth2 gate on its first statement.

    mysql-connector runs SET NAMES and SET autocommit right after the handshake. For an OAuth2
    user that is the first statement StarRocks refuses, which would fail the whole connect.
    That setup is deferred instead, and applied by ``finish_login`` once the user has signed in.
    """

    login_url: Optional[str] = None

    def _post_connection(self) -> None:
        try:
            super()._post_connection()
        except MySQLError as e:
            url = extract_login_url(e)
            if url is None:
                raise
            self.login_url = url

    def finish_login(self) -> bool:
        """Apply the deferred setup. Returns True once StarRocks accepts statements."""
        if self.login_url is None:
            return True
        try:
            super()._post_connection()
        except MySQLError as e:
            url = extract_login_url(e)
            if url is None:
                raise
            self.login_url = url
            return False
        self.login_url = None
        return True


@dataclass
class OAuth2Session:
    user: str
    conn: OAuth2GatedConnection
    # One connection per session, so statements of a session run one at a time.
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)

    def login_required_message(self) -> str:
        return (f"StarRocks OAuth2 login required for user '{self.user}'. "
                f"Open this URL in a browser and sign in, then retry: {self.conn.login_url}")


class OAuth2SessionManager:
    """Owns the dedicated connection of each MCP session, keyed by MCP session id."""

    def __init__(self, connect_params: dict, idle_timeout: int):
        self._connect_params = connect_params
        self._idle_timeout = idle_timeout
        self._sessions: dict[str, OAuth2Session] = {}
        self._lock = threading.Lock()

    def _connect(self, user: str) -> OAuth2GatedConnection:
        return OAuth2GatedConnection(user=user, password='', **self._connect_params)

    def start(self, key: str, user: str) -> OAuth2Session:
        """Open a new connection for ``user`` on session ``key``, replacing any previous one."""
        session = OAuth2Session(user=user, conn=self._connect(user))
        with self._lock:
            previous = self._sessions.get(key)
            self._sessions[key] = session
        if previous is not None:
            _close_quietly(previous.conn)
        self.evict_idle()
        logger.info(f"Started OAuth2 session for user '{user}'")
        return session

    def get(self, key: str) -> Optional[OAuth2Session]:
        self.evict_idle()
        with self._lock:
            session = self._sessions.get(key)
        if session is not None:
            session.last_used = time.time()
        return session

    def end(self, key: str) -> Optional[OAuth2Session]:
        with self._lock:
            session = self._sessions.pop(key, None)
        if session is not None:
            _close_quietly(session.conn)
        return session

    def evict_idle(self) -> None:
        cutoff = time.time() - self._idle_timeout
        with self._lock:
            stale = [k for k, s in self._sessions.items() if s.last_used < cutoff]
            evicted = [self._sessions.pop(k) for k in stale]
        for session in evicted:
            logger.info(f"Closing idle OAuth2 session of user '{session.user}'")
            _close_quietly(session.conn)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            _close_quietly(session.conn)


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass
