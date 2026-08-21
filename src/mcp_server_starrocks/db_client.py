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

import io
import os
import time
import re
import json
import threading
from contextlib import nullcontext
from typing import Optional, List, Any, Union, Literal, TypedDict, NotRequired
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error as MySQLError
import adbc_driver_manager
import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_manager import Error as adbcError
import pandas as pd
from .secret_resolver import resolve_password


# JavaScript Number.MAX_SAFE_INTEGER (2^53 - 1)
# Integers beyond this range lose precision in JSON when parsed by JavaScript clients.
MAX_SAFE_INTEGER = 2**53 - 1

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROC_PATH_RE = re.compile(r"^[A-Za-z0-9_/-]*$")
_QUERY_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def validate_sql_identifier(value: str, kind: str) -> str:
    """Reject anything but a plain identifier before it is interpolated unquoted into SQL."""
    if not value or not _SQL_IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {kind} name: {value!r}")
    return value


def validate_proc_path(value: str) -> str:
    """Reject anything but path-shaped characters before interpolating into show proc '...'."""
    if not _PROC_PATH_RE.match(value):
        raise ValueError(f"Invalid proc path: {value!r}")
    return value


def validate_query_uuid(value: str) -> str:
    """Reject anything not matching the documented 8-4-4-4-12 hex query id format."""
    if not _QUERY_UUID_RE.match(value):
        raise ValueError(f"Invalid query UUID: {value!r}")
    return value


def _safe_json_value(v):
    """Convert integers outside JavaScript's safe integer range to strings.

    JavaScript ``Number`` is IEEE 754 double-precision (53-bit mantissa).
    Integers with absolute value larger than ``2**53 - 1`` silently lose
    precision when a JS client parses the JSON.  Converting them to strings
    preserves the exact value.
    """
    if isinstance(v, int) and not isinstance(v, bool) and (
        v > MAX_SAFE_INTEGER or v < -MAX_SAFE_INTEGER
    ):
        return str(v)
    return v


@dataclass
class ResultSet:
    """Database query result set."""
    success: bool
    column_names: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    rows_affected: Optional[int] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    pandas: Optional[pd.DataFrame] = None
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert ResultSet to pandas DataFrame."""
        if self.pandas is not None:
            return self.pandas
            
        if not self.success:
            raise ValueError(f"Cannot convert failed result to DataFrame: {self.error_message}")
        
        if self.column_names is None or self.rows is None:
            raise ValueError("No data available to convert to DataFrame")
            
        return pd.DataFrame(self.rows, columns=self.column_names)
    
    def to_string(self, limit: Optional[int] = None) -> str:
        """Format rows as CSV-like string with column names as first row."""
        if not self.success:
            return f"Error: {self.error_message}"
        if self.column_names is None or self.rows is None:
            return "No data"
        def to_csv_line(row):
            return ",".join(
                str(item).replace("\"", "\"\"") if isinstance(item, str) else str(item) for item in row)
        output = io.StringIO()
        output.write(to_csv_line(self.column_names) + "\n")
        for row in self.rows:
            line = to_csv_line(row) + "\n"
            if limit is not None and output.tell() + len(line) > limit:
                output.write("...\n")
                break
            output.write(line)
        output.write(f"Total rows: {len(self.rows)}\n")
        output.write(f"Execution time: {self.execution_time:.3f}s\n");
        return output.getvalue()

    def to_dict(self) -> dict:
        ret = {
            "success": self.success,
            "execution_time": self.execution_time,
        }
        if self.column_names is not None:
            ret["column_names"] = self.column_names
            ret["rows"] = [
                [_safe_json_value(v) for v in row] for row in self.rows
            ]
        if self.rows_affected is not None:
            ret["rows_affected"] = self.rows_affected
        if self.error_message:
            ret["error_message"] = self.error_message
        return ret


class PerfAnalysisInput(TypedDict):
    error_message: NotRequired[Optional[str]]
    query_id: NotRequired[Optional[str]]
    rows_returned: NotRequired[Optional[int]]
    duration: NotRequired[Optional[float]]
    query_dump: NotRequired[Optional[dict]]
    profile: NotRequired[Optional[str]]
    analyze_profile: NotRequired[Optional[str]]


def _parse_connection_url_details(connection_url: str) -> tuple[dict, bool]:
    """
    Parse connection URL into dict with user, password, host, port, database.
    
    Supports flexible formats:
    - [<schema>://]<user>[:<password>]@<host>[:<port>][/<database>]
    - Empty passwords: user:@host:port or user@host:port  
    - Missing ports (uses default 9030): user:pass@host
    - All components are optional except user and host
    """
    # More flexible regex pattern that handles optional password and port
    pattern = re.compile(
        r'^(?:(?P<schema>[\w+]+)://)?'           # Optional schema://
        r'(?P<user>[^:@]+)'                     # Required username (no : or @)
        r'(?::(?P<password>[^@]*))?'            # Optional :password (can be empty)
        r'@(?P<host>[^:/]+)'                    # Required @host 
        r'(?::(?P<port>\d+))?'                  # Optional :port
        r'(?:/(?P<database>[\w-]+))?$'          # Optional /database
    )
    
    match = pattern.match(connection_url)
    if not match:
        raise ValueError(f"Invalid connection URL: {connection_url}")
    
    result = match.groupdict()
    password_provided = result['password'] is not None

    # Only keep connection parameters that mysql.connector supports
    # Filter out None values and schema (which is not a mysql.connector parameter)
    filtered_result = {}

    # Always include user and host as they are required
    filtered_result['user'] = result['user']
    filtered_result['host'] = result['host']

    # Include password (default to empty string if None)
    filtered_result['password'] = result['password'] if result['password'] is not None else ''

    # Include port (default to 9030 if None)
    filtered_result['port'] = result['port'] if result['port'] is not None else '9030'

    # Always include database (None if not provided in URL)
    filtered_result['database'] = result['database']

    # Note: schema is intentionally excluded as it's not supported by mysql.connector

    return filtered_result, password_provided


def parse_connection_url(connection_url: str) -> dict:
    """Parse connection URL into dict with user, password, host, port, database."""
    filtered_result, _ = _parse_connection_url_details(connection_url)
    return filtered_result

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('true', '1', 'yes', 'on')


def _build_mysql_ssl_options() -> dict:
    """
    Build TLS/SSL connection options for mysql.connector from environment variables.

    Supported environment variables:
    - STARROCKS_SSL_DISABLED:        'true' to force-disable TLS (overrides everything else).
    - STARROCKS_SSL_CA:              path to the CA certificate (PEM) used to verify the server.
    - STARROCKS_SSL_CERT:            path to the client certificate (PEM) for mutual TLS.
    - STARROCKS_SSL_KEY:             path to the client private key (PEM) for mutual TLS.
    - STARROCKS_SSL_VERIFY_CERT:     'true' to verify the server certificate against the CA.
    - STARROCKS_SSL_VERIFY_IDENTITY: 'true' to verify the server hostname against the certificate.
    - STARROCKS_TLS_VERSIONS:        comma-separated list, e.g. "TLSv1.2,TLSv1.3".

    Notes:
    - When none of these are set, no SSL keys are injected and mysql.connector keeps its
      default behavior (ssl-mode PREFERRED: encrypted if the server supports it, no verification).
    - For real security, set STARROCKS_SSL_CA together with STARROCKS_SSL_VERIFY_CERT=true
      (and STARROCKS_SSL_VERIFY_IDENTITY=true to also check the hostname).
    """
    if _env_flag('STARROCKS_SSL_DISABLED', False):
        return {'ssl_disabled': True}

    opts: dict = {}
    ssl_ca = os.getenv('STARROCKS_SSL_CA')
    ssl_cert = os.getenv('STARROCKS_SSL_CERT')
    ssl_key = os.getenv('STARROCKS_SSL_KEY')
    if ssl_ca:
        opts['ssl_ca'] = ssl_ca
    if ssl_cert:
        opts['ssl_cert'] = ssl_cert
    if ssl_key:
        opts['ssl_key'] = ssl_key
    if _env_flag('STARROCKS_SSL_VERIFY_CERT', False):
        opts['ssl_verify_cert'] = True
    if _env_flag('STARROCKS_SSL_VERIFY_IDENTITY', False):
        opts['ssl_verify_identity'] = True

    tls_versions = os.getenv('STARROCKS_TLS_VERSIONS')
    if tls_versions:
        versions = [v.strip() for v in tls_versions.split(',') if v.strip()]
        if versions:
            opts['tls_versions'] = versions

    return opts


def _build_flight_sql_tls(fe_host: str, fe_port: str) -> tuple:
    """
    Build the Arrow Flight SQL URI and TLS db_kwargs from environment variables.

    Supported environment variables:
    - STARROCKS_FE_ARROW_FLIGHT_SQL_USE_TLS: 'true' to use grpc+tls:// instead of plaintext grpc://.
    - STARROCKS_SSL_CA:                      path to the CA certificate (PEM) used as TLS root certs.
    - STARROCKS_SSL_VERIFY_CERT:             when 'false' (default), skip server certificate verification.

    Returns a tuple of (uri, tls_db_kwargs).
    """
    use_tls = _env_flag('STARROCKS_FE_ARROW_FLIGHT_SQL_USE_TLS', False)
    if not use_tls:
        return f"grpc://{fe_host}:{fe_port}", {}

    # Option keys defined by the ADBC Flight SQL driver.
    tls_skip_verify_key = "adbc.flight.sql.client_option.tls_skip_verify"
    tls_root_certs_key = "adbc.flight.sql.client_option.tls_root_certs"

    tls_db_kwargs: dict = {}
    verify_cert = _env_flag('STARROCKS_SSL_VERIFY_CERT', False)
    tls_db_kwargs[tls_skip_verify_key] = "false" if verify_cert else "true"

    ssl_ca = os.getenv('STARROCKS_SSL_CA')
    if ssl_ca:
        try:
            with open(ssl_ca, 'r') as f:
                tls_db_kwargs[tls_root_certs_key] = f.read()
        except OSError as e:
            print(f"Warning: could not read STARROCKS_SSL_CA '{ssl_ca}': {e}")

    return f"grpc+tls://{fe_host}:{fe_port}", tls_db_kwargs


ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def remove_ansi_codes(text):
  return ANSI_ESCAPE_PATTERN.sub('', text)


def _resolve_connection_password(user: str, explicit_password: str, explicit_password_provided: bool) -> str:
    """Resolve password from explicit config first, then optional Keychain lookup."""
    if not explicit_password_provided and 'STARROCKS_PASSWORD' in os.environ:
        explicit_password = os.environ['STARROCKS_PASSWORD']
        explicit_password_provided = True

    return resolve_password(
        user=user,
        explicit_password=explicit_password,
        explicit_password_provided=explicit_password_provided
    )


class DBClient:
    """Simplified database client for StarRocks connection and query execution."""
    
    def __init__(self):
        self.enable_dummy_test = bool(os.getenv('STARROCKS_DUMMY_TEST'))
        self.enable_arrow_flight_sql = bool(os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT'))
        if os.getenv('STARROCKS_URL'):
            self.connection_params, url_password_provided = _parse_connection_url_details(os.getenv('STARROCKS_URL'))
            self._password_resolution = {
                'user': self.connection_params['user'],
                'explicit_password': self.connection_params['password'],
                'explicit_password_provided': url_password_provided,
            }
            # Convert port to integer for mysql.connector
            self.connection_params['port'] = int(self.connection_params['port'])
        else:
            user = os.getenv('STARROCKS_USER', 'root')
            explicit_password = os.getenv('STARROCKS_PASSWORD', '')
            explicit_password_provided = 'STARROCKS_PASSWORD' in os.environ
            self.connection_params = {
                'host': os.getenv('STARROCKS_HOST', 'localhost'),
                'port': int(os.getenv('STARROCKS_PORT', '9030')),
                'user': user,
                'password': explicit_password,
                'database': os.getenv('STARROCKS_DB', None),
            }
            self._password_resolution = {
                'user': user,
                'explicit_password': explicit_password,
                'explicit_password_provided': explicit_password_provided,
            }
        self.connection_params.update(**{
            'auth_plugin': os.getenv('STARROCKS_MYSQL_AUTH_PLUGIN', 'mysql_native_password'),
            'pool_size': int(os.getenv('STARROCKS_POOL_SIZE', '10')),
            'pool_name': 'mcp_starrocks_pool',
            'pool_reset_session': True,
            'autocommit': True,
            'connection_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
            'connect_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
            'use_pure': os.getenv('STARROCKS_USE_PURE', 'false').lower() in ('true', '1', 'yes'),
        })
        # Apply optional TLS/SSL options for the MySQL protocol connection.
        self.connection_params.update(_build_mysql_ssl_options())
        self.default_database = self.connection_params.get('database')

        # MySQL connection pool
        self._connection_pool = None

        # ADBC connection (singleton). The shared connection is not safe for
        # concurrent use, so all access is serialized through _adbc_lock.
        self._adbc_connection = None
        self._adbc_lock = threading.Lock()

        # Per-MCP-session default database, keyed by FastMCP session_id.
        # Falls back to self.default_database when no entry is set.
        self._session_default_dbs: dict[str, str] = {}
        self._session_dbs_lock = threading.Lock()

    def get_session_default_db(self, session_id: Optional[str]) -> Optional[str]:
        """Return the effective default db for a session, falling back to the global default."""
        if session_id:
            with self._session_dbs_lock:
                if session_id in self._session_default_dbs:
                    return self._session_default_dbs[session_id]
        return self.default_database

    def set_session_default_db(self, session_id: str, db: Optional[str]) -> None:
        """Set or clear the per-session default db. Pass db=None to clear."""
        with self._session_dbs_lock:
            if db is None:
                self._session_default_dbs.pop(session_id, None)
            else:
                self._session_default_dbs[session_id] = db

    def _resolve_target_db(self, session_id: Optional[str], db_param: Optional[str]) -> Optional[str]:
        """Resolve the database to USE for a call: explicit param > session default > global default."""
        if db_param:
            return db_param
        return self.get_session_default_db(session_id)

    def _get_connection_params(self):
        """Return connection parameters with password resolved only when needed."""
        params = dict(self.connection_params)
        params['password'] = _resolve_connection_password(**self._password_resolution)
        return params
    
    def _get_connection_pool(self):
        """Get or create a connection pool for MySQL connections."""
        if self._connection_pool is None:
            try:
                self._connection_pool = mysql.connector.pooling.MySQLConnectionPool(**self._get_connection_params())
            except MySQLError as conn_err:
                raise conn_err
        
        return self._connection_pool
    
    def _validate_connection(self, conn):
        """Validate that a MySQL connection is still alive and working."""
        try:
            conn.ping(reconnect=True, attempts=1, delay=0)
            return True
        except MySQLError:
            return False
    
    def _get_pooled_connection(self):
        """Get a MySQL connection from the pool with timeout and retry logic."""
        pool = self._get_connection_pool()
        try:
            conn = pool.get_connection()
            if not self._validate_connection(conn):
                conn.close()
                conn = pool.get_connection()
            return conn
        except mysql.connector.errors.PoolError as pool_err:
            if "Pool is exhausted" in str(pool_err):
                time.sleep(0.1)
                try:
                    return pool.get_connection()
                except mysql.connector.errors.PoolError:
                    self._connection_pool = None
                    new_pool = self._get_connection_pool()
                    return new_pool.get_connection()
            raise pool_err
    
    def _create_adbc_connection(self):
        """Create a new ADBC connection."""
        connection_params = self._get_connection_params()
        fe_host = connection_params['host']
        fe_port = os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT', '')
        user = connection_params['user']
        password = connection_params['password']
        
        uri, tls_db_kwargs = _build_flight_sql_tls(fe_host, fe_port)
        db_kwargs = {
            adbc_driver_manager.DatabaseOptions.USERNAME.value: user,
            adbc_driver_manager.DatabaseOptions.PASSWORD.value: password,
        }
        db_kwargs.update(tls_db_kwargs)

        try:
            connection = flight_sql.connect(
                uri=uri,
                db_kwargs=db_kwargs,
            )
            
            # Switch to default database if set
            if self.default_database:
                try:
                    cursor = connection.cursor()
                    cursor.execute(f"USE {self.default_database}")
                    cursor.close()
                except adbcError as db_err:
                    print(f"Warning: Could not switch to default database '{self.default_database}': {db_err}")
            
            return connection
        except adbcError:
            print(f"Error creating ADBC connection: {adbcError}")
            raise
    
    def _get_adbc_connection(self):
        """Get or create an ADBC connection with health check."""
        if self._adbc_connection is None:
            self._adbc_connection = self._create_adbc_connection()
        
        # Health check for ADBC connection
        if self._adbc_connection is not None:
            try:
                self._adbc_connection.adbc_get_info()
            except adbcError as check_err:
                print(f"Connection check failed: {check_err}, creating new ADBC connection.")
                self._reset_adbc_connection()
                self._adbc_connection = self._create_adbc_connection()
        
        return self._adbc_connection
    
    def _get_connection(self):
        """Get appropriate connection based on configuration."""
        if self.enable_arrow_flight_sql:
            return self._get_adbc_connection()
        else:
            return self._get_pooled_connection()
    
    def _reset_adbc_connection(self):
        """Reset ADBC connection."""
        if self._adbc_connection is not None:
            try:
                self._adbc_connection.close()
            except Exception as e:
                print(f"Error closing ADBC connection: {e}")
            finally:
                self._adbc_connection = None
    
    def _reset_connection(self):
        """Reset connections based on configuration."""
        if self.enable_arrow_flight_sql:
            self._reset_adbc_connection()
        else:
            self._connection_pool = None
    
    def _handle_db_error(self, error):
        """Handle database errors and reset connections as needed."""
        if not self.enable_arrow_flight_sql and ("MySQL Connection not available" in str(error) or "Lost connection" in str(error)):
            self._connection_pool = None
        elif self.enable_arrow_flight_sql:
            self._reset_adbc_connection()


    def _execute(self, conn, statement: str, params=None, return_format:str="raw") -> ResultSet:
        cursor = None
        start_time = time.time()
        try:
            cursor = conn.cursor()
            cursor.execute(statement, params)
            # Initialize variables to track the last result set
            last_result = None
            last_affected_rows = None
            # Process first result set
            if cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                if self.enable_arrow_flight_sql:
                    arrow_result = cursor.fetchallarrow()
                    pandas_df = arrow_result.to_pandas() if return_format == "pandas" else None
                    rows = arrow_result.to_pandas().values.tolist()

                    # Check if this is a status result for DML operations (INSERT/UPDATE/DELETE)
                    # Arrow Flight SQL returns status results as a single column 'StatusResult'
                    # Note: StarRocks Arrow Flight SQL seems to always return '0' in StatusResult,
                    # so we use cursor.rowcount when available as a fallback
                    if (len(column_names) == 1 and column_names[0] == 'StatusResult' and
                            len(rows) == 1 and len(rows[0]) == 1):
                        try:
                            status_value = int(rows[0][0])
                            # If status_value is 0 but we have cursor.rowcount, prefer that
                            if status_value == 0 and hasattr(cursor, 'rowcount') and cursor.rowcount > 0:
                                last_affected_rows = cursor.rowcount
                            else:
                                last_affected_rows = status_value
                            last_result = None  # Don't treat this as a regular result set
                        except (ValueError, TypeError):
                            # If we can't parse the status result as an integer, treat it as a regular result
                            last_result = ResultSet(
                                success=True,
                                column_names=column_names,
                                rows=rows,
                                execution_time=0,  # Will be set at the end
                                pandas=pandas_df
                            )
                    else:
                        last_result = ResultSet(
                            success=True,
                            column_names=column_names,
                            rows=rows,
                            execution_time=0,  # Will be set at the end
                            pandas=pandas_df
                        )
                else:
                    rows = cursor.fetchall()
                    pandas_df = pd.DataFrame(rows, columns=column_names) if return_format == "pandas" else None

                    last_result = ResultSet(
                        success=True,
                        column_names=column_names,
                        rows=rows,
                        execution_time=0,  # Will be set at the end
                        pandas=pandas_df
                    )
            else:
                last_affected_rows = cursor.rowcount if cursor.rowcount >= 0 else None
            # Process additional result sets (for multi-statement queries)
            # Note: Arrow Flight SQL may not support nextset(), so we check for it
            if not self.enable_arrow_flight_sql and hasattr(cursor, 'nextset'):
                while cursor.nextset():
                    if cursor.description:
                        column_names = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()
                        pandas_df = pd.DataFrame(rows, columns=column_names) if return_format == "pandas" else None

                        last_result = ResultSet(
                            success=True,
                            column_names=column_names,
                            rows=rows,
                            execution_time=0,  # Will be set at the end
                            pandas=pandas_df
                        )
                    else:
                        last_affected_rows = cursor.rowcount if cursor.rowcount >= 0 else None
                        last_result = None
            # Return the last result set found
            if last_result is not None:
                last_result.execution_time = time.time() - start_time
                return last_result
            else:
                return ResultSet(
                    success=True,
                    rows_affected=last_affected_rows,
                    execution_time=time.time() - start_time
                )
        except (MySQLError, adbcError) as e:
            self._handle_db_error(e)
            return ResultSet(
                success=False,
                error_message=f"Error executing statement '{statement}': {str(e)}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ResultSet(
                success=False,
                error_message=f"Unexpected error executing statement '{statement}': {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass


    def execute(
        self,
        statement: str,
        db: Optional[str] = None,
        return_format: Literal["raw", "pandas"] = "raw",
        session_id: Optional[str] = None,
    ) -> ResultSet:
        """
        Execute a SQL statement and return results.

        Args:
            statement: SQL statement to execute
            db: Optional database to USE before this statement (overrides session/global default)
            return_format: "raw" returns ResultSet with rows, "pandas" also populates pandas field
            session_id: FastMCP session id; resolves the per-session default db when ``db`` is None

        Returns:
            ResultSet with column_names and rows, optionally with pandas DataFrame
        """
        # If dummy test mode is enabled, return dummy data without connecting to database
        if self.enable_dummy_test:
            column_names = ['name']
            rows = [['aaa'], ['bbb'], ['ccc']]
            pandas_df = None

            if return_format == "pandas":
                pandas_df = pd.DataFrame(rows, columns=column_names)

            return ResultSet(
                success=True,
                column_names=column_names,
                rows=rows,
                execution_time=0.1,
                pandas=pandas_df
            )

        target_db = self._resolve_target_db(session_id, db)
        # ADBC uses a singleton connection shared across MCP sessions; serialize
        # USE+query as one atomic block so concurrent callers don't see each
        # other's USE state. Pool mode hands out a private connection per call,
        # so no extra lock is needed there.
        outer_lock = self._adbc_lock if self.enable_arrow_flight_sql else nullcontext()
        with outer_lock:
            conn = None
            try:
                conn = self._get_connection()
                # Always USE the resolved target db. The pool's
                # pool_reset_session=True is not reliable for the current schema
                # on StarRocks, and the ADBC singleton outright leaks state, so
                # we re-set it on every call.
                if target_db:
                    cursor_temp = conn.cursor()
                    try:
                        cursor_temp.execute(f"USE `{target_db}`")
                    except (MySQLError, adbcError) as db_err:
                        cursor_temp.close()
                        return ResultSet(
                            success=False,
                            error_message=f"Error switching to database '{target_db}': {str(db_err)}",
                            execution_time=0
                        )
                    cursor_temp.close()
                return self._execute(conn, statement, None, return_format)
            except (MySQLError, adbcError) as e:
                self._handle_db_error(e)
                return ResultSet(
                    success=False,
                    error_message=f"Error executing statement '{statement}': {str(e)}",
                )
            except Exception as e:
                return ResultSet(
                    success=False,
                    error_message=f"Unexpected error executing statement '{statement}': {str(e)}",
                )
            finally:
                if conn and not self.enable_arrow_flight_sql:
                    try:
                        conn.close()
                    except:
                        pass

    def collect_perf_analysis_input(self, query: str, db: Optional[str] = None,
                                    session_id: Optional[str] = None) -> PerfAnalysisInput:
        target_db = self._resolve_target_db(session_id, db)
        outer_lock = self._adbc_lock if self.enable_arrow_flight_sql else nullcontext()
        with outer_lock:
            return self._collect_perf_analysis_input_locked(query, target_db)

    def _collect_perf_analysis_input_locked(self, query: str, target_db: Optional[str]) -> PerfAnalysisInput:
        conn = None
        try:
            conn = self._get_connection()
            # Always USE the resolved target db (see execute() for rationale).
            if target_db:
                cursor_temp = conn.cursor()
                try:
                    cursor_temp.execute(f"USE `{target_db}`")
                except (MySQLError, adbcError) as db_err:
                    return {"error_message": str(db_err)}
                finally:
                    cursor_temp.close()
            query_dump_result = self._execute(conn, "select get_query_dump(%s, %s)", (query, False))
            if not query_dump_result.success:
                return {"error_message":query_dump_result.error_message}
            ret = {
                "query_dump": json.loads(query_dump_result.rows[0][0]),
            }
            start_ts = time.time()
            profile_query = "/*+ SET_VAR (enable_profile='true') */ " + query
            query_result = self._execute(conn, profile_query)
            duration = time.time() - start_ts
            ret["duration"] = duration
            if not query_result.success:
                ret["error_message"] = query_result.error_message
                return ret
            ret["rows_returned"] = len(query_result.rows) if query_result.rows else 0
            # Try to get query id
            query_id_result = self._execute(conn, "select last_query_id()")
            if not query_id_result.success:
                ret["error_message"] = query_id_result.error_message
                return ret
            ret["query_id"] = query_id_result.rows[0][0]
            # Try to get query profile with retries
            query_profile = ''
            retry_count = 0
            while not query_profile and retry_count < 3:
                time.sleep(1+retry_count)
                query_profile_result = self._execute(conn,"select get_query_profile(%s)", (ret["query_id"],))
                if query_profile_result.success:
                    query_profile = query_profile_result.rows[0][0]
                retry_count += 1
            if not query_profile:
                ret['error_message'] = "Failed to get query profile after 3 retries"
                return ret
            ret['profile'] = query_profile
            analyze_profile_result = self._execute(conn,"ANALYZE PROFILE FROM %s", (ret["query_id"],))
            if not analyze_profile_result.success:
                ret["error_message"] = analyze_profile_result.error_message
                return ret
            analyze_text = '\n'.join(row[0] for row in analyze_profile_result.rows)
            ret['analyze_profile'] = remove_ansi_codes(analyze_text)
            return ret
        except (MySQLError, adbcError) as e:
            self._handle_db_error(e)
            return {"error_message":str(e)}
        except Exception as e:
            return {"error_message":str(e)}
        finally:
            if conn and not self.enable_arrow_flight_sql:
                try:
                    conn.close()
                except:
                    pass

    def reset_connections(self):
        """Public method to reset all connections."""
        self._reset_connection()


# Global singleton instance
_db_client_instance: Optional[DBClient] = None


def get_db_client() -> DBClient:
    """Get or create the global DBClient instance."""
    global _db_client_instance
    if _db_client_instance is None:
        _db_client_instance = DBClient()
    return _db_client_instance


def reset_db_connections():
    """Reset all database connections (useful for error recovery)."""
    global _db_client_instance
    if _db_client_instance is not None:
        _db_client_instance.reset_connections()
