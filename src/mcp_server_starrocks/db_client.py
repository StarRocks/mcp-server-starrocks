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
from typing import Optional, List, Any, Union, Literal, TypedDict, NotRequired
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error as MySQLError
import adbc_driver_manager
import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_manager import Error as adbcError
import pandas as pd


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
            ret["rows"] = self.rows
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


def parse_connection_url(connection_url: str) -> dict:
    """
    Parse `(<schema>://)?user:password@host:port/database` into dict with user, password, host, port, database.
    <schema> is optional.
    """
    pattern = re.compile(
        r'^(?:(?P<schema>[\w+]+)://)?'
        r'(?P<user>[^:]+):(?P<password>[^@]+)@'
        r'(?P<host>[^:/]+):(?P<port>\d+)'
        r'(?:/(?P<database>[\w-]+))?$'
    )
    match = pattern.match(connection_url)
    if not match:
        raise ValueError(f"Invalid connection URL: {connection_url}")
    return match.groupdict()

ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def remove_ansi_codes(text):
  return ANSI_ESCAPE_PATTERN.sub('', text)


class DBClient:
    """Simplified database client for StarRocks connection and query execution."""
    
    def __init__(self):
        self.enable_dummy_test = bool(os.getenv('STARROCKS_DUMMY_TEST'))
        self.enable_arrow_flight_sql = bool(os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT'))
        if os.getenv('STARROCKS_URL'):
            self.connection_params = parse_connection_url(os.getenv('STARROCKS_URL'))
        else:
            self.connection_params = {
                'host': os.getenv('STARROCKS_HOST', 'localhost'),
                'port': int(os.getenv('STARROCKS_PORT', '9030')),
                'user': os.getenv('STARROCKS_USER', 'root'),
                'password': os.getenv('STARROCKS_PASSWORD', ''),
                'database': os.getenv('STARROCKS_DB', None),
            }
        self.connection_params.update(**{
            'auth_plugin': os.getenv('STARROCKS_MYSQL_AUTH_PLUGIN', 'mysql_native_password'),
            'pool_size': int(os.getenv('STARROCKS_POOL_SIZE', '10')),
            'pool_name': 'mcp_starrocks_pool',
            'pool_reset_session': True,
            'autocommit': True,
            'connection_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
            'connect_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
        })
        self.default_database = self.connection_params.get('database')

        # MySQL connection pool
        self._connection_pool = None
        
        # ADBC connection (singleton)
        self._adbc_connection = None
    
    def _get_connection_pool(self):
        """Get or create a connection pool for MySQL connections."""
        if self._connection_pool is None:
            try:
                self._connection_pool = mysql.connector.pooling.MySQLConnectionPool(**self.connection_params)
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
        fe_host = self.connection_params['host']
        fe_port = os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT', '')
        user = self.connection_params['user']
        password = self.connection_params['password']
        
        try:
            connection = flight_sql.connect(
                uri=f"grpc://{fe_host}:{fe_port}",
                db_kwargs={
                    adbc_driver_manager.DatabaseOptions.USERNAME.value: user,
                    adbc_driver_manager.DatabaseOptions.PASSWORD.value: password,
                }
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
        return_format: Literal["raw", "pandas"] = "raw"
    ) -> ResultSet:
        """
        Execute a SQL statement and return results.
        
        Args:
            statement: SQL statement to execute
            db: Optional database to use (overrides default)
            return_format: "raw" returns ResultSet with rows, "pandas" also populates pandas field
            
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
        conn = None
        try:
            conn = self._get_connection()
            # Switch database if specified
            if db and db != self.default_database:
                cursor_temp = conn.cursor()
                try:
                    cursor_temp.execute(f"USE `{db}`")
                except (MySQLError, adbcError) as db_err:
                    cursor_temp.close()
                    return ResultSet(
                        success=False,
                        error_message=f"Error switching to database '{db}': {str(db_err)}",
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

    def collect_perf_analysis_input(self, query: str, db:Optional[str]=None) -> PerfAnalysisInput:
        conn = None
        try:
            conn = self._get_connection()
            # Switch database if specified
            if db and db != self.default_database:
                cursor_temp = conn.cursor()
                try:
                    cursor_temp.execute(f"USE `{db}`")
                except (MySQLError, adbcError) as db_err:
                    return {"error_message":str(db_err)}
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