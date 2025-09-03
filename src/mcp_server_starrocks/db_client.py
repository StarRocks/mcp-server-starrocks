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
from typing import Optional, List, Any, Union, Literal
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
                break
            output.write(line)
        return output.getvalue()


class DBClient:
    """Simplified database client for StarRocks connection and query execution."""
    
    def __init__(self):
        self.enable_arrow_flight_sql = bool(os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT'))
        self.default_database = os.getenv('STARROCKS_DB')
        
        # MySQL connection pool
        self._connection_pool = None
        
        # ADBC connection (singleton)
        self._adbc_connection = None
    
    def _get_connection_pool(self):
        """Get or create a connection pool for MySQL connections."""
        if self._connection_pool is None:
            connection_params = {
                'host': os.getenv('STARROCKS_HOST', 'localhost'),
                'port': int(os.getenv('STARROCKS_PORT', '9030')),
                'user': os.getenv('STARROCKS_USER', 'root'),
                'password': os.getenv('STARROCKS_PASSWORD', ''),
                'auth_plugin': os.getenv('STARROCKS_MYSQL_AUTH_PLUGIN', 'mysql_native_password'),
                'pool_size': int(os.getenv('STARROCKS_POOL_SIZE', '10')),
                'pool_name': 'mcp_starrocks_pool',
                'pool_reset_session': True,
                'autocommit': True,
                'connection_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
                'connect_timeout': int(os.getenv('STARROCKS_CONNECTION_TIMEOUT', '10')),
            }
            
            if self.default_database:
                connection_params['database'] = self.default_database
            
            try:
                self._connection_pool = mysql.connector.pooling.MySQLConnectionPool(**connection_params)
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
        fe_host = os.getenv('STARROCKS_HOST', 'localhost')
        fe_port = os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT', '')
        user = os.getenv('STARROCKS_USER', 'root')
        password = os.getenv('STARROCKS_PASSWORD', '')
        
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
        conn = None
        cursor = None
        start_time = time.time()
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
                        execution_time=time.time() - start_time
                    )
                cursor_temp.close()
            cursor = conn.cursor()
            cursor.execute(statement)
            # Handle result set queries (SELECT, SHOW, DESCRIBE, etc.)
            if cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                pandas_df = None
                
                if self.enable_arrow_flight_sql:
                    arrow_result = cursor.fetchallarrow()
                    if return_format == "pandas":
                        pandas_df = arrow_result.to_pandas()
                        rows = pandas_df.values.tolist()
                    else:
                        pandas_df = None
                        rows = arrow_result.to_pandas().values.tolist()
                else:
                    rows = cursor.fetchall()
                    if return_format == "pandas":
                        pandas_df = pd.DataFrame(rows, columns=column_names)
                    else:
                        pandas_df = None
                
                return ResultSet(
                    success=True,
                    column_names=column_names,
                    rows=rows,
                    execution_time=time.time() - start_time,
                    pandas=pandas_df
                )
            else:
                affected_rows = cursor.rowcount if cursor.rowcount >= 0 else None
                return ResultSet(
                    success=True,
                    rows_affected=affected_rows,
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