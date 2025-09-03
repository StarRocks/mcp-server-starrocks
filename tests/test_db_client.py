"""
Tests for db_client module.

These tests assume a StarRocks cluster is running on localhost with default configurations:
- Host: localhost
- Port: 9030 (MySQL protocol)
- User: root
- Password: (empty)
- No default database set

Run tests with: pytest tests/test_db_client.py -v
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Set up test environment variables
os.environ.pop('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT', None)  # Force MySQL mode for tests
os.environ.pop('STARROCKS_DB', None)  # No default database

from src.mcp_server_starrocks.db_client import (
    DBClient, 
    ResultSet, 
    get_db_client, 
    reset_db_connections
)


class TestDBClient:
    """Test cases for DBClient class."""
    
    @pytest.fixture
    def db_client(self):
        """Create a fresh DBClient instance for each test."""
        # Reset global state
        reset_db_connections()
        return DBClient()
    
    def test_client_initialization(self, db_client):
        """Test DBClient initialization with default settings."""
        assert db_client.enable_arrow_flight_sql is False
        assert db_client.default_database is None
        assert db_client._connection_pool is None
        assert db_client._adbc_connection is None
    
    def test_singleton_pattern(self):
        """Test that get_db_client returns the same instance."""
        client1 = get_db_client()
        client2 = get_db_client()
        assert client1 is client2
    
    def test_execute_show_databases(self, db_client):
        """Test executing SHOW DATABASES query."""
        result = db_client.execute("SHOW DATABASES")
        
        assert isinstance(result, ResultSet)
        assert result.success is True
        assert result.column_names is not None
        assert len(result.column_names) == 1
        assert result.rows is not None
        assert len(result.rows) > 0
        assert result.execution_time is not None
        assert result.execution_time > 0
        
        # Check that information_schema is present (standard in StarRocks)
        database_names = [row[0] for row in result.rows]
        assert 'information_schema' in database_names
    
    def test_execute_show_databases_pandas(self, db_client):
        """Test executing SHOW DATABASES with pandas return format."""
        result = db_client.execute("SHOW DATABASES", return_format="pandas")
        
        assert isinstance(result, ResultSet)
        assert result.success is True
        assert result.pandas is not None
        assert isinstance(result.pandas, pd.DataFrame)
        assert len(result.pandas.columns) == 1
        assert len(result.pandas) > 0
        
        # Test that to_pandas() returns the same DataFrame
        df = result.to_pandas()
        assert df is result.pandas
    
    def test_execute_invalid_query(self, db_client):
        """Test executing an invalid SQL query."""
        result = db_client.execute("SELECT * FROM nonexistent_table_12345")
        
        assert isinstance(result, ResultSet)
        assert result.success is False
        assert result.error_message is not None
        assert "nonexistent_table_12345" in result.error_message or "doesn't exist" in result.error_message.lower()
        assert result.execution_time is not None
    
    def test_execute_create_and_drop_database(self, db_client):
        """Test creating and dropping a test database."""
        test_db_name = "test_mcp_db_client"
        
        # Clean up first (in case previous test failed)
        db_client.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
        
        # Create database
        create_result = db_client.execute(f"CREATE DATABASE {test_db_name}")
        assert create_result.success is True
        assert create_result.rows_affected is not None  # DDL returns row count (usually 0)
        
        # Verify database exists
        show_result = db_client.execute("SHOW DATABASES")
        database_names = [row[0] for row in show_result.rows]
        assert test_db_name in database_names
        
        # Drop database
        drop_result = db_client.execute(f"DROP DATABASE {test_db_name}")
        assert drop_result.success is True
        
        # Verify database is gone
        show_result = db_client.execute("SHOW DATABASES")
        database_names = [row[0] for row in show_result.rows]
        assert test_db_name not in database_names
    
    def test_execute_with_specific_database(self, db_client):
        """Test executing query with specific database context."""
        # Use information_schema which should always be available
        result = db_client.execute("SHOW TABLES", db="information_schema")
        
        assert result.success is True
        assert result.column_names is not None
        assert result.rows is not None
        assert len(result.rows) > 0  # information_schema should have tables
        
        # Check for expected information_schema tables
        table_names = [row[0] for row in result.rows]
        expected_tables = ['tables', 'columns', 'schemata']
        found_expected = any(table in table_names for table in expected_tables)
        assert found_expected, f"Expected at least one of {expected_tables} in {table_names}"
    
    def test_execute_with_invalid_database(self, db_client):
        """Test executing query with non-existent database."""
        result = db_client.execute("SHOW TABLES", db="nonexistent_db_12345")
        
        assert result.success is False
        assert result.error_message is not None
        assert "nonexistent_db_12345" in result.error_message
    
    def test_execute_table_operations(self, db_client):
        """Test creating, inserting, querying, and dropping a table."""
        test_db = "test_mcp_table_ops"
        test_table = "test_table"
        
        try:
            # Create database
            create_db_result = db_client.execute(f"CREATE DATABASE IF NOT EXISTS {test_db}")
            assert create_db_result.success is True
            
            # Create table (with replication_num=1 for single-node setup)
            create_table_sql = f"""
            CREATE TABLE {test_db}.{test_table} (
                id INT,
                name STRING,
                value DOUBLE
            )
            PROPERTIES ("replication_num" = "1")
            """
            create_result = db_client.execute(create_table_sql)
            assert create_result.success is True
            
            # Insert data
            insert_sql = f"""
            INSERT INTO {test_db}.{test_table} VALUES 
            (1, 'test1', 1.5),
            (2, 'test2', 2.5),
            (3, 'test3', 3.5)
            """
            insert_result = db_client.execute(insert_sql)
            assert insert_result.success is True
            assert insert_result.rows_affected == 3
            
            # Query data
            select_result = db_client.execute(f"SELECT * FROM {test_db}.{test_table} ORDER BY id")
            assert select_result.success is True
            assert len(select_result.column_names) == 3
            assert select_result.column_names == ['id', 'name', 'value']
            assert len(select_result.rows) == 3
            # MySQL connector returns tuples, convert to lists for comparison
            assert list(select_result.rows[0]) == [1, 'test1', 1.5]
            assert list(select_result.rows[1]) == [2, 'test2', 2.5]
            assert list(select_result.rows[2]) == [3, 'test3', 3.5]
            
            # Test COUNT query
            count_result = db_client.execute(f"SELECT COUNT(*) as cnt FROM {test_db}.{test_table}")
            assert count_result.success is True
            assert count_result.rows[0][0] == 3
            
            # Test with specific database context
            ctx_result = db_client.execute(f"SELECT * FROM {test_table}", db=test_db)
            assert ctx_result.success is True
            assert len(ctx_result.rows) == 3
            
        finally:
            # Clean up
            db_client.execute(f"DROP DATABASE IF EXISTS {test_db}")
    
    def test_execute_pandas_format_with_data(self, db_client):
        """Test pandas format with actual data."""
        test_db = "test_mcp_pandas"
        
        try:
            # Setup test data
            db_client.execute(f"CREATE DATABASE IF NOT EXISTS {test_db}")
            db_client.execute(f"""
                CREATE TABLE {test_db}.pandas_test (
                    id INT,
                    category STRING,
                    amount DECIMAL(10,2)
                )
                PROPERTIES ("replication_num" = "1")
            """)
            db_client.execute(f"""
                INSERT INTO {test_db}.pandas_test VALUES 
                (1, 'A', 100.50),
                (2, 'B', 200.75),
                (3, 'A', 150.25)
            """)
            
            # Test executing query with pandas format
            result = db_client.execute(f"SELECT * FROM {test_db}.pandas_test ORDER BY id", return_format="pandas")
            
            assert isinstance(result, ResultSet)
            assert result.success is True
            assert result.pandas is not None
            assert isinstance(result.pandas, pd.DataFrame)
            assert len(result.pandas) == 3
            assert list(result.pandas.columns) == ['id', 'category', 'amount']
            assert result.pandas.iloc[0]['id'] == 1
            assert result.pandas.iloc[0]['category'] == 'A'
            assert float(result.pandas.iloc[0]['amount']) == 100.50
            
            # Test that to_pandas() returns the same DataFrame
            df = result.to_pandas()
            assert df is result.pandas
        
        finally:
            db_client.execute(f"DROP DATABASE IF EXISTS {test_db}")
    
    def test_connection_error_handling(self, db_client):
        """Test error handling when connection fails."""
        # Mock a connection failure
        with patch.object(db_client, '_get_connection', side_effect=Exception("Connection failed")):
            result = db_client.execute("SHOW DATABASES")
            
            assert result.success is False
            assert "Connection failed" in result.error_message
            assert result.execution_time is not None
    
    def test_reset_connections(self, db_client):
        """Test connection reset functionality."""
        # First execute a query to establish connection
        result1 = db_client.execute("SHOW DATABASES")
        assert result1.success is True
        
        # Reset connections
        db_client.reset_connections()
        
        # Should still work after reset
        result2 = db_client.execute("SHOW DATABASES")
        assert result2.success is True
    
    def test_describe_table(self, db_client):
        """Test DESCRIBE table functionality."""
        test_db = "test_mcp_describe"
        test_table = "describe_test"
        
        try:
            # Create test table
            db_result = db_client.execute(f"CREATE DATABASE IF NOT EXISTS {test_db}")
            assert db_result.success, f"Failed to create database: {db_result.error_message}"
            
            table_result = db_client.execute(f"""
                CREATE TABLE {test_db}.{test_table} (
                    id BIGINT NOT NULL COMMENT 'Primary key',
                    name VARCHAR(100) COMMENT 'Name field',
                    created_at DATETIME,
                    is_active BOOLEAN
                )
                PROPERTIES ("replication_num" = "1")
            """)
            assert table_result.success, f"Failed to create table: {table_result.error_message}"
            
            # Verify table exists first
            show_result = db_client.execute(f"SHOW TABLES", db=test_db)
            assert show_result.success, f"Failed to show tables: {show_result.error_message}"
            table_names = [row[0] for row in show_result.rows]
            assert test_table in table_names, f"Table {test_table} not found in {table_names}"
            
            # Describe table (use full table name for clarity)
            result = db_client.execute(f"DESCRIBE {test_db}.{test_table}")
            
            assert result.success is True
            assert result.column_names is not None
            assert len(result.rows) == 4  # 4 columns
            
            # Check column names in result (should include Field, Type, etc.)
            expected_columns = ['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']
            for expected_col in expected_columns[:len(result.column_names)]:
                assert expected_col in result.column_names
            
            # Check that our table columns are present
            field_names = [row[0] for row in result.rows]
            assert 'id' in field_names
            assert 'name' in field_names
            assert 'created_at' in field_names
            assert 'is_active' in field_names
        
        finally:
            db_client.execute(f"DROP DATABASE IF EXISTS {test_db}")


class TestDBClientWithArrowFlight:
    """Test cases for DBClient with Arrow Flight SQL (if configured)."""
    
    @pytest.fixture
    def arrow_client(self):
        """Create DBClient with Arrow Flight SQL if available."""
        # Check if Arrow Flight SQL port is configured (either from env or default test port)
        arrow_port = os.getenv('STARROCKS_FE_ARROW_FLIGHT_SQL_PORT', '9408')
        
        # Test if Arrow Flight SQL is actually available by trying to connect
        try:
            with patch.dict(os.environ, {'STARROCKS_FE_ARROW_FLIGHT_SQL_PORT': arrow_port}):
                reset_db_connections()
                client = DBClient()
                assert client.enable_arrow_flight_sql is True
                
                # Test basic connectivity
                result = client.execute("SHOW DATABASES")
                if not result.success:
                    pytest.skip(f"Arrow Flight SQL not available on port {arrow_port}: {result.error_message}")
                
                return client
        except Exception as e:
            pytest.skip(f"Arrow Flight SQL not available: {e}")
    
    def test_arrow_flight_basic_query(self, arrow_client):
        """Test basic query with Arrow Flight SQL."""
        result = arrow_client.execute("SHOW DATABASES")
        
        assert isinstance(result, ResultSet)
        assert result.success is True
        assert result.column_names is not None
        assert result.rows is not None
        assert len(result.rows) > 0
        
        # Verify we're actually using Arrow Flight SQL
        assert arrow_client.enable_arrow_flight_sql is True
    
    def test_arrow_flight_pandas_format(self, arrow_client):
        """Test pandas format with Arrow Flight SQL."""
        result = arrow_client.execute("SHOW DATABASES", return_format="pandas")
        
        assert isinstance(result, ResultSet)
        assert result.success is True
        assert result.pandas is not None
        assert isinstance(result.pandas, pd.DataFrame)
        assert len(result.pandas) > 0
        assert len(result.pandas.columns) == 1
        
        # Test that to_pandas() returns the same DataFrame
        df = result.to_pandas()
        assert df is result.pandas
        
        # Verify we're actually using Arrow Flight SQL
        assert arrow_client.enable_arrow_flight_sql is True
    
    def test_arrow_flight_table_operations(self, arrow_client):
        """Test table operations with Arrow Flight SQL."""
        test_db = "test_arrow_flight"
        test_table = "arrow_test"
        
        try:
            # Create database
            create_db_result = arrow_client.execute(f"CREATE DATABASE IF NOT EXISTS {test_db}")
            assert create_db_result.success is True
            
            # Create table
            create_table_sql = f"""
            CREATE TABLE {test_db}.{test_table} (
                id INT,
                name STRING,
                value DOUBLE
            )
            PROPERTIES ("replication_num" = "1")
            """
            create_result = arrow_client.execute(create_table_sql)
            assert create_result.success is True
            
            # Insert data
            insert_sql = f"""
            INSERT INTO {test_db}.{test_table} VALUES 
            (1, 'arrow1', 1.1),
            (2, 'arrow2', 2.2)
            """
            insert_result = arrow_client.execute(insert_sql)
            assert insert_result.success is True
            assert insert_result.rows_affected == 2
            
            # Query data with pandas format
            select_result = arrow_client.execute(f"SELECT * FROM {test_db}.{test_table} ORDER BY id", return_format="pandas")
            assert isinstance(select_result, ResultSet)
            assert select_result.success is True
            assert select_result.pandas is not None
            assert isinstance(select_result.pandas, pd.DataFrame)
            assert len(select_result.pandas) == 2
            assert list(select_result.pandas.columns) == ['id', 'name', 'value']
            assert select_result.pandas.iloc[0]['id'] == 1
            assert select_result.pandas.iloc[0]['name'] == 'arrow1'
            assert select_result.pandas.iloc[0]['value'] == 1.1
            
            # Test that to_pandas() returns the same DataFrame
            df = select_result.to_pandas()
            assert df is select_result.pandas
            
            # Query data with raw format
            raw_result = arrow_client.execute(f"SELECT * FROM {test_db}.{test_table} ORDER BY id")
            assert raw_result.success is True
            assert len(raw_result.rows) == 2
            assert raw_result.column_names == ['id', 'name', 'value']
            
        finally:
            # Clean up
            arrow_client.execute(f"DROP DATABASE IF EXISTS {test_db}")
    
    def test_arrow_flight_error_handling(self, arrow_client):
        """Test error handling with Arrow Flight SQL."""
        # Test invalid query
        result = arrow_client.execute("SELECT * FROM nonexistent_arrow_table")
        assert result.success is False
        assert result.error_message is not None
        
        # Test invalid database
        result = arrow_client.execute("SHOW TABLES", db="nonexistent_arrow_db")
        assert result.success is False
        assert "nonexistent_arrow_db" in result.error_message


class TestResultSet:
    """Test cases for ResultSet dataclass."""
    
    def test_result_set_creation(self):
        """Test ResultSet creation with various parameters."""
        # Success case
        result = ResultSet(
            success=True,
            column_names=['id', 'name'],
            rows=[[1, 'test'], [2, 'test2']],
            execution_time=0.5
        )
        
        assert result.success is True
        assert result.column_names == ['id', 'name']
        assert result.rows == [[1, 'test'], [2, 'test2']]
        assert result.execution_time == 0.5
        assert result.rows_affected is None
        assert result.error_message is None
    
    def test_result_set_to_pandas_from_rows(self):
        """Test ResultSet to_pandas conversion from rows."""
        result = ResultSet(
            success=True,
            column_names=['id', 'name', 'value'],
            rows=[[1, 'test1', 10.5], [2, 'test2', 20.5]],
            execution_time=0.1
        )
        
        df = result.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']
        assert df.iloc[0]['id'] == 1
        assert df.iloc[0]['name'] == 'test1'
        assert df.iloc[0]['value'] == 10.5
        assert df.iloc[1]['id'] == 2
        assert df.iloc[1]['name'] == 'test2'
        assert df.iloc[1]['value'] == 20.5
    
    def test_result_set_to_pandas_from_pandas_field(self):
        """Test ResultSet to_pandas returns existing pandas field if available."""
        original_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['test1', 'test2'],
            'value': [10.5, 20.5]
        })
        
        result = ResultSet(
            success=True,
            column_names=['id', 'name', 'value'],
            rows=[[1, 'test1', 10.5], [2, 'test2', 20.5]],
            pandas=original_df,
            execution_time=0.1
        )
        
        df = result.to_pandas()
        assert df is original_df  # Should return the same object
    
    def test_result_set_to_string(self):
        """Test ResultSet to_string conversion."""
        result = ResultSet(
            success=True,
            column_names=['id', 'name', 'value'],
            rows=[[1, 'test1', 10.5], [2, 'test2', 20.5]],
            execution_time=0.1
        )
        
        string_output = result.to_string()
        expected_lines = [
            'id,name,value',
            '1,test1,10.5',
            '2,test2,20.5',
            ''
        ]
        assert string_output == '\n'.join(expected_lines)
    
    def test_result_set_to_string_with_limit(self):
        """Test ResultSet to_string with limit."""
        result = ResultSet(
            success=True,
            column_names=['id', 'name'],
            rows=[[1, 'very_long_test_string'], [2, 'another_long_string']],
            execution_time=0.1
        )
        
        # Test with very small limit
        string_output = result.to_string(limit=20)
        lines = string_output.split('\n')
        assert lines[0] == 'id,name'  # Header should always be included
        # Should stop before all rows due to limit
        assert len(lines) < 4  # Should be less than header + 2 rows + empty line
    
    def test_result_set_to_string_error_cases(self):
        """Test ResultSet to_string error handling."""
        # Test with failed result
        failed_result = ResultSet(
            success=False,
            error_message="Test error"
        )
        
        string_output = failed_result.to_string()
        assert string_output == "Error: Test error"
        
        # Test with no data
        no_data_result = ResultSet(
            success=True,
            column_names=None,
            rows=None
        )
        
        string_output = no_data_result.to_string()
        assert string_output == "No data"
    
    def test_result_set_to_pandas_error_cases(self):
        """Test ResultSet to_pandas error handling."""
        # Test with failed result
        failed_result = ResultSet(
            success=False,
            error_message="Test error"
        )
        
        with pytest.raises(ValueError, match="Cannot convert failed result to DataFrame"):
            failed_result.to_pandas()
        
        # Test with no data
        no_data_result = ResultSet(
            success=True,
            column_names=None,
            rows=None
        )
        
        with pytest.raises(ValueError, match="No data available to convert to DataFrame"):
            no_data_result.to_pandas()
    
    def test_result_set_error_case(self):
        """Test ResultSet for error cases."""
        result = ResultSet(
            success=False,
            error_message="Test error",
            execution_time=0.1
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.execution_time == 0.1
        assert result.column_names is None
        assert result.rows is None
        assert result.rows_affected is None
    
    def test_result_set_write_operation(self):
        """Test ResultSet for write operations."""
        result = ResultSet(
            success=True,
            rows_affected=5,
            execution_time=0.2
        )
        
        assert result.success is True
        assert result.rows_affected == 5
        assert result.execution_time == 0.2
        assert result.column_names is None
        assert result.rows is None
        assert result.error_message is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])