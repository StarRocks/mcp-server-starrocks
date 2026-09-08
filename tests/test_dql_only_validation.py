"""
Unit tests for validate_dql_only and the read_query / query_and_plotly_chart
guards that use it (issue #27): read_query executed arbitrary SQL with no
statement-type enforcement, so an LLM agent (or a prompt injection in
retrieved data) could mutate data through the nominally read-only tool.

These tests are pure-function / mocked-db_client tests and do NOT require a
running StarRocks cluster.

Run with: pytest tests/test_dql_only_validation.py -v
"""

from unittest.mock import MagicMock

import pytest

from src.mcp_server_starrocks.db_client import validate_dql_only
from src.mcp_server_starrocks import server


class TestValidateDqlOnly:
    @pytest.mark.parametrize("value", [
        "SELECT * FROM t",
        "  select 1",
        "--comment\nSELECT 1",
        "/* block comment */ SELECT 1",
        "SHOW TABLES",
        "DESCRIBE t",
        "desc t",
        "EXPLAIN SELECT 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
    ])
    def test_accepts_read_only_statements(self, value):
        assert validate_dql_only(value) == value

    @pytest.mark.parametrize("value", [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "TRUNCATE TABLE t",
        "CREATE TABLE t (a int)",
        "ALTER TABLE t ADD COLUMN b int",
        "--comment\nDELETE FROM t",
        "",
        "   ",
        None,
    ])
    def test_rejects_anything_else(self, value):
        with pytest.raises(ValueError):
            validate_dql_only(value)


def _mock_db_client():
    captured = []

    def fake_execute(query, *args, **kwargs):
        captured.append(query)
        result = MagicMock()
        result.success = True
        result.rows = []
        result.to_string.return_value = "<ok>"
        result.to_dict.return_value = {}
        return result

    client = MagicMock()
    client.execute = fake_execute
    return client, captured


class TestReadQueryGuard:
    def setup_method(self):
        self.original_db_client = server.db_client
        server.db_client, self.captured = _mock_db_client()

    def teardown_method(self):
        server.db_client = self.original_db_client

    def test_rejects_write_before_execute(self):
        with pytest.raises(ValueError):
            server.read_query(query="DELETE FROM users")
        assert self.captured == []

    def test_allows_select(self):
        server.read_query(query="SELECT * FROM users")
        assert self.captured == ["SELECT * FROM users"]


class TestQueryAndPlotlyChartGuard:
    def setup_method(self):
        self.original_db_client = server.db_client
        server.db_client, self.captured = _mock_db_client()

    def teardown_method(self):
        server.db_client = self.original_db_client

    def test_rejects_write_before_execute(self):
        result = server.query_and_plotly_chart(
            query="DROP TABLE users", plotly_expr="px.scatter(df)"
        )
        assert self.captured == []
        assert result.structured_content["success"] is False
