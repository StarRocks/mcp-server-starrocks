"""
Unit tests for the identifier/path/UUID validators in db_client, and for the
server.py / db_summary_manager call sites that must reject a malicious value
before it reaches SQL text (issue #33).

These tests are pure-function / mocked-db_client tests and do NOT require a
running StarRocks cluster.

Run with: pytest tests/test_sql_identifier_validation.py -v
"""

from unittest.mock import MagicMock

import pytest

from src.mcp_server_starrocks.db_client import (
    validate_sql_identifier,
    validate_proc_path,
    validate_query_uuid,
)
from src.mcp_server_starrocks import server
from src.mcp_server_starrocks.db_summary_manager import DatabaseSummaryManager


class TestValidateSqlIdentifier:
    @pytest.mark.parametrize("value", ["db1", "_db", "My_Table", "a" * 5])
    def test_accepts_plain_identifiers(self, value):
        assert validate_sql_identifier(value, "database") == value

    @pytest.mark.parametrize("value", [
        "db; DROP TABLE secrets;",
        "db`; DROP TABLE secrets;#",
        "db.other",
        "db name",
        "",
        None,
    ])
    def test_rejects_anything_else(self, value):
        with pytest.raises(ValueError):
            validate_sql_identifier(value, "database")


class TestValidateProcPath:
    @pytest.mark.parametrize("value", ["/backends", "/dbs/10001/10002/partitions", ""])
    def test_accepts_path_shaped_values(self, value):
        assert validate_proc_path(value) == value

    @pytest.mark.parametrize("value", [
        "x' UNION SELECT password FROM mysql.user#",
        "/backends'; DROP TABLE secrets;#",
        "/backends\\",
    ])
    def test_rejects_anything_with_sql_metacharacters(self, value):
        with pytest.raises(ValueError):
            validate_proc_path(value)


class TestValidateQueryUuid:
    def test_accepts_documented_format(self):
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_query_uuid(uuid) == uuid

    @pytest.mark.parametrize("value", [
        "x' UNION SELECT password FROM mysql.user#",
        "550e8400-e29b-41d4-a716-44665544000",  # one digit short
        "not-a-uuid",
    ])
    def test_rejects_anything_else(self, value):
        with pytest.raises(ValueError):
            validate_query_uuid(value)


def _mock_db_client():
    captured = []

    def fake_execute(query, *args, **kwargs):
        captured.append(query)
        result = MagicMock()
        result.to_string.return_value = "<ok>"
        return result

    client = MagicMock()
    client.execute = fake_execute
    return client, captured


class TestServerResourceAndToolGuards:
    """The four fault sites named in issue #33 must reject a malicious value
    before db_client.execute is ever called, and must still build the exact
    same SQL as before for a legitimate value."""

    def setup_method(self):
        self.original_db_client = server.db_client
        server.db_client, self.captured = _mock_db_client()

    def teardown_method(self):
        server.db_client = self.original_db_client

    def test_get_table_schema_rejects_malicious_table(self):
        with pytest.raises(ValueError):
            server.get_table_schema(db="mydb", table="x`; DROP TABLE secrets;#")
        assert self.captured == []

    def test_get_table_schema_legitimate(self):
        server.get_table_schema(db="mydb", table="mytable")
        assert self.captured == ["SHOW CREATE TABLE mydb.mytable"]

    def test_get_database_tables_rejects_malicious_db(self):
        with pytest.raises(ValueError):
            server.get_database_tables(db="mydb; DROP TABLE secrets;#")
        assert self.captured == []

    def test_get_database_tables_legitimate(self):
        server.get_database_tables(db="mydb")
        assert self.captured == ["SHOW TABLES FROM mydb"]

    def test_get_system_internal_information_rejects_malicious_path(self):
        with pytest.raises(ValueError):
            server.get_system_internal_information(path="x' UNION SELECT password FROM mysql.user#")
        assert self.captured == []

    def test_get_system_internal_information_legitimate(self):
        server.get_system_internal_information(path="/backends")
        assert self.captured == ["show proc '/backends'"]

    def test_analyze_query_rejects_malicious_uuid(self):
        with pytest.raises(ValueError):
            server.analyze_query(uuid="x' UNION SELECT password FROM mysql.user#")
        assert self.captured == []

    def test_analyze_query_legitimate(self):
        server.analyze_query(uuid="550e8400-e29b-41d4-a716-446655440000")
        assert self.captured == ["ANALYZE PROFILE FROM '550e8400-e29b-41d4-a716-446655440000'"]


class TestDbSummaryManagerGuard:
    def test_get_database_summary_rejects_malicious_database_before_any_sql(self):
        client, captured = _mock_db_client()
        manager = DatabaseSummaryManager(client)
        result = manager.get_database_summary("mydb' UNION SELECT password FROM mysql.user#", refresh=True)
        assert captured == []
        assert result.startswith("Error:")

    def test_get_database_summary_legitimate(self):
        client, captured = _mock_db_client()

        def fake_execute(query, *args, **kwargs):
            captured.append(query)
            result = MagicMock()
            result.success = True
            result.rows = [["mytable", "1.0KB", "3"]] if query == "SHOW DATA" else []
            return result

        client.execute = fake_execute
        manager = DatabaseSummaryManager(client)
        manager.get_database_summary("mydb", refresh=True)
        assert captured[0] == "SHOW DATA"
