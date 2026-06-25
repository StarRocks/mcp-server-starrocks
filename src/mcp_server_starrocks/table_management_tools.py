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

from datetime import datetime
from typing import Any, Optional


AUDIT_LOG_TABLE = "starrocks_audit_db__.starrocks_audit_tbl__"
SYSTEM_SCHEMAS = ("information_schema", "_statistics_", "sys")


def _sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _format_epoch_millis(value: int) -> str:
    return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def _bounded_limit(value: Optional[int], default: int = 20, maximum: int = 100) -> int:
    if value is None:
        return default
    return max(1, min(int(value), maximum))

def _display_number(value: Any) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number.is_integer():
        return int(number)
    return round(number, 2)


class TableManagementTools:
    def __init__(self, db_client):
        self.db_client = db_client

    def get_top_hot_tables(
        self,
        db: Optional[str] = None,
        table: Optional[str] = None,
        min_start_time_ms: Optional[int] = None,
        max_start_time_ms: Optional[int] = None,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Get top hot tables."""
        limit = _bounded_limit(top_n)
        now = datetime.now()
        sql = self._build_top_hot_tables_sql(
            db=db,
            table=table,
            min_start_time_ms=min_start_time_ms,
            max_start_time_ms=max_start_time_ms,
            top_n=limit,
        )
        result = self.db_client.execute(sql)
        if not result.success:
            return {
                "success": False,
                "error": result.error_message or "Failed to get top hot tables.",
                "analysis_timestamp": now.isoformat(),
                "data_source": AUDIT_LOG_TABLE,
                "top_hot_tables": [],
            }

        column_names = result.column_names or []
        rows = result.rows or []
        top_tables = []
        for index, row in enumerate(rows[:limit]):
            item = dict(zip(column_names, row))
            top_tables.append(
                {
                    "rank": index + 1,
                    "db": item.get("table_schema"),
                    "table": item.get("table_name"),
                    "visit_count": _display_number(item.get("visit_count")),
                }
            )

        return {
            "success": True,
            "analysis_timestamp": now.isoformat(),
            "data_source": AUDIT_LOG_TABLE,
            "filters": {
                "db": db,
                "table": table,
                "min_start_time_ms": min_start_time_ms,
                "max_start_time_ms": max_start_time_ms,
                "top_n": limit,
            },
            "summary": {
                "top_n_requested": limit,
                "top_n_returned": len(top_tables),
                "execution_time": result.execution_time,
            },
            "top_hot_tables": top_tables,
        }

    def get_top_bad_tables(
        self,
        db: Optional[str] = None,
        table: Optional[str] = None,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Get top bad tables."""
        limit = _bounded_limit(top_n)
        now = datetime.now()
        sql = self._build_top_bad_tables_sql(db=db, table=table, top_n=limit)
        result = self.db_client.execute(sql)
        if not result.success:
            return {
                "success": False,
                "error": result.error_message or "Failed to get top bad tables.",
                "analysis_timestamp": now.isoformat(),
                "top_bad_tables": [],
            }

        column_names = result.column_names or []
        rows = result.rows or []
        top_tables = []
        for index, row in enumerate(rows[:limit]):
            item = dict(zip(column_names, row))
            top_tables.append(
                {
                    "rank": index + 1,
                    "table_id": _display_number(item.get("table_id")),
                    "db": item.get("db_name"),
                    "table": item.get("table_name"),
                    "partition_num": _display_number(item.get("partition_num")),
                    "bucket_num": _display_number(item.get("bucket_num")),
                    "replication_num": _display_number(item.get("replication_num")),
                    "tablet_num": _display_number(item.get("tablet_num")),
                    "replica_score": _display_number(item.get("replica_score")),
                    "segment_score": _display_number(item.get("segment_score")),
                    "version_score": _display_number(item.get("version_score")),
                    "tablet_score": _display_number(item.get("tablet_score")),
                    "table_health_score": _display_number(item.get("table_health_score")),
                }
            )

        return {
            "success": True,
            "analysis_timestamp": now.isoformat(),
            "filters": {
                "db": db,
                "table": table,
                "top_n": limit,
            },
            "summary": {
                "top_n_requested": limit,
                "top_n_returned": len(top_tables),
                "execution_time": result.execution_time,
            },
            "top_bad_tables": top_tables,
        }

    @staticmethod
    def _build_top_hot_tables_sql(
        db: Optional[str],
        table: Optional[str],
        min_start_time_ms: Optional[int],
        max_start_time_ms: Optional[int],
        top_n: int,
    ) -> str:
        conditions = [
            "t.table_schema NOT IN ('information_schema', '_statistics_', 'sys')",
        ]
        if db:
            conditions.append(f"t.table_schema = {_sql_literal(db)}")
        if table:
            conditions.append(f"t.table_name LIKE {_sql_literal('%' + table + '%')}")
        if min_start_time_ms is not None and max_start_time_ms is not None:
            conditions.append(
                "a.`timestamp` BETWEEN "
                f"{_sql_literal(_format_epoch_millis(min_start_time_ms))} "
                f"AND {_sql_literal(_format_epoch_millis(max_start_time_ms))}"
            )

        where_sql = "\n AND ".join(conditions)
        return f"""
            SELECT
                t.table_name,
                t.table_schema,
                COUNT(a.`stmt`) AS visit_count
            FROM information_schema.tables t
            JOIN {AUDIT_LOG_TABLE} a
                ON a.`user` != 'root'
                AND LOWER(a.`stmt`) NOT LIKE '%show%'
                AND a.`stmt` LIKE CONCAT('%', t.table_name, '%')
            WHERE {where_sql}
            GROUP BY t.table_name, t.table_schema
            ORDER BY visit_count DESC
            LIMIT {top_n}
            """

    @staticmethod
    def _build_top_bad_tables_sql(
        db: Optional[str],
        table: Optional[str],
        top_n: int,
    ) -> str:
        conditions = [
            "db_name NOT IN ('information_schema', '_statistics_', 'sys')",
            "table_health_score IS NOT NULL",
        ]
        if db:
            conditions.append(f"db_name = {_sql_literal(db)}")
        if table:
            conditions.append(f"table_name LIKE {_sql_literal('%' + table + '%')}")

        where_sql = "\n            AND ".join(conditions)
        return f"""
            SELECT *
            FROM (
                {TableManagementTools._build_table_health_base_sql()}
            ) table_health
            WHERE {where_sql}
            ORDER BY table_health_score ASC
            LIMIT {top_n}
            OFFSET 0
            """

    @staticmethod
    def _build_table_health_base_sql() -> str:
        return """
            WITH tablet_base AS (
                SELECT
                    table_id,
                    partition_id,
                    tablet_id,
                    MAX(num_segment) AS segment_num,
                    MAX(num_version) AS version_num,
                    MAX(data_size) AS tablet_size
                FROM information_schema.be_tablets
                GROUP BY table_id, partition_id, tablet_id
            ),
            tablet_stat AS (
                SELECT
                    table_id,
                    AVG(segment_num) AS avg_segment,
                    MAX(segment_num) AS max_segment,
                    AVG(version_num) AS avg_version,
                    MAX(version_num) AS max_version,
                    AVG(tablet_size) AS avg_size,
                    MAX(tablet_size) AS max_size,
                    STDDEV_POP(tablet_size) AS size_std
                FROM tablet_base
                GROUP BY table_id
            ),
            replica_stat AS (
                SELECT
                    table_id,
                    SUM(expected_replica) AS expected_replica,
                    SUM(actual_replica) AS actual_replica
                FROM (
                    SELECT
                        MAX(bt.table_id) AS table_id,
                        pm.partition_id,
                        pm.buckets * pm.replication_num AS expected_replica,
                        COUNT(bt.tablet_id) AS actual_replica
                    FROM information_schema.partitions_meta pm
                    LEFT JOIN information_schema.be_tablets bt
                        ON pm.partition_id = bt.partition_id
                    WHERE pm.storage_path NOT LIKE 's3://%'
                    GROUP BY pm.partition_id, pm.buckets, pm.replication_num
                ) t
                GROUP BY table_id
            ),
            table_meta AS (
                SELECT
                    db_name AS db_name,
                    table_name AS table_name,
                    bt.table_id AS table_id,
                    COUNT(DISTINCT pm.partition_id) AS partition_num,
                    MAX(buckets) AS bucket_num,
                    MAX(replication_num) AS replication_num,
                    COUNT(tablet_id) AS tablet_num
                FROM information_schema.partitions_meta pm
                LEFT JOIN information_schema.be_tablets bt
                    ON pm.partition_id = bt.partition_id
                WHERE db_name NOT IN ('information_schema','_statistics_','sys')
                GROUP BY db_name, table_name, bt.table_id
            )
            SELECT
                tm.db_name,
                tm.table_name,
                ts.table_id,
                tm.partition_num,
                tm.bucket_num,
                tm.replication_num,
                tm.tablet_num,
                avg_segment,
                max_segment,
                avg_version,
                max_version,
                avg_size / (1024 * 1024 * 1024) AS avg_bucket_gb,
                max_size / (1024 * 1024 * 1024) AS max_bucket_gb,
                COALESCE(ROUND(rs.actual_replica / NULLIF(rs.expected_replica, 0) * 100, 2), 100) AS replica_score,
                GREATEST(0, 100 - FLOOR(avg_segment / 2000) * 5) AS segment_score,
                GREATEST(0, 100 - FLOOR((max_version - avg_version) / 10) * 5) AS version_score,
                (
                    GREATEST(0, 30 - FLOOR(tm.bucket_num / 1000) * 3)
                    + GREATEST(0, 30 - FLOOR((avg_size / (1024 * 1024 * 1024)) / 2))
                    + GREATEST(0, 40 - (size_std / avg_size) * 40)
                ) AS tablet_score,
                (
                    0.35 * COALESCE((rs.actual_replica / NULLIF(rs.expected_replica, 0) * 100), 100)
                    + 0.35 * (
                        GREATEST(0, 30 - FLOOR(tm.bucket_num / 1000) * 3)
                        + GREATEST(0, 30 - FLOOR((avg_size / (1024 * 1024 * 1024)) / 2))
                        + GREATEST(0, 40 - (size_std / avg_size) * 40)
                    )
                    + 0.10 * GREATEST(0, 100 - FLOOR(avg_segment / 2000) * 5)
                    + 0.20 * GREATEST(0, 100 - FLOOR((max_version - avg_version) / 10) * 5)
                ) AS table_health_score
            FROM tablet_stat ts
            LEFT JOIN replica_stat rs ON ts.table_id = rs.table_id
            LEFT JOIN table_meta tm ON ts.table_id = tm.table_id
            WHERE table_name IS NOT NULL
            """


def format_top_hot_tables_analysis(result: dict[str, Any]) -> str:
    """Format top hot table information."""
    if not result.get("success"):
        return f"Error getting top hot tables: {result.get('error', 'unknown error')}"

    summary = result.get("summary", {})
    lines = [
        "Top hot tables analysis completed.",
        (
            f"Returned {summary.get('top_n_returned', 0)} "
            f"of {summary.get('top_n_requested', 0)} requested table(s)."
        ),
    ]

    top_tables = result.get("top_hot_tables", [])
    if not top_tables:
        lines.append("No hot tables found for the specified criteria.")
        return "\n".join(lines)

    lines.append("Top tables:")
    for table in top_tables[:10]:
        lines.append(
            f"#{table.get('rank')} {table.get('db')}.{table.get('table')} "
            f"visits={table.get('visit_count')}"
        )

    return "\n".join(lines)


def format_top_bad_tables_analysis(result: dict[str, Any]) -> str:
    """Create a compact text summary for table health analysis."""
    if not result.get("success"):
        return f"Error getting top bad tables: {result.get('error', 'unknown error')}"

    summary = result.get("summary", {})
    lines = [
        "Top bad tables analysis completed.",
        (
            f"Returned {summary.get('top_n_returned', 0)} "
            f"of {summary.get('top_n_requested', 0)} requested table(s)."
        ),
    ]

    top_tables = result.get("top_bad_tables", [])
    if not top_tables:
        lines.append("No bad tables found for the specified criteria.")
        return "\n".join(lines)

    lines.append("Top tables:")
    for table in top_tables[:10]:
        lines.append(
            f"#{table.get('rank')} {table.get('db')}.{table.get('table')} "
            f"health_score={table.get('table_health_score')} "
            f"tablet_score={table.get('tablet_score')} "
            f"replica_score={table.get('replica_score')}"
        )

    return "\n".join(lines)
