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

import math
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, TypedDict

from loguru import logger


Number = int | float

_LARGE_SCAN_BYTES = 1024**3
_LARGE_ROW_SCAN_ROWS = 10_000_000
_LARGE_RESULT_ROWS = 1_000_000
_LARGE_SCAN_NO_RESULT_ROWS = 1_000_000
_EXPENSIVE_SCAN_BYTES_PER_ROW = 64 * 1024
_SMALL_SCAN_BYTES = 100 * 1024**2
_SMALL_SCAN_ROWS = 100_000
_HIGH_EXECUTION_TIME_MS = 10_000
_EXCESSIVE_EXECUTION_TIME_MS = 60_000


class SlowQueryRow(TypedDict, total=False):
    user_name: str
    host: str
    query_time: Any
    query_type: str
    sql_statement: str
    execution_time_ms: Any
    scan_bytes: Any
    scan_rows: Any
    return_rows: Any


class AnalysisTimeRange(TypedDict):
    days: int
    start_date: str
    end_date: str


class AnalysisFilters(TypedDict):
    top_n: int
    min_execution_time_ms: int


class AnalysisMetadata(TypedDict, total=False):
    analysis_timestamp: str
    time_range: AnalysisTimeRange
    filters: AnalysisFilters


class SlowQuerySummary(TypedDict):
    total_slow_queries: int
    top_n_analyzed: int


class TopSlowQueryClient(TypedDict):
    user_name: str
    host: Any


class TopSlowQueryDetails(TypedDict):
    query_type: str
    sql_statement: str


class TopSlowQueryExecution(TypedDict):
    execution_time_ms: Number


class TopSlowQueryScanEfficiency(TypedDict):
    scan_bytes: Number
    scan_rows: Number
    return_rows: Number
    filter_ratio: Number
    scan_bytes_per_row: Number


class TopSlowQuery(TypedDict):
    rank: int
    query_time: str
    client: TopSlowQueryClient
    query: TopSlowQueryDetails
    execution: TopSlowQueryExecution
    scan_efficiency: TopSlowQueryScanEfficiency
    performance_issues: list[str]


class ExecutionTimeStatistics(TypedDict):
    avg_ms: Number
    p95_ms: Number
    max_ms: Number
    min_ms: Number


class ScanEfficiencyStatistics(TypedDict):
    avg_scan_bytes: Number
    total_scan_bytes: Number
    max_scan_bytes: Number
    min_scan_bytes: Number
    avg_filter_ratio: Number
    max_filter_ratio: Number


class UserActivity(TypedDict):
    top_slow_query_users: dict[str, int]
    unique_user_count: int


class WorkloadInsights(TypedDict, total=False):
    execution_time_statistics: ExecutionTimeStatistics
    scan_efficiency_statistics: ScanEfficiencyStatistics
    user_activity: UserActivity
    query_type_distribution: dict[str, int]
    performance_issue_distribution: dict[str, int]
    frequently_accessed_tables: dict[str, int]


class SlowQueryAnalysisResult(TypedDict, total=False):
    success: bool
    analysis_metadata: AnalysisMetadata
    message: str
    summary: SlowQuerySummary
    workload_insights: WorkloadInsights
    top_slow_queries: list[TopSlowQuery]
    recommendations: list[str]
    error: str


def _as_number(value: Any, default: float = 0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display_number(value: Any) -> int | float:
    number = _as_number(value)
    if number.is_integer():
        return int(number)
    return round(number, 2)


def _ratio(numerator: Any, denominator: Any) -> int | float:
    denominator_number = _as_number(denominator)
    if denominator_number <= 0:
        return 0
    return _display_number(_as_number(numerator) / denominator_number)


def _percentile(values: list[float], percentile: float) -> int | float:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * percentile) - 1)
    return _display_number(sorted_values[index])


def _empty_workload_insights() -> WorkloadInsights:
    return {
        "execution_time_statistics": {
            "avg_ms": 0,
            "p95_ms": 0,
            "max_ms": 0,
            "min_ms": 0,
        },
        "scan_efficiency_statistics": {
            "avg_scan_bytes": 0,
            "total_scan_bytes": 0,
            "max_scan_bytes": 0,
            "min_scan_bytes": 0,
            "avg_filter_ratio": 0,
            "max_filter_ratio": 0,
        },
        "user_activity": {
            "top_slow_query_users": {},
            "unique_user_count": 0,
        },
        "query_type_distribution": {},
        "performance_issue_distribution": {},
        "frequently_accessed_tables": {},
    }


class QueryProfileAnalyticsTools:

    def __init__(self, db_client):
        self.db_client = db_client

    def analyze_slow_queries_topn(
        self,
        days: int = 7,
        top_n: int = 20,
        min_execution_time_ms: int = 1000,
    ) -> SlowQueryAnalysisResult:
        """Analyze top N slow queries and performance patterns."""
        try:
            slow_queries = self._get_slow_query_data(
                days=days,
                top_n=top_n,
                min_execution_time_ms=min_execution_time_ms,
            )

            now = datetime.now()
            base_result: SlowQueryAnalysisResult = {
                "success": True,
                "analysis_metadata": {
                    "analysis_timestamp": now.isoformat(),
                    "time_range": {
                        "days": days,
                        "start_date": (now - timedelta(days=days)).isoformat(),
                        "end_date": now.isoformat(),
                    },
                    "filters": {
                        "top_n": top_n,
                        "min_execution_time_ms": min_execution_time_ms,
                    },
                },
            }

            if not slow_queries:
                return {
                    **base_result,
                    "message": "No slow queries found for the specified criteria",
                    "summary": {
                        "total_slow_queries": 0,
                        "top_n_analyzed": 0,
                    },
                    "workload_insights": _empty_workload_insights(),
                    "top_slow_queries": [],
                    "recommendations": [],
                }

            top_queries = self._analyze_top_slow_queries(slow_queries, top_n)
            workload_insights = self._generate_performance_insights(slow_queries)
            workload_insights.update(self._analyze_query_patterns(slow_queries))

            return {
                **base_result,
                "summary": {
                    "total_slow_queries": len(slow_queries),
                    "top_n_analyzed": min(top_n, len(slow_queries)),
                },
                "workload_insights": workload_insights,
                "top_slow_queries": top_queries,
                "recommendations": self._generate_performance_recommendations(
                    workload_insights
                ),
            }
        except Exception as exc:
            logger.exception("Slow query analysis failed")
            return {
                "success": False,
                "error": str(exc),
                "analysis_metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                },
            }

    def _get_slow_query_data(
        self,
        days: int,
        top_n: int,
        min_execution_time_ms: int,
    ) -> list[SlowQueryRow]:
        """Get slow query data from the internal audit log."""
        try:
            limit = max(5000, top_n)
            start_date = datetime.now() - timedelta(days=days)
            slow_query_sql = f"""
            SELECT
                `user` as user_name,
                `clientIp` as host,
                `timestamp` as query_time,
                `queryType` as query_type,
                `stmt` as sql_statement,
                `queryTime` as execution_time_ms,
                `scanBytes` as scan_bytes,
                `scanRows` as scan_rows,
                `returnRows` as return_rows
            FROM starrocks_audit_db__.starrocks_audit_tbl__
            WHERE `timestamp` >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
                AND `queryTime` >= {min_execution_time_ms}
                AND `stmt` IS NOT NULL
                AND `stmt` != ''
                AND `state` != 'ERR'
            ORDER BY `queryTime` DESC
            LIMIT {limit}
            """

            result = self.db_client.execute(slow_query_sql)
            if not result.success:
                logger.warning(
                    "Failed to get slow query data: {}",
                    result.error_message or "Unknown query failure",
                )
                return []

            rows = result.rows or []
            column_names = result.column_names or []
            return [dict(zip(column_names, row)) for row in rows]
        except Exception as exc:
            logger.warning("Failed to get slow query data: {}", str(exc))
            return []

    def _analyze_top_slow_queries(
        self,
        slow_queries: list[SlowQueryRow],
        top_n: int,
    ) -> list[TopSlowQuery]:
        sorted_queries = sorted(
            slow_queries,
            key=lambda row: _as_number(row.get("execution_time_ms")),
            reverse=True,
        )[:top_n]

        analyzed_queries = []
        for index, query in enumerate(sorted_queries):
            sql = str(query.get("sql_statement") or "")
            execution_time_ms = _display_number(query.get("execution_time_ms"))
            scan_bytes = query.get("scan_bytes")
            scan_rows = query.get("scan_rows")
            return_rows = query.get("return_rows")
            analyzed_queries.append(
                {
                    "rank": index + 1,
                    "query_time": str(query.get("query_time") or ""),
                    "client": {
                        "user_name": query.get("user_name") or "unknown",
                        "host": query.get("host"),
                    },
                    "query": {
                        "query_type": query.get("query_type") or "unknown",
                        "sql_statement": sql[:500] + "..." if len(sql) > 500 else sql,
                    },
                    "execution": {
                        "execution_time_ms": execution_time_ms,
                    },
                    "scan_efficiency": {
                        "scan_bytes": _display_number(scan_bytes),
                        "scan_rows": _display_number(scan_rows),
                        "return_rows": _display_number(return_rows),
                        "filter_ratio": _ratio(scan_rows, return_rows),
                        "scan_bytes_per_row": _ratio(scan_bytes, scan_rows),
                    },
                    "performance_issues": self._identify_performance_issues(query),
                }
            )

        return analyzed_queries

    def _identify_performance_issues(self, query: SlowQueryRow) -> list[str]:
        issues = []
        sql = str(query.get("sql_statement") or "").upper()
        execution_time = _as_number(query.get("execution_time_ms"))
        scan_bytes = _as_number(query.get("scan_bytes"))
        scan_rows = _as_number(query.get("scan_rows"))
        return_rows = _as_number(query.get("return_rows"))
        filter_ratio = scan_rows / return_rows if return_rows > 0 else 0
        scan_bytes_per_row = scan_bytes / scan_rows if scan_rows > 0 else 0

        if execution_time > _EXCESSIVE_EXECUTION_TIME_MS:
            issues.append("excessive_execution_time")
        elif execution_time > _HIGH_EXECUTION_TIME_MS:
            issues.append("high_execution_time")

        if scan_bytes > _LARGE_SCAN_BYTES:
            issues.append("large_scan_volume")

        if scan_rows > _LARGE_ROW_SCAN_ROWS:
            issues.append("large_row_scan")

        if return_rows > _LARGE_RESULT_ROWS:
            issues.append("large_result_set")

        if scan_rows > _LARGE_SCAN_NO_RESULT_ROWS and return_rows == 0:
            issues.append("large_scan_no_result")

        if filter_ratio > 1000:
            issues.append("low_filter_selectivity")

        if scan_bytes_per_row > _EXPENSIVE_SCAN_BYTES_PER_ROW:
            issues.append("expensive_row_materialization")

        if (
            execution_time > _HIGH_EXECUTION_TIME_MS
            and scan_bytes < _SMALL_SCAN_BYTES
            and scan_rows < _SMALL_SCAN_ROWS
        ):
            issues.append("slow_without_large_scan")

        if "SELECT *" in sql:
            issues.append("wildcard_projection")

        if "ORDER BY" in sql and "LIMIT" not in sql:
            issues.append("unbounded_sort")

        return issues

    def _generate_performance_insights(
        self,
        slow_queries: list[SlowQueryRow],
    ) -> WorkloadInsights:
        execution_times = [_as_number(q.get("execution_time_ms")) for q in slow_queries]
        scan_bytes = [
            _as_number(q.get("scan_bytes")) for q in slow_queries if _as_number(q.get("scan_bytes")) > 0
        ]
        filter_ratios = [
            _ratio(q.get("scan_rows"), q.get("return_rows"))
            for q in slow_queries
            if _as_number(q.get("scan_rows")) > 0 and _as_number(q.get("return_rows")) > 0
        ]
        user_query_counts = Counter(q.get("user_name") or "unknown" for q in slow_queries)
        query_types = Counter(q.get("query_type") or "unknown" for q in slow_queries)

        return {
            "execution_time_statistics": {
                "avg_ms": round(statistics.mean(execution_times), 2) if execution_times else 0,
                "p95_ms": _percentile(execution_times, 0.95),
                "max_ms": _display_number(max(execution_times)) if execution_times else 0,
                "min_ms": _display_number(min(execution_times)) if execution_times else 0,
            },
            "scan_efficiency_statistics": {
                "avg_scan_bytes": round(statistics.mean(scan_bytes), 2) if scan_bytes else 0,
                "total_scan_bytes": _display_number(sum(scan_bytes)) if scan_bytes else 0,
                "max_scan_bytes": _display_number(max(scan_bytes)) if scan_bytes else 0,
                "min_scan_bytes": _display_number(min(scan_bytes)) if scan_bytes else 0,
                "avg_filter_ratio": (
                    round(statistics.mean(filter_ratios), 2) if filter_ratios else 0
                ),
                "max_filter_ratio": (
                    _display_number(max(filter_ratios)) if filter_ratios else 0
                ),
            },
            "user_activity": {
                "top_slow_query_users": dict(user_query_counts.most_common(10)),
                "unique_user_count": len(user_query_counts),
            },
            "query_type_distribution": dict(query_types),
        }

    def _analyze_query_patterns(
        self,
        slow_queries: list[SlowQueryRow],
    ) -> WorkloadInsights:
        common_issues = Counter()
        table_access_patterns = Counter()

        for query in slow_queries:
            sql = str(query.get("sql_statement") or "")
            common_issues.update(self._identify_performance_issues(query))
            table_access_patterns.update(self._extract_table_names(sql))

        return {
            "performance_issue_distribution": dict(common_issues.most_common(10)),
            "frequently_accessed_tables": dict(table_access_patterns.most_common(15)),
        }

    @staticmethod
    def _extract_table_names(sql: str) -> list[str]:
        if not sql:
            return []
        patterns = [
            r"\bFROM\s+([`\"\w.]+)",
            r"\bJOIN\s+([`\"\w.]+)",
            r"\bINTO\s+([`\"\w.]+)",
            r"\bUPDATE\s+([`\"\w.]+)",
        ]
        tables = []
        for pattern in patterns:
            for match in re.findall(pattern, sql, re.IGNORECASE):
                table = match.strip("`\"").lower()
                if table:
                    tables.append(table)
        return tables

    @staticmethod
    def _generate_performance_recommendations(
        workload_insights: WorkloadInsights,
    ) -> list[str]:
        recommendations = []
        execution_stats = workload_insights.get("execution_time_statistics", {})
        scan_stats = workload_insights.get("scan_efficiency_statistics", {})
        common_issues = workload_insights.get("performance_issue_distribution", {})

        def has_issue(*issue_names: str) -> bool:
            return any(common_issues.get(issue_name, 0) > 0 for issue_name in issue_names)

        if execution_stats.get("max_ms", 0) > _EXCESSIVE_EXECUTION_TIME_MS:
            recommendations.append(
                "Investigate queries running longer than 60 seconds with EXPLAIN ANALYZE or query profiles."
            )
        if (
            scan_stats.get("max_scan_bytes", 0) > _LARGE_SCAN_BYTES
            or has_issue("large_scan_volume")
        ):
            recommendations.append(
                "Review partition pruning, predicate selectivity, and materialized views for queries scanning more than 1 GB."
            )
        if has_issue("large_row_scan", "low_filter_selectivity"):
            recommendations.append(
                "Review partition pruning, predicate selectivity, and table statistics for queries scanning too many rows."
            )
        if has_issue("large_scan_no_result"):
            recommendations.append(
                "Check predicates and partition filters for queries scanning many rows while returning no rows."
            )
        if has_issue("wildcard_projection", "expensive_row_materialization"):
            recommendations.append(
                "Reduce scanned wide columns by replacing SELECT * with explicit column lists and avoiding unused large fields."
            )
        if has_issue("large_result_set"):
            recommendations.append(
                "Use LIMIT, pagination, or downstream aggregation to page or aggregate large result sets before returning them to clients."
            )
        if has_issue("unbounded_sort"):
            recommendations.append(
                "Add a LIMIT or stronger filters to ORDER BY queries when the full sorted output is not required."
            )
        if has_issue("slow_without_large_scan"):
            recommendations.append(
                "Inspect the query profile for joins, sorts, aggregations, queueing, metadata latency, or external I/O because the query is slow without a large scan."
            )
        if not recommendations:
            recommendations.append(
                "Review the top slow queries' execution profiles and table statistics for workload-specific optimizations."
            )
        return recommendations


def format_slow_query_analysis(result: SlowQueryAnalysisResult) -> str:
    """Create a compact text summary for MCP clients that only show text."""
    if not result.get("success"):
        return f"Error analyzing slow queries: {result.get('error', 'unknown error')}"

    if result.get("message"):
        metadata = result.get("analysis_metadata", {})
        period = metadata.get("time_range", {})
        return (
            f"{result['message']} "
            f"(days={period.get('days')})."
        )

    summary = result.get("summary", {})
    lines = [
        "Slow query analysis completed.",
        (
            "Summary: "
            f"{summary.get('total_slow_queries', 0)} slow queries, "
            f"top {summary.get('top_n_analyzed', 0)} analyzed."
        ),
    ]
    for query in result.get("top_slow_queries", [])[:5]:
        execution = query.get("execution", {})
        query_details = query.get("query", {})
        client = query.get("client", {})
        lines.append(
            f"#{query['rank']} {execution.get('execution_time_ms')}ms "
            f"{query_details.get('query_type')} user={client.get('user_name')} "
            f"issues={','.join(query['performance_issues']) or 'none'}"
        )
    recommendations = result.get("recommendations", [])
    if recommendations:
        lines.append("Recommendations:")
        lines.extend(f"- {item}" for item in recommendations[:5])
    return "\n".join(lines)
