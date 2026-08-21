import inspect

from src.mcp_server_starrocks import server
from src.mcp_server_starrocks import query_profile_analytics
from src.mcp_server_starrocks.db_client import ResultSet
from src.mcp_server_starrocks.query_profile_analytics import (
    QueryProfileAnalyticsTools,
    format_slow_query_analysis,
)


class FakeDBClient:
    def execute(self, statement):
        self.statement = statement
        return ResultSet(
            success=True,
            column_names=[
                "user_name",
                "host",
                "query_time",
                "query_type",
                "sql_statement",
                "execution_time_ms",
                "scan_bytes",
                "scan_rows",
                "return_rows",
            ],
            rows=[
                [
                    "alice",
                    "127.0.0.1",
                    "2026-08-18 10:00:00",
                    "Query",
                    "SELECT * FROM db.orders ORDER BY created_at",
                    65000,
                    2 * 1024**3,
                    2_000_000,
                    100,
                ],
                [
                    "bob",
                    "127.0.0.2",
                    "2026-08-18 10:05:00",
                    "Query",
                    "SELECT id FROM db.users LIMIT 10",
                    12000,
                    1024,
                    10,
                    10,
                ],
            ],
            execution_time=0.1,
        )


def test_profile_text_analyzer_is_not_public_api():
    assert not hasattr(query_profile_analytics, "analyze_profile_text")
    assert not hasattr(query_profile_analytics, "format_profile_analysis")


def test_slow_query_analysis_does_not_expose_include_patterns_parameter():
    assert (
        "include_patterns"
        not in inspect.signature(
            QueryProfileAnalyticsTools.analyze_slow_queries_topn
        ).parameters
    )
    assert (
        "include_patterns"
        not in inspect.signature(server.analyze_slow_queries_topn).parameters
    )


def test_analyze_slow_queries_topn_returns_clean_summary_and_patterns():
    tools = QueryProfileAnalyticsTools(FakeDBClient())

    result = tools.analyze_slow_queries_topn(
        days=1,
        top_n=2,
        min_execution_time_ms=1000,
    )

    assert result["success"] is True
    assert "analysis_time_range" not in result
    assert "performance_insights" not in result
    assert "query_patterns" not in result
    assert result["analysis_metadata"]["time_range"]["days"] == 1
    assert result["analysis_metadata"]["filters"] == {
        "top_n": 2,
        "min_execution_time_ms": 1000,
    }
    assert result["summary"] == {
        "total_slow_queries": 2,
        "top_n_analyzed": 2,
    }
    assert result["top_slow_queries"][0]["client"]["user_name"] == "alice"
    assert result["top_slow_queries"][0]["client"]["host"] == "127.0.0.1"
    assert result["top_slow_queries"][0]["query"] == {
        "query_type": "Query",
        "sql_statement": "SELECT * FROM db.orders ORDER BY created_at",
    }
    assert result["top_slow_queries"][0]["execution"] == {
        "execution_time_ms": 65000,
    }
    assert result["top_slow_queries"][0]["performance_issues"] == [
        "excessive_execution_time",
        "large_scan_volume",
        "low_filter_selectivity",
        "wildcard_projection",
        "unbounded_sort",
    ]
    assert result["top_slow_queries"][0]["scan_efficiency"]["filter_ratio"] == 20000
    assert result["top_slow_queries"][0]["scan_efficiency"]["scan_bytes_per_row"] == 1073.74
    workload_insights = result["workload_insights"]
    assert workload_insights["execution_time_statistics"]["p95_ms"] == 65000
    assert workload_insights["scan_efficiency_statistics"]["max_scan_bytes"] == 2147483648
    assert workload_insights["scan_efficiency_statistics"]["total_scan_bytes"] == 2147484672
    assert workload_insights["scan_efficiency_statistics"]["avg_filter_ratio"] == 10000.5
    assert workload_insights["scan_efficiency_statistics"]["max_filter_ratio"] == 20000
    assert workload_insights["user_activity"] == {
        "top_slow_query_users": {"alice": 1, "bob": 1},
        "unique_user_count": 2,
    }
    assert workload_insights["query_type_distribution"] == {"Query": 2}
    assert workload_insights["performance_issue_distribution"][
        "slow_without_large_scan"
    ] == 1
    assert workload_insights["frequently_accessed_tables"] == {
        "db.orders": 1,
        "db.users": 1,
    }


def test_identify_performance_issues_reports_additional_scan_shapes():
    tools = QueryProfileAnalyticsTools(None)
    cases = [
        (
            {
                "execution_time_ms": 1000,
                "scan_bytes": 1024,
                "scan_rows": 10_000_001,
                "return_rows": 10_000_001,
            },
            "large_row_scan",
        ),
        (
            {
                "execution_time_ms": 1000,
                "scan_bytes": 1024,
                "scan_rows": 1_000_001,
                "return_rows": 1_000_001,
            },
            "large_result_set",
        ),
        (
            {
                "execution_time_ms": 1000,
                "scan_bytes": 1024,
                "scan_rows": 1_000_001,
                "return_rows": 0,
            },
            "large_scan_no_result",
        ),
        (
            {
                "execution_time_ms": 1000,
                "scan_bytes": 65 * 1024,
                "scan_rows": 1,
                "return_rows": 1,
            },
            "expensive_row_materialization",
        ),
        (
            {
                "execution_time_ms": 12_000,
                "scan_bytes": 1024,
                "scan_rows": 10,
                "return_rows": 1,
            },
            "slow_without_large_scan",
        ),
    ]

    for query, expected_issue in cases:
        assert expected_issue in tools._identify_performance_issues(query)


def test_generate_performance_recommendations_covers_additional_issues():
    recommendations = QueryProfileAnalyticsTools._generate_performance_recommendations(
        {
            "execution_time_statistics": {},
            "scan_efficiency_statistics": {},
            "performance_issue_distribution": {
                "large_row_scan": 1,
                "large_result_set": 1,
                "large_scan_no_result": 1,
                "expensive_row_materialization": 1,
                "slow_without_large_scan": 1,
            }
        },
    )

    text = "\n".join(recommendations)
    assert "partition pruning" in text
    assert "page or aggregate" in text
    assert "returning no rows" in text
    assert "wide columns" in text
    assert "query profile" in text


def test_format_slow_query_analysis_handles_empty_result():
    text = format_slow_query_analysis(
        {
            "success": True,
            "message": "No slow queries found for the specified criteria",
            "analysis_metadata": {
                "time_range": {"days": 7},
            },
        }
    )

    assert text == "No slow queries found for the specified criteria (days=7)."


def test_format_slow_query_analysis_omits_missing_data_source():
    text = format_slow_query_analysis(
        {
            "success": True,
            "analysis_timestamp": "2026-08-18T10:00:00",
            "summary": {
                "total_slow_queries": 1,
                "top_n_analyzed": 1,
            },
            "top_slow_queries": [],
            "recommendations": [],
        }
    )

    assert "Data source: None" not in text
