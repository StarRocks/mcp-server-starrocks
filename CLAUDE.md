# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarRocks Official MCP Server - A bridge between AI assistants and StarRocks databases, built using FastMCP framework. Enables direct SQL execution, database exploration, data visualization, and schema introspection through the Model Context Protocol (MCP).

## Development Commands

**Local Development:**
```bash
# Run the server directly for testing
uv run mcp-server-starrocks

# Run with test mode to verify table overview functionality
uv run mcp-server-starrocks --test

# Run in Streamable HTTP mode (recommended for integration)
export MCP_TRANSPORT_MODE=streamable-http
uv run mcp-server-starrocks
```

**Package Management:**
```bash
# Install dependencies (handled by uv automatically)
uv sync

# Build package
uv build
```

## Architecture Overview

### Core Components

- **`src/mcp_server_starrocks/server.py`**: Main server implementation containing all MCP tools, resources, and database connection logic
- **`src/mcp_server_starrocks/__init__.py`**: Entry point that starts the async server

### Connection Architecture

The server supports two connection modes:
- **Standard MySQL Protocol**: Default connection using `mysql.connector` 
- **Arrow Flight SQL**: High-performance connection using ADBC drivers (enabled when `STARROCKS_FE_ARROW_FLIGHT_SQL_PORT` is set)

Connection management uses a global singleton pattern with automatic reconnection handling.

### Tool Categories

1. **Query Execution Tools**:
   - `read_query`: Execute SELECT and other result-returning queries
   - `write_query`: Execute DDL/DML commands
   - `analyze_query`: Query performance analysis via EXPLAIN ANALYZE

2. **Overview Tools with Caching**:
   - `table_overview`: Get table schema, row count, and sample data (cached)
   - `db_overview`: Get overview of all tables in a database (uses table cache)
   
3. **Visualization Tool**:
   - `query_and_plotly_chart`: Execute query and generate Plotly charts from results

### Resource Endpoints

- `starrocks:///databases`: List all databases
- `starrocks:///{db}/tables`: List tables in a database  
- `starrocks:///{db}/{table}/schema`: Get table CREATE statement
- `proc:///{path}`: Access StarRocks internal system information (similar to Linux /proc)

### Caching System

In-memory cache for table overviews using `(database_name, table_name)` as cache keys. Cache includes both successful results and error messages. Controlled by `STARROCKS_OVERVIEW_LIMIT` environment variable (default: 20000 characters).

## Configuration

Environment variables for database connection:
- `STARROCKS_HOST`: Database host (default: localhost)
- `STARROCKS_PORT`: MySQL port (default: 9030)  
- `STARROCKS_USER`: Username (default: root)
- `STARROCKS_PASSWORD`: Password (default: empty)
- `STARROCKS_DB`: Default database for session
- `STARROCKS_MYSQL_AUTH_PLUGIN`: Auth plugin (e.g., mysql_clear_password)
- `STARROCKS_FE_ARROW_FLIGHT_SQL_PORT`: Enables Arrow Flight SQL mode
- `MCP_TRANSPORT_MODE`: Communication mode (stdio/streamable-http/sse)

## Code Patterns

### Error Handling
- Database errors trigger connection reset via `reset_connection()`
- All tools return string error messages rather than raising exceptions
- Cursors are always closed in finally blocks

### Security
- SQL injection prevention through parameterized queries and backtick escaping
- Plotly expressions are validated using AST parsing to prevent code injection
- Limited `eval()` usage with restricted scope for chart generation

### Async Patterns
- Tools are defined as async functions even though database operations are synchronous
- Main server runs in async context using `FastMCP.run_async()`

## Package Structure

This is a simple Python package built with hatchling:
- Single module in `src/mcp_server_starrocks/`
- Entry point defined in pyproject.toml as `mcp-server-starrocks` command
- Dependencies managed through pyproject.toml, no requirements.txt files