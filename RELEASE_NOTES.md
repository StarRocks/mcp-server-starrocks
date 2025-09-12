# StarRocks MCP Server Release Notes

## Version 0.1.5

Major Features and Enhancements

1. Connection Pooling and Architecture Refactor (commit 0fc372d)
  - Major refactor introducing connection pooling for improved performance
  - Extracted database client logic into separate db_client.py module
  - Enhanced connection management and reliability
2. Enhanced Arrow Flight SQL Support (commit 877338f)
  - Improved Arrow Flight SQL connection handling
  - Better result processing for high-performance queries
  - Enhanced error handling for Arrow Flight connections
3. New Query Analysis Tools (commit 60ca975)
  - Added collect_query_dump_and_profile functionality
  - Enhanced query performance analysis capabilities
4. Database Summary Management (commits d269ebe, 5b2ca59)
  - Added new db_summary_manager.py module
  - Implemented database summary functionality for better overview capabilities
  - Enhanced database exploration features
5. Configuration Enhancements (commit fb09271)
  - Added STARROCKS_URL configuration option
  - Improved connection configuration flexibility

  Testing and Infrastructure

- Updated test suite with new test cases for database client functionality
- Added comprehensive testing for Arrow Flight SQL features
- Improved test infrastructure with new README documentation

  Breaking Changes

- Major refactor may require configuration updates for some deployment scenarios
- Connection handling has been restructured (though backwards compatibility is maintained)

## Version 0.1.4


## Version 0.1.3

1. refactor using fastmcp
2. add new config STARROCKS_MYSQL_AUTH_PLUGIN

## Version 0.1.2

Fix accidental extra import of sqlalalchemy

## Version 0.1.1

1. add new tool query_and_plotly_chart
2. add new tool table_overview & db_overview
3. add env config STARROCKS_DB and STARROCKS_OVERVIEW_LIMIT, both optional


## Version 0.1.0 (Initial Release)

We are excited to announce the first release of the StarRocks MCP (Model Context Protocol) Server. This server enables AI assistants to interact directly with StarRocks databases, providing a seamless interface for executing queries and retrieving database information.

### Description

The StarRocks MCP Server acts as a bridge between AI assistants and StarRocks databases, allowing for direct SQL execution and database exploration without requiring complex setup or configuration. This initial release provides essential functionality for database interaction while maintaining security and performance.

### Features

- **SQL Query Execution**
  - `read_query` tool for executing SELECT queries and commands that return result sets
  - `write_query` tool for executing DDL/DML statements and other StarRocks commands
  - Proper error handling and connection management

- **Database Exploration**
  - List all databases in a StarRocks instance
  - View table schemas using SHOW CREATE TABLE
  - List all tables within a specific database

- **System Information Access**
  - Access to StarRocks internal system information via proc-like interface
  - Visibility into FE nodes, BE nodes, CN nodes, databases, tables, partitions, transactions, jobs, and more

- **Flexible Configuration**
  - Configurable connection parameters (host, port, user, password)
  - Support for both package installation and local directory execution

### Requirements

- Python 3.10 or higher
- Dependencies:
  - mcp >= 1.0.0
  - mysql-connector-python >= 9.2.0

### Configuration

The server can be configured through environment variables:

- `STARROCKS_HOST` (default: localhost)
- `STARROCKS_PORT` (default: 9030)
- `STARROCKS_USER` (default: root)
- `STARROCKS_PASSWORD` (default: empty)
- `STARROCKS_MYSQL_AUTH_PLUGIN` (default: mysql_native_password) user can also pass different auth plugins like `mysql_clear_password`

### Installation

The server can be installed as a Python package:

```bash
pip install mcp-server-starrocks
```

Or run directly from the source:

```bash
uv --directory path/to/mcp-server-starrocks run mcp-server-starrocks
```

### MCP Integration

Add the following configuration to your MCP settings file:

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp-server-starrocks",
        "mcp-server-starrocks"
      ],
      "env": {
        "STARROCKS_HOST": "localhost",
        "STARROCKS_PORT": "9030",
        "STARROCKS_USER": "root",
        "STARROCKS_PASSWORD": "",
        "STARROCKS_MYSQL_AUTH_PLUGIN":"mysql_clear_password"
      }
    }
  }
}
```

---

We welcome feedback and contributions to improve the StarRocks MCP Server. Please report any issues or suggestions through our GitHub repository.
