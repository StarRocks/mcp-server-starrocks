# Tests for mcp-server-starrocks

## Prerequisites

1. **StarRocks cluster running on localhost** with default configuration:
   - Host: localhost
   - Port: 9030 (MySQL protocol) 
   - User: root
   - Password: (empty)
   - At least one BE node available

2. **Optional: Arrow Flight SQL enabled** (for Arrow Flight tests):
   - Port: 9408 (or custom port)
   - Add `arrow_flight_sql_port = 9408` to `fe.conf`
   - Restart FE service
   - Verify with: `python test_arrow_flight.py`

3. **Test dependencies installed**:
   ```bash
   uv add --optional test pytest pytest-cov
   ```

## Running Tests

### Quick Connection Test
First, verify your StarRocks connection:
```bash
# Test MySQL connection and basic operations
python test_connection.py

# Test Arrow Flight SQL connectivity (if enabled)
python test_arrow_flight.py
```

The MySQL test will verify basic connectivity and table operations. The Arrow Flight test will diagnose Arrow Flight SQL availability and performance.

### Full Test Suite
Run the complete db_client test suite:
```bash
# Run all tests (MySQL only)
uv run pytest tests/test_db_client.py::TestDBClient -v

# Run Arrow Flight SQL tests (if enabled)
STARROCKS_FE_ARROW_FLIGHT_SQL_PORT=9408 uv run pytest tests/test_db_client.py::TestDBClientWithArrowFlight -v

# Run all tests (both MySQL and Arrow Flight if available)
uv run pytest tests/test_db_client.py -v

# Run specific test
uv run pytest tests/test_db_client.py::TestDBClient::test_execute_show_databases -v
```

### Test Coverage

The test suite covers:

- **Connection Management**: MySQL pooled connections and ADBC Arrow Flight SQL
- **Query Execution**: SELECT, DDL, DML operations with both success and error cases
- **Result Formats**: Raw ResultSet and pandas DataFrame outputs
- **Database Context**: Switching databases for queries
- **Error Handling**: Connection failures, invalid queries, malformed SQL
- **Resource Management**: Connection pooling, cursor cleanup, connection reset
- **Edge Cases**: Empty results, type conversion, schema operations

### Test Configuration

- **Single-node setup**: Tests create tables with `PROPERTIES ("replication_num" = "1")`
- **Temporary databases**: Tests create and clean up test databases automatically
- **Arrow Flight SQL**: Tests are skipped if `STARROCKS_FE_ARROW_FLIGHT_SQL_PORT` is not set
- **Isolation**: Each test uses a fresh DBClient instance with reset connections

## Test Results

When all tests pass, you should see:
```
======================== 16 passed, 2 skipped in 1.30s =========================
```

The 2 skipped tests are Arrow Flight SQL tests that only run when the environment variable is configured.

## Troubleshooting

**Connection issues**:
- Ensure StarRocks FE is running on localhost:9030
- Check that the `root` user has no password set
- Verify at least one BE node is available

**Table creation failures**:
- Single-node clusters need `replication_num=1`
- Check StarRocks logs for detailed error messages

**Import errors**:
- Ensure you're running from the project root directory
- Check that `src/mcp_server_starrocks` is in your Python path