[![MseeP.ai Security Assessment Badge](https://mseep.net/mseep-audited.png)](https://mseep.ai/app/starrocks-mcp-server-starrocks)

# StarRocks Official MCP Server

The StarRocks MCP Server acts as a bridge between AI assistants and StarRocks databases. It allows for direct SQL execution, database exploration, data visualization via charts, and retrieving detailed schema/data overviews without requiring complex client-side setup.

<a href="https://glama.ai/mcp/servers/@StarRocks/mcp-server-starrocks">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@StarRocks/mcp-server-starrocks/badge" alt="StarRocks Server MCP server" />
</a>

## Features

- **Direct SQL Execution:** Run `SELECT` queries (`read_query`) and DDL/DML commands (`write_query`).
- **Database Exploration:** List databases and tables, retrieve table schemas (`starrocks://` resources).
- **System Information:** Access internal StarRocks metrics and states via the `proc://` resource path.
- **Detailed Overviews:** Get comprehensive summaries of tables (`table_overview`) or entire databases (`db_overview`), including column definitions, row counts, and sample data.
- **Data Visualization:** Execute a query and generate a Plotly chart directly from the results (`query_and_plotly_chart`).
- **Intelligent Caching:** Table and database overviews are cached in memory to speed up repeated requests. Cache can be bypassed when needed.
- **Flexible Configuration:** Set connection details and behavior via environment variables.

## Prerequisites

- **Python 3.11 or newer.**
- A reachable **StarRocks** cluster (FE service). By default the server connects to `localhost:9030` over the MySQL protocol.
- **[`uv`](https://docs.astral.sh/uv/)** — a fast Python package and project manager (a modern replacement for `pip` + `virtualenv`) from Astral. This project uses `uv` to resolve dependencies, create the virtual environment, and launch the server. The `uv run` commands throughout this README automatically create an isolated environment and install the required dependencies on first use, so no manual `pip install` step is needed.

### Installing `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew / pipx / pip
brew install uv
# pipx install uv
# pip install uv
```

See the [official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options. After installing, verify it is on your `PATH`:

```bash
uv --version
```

## Installation

You generally do **not** need to install the package manually — the MCP host launches it for you via `uv` (see [Configuration](#configuration) below). `uv` fetches the package and its dependencies on demand.

To run it directly for testing or development:

```bash
# Run the published package in a throwaway environment
uv run --with mcp-server-starrocks mcp-server-starrocks --help

# Or, from a local checkout of this repository
git clone https://github.com/starrocks/mcp-server-starrocks.git
cd mcp-server-starrocks
uv sync                      # create the virtual environment and install dependencies
uv run mcp-server-starrocks --help
```

## Configuration

The MCP server is typically run via an MCP host. Configuration is passed to the host, specifying how to launch the StarRocks MCP server process.

**Using Streamable HTTP (recommended):**

To start the server in Streamable HTTP mode:

First test that the connection to StarRocks is OK (`9030` is the StarRocks MySQL protocol port, not the HTTP server port):
```
$ STARROCKS_URL=root:@localhost:9030 uv run mcp-server-starrocks --test
```

Start the server:

```
uv run mcp-server-starrocks --mode streamable-http --port 8000
```

Then config the MCP like this:

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```


**Using `uv` with installed package (individual environment variables):**

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "command": "uv",
      "args": ["run", "--with", "mcp-server-starrocks", "mcp-server-starrocks"],
      "env": {
        "STARROCKS_HOST": "default localhost",
        "STARROCKS_PORT": "default 9030",
        "STARROCKS_USER": "default root",
        "STARROCKS_PASSWORD": "default empty",
        "STARROCKS_DB": "default empty"
      }
    }
  }
}
```

**Using `uv` with installed package (connection URL):**

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "command": "uv",
      "args": ["run", "--with", "mcp-server-starrocks", "mcp-server-starrocks"],
      "env": {
        "STARROCKS_URL": "root:password@localhost:9030/my_database"
      }
    }
  }
}
```

**Using `uv` with local directory (for development):**

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "command": "uv",
      "args": [
        "--directory",
        "path/to/mcp-server-starrocks", // <-- Update this path
        "run",
        "mcp-server-starrocks"
      ],
      "env": {
        "STARROCKS_HOST": "default localhost",
        "STARROCKS_PORT": "default 9030",
        "STARROCKS_USER": "default root",
        "STARROCKS_PASSWORD": "default empty",
        "STARROCKS_DB": "default empty"
      }
    }
  }
}
```

**Using `uv` with local directory and connection URL:**

```json
{
  "mcpServers": {
    "mcp-server-starrocks": {
      "command": "uv",
      "args": [
        "--directory",
        "path/to/mcp-server-starrocks", // <-- Update this path
        "run",
        "mcp-server-starrocks"
      ],
      "env": {
        "STARROCKS_URL": "root:password@localhost:9030/my_database"
      }
    }
  }
}
```

**Command-line Arguments:**

The server supports the following command-line arguments:

```bash
uv run mcp-server-starrocks --help
```

- `--mode {stdio,sse,http,streamable-http}`: Transport mode (default: stdio or MCP_TRANSPORT_MODE env var)
- `--host HOST`: Server host for HTTP modes (default: localhost)
- `--port PORT`: Server port for HTTP modes
- `--test`: Run in test mode to verify functionality

Examples:

```bash
# Start in streamable HTTP mode on custom host/port
uv run mcp-server-starrocks --mode streamable-http --host 0.0.0.0 --port 8080

# Start in stdio mode (default)
uv run mcp-server-starrocks --mode stdio

# Run test mode
uv run mcp-server-starrocks --test
```

- The `url` field should point to the Streamable HTTP endpoint of your MCP server (adjust host/port as needed).
- With this configuration, clients can interact with the server using standard JSON over HTTP POST requests. No special SDK is required.
- All tool APIs accept and return standard JSON as described above.

> **Note:**
> The `sse` (Server-Sent Events) mode is deprecated and no longer maintained. Please use Streamable HTTP mode for all new integrations.

**Environment Variables:**

### Connection Configuration

You can configure StarRocks connection using either individual environment variables or a single connection URL:

**Option 1: Individual Environment Variables**

- `STARROCKS_HOST`: (Optional) Hostname or IP address of the StarRocks FE service. Defaults to `localhost`.
- `STARROCKS_PORT`: (Optional) MySQL protocol port of the StarRocks FE service. Defaults to `9030`.
- `STARROCKS_USER`: (Optional) StarRocks username. Defaults to `root`.
- `STARROCKS_PASSWORD`: (Optional) StarRocks password. Defaults to empty string.
- `STARROCKS_PASSWORD_KEYCHAIN_SERVICE`: (Optional, macOS only) Generic password service name to use when reading the password from Keychain. This is only used when no explicit password is provided via `STARROCKS_PASSWORD` or `STARROCKS_URL`.
- `STARROCKS_PASSWORD_KEYCHAIN_ACCOUNT`: (Optional, macOS only) Generic password account name to use when reading the password from Keychain. Defaults to the resolved StarRocks user.
- `STARROCKS_DB`: (Optional) Default database to use if not specified in tool arguments or resource URIs. If set, the connection will attempt to `USE` this database. Tools like `table_overview` and `db_overview` will use this if the database part is omitted in their arguments. Defaults to empty (no default database).

**Option 2: Connection URL (takes precedence over individual variables)**

- `STARROCKS_URL`: (Optional) A connection URL string that contains all connection parameters in a single variable. Format: `[<schema>://]user:password@host:port/database`. The schema part is optional. When this variable is set, it takes precedence over the individual `STARROCKS_HOST`, `STARROCKS_PORT`, `STARROCKS_USER`, `STARROCKS_PASSWORD`, and `STARROCKS_DB` variables.

  Examples:
  - `root:mypass@localhost:9030/test_db`
  - `mysql://admin:secret@db.example.com:9030/production`  
  - `starrocks://user:pass@192.168.1.100:9030/analytics`

Password precedence:
- A password embedded in `STARROCKS_URL` wins, including an explicit empty password like `user:@host:9030/db`.
- If `STARROCKS_URL` omits the password, `STARROCKS_PASSWORD` is used when set.
- If neither explicit password source is set and `STARROCKS_PASSWORD_KEYCHAIN_SERVICE` is configured, the password is read from macOS Keychain.

**macOS Keychain example**

Store the password:

```bash
security add-generic-password -U -a root -s mcp-server-starrocks -w 'secret'
```

Verify the stored password:

```bash
security find-generic-password -a root -s mcp-server-starrocks -w
```

Use it with this server:

```bash
export STARROCKS_URL=root@localhost:9030/test_db
export STARROCKS_PASSWORD_KEYCHAIN_SERVICE=mcp-server-starrocks
export STARROCKS_PASSWORD_KEYCHAIN_ACCOUNT=root
```

### Additional Configuration

- `STARROCKS_FE_ARROW_FLIGHT_SQL_PORT`: (Optional) Arrow Flight SQL port of the StarRocks FE service. When set, the server connects using the high-performance Arrow Flight SQL protocol (via ADBC drivers) instead of the standard MySQL protocol. Leave unset to use the default MySQL connection. The host, user, and password are taken from the same connection settings described above.

- `STARROCKS_OVERVIEW_LIMIT`: (Optional) An _approximate_ character limit for the _total_ text generated by overview tools (`table_overview`, `db_overview`) when fetching data to populate the cache. This helps prevent excessive memory usage for very large schemas or numerous tables. Defaults to `20000`.

- `STARROCKS_MCP_OUTPUT_DIR`: (Optional) Directory used by `read_query` when its `output_file` argument is a relative path. Defaults to `~/.mcp-server-starrocks/output/`. The directory is created on demand. Absolute paths passed to `output_file` (including `~`-prefixed paths) bypass this setting. **Note:** files are written on the machine where the MCP server runs. For Claude Code / Claude Desktop the server runs locally, so files land on your laptop. For remote/http deployments the file lands on the server, not the client.

- `STARROCKS_CHART_OUTPUT_DIR`: (Optional) Directory where `query_and_plotly_chart` writes interactive HTML charts (when `format="html"`). Defaults to the system temp directory. The directory is created on demand. **Note:** like other output files, charts are written on the machine where the MCP server runs.

- `STARROCKS_CHART_INCLUDE_PLOTLYJS`: (Optional) Controls how `plotly.js` is bundled into HTML charts. `cdn` (default) keeps files small but needs network access when viewing; `inline`/`true` embeds the full library for offline use; `directory` and `false` are also accepted (passed through to Plotly's `write_html`).

- `STARROCKS_CHART_DEFAULT_FORMAT`: (Optional) Default output format for `query_and_plotly_chart` when the `format` argument is omitted. One of `json`, `png`, `jpeg` (default), or `html`. Set to `html` to always write an interactive chart file to `STARROCKS_CHART_OUTPUT_DIR` (with an inline PNG preview) without passing `format` on every call. Invalid values fall back to `jpeg` with a warning.

- `STARROCKS_MYSQL_AUTH_PLUGIN`: (Optional) Specifies the authentication plugin to use when connecting to the StarRocks FE service. For example, set to `mysql_clear_password` if your StarRocks deployment requires clear text password authentication (such as when using certain LDAP or external authentication setups). Only set this if your environment specifically requires it; otherwise, the default auth_plugin is used.

### TLS / SSL Configuration

These variables control TLS for the connection. When none of them are set, the underlying `mysql.connector` keeps its default behavior (`ssl-mode=PREFERRED`): the connection is encrypted if the server supports TLS, but the server certificate is **not** verified. For real security, provide a CA certificate and enable verification.

- `STARROCKS_SSL_DISABLED`: (Optional) Set to `true` to force-disable TLS. Overrides all other SSL settings. Defaults to `false`.
- `STARROCKS_SSL_CA`: (Optional) Path to the CA certificate (PEM) used to verify the StarRocks server certificate.
- `STARROCKS_SSL_CERT`: (Optional) Path to the client certificate (PEM) for mutual TLS (mTLS).
- `STARROCKS_SSL_KEY`: (Optional) Path to the client private key (PEM) for mutual TLS (mTLS).
- `STARROCKS_SSL_VERIFY_CERT`: (Optional) Set to `true` to verify the server certificate against the CA. Defaults to `false`.
- `STARROCKS_SSL_VERIFY_IDENTITY`: (Optional) Set to `true` to also verify that the server hostname matches the certificate. Defaults to `false`.
- `STARROCKS_TLS_VERSIONS`: (Optional) Comma-separated list of allowed TLS versions, e.g. `TLSv1.2,TLSv1.3`.

Example (verify the server against a CA certificate):

```json
"env": {
  "STARROCKS_HOST": "your-fe-host",
  "STARROCKS_PORT": "9030",
  "STARROCKS_USER": "root",
  "STARROCKS_PASSWORD": "your-password",
  "STARROCKS_SSL_CA": "/path/to/ca.pem",
  "STARROCKS_SSL_VERIFY_CERT": "true",
  "STARROCKS_SSL_VERIFY_IDENTITY": "true"
}
```

For the high-performance **Arrow Flight SQL** connection (enabled via `STARROCKS_FE_ARROW_FLIGHT_SQL_PORT`), TLS is controlled separately:

- `STARROCKS_FE_ARROW_FLIGHT_SQL_USE_TLS`: (Optional) Set to `true` to use `grpc+tls://` instead of plaintext `grpc://`. When enabled, `STARROCKS_SSL_CA` is used as the TLS root certificate and `STARROCKS_SSL_VERIFY_CERT=false` (default) skips server certificate verification.

> Security note: avoid storing plaintext passwords directly in `mcp.json`. Prefer injecting `STARROCKS_PASSWORD` (and certificate paths) from a secrets manager or environment, and never commit credentials to version control.

- `MCP_TRANSPORT_MODE`: (Optional) Communication mode that specifies how the MCP Server exposes its services. Available options:
  - `stdio` (default): Communicates through standard input/output, suitable for MCP Host hosting.
  - `streamable-http` (Streamable HTTP): Starts as a Streamable HTTP Server, supporting RESTful API calls.
  - `sse`: **(Deprecated, not recommended)** Starts in Server-Sent Events (SSE) streaming mode, suitable for scenarios requiring streaming responses. **Note: SSE mode is no longer maintained, it is recommended to use Streamable HTTP mode uniformly.**

## Components

### Tools

- `read_query`

  - **Description:** Execute a SELECT query or other commands that return a ResultSet (e.g., `SHOW`, `DESCRIBE`). Optionally write the full result to a local file instead of returning it inline — useful for results too large to fit in the model context.
  - **Input:**
    ```json
    {
      "query": "SQL query string",
      "db": "database name (optional, uses default database if not specified)",
      "output_file": "optional path; if set, writes the full result to disk and returns only a summary + small preview. Relative paths resolve against STARROCKS_MCP_OUTPUT_DIR (default: ~/.mcp-server-starrocks/output/); absolute paths and ~ are used as-is",
      "output_format": "optional: csv | tsv | json | jsonl. If omitted, inferred from output_file extension (.csv/.tsv/.json/.jsonl/.ndjson); defaults to csv"
    }
    ```
  - **Output:** Without `output_file`, text content containing the query results in CSV-like format with a header row and row count summary. With `output_file`, a short summary including the resolved absolute path, byte count, and row count, plus a small preview. Returns an error message on failure.

- `write_query`

  - **Description:** Execute a DDL (`CREATE`, `ALTER`, `DROP`), DML (`INSERT`, `UPDATE`, `DELETE`), or other StarRocks command that does not return a ResultSet.
  - **Input:** 
    ```json
    {
      "query": "SQL command string",
      "db": "database name (optional, uses default database if not specified)"
    }
    ```
  - **Output:** Text content confirming success (e.g., "Query OK, X rows affected") or reporting an error. Changes are committed automatically on success.

- `analyze_query`

  - **Description:** Analyze a query and get analyze result using query profile or explain analyze.
  - **Input:**
    ```json
    {
      "uuid": "Query ID, a string composed of 32 hexadecimal digits formatted as 8-4-4-4-12",
      "sql": "Query SQL to analyze",
      "db": "database name (optional, uses default database if not specified)"
    }
    ```
  - **Output:** Text content containing the query analysis results. Uses `ANALYZE PROFILE FROM` if uuid is provided, otherwise uses `EXPLAIN ANALYZE` if sql is provided.

- `top_hot_tables`

  - **Description:** Get top hot tables by audit-log visit count. It joins `information_schema.tables` with `starrocks_audit_db__.starrocks_audit_tbl__`, excludes `root` and `SHOW` statements, matches audit SQL text against table names, and orders by `visit_count` descending.
  - **Input:**
    ```json
    {
      "db": "optional database/schema filter",
      "table": "optional table name substring filter",
      "min_start_time_ms": 1704067200000,
      "max_start_time_ms": 1704153600000,
      "top_n": 20
    }
    ```
  - **Output:** Text summary plus structured content containing ranked rows with `db`, `table`, and `visit_count`.

- `top_bad_tables`

  - **Description:** Get top bad tables by table health score, following Star Management Studio's `top-bad-tables` logic. It reuses the table-health calculation based on `information_schema.be_tablets` and `information_schema.partitions_meta`, filters out system schemas, orders by `table_health_score` ascending, and returns the lowest-scoring tables.
  - **Input:**
    ```json
    {
      "db": "optional database/schema filter",
      "table": "optional table name substring filter",
      "top_n": 20
    }
    ```
  - **Output:** Text summary plus structured content containing ranked rows with table health fields such as `db`, `table`, `tablet_num`, `replica_score`, `tablet_score`, and `table_health_score`.

- `query_and_plotly_chart`

  - **Description:** Executes a SQL query, loads the results into a Pandas DataFrame, and generates a Plotly chart using a provided Python expression. Designed for visualization in supporting UIs.
  - **Input:**
    ```json
    {
      "query": "SQL query to fetch data",
      "plotly_expr": "Python expression string using 'px' (Plotly Express) and 'df' (DataFrame). Example: 'px.scatter(df, x=\"col1\", y=\"col2\")'",
      "db": "database name (optional, uses default database if not specified)"
    }
    ```
  - **Output:** A list containing:
    1.  `TextContent`: A text representation of the DataFrame and a note that the chart is for UI display.
    2.  `ImageContent`: The generated Plotly chart encoded as a base64 PNG image (`image/png`). Returns text error message on failure or if the query yields no data.

- `table_overview`

  - **Description:** Get an overview of a specific table: columns (from `DESCRIBE`), total row count, and sample rows (`LIMIT 3`). Uses an in-memory cache unless `refresh` is true.
  - **Input:**
    ```json
    {
      "table": "Table name, optionally prefixed with database name (e.g., 'db_name.table_name' or 'table_name'). If database is omitted, uses STARROCKS_DB environment variable if set.",
      "refresh": false // Optional, boolean. Set to true to bypass the cache. Defaults to false.
    }
    ```
  - **Output:** Text content containing the formatted overview (columns, row count, sample data) or an error message. Cached results include previous errors if applicable.

- `db_overview`
  - **Description:** Get an overview (columns, row count, sample rows) for _all_ tables within a specified database. Uses the table-level cache for each table unless `refresh` is true.
  - **Input:**
    ```json
    {
      "db": "database_name", // Optional if default database is set.
      "refresh": false // Optional, boolean. Set to true to bypass the cache for all tables in the DB. Defaults to false.
    }
    ```
  - **Output:** Text content containing concatenated overviews for all tables found in the database, separated by headers. Returns an error message if the database cannot be accessed or contains no tables.

### Resources

#### Direct Resources

- `starrocks:///databases`
  - **Description:** Lists all databases accessible to the configured user.
  - **Equivalent Query:** `SHOW DATABASES`
  - **MIME Type:** `text/plain`

#### Resource Templates

- `starrocks:///{db}/{table}/schema`

  - **Description:** Gets the schema definition of a specific table.
  - **Equivalent Query:** `SHOW CREATE TABLE {db}.{table}`
  - **MIME Type:** `text/plain`

- `starrocks:///{db}/tables`

  - **Description:** Lists all tables within a specific database.
  - **Equivalent Query:** `SHOW TABLES FROM {db}`
  - **MIME Type:** `text/plain`

- `proc:///{+path}`
  - **Description:** Accesses StarRocks internal system information, similar to Linux `/proc`. The `path` parameter specifies the desired information node.
  - **Equivalent Query:** `SHOW PROC '/{path}'`
  - **MIME Type:** `text/plain`
  - **Common Paths:**
    - `/frontends` - Information about FE nodes.
    - `/backends` - Information about BE nodes (for non-cloud native deployments).
    - `/compute_nodes` - Information about CN nodes (for cloud native deployments).
    - `/dbs` - Information about databases.
    - `/dbs/<DB_ID>` - Information about a specific database by ID.
    - `/dbs/<DB_ID>/<TABLE_ID>` - Information about a specific table by ID.
    - `/dbs/<DB_ID>/<TABLE_ID>/partitions` - Partition information for a table.
    - `/transactions` - Transaction information grouped by database.
    - `/transactions/<DB_ID>` - Transaction information for a specific database ID.
    - `/transactions/<DB_ID>/running` - Running transactions for a database ID.
    - `/transactions/<DB_ID>/finished` - Finished transactions for a database ID.
    - `/jobs` - Information about asynchronous jobs (Schema Change, Rollup, etc.).
    - `/statistic` - Statistics for each database.
    - `/tasks` - Information about agent tasks.
    - `/cluster_balance` - Load balance status information.
    - `/routine_loads` - Information about Routine Load jobs.
    - `/colocation_group` - Information about Colocation Join groups.
    - `/catalog` - Information about configured catalogs (e.g., Hive, Iceberg).

### Prompts

None defined by this server.

## Caching Behavior

- The `table_overview` and `db_overview` tools utilize an in-memory cache to store the generated overview text.
- The cache key is a tuple of `(database_name, table_name)`.
- When `table_overview` is called, it checks the cache first. If a result exists and the `refresh` parameter is `false` (default), the cached result is returned immediately. Otherwise, it fetches the data from StarRocks, stores it in the cache, and then returns it.
- When `db_overview` is called, it lists all tables in the database and then attempts to retrieve the overview for _each table_ using the same caching logic as `table_overview` (checking cache first, fetching if needed and `refresh` is `false` or cache miss). If `refresh` is `true` for `db_overview`, it forces a refresh for _all_ tables in that database.
- The `STARROCKS_OVERVIEW_LIMIT` environment variable provides a _soft target_ for the maximum length of the overview string generated _per table_ when populating the cache, helping to manage memory usage.
- Cached results, including any error messages encountered during the original fetch, are stored and returned on subsequent cache hits.

## Debug

After starting mcp server, you can use inspector to debug:
```
npx @modelcontextprotocol/inspector
```

## Demo

![MCP Demo Image](mcpserverdemo.jpg)
