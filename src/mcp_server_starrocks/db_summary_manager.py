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

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class ColumnInfo:
    name: str
    column_type: str
    ordinal_position: int


@dataclass
class TableInfo:
    name: str
    database: str
    size_bytes: int = 0
    size_str: str = ""
    replica_count: int = 0
    columns: List[ColumnInfo] = field(default_factory=list)
    create_statement: Optional[str] = None
    last_updated: float = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = time.time()
    
    @staticmethod
    def parse_size_string(size_str: str) -> int:
        """Parse size strings like '1.285 GB', '714.433 MB', '2.269 KB' to bytes"""
        if not size_str or size_str == "0" or size_str.lower() == "total":
            return 0
            
        # Handle special cases
        if size_str.lower() in ["quota", "left"]:
            return 0
            
        # Match pattern like "1.285 GB"
        match = re.match(r'([\d.]+)\s*([KMGT]?B)', size_str.strip(), re.IGNORECASE)
        if not match:
            return 0
            
        value, unit = match.groups()
        try:
            num_value = float(value)
        except ValueError:
            return 0
            
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4
        }
        
        multiplier = multipliers.get(unit.upper(), 1)
        return int(num_value * multiplier)
    
    def is_large_table(self) -> bool:
        """Determine if table is considered large (replica_count > 64 OR size > 2GB)"""
        return self.replica_count > 64 or self.size_bytes > (2 * 1024 ** 3)
    
    def priority_score(self) -> float:
        """Calculate priority score combining size and replica count for sorting"""
        # Normalize size to GB and combine with replica count
        size_gb = self.size_bytes / (1024 ** 3)
        return size_gb + (self.replica_count * 0.1)  # Weight replica count less than size
    
    def is_expired(self, expire_seconds: int = 120) -> bool:
        """Check if cache entry is expired (default 2 minutes)"""
        return time.time() - self.last_updated > expire_seconds


class DatabaseSummaryManager:
    def __init__(self, db_client):
        self.db_client = db_client
        # Cache: {(database, table_name): TableInfo}
        self.table_cache: Dict[Tuple[str, str], TableInfo] = {}
        # Database last sync time: {database: timestamp}
        self.db_last_sync: Dict[str, float] = {}
    
    def _sync_table_list(self, database: str, force: bool = False) -> bool:
        """Sync table list using SHOW DATA, detect new/dropped tables"""
        current_time = time.time()
        
        # Check if sync is needed (2min expiration or force)
        if not force and database in self.db_last_sync:
            if current_time - self.db_last_sync[database] < 120:
                return True
        
        logger.debug(f"Syncing table list for database {database}")
        
        try:
            # Execute SHOW DATA to get current table list with sizes
            result = self.db_client.execute("SHOW DATA", db=database)
            if not result.success:
                logger.error(f"Failed to sync table list for {database}: {result.error_message}")
                return False
            
            if not result.rows:
                logger.info(f"No tables found in database {database}")
                # Clear cache for this database
                keys_to_remove = [key for key in self.table_cache.keys() if key[0] == database]
                for key in keys_to_remove:
                    del self.table_cache[key]
                self.db_last_sync[database] = current_time
                return True
            
            # Parse current tables from SHOW DATA
            current_tables = {}
            for row in result.rows:
                table_name = row[0]
                # Skip summary rows (Total, Quota, Left)
                if table_name.lower() in ['total', 'quota', 'left']:
                    continue
                    
                size_str = row[1] if len(row) > 1 else ""
                replica_count = int(row[2]) if len(row) > 2 and str(row[2]).isdigit() else 0
                
                size_bytes = TableInfo.parse_size_string(size_str)
                current_tables[table_name] = {
                    'size_str': size_str,
                    'size_bytes': size_bytes,
                    'replica_count': replica_count
                }
            
            # Update cache: add new tables, update existing, remove dropped
            cache_keys_for_db = {key[1]: key for key in self.table_cache.keys() if key[0] == database}
            
            # Add or update existing tables
            for table_name, table_data in current_tables.items():
                cache_key = (database, table_name)
                
                if cache_key in self.table_cache:
                    # Update existing table info
                    table_info = self.table_cache[cache_key]
                    table_info.size_str = table_data['size_str']
                    table_info.size_bytes = table_data['size_bytes']
                    table_info.replica_count = table_data['replica_count']
                    table_info.last_updated = current_time
                else:
                    # Create new table info
                    self.table_cache[cache_key] = TableInfo(
                        name=table_name,
                        database=database,
                        size_str=table_data['size_str'],
                        size_bytes=table_data['size_bytes'],
                        replica_count=table_data['replica_count'],
                        last_updated=current_time
                    )
            
            # Remove dropped tables
            for table_name in cache_keys_for_db:
                if table_name not in current_tables:
                    cache_key = cache_keys_for_db[table_name]
                    del self.table_cache[cache_key]
                    logger.debug(f"Removed dropped table {database}.{table_name} from cache")
            
            self.db_last_sync[database] = current_time
            logger.debug(f"Synced {len(current_tables)} tables for database {database}")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing table list for {database}: {e}")
            return False
    
    def _fetch_column_info(self, database: str, tables: List[str]) -> Dict[str, List[ColumnInfo]]:
        """Fetch column information for all tables using information_schema.columns"""
        if not tables:
            return {}
        
        logger.debug(f"Fetching column info for {len(tables)} tables in {database}")
        
        try:
            # Build query to get column information for all tables
            table_names_quoted = "', '".join(tables)
            query = f"""
            SELECT table_name, column_name, ordinal_position, column_type 
            FROM information_schema.columns 
            WHERE table_schema = '{database}' 
            AND table_name IN ('{table_names_quoted}')
            ORDER BY table_name, ordinal_position
            """
            
            result = self.db_client.execute(query)
            if not result.success:
                logger.error(f"Failed to fetch column info: {result.error_message}")
                return {}
            
            # Group columns by table
            table_columns = {}
            for row in result.rows:
                table_name = row[0]
                column_name = row[1]
                ordinal_position = int(row[2]) if row[2] else 0
                column_type = row[3]
                
                if table_name not in table_columns:
                    table_columns[table_name] = []
                
                table_columns[table_name].append(ColumnInfo(
                    name=column_name,
                    column_type=column_type,
                    ordinal_position=ordinal_position
                ))
            
            logger.debug(f"Fetched column info for {len(table_columns)} tables")
            return table_columns
            
        except Exception as e:
            logger.error(f"Error fetching column information: {e}")
            return {}
    
    def _fetch_create_statement(self, database: str, table: str) -> Optional[str]:
        """Fetch CREATE TABLE statement for large tables"""
        try:
            result = self.db_client.execute(f"SHOW CREATE TABLE `{database}`.`{table}`")
            if result.success and result.rows and len(result.rows[0]) > 1:
                return result.rows[0][1]  # Second column contains CREATE statement
        except Exception as e:
            logger.error(f"Error fetching CREATE statement for {database}.{table}: {e}")
        return None
    
    def get_database_summary(self, database: str, limit: int = 10000, refresh: bool = False) -> str:
        """Generate comprehensive database summary with intelligent prioritization"""
        if not database:
            return "Error: Database name is required"
        
        logger.info(f"Generating database summary for {database}, limit={limit}, refresh={refresh}")
        
        # Sync table list
        if refresh or not self._sync_table_list(database):
            return f"Error: Failed to sync table information for database '{database}'"
        
        # Get all tables for this database from cache
        tables_info = []
        for (db, table_name), table_info in self.table_cache.items():
            if db == database:
                tables_info.append(table_info)
        
        if not tables_info:
            return f"No tables found in database '{database}'"
        
        # Sort tables by priority (large tables first)
        tables_info.sort(key=lambda t: t.priority_score(), reverse=True)
        
        # Check if any table needs column information refresh
        need_column_refresh = refresh or any(not table_info.columns or table_info.is_expired() for table_info in tables_info)
        
        # If any table needs refresh, fetch ALL tables' columns in one query (more efficient)
        if need_column_refresh:
            all_table_names = [table_info.name for table_info in tables_info]
            table_columns = self._fetch_column_info(database, all_table_names)
            
            # Update cache with column information for all tables
            current_time = time.time()
            for table_info in tables_info:
                if table_info.name in table_columns:
                    table_info.columns = table_columns[table_info.name]
                    table_info.last_updated = current_time
        
        # Identify large tables that need CREATE statements
        large_tables = [t for t in tables_info if t.is_large_table()][:10]  # Top 10 large tables
        for table_info in large_tables:
            if refresh or not table_info.create_statement:
                table_info.create_statement = self._fetch_create_statement(database, table_info.name)
                table_info.last_updated = time.time()
        
        # Generate summary output
        return self._format_database_summary(database, tables_info, limit)
    
    def _format_database_summary(self, database: str, tables_info: List[TableInfo], limit: int) -> str:
        """Format database summary with intelligent truncation"""
        lines = []
        lines.append(f"=== Database Summary: '{database}' ===")
        lines.append(f"Total tables: {len(tables_info)}")
        
        # Calculate totals
        total_size = sum(t.size_bytes for t in tables_info)
        total_replicas = sum(t.replica_count for t in tables_info)
        large_tables = [t for t in tables_info if t.is_large_table()]
        
        lines.append(f"Total size: {self._format_bytes(total_size)}")

        current_length = len("\n".join(lines))
        table_limit = min(len(tables_info), 50)  # Show max 50 tables
        
        # Show large tables first with full details
        if large_tables:
            for i, table_info in enumerate(large_tables):
                if current_length > limit * 0.8:  # Reserve 20% for smaller tables
                    lines.append(f"... and {len(large_tables) - i} more large tables")
                    break
                    
                table_summary = self._format_table_info(table_info, detailed=True)
                lines.append(table_summary)
                lines.append("")
                current_length = len("\n".join(lines))
        
        # Show remaining tables with basic info
        remaining_tables = [t for t in tables_info if not t.is_large_table()]
        if remaining_tables and current_length < limit:
            lines.append("--- Other Tables ---")
            
            for i, table_info in enumerate(remaining_tables):
                if current_length > limit:
                    lines.append(f"... and {len(remaining_tables) - i} more tables (use higher limit to see all)")
                    break
                    
                table_summary = self._format_table_info(table_info, detailed=False)
                lines.append(table_summary)
                current_length = len("\n".join(lines))
        
        return "\n".join(lines)
    
    def _format_table_info(self, table_info: TableInfo, detailed: bool = True) -> str:
        """Format individual table information"""
        lines = []
        
        # Basic info line
        size_info = f"{table_info.size_str} ({table_info.replica_count} replicas)"
        lines.append(f"Table: {table_info.name} - {size_info}")
        
        if table_info.error_message:
            lines.append(f"  Error: {table_info.error_message}")
            return "\n".join(lines)
        
        # Show CREATE statement if available, otherwise show column list
        if table_info.create_statement:
            lines.append(table_info.create_statement)
        elif table_info.columns:
            # Sort columns by ordinal position and show as list
            sorted_columns = sorted(table_info.columns, key=lambda c: c.ordinal_position)
            if detailed or len(sorted_columns) <= 20:
                for col in sorted_columns:
                    lines.append(f" {col.name} ({col.column_type})")
            else:
                lines.append(f"  Columns ({len(sorted_columns)}): {', '.join(col.name for col in sorted_columns[:100])}...")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_bytes(bytes_count: int) -> str:
        """Format bytes to human readable string"""
        if bytes_count == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(bytes_count)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"
    
    def clear_cache(self, database: Optional[str] = None):
        """Clear cache for specific database or all databases"""
        if database:
            keys_to_remove = [key for key in self.table_cache.keys() if key[0] == database]
            for key in keys_to_remove:
                del self.table_cache[key]
            if database in self.db_last_sync:
                del self.db_last_sync[database]
            logger.info(f"Cleared cache for database {database}")
        else:
            self.table_cache.clear()
            self.db_last_sync.clear()
            logger.info("Cleared all cache")


# Global instance (will be initialized in server.py)
_db_summary_manager: Optional[DatabaseSummaryManager] = None


def get_db_summary_manager(db_client) -> DatabaseSummaryManager:
    """Get or create global database summary manager instance"""
    global _db_summary_manager
    if _db_summary_manager is None:
        _db_summary_manager = DatabaseSummaryManager(db_client)
    return _db_summary_manager