"""
QueryAgent: Structured policy query agent
Handles multi-condition queries with pagination and result formatting
"""
import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection


class QueryAgent:
    """Agent for structured policy queries"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize QueryAgent
        
        Args:
            db_path: Optional database path (uses config default if not provided)
        """
        if db_path:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        else:
            self.conn = get_connection()
    
    def query(
        self,
        filters: Dict[str, Any],
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        Execute structured query with multiple conditions
        
        Args:
            filters: Query conditions dictionary with keys:
                - country_code: str or List[str] - Country code(s)
                - year_range: Tuple[int, int] - (start_year, end_year)
                - year: int - Specific year
                - target_sector: str or List[str] - Target sector(s)
                - instrument_type: str or List[str] - Instrument type(s)
                - geographic_scope: str or List[str] - Geographic scope(s)
                - policy_name: str - Policy name (partial match)
            page: Page number (1-indexed)
            page_size: Number of results per page
        
        Returns:
            Dictionary with:
                - results: List of policy records
                - count: Total number of matching records
                - page: Current page number
                - page_size: Results per page
                - total_pages: Total number of pages
        """
        # Build WHERE clause and parameters
        conditions, params = self._build_conditions(filters)
        
        # Count total results
        count_query = f"""
        SELECT COUNT(*) as total
        FROM policies
        WHERE {conditions}
        """
        cursor = self.conn.execute(count_query, params)
        total_count = cursor.fetchone()['total']
        
        # Calculate pagination
        offset = (page - 1) * page_size
        total_pages = (total_count + page_size - 1) // page_size
        
        # Execute main query with pagination
        query = f"""
        SELECT 
            id,
            country_code,
            policy_name,
            policy_content,
            year,
            instrument_type,
            target_sector,
            geographic_scope,
            temporal_scope,
            quantitative_targets,
            policy_actors,
            implementation_mechanisms,
            extracted_entities
        FROM policies
        WHERE {conditions}
        ORDER BY year DESC, id DESC
        LIMIT ? OFFSET ?
        """
        
        cursor = self.conn.execute(query, params + [page_size, offset])
        results = [self._format_result(dict(row)) for row in cursor.fetchall()]
        
        return {
            'results': results,
            'count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages
        }
    
    def _build_conditions(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build SQL WHERE conditions from filters
        
        Args:
            filters: Query filters dictionary
        
        Returns:
            Tuple of (conditions_string, parameters_list)
        """
        conditions = []
        params = []
        
        # Country code filter
        if 'country_code' in filters:
            country_code = filters['country_code']
            if isinstance(country_code, list):
                placeholders = ','.join(['?' for _ in country_code])
                conditions.append(f"country_code IN ({placeholders})")
                params.extend(country_code)
            else:
                conditions.append("country_code = ?")
                params.append(country_code)
        
        # Year range filter
        if 'year_range' in filters:
            start_year, end_year = filters['year_range']
            conditions.append("year BETWEEN ? AND ?")
            params.extend([start_year, end_year])
        
        # Specific year filter
        if 'year' in filters:
            conditions.append("year = ?")
            params.append(filters['year'])
        
        # Target sector filter (JSON array contains)
        if 'target_sector' in filters:
            target_sector = filters['target_sector']
            if isinstance(target_sector, list):
                sector_conditions = []
                for sector in target_sector:
                    sector_conditions.append("target_sector LIKE ?")
                    params.append(f"%{sector}%")
                conditions.append(f"({' OR '.join(sector_conditions)})")
            else:
                conditions.append("target_sector LIKE ?")
                params.append(f"%{target_sector}%")
        
        # Instrument type filter (JSON array contains)
        if 'instrument_type' in filters:
            instrument_type = filters['instrument_type']
            if isinstance(instrument_type, list):
                instrument_conditions = []
                for instrument in instrument_type:
                    instrument_conditions.append("instrument_type LIKE ?")
                    params.append(f"%{instrument}%")
                conditions.append(f"({' OR '.join(instrument_conditions)})")
            else:
                conditions.append("instrument_type LIKE ?")
                params.append(f"%{instrument_type}%")
        
        # Geographic scope filter (JSON array contains)
        if 'geographic_scope' in filters:
            geographic_scope = filters['geographic_scope']
            if isinstance(geographic_scope, list):
                geo_conditions = []
                for geo in geographic_scope:
                    geo_conditions.append("geographic_scope LIKE ?")
                    params.append(f"%{geo}%")
                conditions.append(f"({' OR '.join(geo_conditions)})")
            else:
                conditions.append("geographic_scope LIKE ?")
                params.append(f"%{geographic_scope}%")
        
        # Policy name filter (partial match)
        if 'policy_name' in filters:
            conditions.append("policy_name LIKE ?")
            params.append(f"%{filters['policy_name']}%")
        
        # If no conditions, return "1=1" (match all)
        if not conditions:
            return "1=1", []
        
        return " AND ".join(conditions), params
    
    def _format_result(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format query result by parsing JSON fields
        
        Args:
            row: Raw database row as dictionary
        
        Returns:
            Formatted result dictionary
        """
        result = {
            'id': row['id'],
            'country_code': row['country_code'],
            'policy_name': row['policy_name'],
            'policy_content': row['policy_content'],
            'year': row['year']
        }
        
        # Parse JSON fields
        json_fields = [
            'instrument_type',
            'target_sector',
            'geographic_scope',
            'temporal_scope',
            'quantitative_targets',
            'policy_actors',
            'implementation_mechanisms',
            'extracted_entities'
        ]
        
        for field in json_fields:
            if row[field]:
                try:
                    result[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = row[field]
            else:
                result[field] = None
        
        return result
    
    def get_by_id(self, policy_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single policy by ID
        
        Args:
            policy_id: Policy ID
        
        Returns:
            Formatted policy record or None if not found
        """
        query = """
        SELECT 
            id,
            country_code,
            policy_name,
            policy_content,
            year,
            instrument_type,
            target_sector,
            geographic_scope,
            temporal_scope,
            quantitative_targets,
            policy_actors,
            implementation_mechanisms,
            extracted_entities
        FROM policies
        WHERE id = ?
        """
        
        cursor = self.conn.execute(query, [policy_id])
        row = cursor.fetchone()
        
        if row:
            return self._format_result(dict(row))
        return None
    
    def insert_policy(self, policy_data: Dict[str, Any]) -> int:
        """
        Insert a new policy into the database
        
        Args:
            policy_data: Policy data dictionary with keys:
                - country_code: str
                - policy_name: str
                - policy_content: str
                - year: int
                - extracted_entities: str (JSON string)
        
        Returns:
            ID of the inserted policy
        """
        # Parse entities if provided
        entities = {}
        if 'extracted_entities' in policy_data:
            if isinstance(policy_data['extracted_entities'], str):
                entities = json.loads(policy_data['extracted_entities'])
            else:
                entities = policy_data['extracted_entities']
        
        query = """
        INSERT INTO policies (
            country_code,
            policy_name,
            policy_content,
            year,
            instrument_type,
            target_sector,
            geographic_scope,
            temporal_scope,
            quantitative_targets,
            policy_actors,
            implementation_mechanisms,
            extracted_entities
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = [
            policy_data.get('country_code'),
            policy_data.get('policy_name'),
            policy_data.get('policy_content'),
            policy_data.get('year'),
            json.dumps(entities.get('instrument_type', []), ensure_ascii=False),
            json.dumps(entities.get('target_sector', []), ensure_ascii=False),
            json.dumps(entities.get('geographic_scope', []), ensure_ascii=False),
            json.dumps(entities.get('temporal_scope', {}), ensure_ascii=False),
            json.dumps(entities.get('quantitative_targets', []), ensure_ascii=False),
            json.dumps(entities.get('policy_actors', {}), ensure_ascii=False),
            json.dumps(entities.get('implementation_mechanisms', []), ensure_ascii=False),
            json.dumps(entities, ensure_ascii=False) if entities else None
        ]
        
        cursor = self.conn.execute(query, params)
        self.conn.commit()
        
        return cursor.lastrowid
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
