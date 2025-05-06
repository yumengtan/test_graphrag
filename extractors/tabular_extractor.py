import pandas as pd
import csv
from typing import Dict, Any, Optional, List, Tuple, Generator, Iterator
from base.extractor import BaseExtractor
from base.chunk import Entity, Relationship
import logging


class TabularExtractor(BaseExtractor):
    """Extract structured data from CSV/Excel files and convert to graph format"""
    
    def __init__(self, llm=None):
        self.column_relationships = {
            # Common column relationship patterns
            'id': ['name', 'email', 'title', 'department'],
            'employee_id': ['name', 'position', 'department'],
            'user_id': ['username', 'email', 'role'],
            'customer_id': ['company', 'contact_person', 'address'],
            'project_id': ['project_name', 'owner', 'status'],
            'order_id': ['customer_id', 'product_id', 'quantity'],
            'product_id': ['product_name', 'category', 'price']
        }
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.chunk_size = 1000  # Number of rows to process at once
    
    def validate(self, source: str) -> bool:
        """Validate if file is CSV or Excel"""
        return source.lower().endswith(('.csv', '.xlsx', '.xls'))
    
    def extract(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Extract data from CSV/Excel and return as structured text
        
        Uses chunking to process large files in manageable pieces.
        """
        self.logger.info(f"Extracting data from {source}")
        
        if source.lower().endswith('.csv'):
            return self._extract_csv(source)
        else:
            return self._extract_excel(source)
    
    def _extract_csv(self, source: str) -> str:
        """Extract data from CSV file with memory-efficient processing"""
        try:
            # First pass to get column information without loading the whole file
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                columns = next(reader)  # Get header row
            
            # Generate structured text for the file metadata
            text_parts = []
            text_parts.append(f"Table: {source}")
            text_parts.append(f"Columns: {', '.join(columns)}")
            
            # Count rows efficiently without loading the whole file
            row_count = 0
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for _ in reader:
                    row_count += 1
            
            text_parts.append(f"Rows: {row_count}")
            text_parts.append("\nData Sample:")
            
            # Process the file in chunks to extract sample data
            chunk_count = 0
            sample_rows = []
            
            for chunk in self._read_csv_in_chunks(source, chunk_size=10):  # Get first 10 rows for sample
                if chunk_count == 0:  # Only process the first chunk for the sample
                    sample_rows = chunk.values.tolist()
                    break
                chunk_count += 1
            
            # Format sample data
            for i, row in enumerate(sample_rows):
                row_str = ", ".join([f"{col}: {val}" for col, val in zip(columns, row)])
                text_parts.append(f"Row {i+1}: {row_str}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error extracting CSV data: {e}")
            return f"Error extracting data from {source}: {str(e)}"
    
    def _extract_excel(self, source: str) -> str:
        """Extract data from Excel file with memory-efficient processing"""
        try:
            # Load Excel file with only sheet names and metadata
            xlsx = pd.ExcelFile(source)
            sheet_names = xlsx.sheet_names
            
            text_parts = []
            text_parts.append(f"Excel File: {source}")
            text_parts.append(f"Sheets: {', '.join(sheet_names)}")
            text_parts.append("\nSheet Data:")
            
            # Process each sheet
            for sheet_name in sheet_names:
                # Read just the first few rows to get column information
                df_sample = pd.read_excel(xlsx, sheet_name=sheet_name, nrows=10)
                columns = df_sample.columns.tolist()
                
                # Get row count efficiently without loading entire sheet
                row_count = 0
                for chunk in pd.read_excel(xlsx, sheet_name=sheet_name, chunksize=1000):
                    row_count += len(chunk)
                
                text_parts.append(f"\nSheet: {sheet_name}")
                text_parts.append(f"Columns: {', '.join(map(str, columns))}")
                text_parts.append(f"Rows: {row_count}")
                text_parts.append("Sample Data:")
                text_parts.append(df_sample.head(5).to_string())
                text_parts.append("\n" + "-" * 50)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error extracting Excel data: {e}")
            return f"Error extracting data from {source}: {str(e)}"
    
    def _read_csv_in_chunks(self, source: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks to avoid loading entire file into memory"""
        try:
            for chunk in pd.read_csv(source, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error reading CSV in chunks: {e}")
            yield pd.DataFrame()  # Empty dataframe on error
    
    def _read_excel_in_chunks(self, source: str, sheet_name: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        """Read Excel file in chunks (approximate, as pandas doesn't support native Excel chunking)"""
        try:
            # This is a simple implementation that loads the entire sheet
            # For very large Excel files, consider converting to CSV first
            df = pd.read_excel(source, sheet_name=sheet_name)
            
            # Simulate chunking by yielding portions of the dataframe
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i:i+chunk_size]
                
        except Exception as e:
            self.logger.error(f"Error reading Excel in chunks: {e}")
            yield pd.DataFrame()  # Empty dataframe on error
    
    def extract_entities_and_relationships(self, source: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from tabular data with memory-efficient processing"""
        all_entities = []
        all_relationships = []
        
        if source.lower().endswith('.csv'):
            # Process CSV in chunks
            chunk_index = 0
            for chunk in self._read_csv_in_chunks(source, chunk_size=self.chunk_size):
                self.logger.info(f"Processing CSV chunk {chunk_index} with {len(chunk)} rows")
                
                # Detect primary keys for this chunk
                primary_keys = self._detect_primary_keys(chunk)
                
                # Process chunk
                for idx, row in chunk.iterrows():
                    row_entities, row_relationships = self._extract_from_row(chunk, row, idx, primary_keys)
                    all_entities.extend(row_entities)
                    all_relationships.extend(row_relationships)
                
                chunk_index += 1
                
        else:  # Excel file
            xlsx = pd.ExcelFile(source)
            
            for sheet_name in xlsx.sheet_names:
                self.logger.info(f"Processing Excel sheet: {sheet_name}")
                
                # Process sheet in chunks
                chunk_index = 0
                for chunk in self._read_excel_in_chunks(source, sheet_name, chunk_size=self.chunk_size):
                    self.logger.info(f"Processing Excel chunk {chunk_index} with {len(chunk)} rows")
                    
                    # Detect primary keys for this chunk
                    primary_keys = self._detect_primary_keys(chunk)
                    
                    # Process chunk
                    for idx, row in chunk.iterrows():
                        row_entities, row_relationships = self._extract_from_row(chunk, row, idx, primary_keys)
                        all_entities.extend(row_entities)
                        all_relationships.extend(row_relationships)
                    
                    chunk_index += 1
        
        self.logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
        return all_entities, all_relationships
    
    # The rest of the methods remain the same as in your original implementation
    # ...
    
    def _extract_using_patterns(self, df: pd.DataFrame, source: str) -> Tuple[List[Entity], List[Relationship], float]:
        """Extract entities and relationships using pattern matching
        
        Args:
            df: DataFrame with tabular data
            source: Original source path
            
        Returns:
            Tuple of entities, relationships, and confidence score
        """
        entities = []
        relationships = []
        
        # Detect primary key columns
        primary_keys = self._detect_primary_keys(df)
        
        # Calculate initial confidence based on primary key detection
        confidence = len(primary_keys) / max(1, len(df.columns)) if df.shape[1] > 0 else 0
        
        # Extract entities from each row
        rows_with_entities = 0
        for idx, row in df.iterrows():
            row_entities, row_relationships = self._extract_from_row(df, row, idx, primary_keys)
            if row_entities:
                rows_with_entities += 1
            entities.extend(row_entities)
            relationships.extend(row_relationships)
        
        # Adjust confidence based on percentage of rows with detected entities
        if len(df) > 0:
            row_coverage = rows_with_entities / len(df)
            confidence = (confidence + row_coverage) / 2
        
        # Extract column relationships
        column_relationships = self._extract_column_relationships(df)
        relationships.extend(column_relationships)
        
        # Adjust confidence based on relationship detection
        if len(df.columns) > 1:
            relationship_coverage = len(column_relationships) / ((len(df.columns) * (len(df.columns) - 1)) / 2)
            confidence = (confidence + relationship_coverage) / 2
        
        # Create source entity
        source_entity = Entity(
            id=f"source_{hash(source)}",
            type="DataSource",
            name=source.split('/')[-1],
            properties={"file_path": source, "rows": len(df), "columns": len(df.columns)},
            description=f"Data source: {source}"
        )
        entities.append(source_entity)
        
        return entities, relationships, confidence
    
    def _extract_using_llm(self, df: pd.DataFrame, source: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using LLM
        
        Args:
            df: DataFrame with tabular data
            source: Original source path
            
        Returns:
            Tuple of entities and relationships
        """
        if self.llm is None:
            self.logger.warning("LLM not available for extraction")
            return [], []
        
        try:
            # Prepare sample data for LLM
            sample_size = min(5, len(df))
            sample_df = df.head(sample_size)
            
            # Convert sample to string representation
            sample_str = sample_df.to_string()
            
            # Create prompt for LLM
            prompt = self._create_llm_extraction_prompt(df.columns.tolist(), sample_str, source)
            
            # Generate LLM response
            llm_response = self.llm.generate(prompt)
            
            # Parse LLM response
            entities, relationships = self._parse_llm_response(llm_response, source)
            
            return entities, relationships
            
        except Exception as e:
            self.logger.error(f"Error in LLM extraction: {e}")
            return [], []
    
    def _create_llm_extraction_prompt(self, columns: List[str], sample_data: str, source: str) -> str:
        """Create prompt for LLM extraction
        
        Args:
            columns: List of column names
            sample_data: String representation of sample data
            source: Source path
            
        Returns:
            Prompt string
        """
        prompt = f"""Analyze this tabular data and identify entities and relationships for a graph database.

Source file: {source}
Columns: {', '.join(columns)}

Sample data:
{sample_data}

Please identify:
1. The main entities (nodes) in this data
2. Relationships between these entities
3. Properties for each entity
4. Any primary keys or identifiers

Format your response as a valid JSON object with the following structure:
{{
    "entities": [
        {{
            "id": "entity_type_1",
            "type": "EntityType",
            "name": "descriptive_name",
            "properties": ["property1", "property2"],
            "primary_key": "key_column_name"
        }}
    ],
    "relationships": [
        {{
            "source_entity": "entity_type_1",
            "target_entity": "entity_type_2",
            "type": "RELATIONSHIP_TYPE",
            "properties": ["property1", "property2"]
        }}
    ]
}}

Focus on creating a meaningful graph structure that represents the semantic relationships in this data.
"""
        return prompt
    
    def _parse_llm_response(self, response: str, source: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response to extract entities and relationships
        
        Args:
            response: LLM response string
            source: Source path
            
        Returns:
            Tuple of entities and relationships
        """
        entities = []
        relationships = []
        
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Process entities
                for ent_data in data.get('entities', []):
                    entity_id = f"{ent_data.get('id', 'unknown')}_{hash(source)}"
                    entity_type = ent_data.get('type', 'Unknown')
                    entity_name = ent_data.get('name', '')
                    
                    # Create properties dictionary
                    properties = {}
                    for prop in ent_data.get('properties', []):
                        properties[prop] = prop
                    
                    if 'primary_key' in ent_data and ent_data['primary_key']:
                        properties['primary_key'] = ent_data['primary_key']
                    
                    entity = Entity(
                        id=entity_id,
                        type=entity_type,
                        name=entity_name,
                        properties=properties,
                        description=f"Generated from {source}"
                    )
                    entities.append(entity)
                
                # Process relationships
                for rel_data in data.get('relationships', []):
                    source_entity = f"{rel_data.get('source_entity', 'unknown')}_{hash(source)}"
                    target_entity = f"{rel_data.get('target_entity', 'unknown')}_{hash(source)}"
                    rel_type = rel_data.get('type', 'RELATED_TO')
                    
                    # Create properties dictionary
                    properties = {}
                    for prop in rel_data.get('properties', []):
                        properties[prop] = prop
                    
                    relationship = Relationship(
                        id=f"rel_{source_entity}_{target_entity}",
                        source_entity_id=source_entity,
                        target_entity_id=target_entity,
                        type=rel_type,
                        properties=properties,
                        description=f"Generated from {source}"
                    )
                    relationships.append(relationship)
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            self.logger.debug(f"Problematic response: {response}")
        
        return entities, relationships
    
    def _merge_extraction_results(
        self, 
        pattern_entities: List[Entity], 
        pattern_relationships: List[Relationship],
        llm_entities: List[Entity], 
        llm_relationships: List[Relationship],
        pattern_confidence: float
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Merge results from pattern-based and LLM-based extraction
        
        Args:
            pattern_entities: Entities from pattern matching
            pattern_relationships: Relationships from pattern matching
            llm_entities: Entities from LLM
            llm_relationships: Relationships from LLM
            pattern_confidence: Confidence in pattern matching results
            
        Returns:
            Merged entities and relationships
        """
        # Start with all LLM results
        merged_entities = llm_entities.copy()
        merged_relationships = llm_relationships.copy()
        
        # Add pattern entities that don't conflict with LLM entities
        llm_entity_ids = {e.id for e in llm_entities}
        llm_entity_types = {e.type for e in llm_entities}
        
        for entity in pattern_entities:
            # If this is a high-confidence entity and doesn't conflict with LLM entities
            if entity.id not in llm_entity_ids and (pattern_confidence > 0.7 or entity.type not in llm_entity_types):
                merged_entities.append(entity)
        
        # Add pattern relationships that don't conflict with LLM relationships
        llm_rel_keys = {(r.source_entity_id, r.target_entity_id, r.type) for r in llm_relationships}
        
        for relationship in pattern_relationships:
            rel_key = (relationship.source_entity_id, relationship.target_entity_id, relationship.type)
            if rel_key not in llm_rel_keys:
                merged_relationships.append(relationship)
        
        return merged_entities, merged_relationships
    
    def _detect_primary_keys(self, df: pd.DataFrame) -> List[str]:
        """Detect potential primary key columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of potential primary key column names
        """
        primary_keys = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column name contains 'id' or other key patterns
            has_id_pattern = any(pattern in col_lower for pattern in self.id_patterns)
            
            # Check if values are unique
            is_unique = df[col].nunique() == len(df) and len(df) > 1
            
            # Check if it's one of our known relationship keys
            is_known_key = any(key in col_lower for key in self.column_relationships.keys())
            
            if (has_id_pattern and is_unique) or (is_unique and is_known_key):
                primary_keys.append(col)
        
        return primary_keys
    
    def _extract_from_row(self, df: pd.DataFrame, row: pd.Series, idx: int, primary_keys: List[str]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single row
        
        Args:
            df: DataFrame containing the data
            row: Series representing a single row
            idx: Row index
            primary_keys: List of detected primary key columns
            
        Returns:
            Tuple of entities and relationships
        """
        entities = []
        relationships = []
        
        # Skip rows with missing values in primary keys
        if primary_keys and all(pd.isna(row[key]) for key in primary_keys if key in row):
            return [], []
        
        # Create entity for the main record (entire row)
        main_entity_id = f"record_{idx}"
        main_entity_type = "Record"
        
        # Determine better entity type from primary keys
        if primary_keys:
            for key in primary_keys:
                if key in df.columns:
                    # Extract entity type from primary key name (e.g., "customer_id" -> "Customer")
                    key_parts = key.lower().split('_')
                    if len(key_parts) > 1 and key_parts[-1] in self.id_patterns:
                        main_entity_type = key_parts[0].capitalize()
                        break
        
        main_entity = Entity(
            id=main_entity_id,
            type=main_entity_type,
            name=f"{main_entity_type} {idx}",
            properties=row.dropna().to_dict(),
            description=f"Row {idx} from the table"
        )
        entities.append(main_entity)
        
        # Extract entities for values in specific columns
        for col, value in row.items():
            if pd.notna(value):
                col_lower = col.lower()
                
                # Create entity for values in id-like columns
                is_id_col = any(pattern in col_lower for pattern in self.id_patterns)
                if is_id_col or col in primary_keys:
                    # Determine entity type from column name
                    entity_type = col.replace('_id', '').replace('_', ' ').title().replace(' ', '')
                    
                    value_entity = Entity(
                        id=f"{col}_{value}",
                        type=entity_type,
                        name=str(value),
                        properties={"column": col, "value": value},
                        description=f"{col}: {value}"
                    )
                    entities.append(value_entity)
                    
                    # Create relationship from main entity to value entity
                    relationship = Relationship(
                        id=f"{main_entity_id}_has_{col}_{value}",
                        source_entity_id=main_entity_id,
                        target_entity_id=value_entity.id,
                        type="HAS",
                        properties={"column": col},
                        description=f"{main_entity_type} has {entity_type}"
                    )
                    relationships.append(relationship)
                
                # Check for column relationships
                for key, related_cols in self.column_relationships.items():
                    if key in col_lower:
                        for related_col in related_cols:
                            # Try both exact match and pattern match
                            matched_cols = [c for c in df.columns if c.lower() == related_col.lower() or related_col.lower() in c.lower()]
                            for matched_col in matched_cols:
                                if matched_col in row.index and pd.notna(row[matched_col]):
                                    related_value = row[matched_col]
                                    
                                    related_entity = Entity(
                                        id=f"{key}_{value}_{matched_col}_{related_value}",
                                        type=matched_col.replace('_', ' ').title().replace(' ', ''),
                                        name=str(related_value),
                                        properties={
                                            "column": matched_col,
                                            "value": related_value,
                                            "parent_column": col,
                                            "parent_value": value
                                        },
                                        description=f"{matched_col}: {related_value}"
                                    )
                                    entities.append(related_entity)
                                    
                                    # Create relationship
                                    relationship = Relationship(
                                        id=f"{col}_{value}_has_{matched_col}_{related_value}",
                                        source_entity_id=f"{col}_{value}",
                                        target_entity_id=related_entity.id,
                                        type="HAS",
                                        properties={"relationship_type": col + "_to_" + matched_col},
                                        description=f"{col} has {matched_col}"
                                    )
                                    relationships.append(relationship)
        
        return entities, relationships
    
    def _extract_column_relationships(self, df: pd.DataFrame) -> List[Relationship]:
        """Extract relationships between columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Find foreign key relationships by analyzing column names
        for col1 in df.columns:
            col1_lower = col1.lower()
            
            for col2 in df.columns:
                col2_lower = col2.lower()
                
                if col1 != col2:
                    # Check if col1 might be a foreign key to col2
                    if self._is_potential_foreign_key(col1, col2, df):
                        # Determine relationship type
                        rel_type = "REFERENCES"
                        
                        # Try to get a more meaningful relationship type
                        if "_id" in col1_lower and col1_lower.replace("_id", "") in col2_lower:
                            rel_type = "IS"
                        elif col1_lower in self.id_patterns and any(pattern in col2_lower for pattern in self.name_patterns):
                            rel_type = "HAS_NAME"
                        
                        relationship = Relationship(
                            id=f"column_{col1}_references_{col2}",
                            source_entity_id=f"column_{col1}",
                            target_entity_id=f"column_{col2}",
                            type=rel_type,
                            properties={
                                "source_column": col1,
                                "target_column": col2,
                                "relationship_type": "foreign_key"
                            },
                            description=f"Column {col1} references {col2}"
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _is_potential_foreign_key(self, col1: str, col2: str, df: pd.DataFrame) -> bool:
        """Check if col1 might be a foreign key referencing col2
        
        Args:
            col1: First column name
            col2: Second column name
            df: DataFrame containing the data
            
        Returns:
            True if col1 might be a foreign key to col2
        """
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Check naming patterns
        name_match = (
            col1_lower.replace('_id', '') in col2_lower or 
            col2_lower.replace('_id', '') in col1_lower
        )
        
        # If no name match, it's probably not a foreign key
        if not name_match:
            return False
        
        # Check if columns have compatible data types
        try:
            col1_values = set(df[col1].dropna())
            col2_values = set(df[col2].dropna())
            
            # Check if values in col1 are subset of col2
            if len(col1_values) > 0 and len(col2_values) > 0:
                # Check if majority of col1 values are in col2
                matches = sum(1 for v in col1_values if v in col2_values)
                match_ratio = matches / len(col1_values) if len(col1_values) > 0 else 0
                
                return match_ratio > 0.5
            
            return False
        except Exception:
            # If comparing values fails, fall back to name matching only
            return '_id' in col1_lower and col1_lower.replace('_id', '') in col2_lower


# Example specific extractors for other file types
class CSVExtractor(TabularExtractor):
    """Specialized CSV extractor"""
    
    def validate(self, source: str) -> bool:
        return source.lower().endswith('.csv')


class ExcelExtractor(TabularExtractor):
    """Specialized Excel extractor"""
    
    def validate(self, source: str) -> bool:
        return source.lower().endswith(('.xlsx', '.xls'))
    
    def extract(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Extract data from all sheets in Excel file"""
        xlsx = pd.ExcelFile(source)
        text_parts = []
        
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            text_parts.append(f"\nSheet: {sheet_name}")
            text_parts.append(f"Columns: {', '.join(df.columns)}")
            text_parts.append(f"Rows: {len(df)}")
            text_parts.append(df.to_string())
            text_parts.append("\n" + "-" * 50)
        
        return "\n".join(text_parts)