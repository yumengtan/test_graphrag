# processors/relationship_extractor.py
import re
import json
from typing import List, Dict, Any, Tuple
from base.chunk import Chunk, Entity, Relationship
from models.local_llm import LocalLLM


class RelationshipExtractor:
    """Extract entities and relationships using local LLM"""
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        
    def extract_entities_and_relationships(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a chunk using LLM"""
        
        # Specialized extraction for tabular data
        if chunk.source_type in ['csv', 'xlsx']:
            return self._extract_from_tabular(chunk)
        
        # Extract from text using LLM
        return self._extract_from_text(chunk)
    
    def _extract_from_tabular(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from tabular data"""
        # This is handled by TabularExtractor
        # Here we just parse the chunk content if it contains structured data
        
        entities = []
        relationships = []
        
        # Simple parsing logic for demonstration
        # In practice, you'd use the TabularExtractor directly
        lines = chunk.content.split('\n')
        for i, line in enumerate(lines):
            if 'Record' in line and '{' in line:
                try:
                    # Extract JSON data from the line
                    json_start = line.find('{')
                    json_end = line.rfind('}') + 1
                    record_data = json.loads(line[json_start:json_end])
                    
                    # Create entity for the record
                    entity = Entity(
                        id=f"record_{i}",
                        type="Record",
                        name=f"Record {i}",
                        properties=record_data,
                        description=f"Extracted from tabular data"
                    )
                    entities.append(entity)
                    
                except json.JSONDecodeError:
                    pass
        
        return entities, relationships
    
    def _extract_from_text(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text using LLM"""
        
        prompt = self._create_extraction_prompt(chunk.content)
        response = self.llm.generate(prompt)
        
        # Parse the LLM response
        entities, relationships = self._parse_llm_response(response, chunk.chunk_id)
        
        return entities, relationships
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM to extract entities and relationships"""
        
        prompt = f"""Extract entities and relationships from the following text in JSON format.
        
Rules:
1. Identify distinct entities (people, organizations, locations, concepts)
2. Identify relationships between entities
3. Return in this exact format:

{{
    "entities": [
        {{
            "id": "entity_1",
            "type": "Person|Organization|Location|Concept",
            "name": "entity name",
            "description": "brief description"
        }}
    ],
    "relationships": [
        {{
            "id": "rel_1",
            "source_entity_id": "entity_1",
            "target_entity_id": "entity_2",
            "type": "relationship_type",
            "description": "relationship description"
        }}
    ]
}}

Text to analyze:
{text}

JSON output:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response into entities and relationships"""
        
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
                    entity = Entity(
                        id=f"{chunk_id}_{ent_data.get('id', 'unknown')}",
                        type=ent_data.get('type', 'Unknown'),
                        name=ent_data.get('name', ''),
                        properties={},
                        description=ent_data.get('description', '')
                    )
                    entities.append(entity)
                
                # Process relationships
                for rel_data in data.get('relationships', []):
                    relationship = Relationship(
                        id=f"{chunk_id}_{rel_data.get('id', 'unknown')}",
                        source_entity_id=f"{chunk_id}_{rel_data.get('source_entity_id', '')}",
                        target_entity_id=f"{chunk_id}_{rel_data.get('target_entity_id', '')}",
                        type=rel_data.get('type', 'RELATED'),
                        properties={},
                        description=rel_data.get('description', '')
                    )
                    relationships.append(relationship)
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response}")
        
        return entities, relationships
    
    def extract_relationships_between_chunks(self, chunks: List[Chunk]) -> List[Relationship]:
        """Extract relationships between different chunks"""
        
        relationships = []
        
        # Extract entities from each chunk first
        chunk_entities = {}
        for chunk in chunks:
            entities, _ = self.extract_entities_and_relationships(chunk)
            chunk_entities[chunk.chunk_id] = entities
        
        # Find cross-chunk relationships
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                # Compare entities from both chunks
                chunk1_entities = chunk_entities.get(chunk1.chunk_id, [])
                chunk2_entities = chunk_entities.get(chunk2.chunk_id, [])
                
                for ent1 in chunk1_entities:
                    for ent2 in chunk2_entities:
                        # Find similar entities across chunks
                        if self._are_entities_related(ent1, ent2):
                            rel_id = f"cross_chunk_{chunk1.chunk_id}_{chunk2.chunk_id}_{ent1.id}_{ent2.id}"
                            relationship = Relationship(
                                id=rel_id,
                                source_entity_id=ent1.id,
                                target_entity_id=ent2.id,
                                type="SIMILAR_TO",
                                properties={
                                    "source_chunk": chunk1.chunk_id,
                                    "target_chunk": chunk2.chunk_id,
                                    "similarity_reason": "Same entity across chunks"
                                },
                                description="Cross-chunk entity similarity"
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _are_entities_related(self, ent1: Entity, ent2: Entity) -> bool:
        """Check if two entities are related (likely the same entity)"""
        
        # Simple similarity checks
        if ent1.name.lower() == ent2.name.lower():
            return True
        
        # Check if one name is contained in the other
        if ent1.name.lower() in ent2.name.lower() or ent2.name.lower() in ent1.name.lower():
            return True
        
        # Check for similar types
        if ent1.type == ent2.type and self._name_similarity(ent1.name, ent2.name) > 0.8:
            return True
        
        return False
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity (simple implementation)"""
        
        # Remove common prefixes/suffixes and special characters
        clean_name1 = re.sub(r'[^a-zA-Z0-9\s]', '', name1.lower())
        clean_name2 = re.sub(r'[^a-zA-Z0-9\s]', '', name2.lower())
        
        # Calculate similarity using simple string methods
        if clean_name1 == clean_name2:
            return 1.0
        
        # Check for partial matches
        words1 = set(clean_name1.split())
        words2 = set(clean_name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2))
        
        return similarity