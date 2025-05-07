# processors/relationship_extractor.py
import re
import json
from typing import List, Dict, Any, Tuple
from base.chunk import Chunk, Entity, Relationship
from models.local_llm import LocalLLM


class RelationshipExtractor:
    """Enhanced extractor with schema-guided extraction and parallel processing"""
    
    def __init__(self, llm, extraction_schemas=None, num_workers=4):
        self.llm = llm
        self.num_workers = num_workers
        self.entity_cache = {}  # For deduplication
        
        # Default extraction schema
        self.schemas = extraction_schemas or [{
            "name": "default",
            "entity_types": ["Person", "Organization", "Location", "Concept"],
            "relationship_types": ["WORKS_FOR", "LOCATED_IN", "RELATED_TO"],
        }]
    
    def extract_entities_and_relationships(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using schema-guided approach"""
        # Select appropriate schema
        schema = self.schemas[0]  # Use default for simplicity
        
        # Create extraction prompt with schema
        prompt = self._create_extraction_prompt_with_schema(
            chunk.content, 
            schema["entity_types"], 
            schema["relationship_types"]
        )
        
        # Get LLM response
        response = self.llm.generate(prompt)
        
        # Parse LLM response
        entities, relationships = self._parse_llm_response_with_deduplication(response, chunk.chunk_id)
        
        return entities, relationships
    
    def extract_entities_and_relationships_batch(self, chunks: List[Chunk]) -> Dict[str, List]:
        """Process multiple chunks in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.extract_entities_and_relationships, chunk): chunk for chunk in chunks}
            
            all_entities = []
            all_relationships = []
            
            for future in concurrent.futures.as_completed(futures):
                chunk = futures[future]
                try:
                    entities, relationships = future.result()
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    
                    # Update chunk
                    chunk.entities = entities
                    chunk.relationships = relationships
                except Exception as e:
                    logging.error(f"Error extracting from chunk {chunk.chunk_id}: {e}")
            
        return {"entities": all_entities, "relationships": all_relationships}
    
    def extract_cross_chunk_relationships(self, chunks: List[Chunk]) -> List[Relationship]:
        """Find relationships between entities in different chunks"""
        cross_relationships = []
        
        # Collect all entities from chunks
        chunk_entities = {}
        for chunk in chunks:
            if not chunk.entities:
                continue
            chunk_entities[chunk.chunk_id] = chunk.entities
        
        # Compare entities across chunks
        for chunk_id1, entities1 in chunk_entities.items():
            for chunk_id2, entities2 in chunk_entities.items():
                if chunk_id1 >= chunk_id2:  # Skip same chunk and avoid duplicate comparisons
                    continue
                
                for entity1 in entities1:
                    for entity2 in entities2:
                        if entity1.type == entity2.type and self._are_entities_similar(entity1, entity2):
                            rel_id = f"cross_{chunk_id1}_{chunk_id2}_{entity1.id}_{entity2.id}"
                            relationship = Relationship(
                                id=rel_id,
                                source_entity_id=entity1.id,
                                target_entity_id=entity2.id,
                                type="SIMILAR_TO",
                                properties={
                                    "source_chunk": chunk_id1,
                                    "target_chunk": chunk_id2
                                },
                                description="Cross-chunk entity similarity"
                            )
                            cross_relationships.append(relationship)
        
        return cross_relationships
    
    def _create_extraction_prompt_with_schema(self, text: str, entity_types: List[str], relationship_types: List[str]) -> str:
        """Create schema-guided extraction prompt"""
        return f"""Extract entities and relationships from the following text in JSON format.
        
Rules:
1. Identify distinct entities of these types: {', '.join(entity_types)}
2. Identify relationships between entities of these types: {', '.join(relationship_types)}
3. Return in this exact format:

{{
    "entities": [
        {{
            "id": "entity_1",
            "type": "One of: {', '.join(entity_types)}",
            "name": "entity name",
            "description": "brief description"
        }}
    ],
    "relationships": [
        {{
            "id": "rel_1",
            "source_entity_id": "entity_1",
            "target_entity_id": "entity_2",
            "type": "One of: {', '.join(relationship_types)}",
            "description": "relationship description"
        }}
    ]
}}

Text to analyze:
{text}

JSON output:"""
    
    def _parse_llm_response_with_deduplication(self, response: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response with entity deduplication"""
        entities = []
        relationships = []
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Process entities with deduplication
                entity_map = {}  # Map from local ID to global ID
                
                for ent_data in data.get('entities', []):
                    entity_type = ent_data.get('type', 'Unknown')
                    entity_name = ent_data.get('name', '')
                    
                    # Create deduplication key
                    dedup_key = f"{entity_type.lower()}:{entity_name.lower()}"
                    
                    # Check if entity already exists in cache
                    if dedup_key in self.entity_cache:
                        # Use existing entity ID for deduplication
                        global_id = self.entity_cache[dedup_key]
                    else:
                        # Create new global ID
                        global_id = f"{chunk_id}_{ent_data.get('id', 'unknown')}"
                        self.entity_cache[dedup_key] = global_id
                    
                    # Map local ID to global ID
                    entity_map[ent_data.get('id', 'unknown')] = global_id
                    
                    # Only add entity if it's new
                    if global_id.startswith(chunk_id):
                        entity = Entity(
                            id=global_id,
                            type=entity_type,
                            name=entity_name,
                            properties={},
                            description=ent_data.get('description', '')
                        )
                        entities.append(entity)
                
                # Process relationships
                for rel_data in data.get('relationships', []):
                    source_local_id = rel_data.get('source_entity_id', '')
                    target_local_id = rel_data.get('target_entity_id', '')
                    
                    # Skip if source or target entity not found
                    if source_local_id not in entity_map or target_local_id not in entity_map:
                        continue
                    
                    # Get global entity IDs
                    source_id = entity_map[source_local_id]
                    target_id = entity_map[target_local_id]
                    
                    relationship = Relationship(
                        id=f"{chunk_id}_{rel_data.get('id', 'unknown')}",
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        type=rel_data.get('type', 'RELATED_TO'),
                        properties={},
                        description=rel_data.get('description', '')
                    )
                    relationships.append(relationship)
                    
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
        
        return entities, relationships
    
    def _are_entities_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities are similar"""
        # Simple name similarity for now
        name1 = entity1.name.lower()
        name2 = entity2.name.lower()
        
        # Exact match
        if name1 == name2:
            return True
        
        # One is contained in the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Word similarity
        words1 = set(name1.split())
        words2 = set(name2.split())
        if words1 and words2:
            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            if similarity > 0.7:  # High threshold
                return True
        
        return False