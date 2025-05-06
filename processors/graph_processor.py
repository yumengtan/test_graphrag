from typing import List, Dict, Any, Optional
from base.chunk import Chunk, Entity, Relationship
from extractors.relationship_extractor import RelationshipExtractor


class GraphProcessor:
    """Process chunks into Neo4j graph structure"""
    
    def __init__(self, relationship_extractor: RelationshipExtractor):
        self.relationship_extractor = relationship_extractor
        
    def create_cypher_statements(self, chunks: List[Chunk]) -> List[str]:
        """Generate Cypher statements for creating nodes and relationships"""
        
        cypher_statements = []
        
        # Process each chunk to extract entities and relationships
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            entities, relationships = self.relationship_extractor.extract_entities_and_relationships(chunk)
            chunk.entities = entities
            chunk.relationships = relationships
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Extract cross-chunk relationships
        cross_relationships = self.relationship_extractor.extract_relationships_between_chunks(chunks)
        all_relationships.extend(cross_relationships)
        
        # Create Cypher statements
        cypher_statements.extend(self._create_document_nodes(chunks))
        cypher_statements.extend(self._create_chunk_nodes(chunks))
        cypher_statements.extend(self._create_entity_nodes(all_entities))
        cypher_statements.extend(self._create_chunk_relationships(chunks))
        cypher_statements.extend(self._create_entity_relationships(all_relationships))
        
        return cypher_statements
    
    def _create_document_nodes(self, chunks: List[Chunk]) -> List[str]:
        """Create document nodes"""
        
        document_nodes = []
        processed_sources = set()
        
        for chunk in chunks:
            if chunk.source not in processed_sources:
                cypher = f"""
                CREATE (:Document {{
                    id: '{self._escape_cypher_string(chunk.source)}',
                    type: '{chunk.source_type}',
                    name: '{self._escape_cypher_string(chunk.source.split('/')[-1])}',
                    created_at: timestamp()
                }})
                """
                document_nodes.append(cypher)
                processed_sources.add(chunk.source)
        
        return document_nodes
    
    def _create_chunk_nodes(self, chunks: List[Chunk]) -> List[str]:
        """Create chunk nodes"""
        
        chunk_nodes = []
        
        for chunk in chunks:
            cypher = f"""
            CREATE (:Chunk {{
                id: '{chunk.chunk_id}',
                content: '{self._escape_cypher_string(chunk.content)}',
                source: '{self._escape_cypher_string(chunk.source)}',
                source_type: '{chunk.source_type}',
                chunk_index: {chunk.metadata.get('chunk_index', 0)},
                created_at: timestamp()
            }})
            """
            chunk_nodes.append(cypher)
        
        return chunk_nodes
    
    def _create_entity_nodes(self, entities: List[Entity]) -> List[str]:
        """Create entity nodes"""
        
        entity_nodes = []
        processed_entities = set()
        
        for entity in entities:
            if entity.id not in processed_entities:
                # Convert properties to string
                props_str = ', '.join([f"{k}: '{self._escape_cypher_string(str(v))}'" for k, v in entity.properties.items()])
                
                cypher = f"""
                CREATE (:{entity.type} {{
                    id: '{entity.id}',
                    name: '{self._escape_cypher_string(entity.name)}',
                    type: '{entity.type}',
                    description: '{self._escape_cypher_string(entity.description or '')}',
                    {props_str}
                }})
                """
                entity_nodes.append(cypher)
                processed_entities.add(entity.id)
        
        return entity_nodes
    
    def _create_chunk_relationships(self, chunks: List[Chunk]) -> List[str]:
        """Create relationships between documents and chunks, and between consecutive chunks"""
        
        relationships = []
        
        for chunk in chunks:
            # Document to Chunk relationship
            doc_chunk_rel = f"""
            MATCH (d:Document {{id: '{self._escape_cypher_string(chunk.source)}'}}), 
                  (c:Chunk {{id: '{chunk.chunk_id}'}})
            CREATE (d)-[:HAS_CHUNK]->(c)
            """
            relationships.append(doc_chunk_rel)
            
            # Chunk to Entity relationships
            if chunk.entities:
                for entity in chunk.entities:
                    chunk_entity_rel = f"""
                    MATCH (c:Chunk {{id: '{chunk.chunk_id}'}}), 
                          (e:{entity.type} {{id: '{entity.id}'}})
                    CREATE (c)-[:CONTAINS]->(e)
                    """
                    relationships.append(chunk_entity_rel)
            
            # Sequential chunk relationships
            chunk_index = chunk.metadata.get('chunk_index', 0)
            if chunk_index > 0:
                prev_chunk_id = f"{chunk.source_type}_{hash(chunk.source)}_{chunk_index - 1}"
                prev_chunk_rel = f"""
                MATCH (c1:Chunk {{id: '{prev_chunk_id}'}}), 
                      (c2:Chunk {{id: '{chunk.chunk_id}'}})
                CREATE (c1)-[:NEXT]->(c2)
                """
                relationships.append(prev_chunk_rel)
        
        return relationships
    
    def _create_entity_relationships(self, relationships: List[Relationship]) -> List[str]:
        """Create relationships between entities"""
        
        cypher_relationships = []
        
        for relationship in relationships:
            # Convert relationship properties to string
            props_str = ', '.join([f"{k}: '{self._escape_cypher_string(str(v))}'" for k, v in relationship.properties.items()])
            
            cypher = f"""
            MATCH (s {{id: '{relationship.source_entity_id}'}}), 
                  (t {{id: '{relationship.target_entity_id}'}})
            CREATE (s)-[:{relationship.type} {{
                id: '{relationship.id}',
                type: '{relationship.type}',
                description: '{self._escape_cypher_string(relationship.description or '')}',
                {props_str}
            }}]->(t)
            """
            cypher_relationships.append(cypher)
        
        return cypher_relationships
    
    def _escape_cypher_string(self, text: str) -> str:
        """Escape special characters for Cypher queries"""
        if not text:
            return ""
        return text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    
    def generate_graph_visualization(self, entities: List[Entity], relationships: List[Relationship]) -> str:
        """Generate a graph visualization of entities and relationships"""
        
        # This would generate a visualization description that can be used in a frontend
        # For simplicity, we'll just return a JSON representation
        
        nodes = []
        edges = []
        
        for entity in entities:
            nodes.append({
                "id": entity.id,
                "label": entity.name,
                "group": entity.type,
                "properties": entity.properties
            })
        
        for relationship in relationships:
            edges.append({
                "id": relationship.id,
                "from": relationship.source_entity_id,
                "to": relationship.target_entity_id,
                "label": relationship.type,
                "properties": relationship.properties
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }