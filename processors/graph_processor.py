from typing import List, Dict, Any, Optional
from base.chunk import Chunk, Entity, Relationship
from extractors.relationship_extractor import RelationshipExtractor


class GraphProcessor:
    """Process chunks into Neo4j graph structure"""
    
    def __init__(self, relationship_extractor: RelationshipExtractor):
        self.relationship_extractor = relationship_extractor
        
    def create_cypher_statements_with_communities(
        self, 
        chunks: List[Chunk], 
        chunk_relationships: List[Dict[str, Any]],
        batch_results: Dict[str, Any]
    ) -> List[str]:
        """Generate Cypher statements with community detection"""
        cypher_statements = []
        
        # Collect all entities and relationships
        all_entities = []
        all_relationships = []
        
        for batch_key, batch_data in batch_results.items():
            if "entities" in batch_data:
                all_entities.extend(batch_data["entities"])
            
            if "relationships" in batch_data:
                all_relationships.extend(batch_data["relationships"])
            
            if "cross_relationships" in batch_data:
                all_relationships.extend(batch_data["cross_relationships"])
        
        # 1. Create Document node
        for source in set(chunk.source for chunk in chunks):
            source_type = chunks[0].source_type  # Assuming all chunks have same source type
            doc_cypher = f"""
            CREATE (:Document {{
                id: '{self._escape_cypher_string(source)}',
                type: '{source_type}',
                name: '{self._escape_cypher_string(source.split('/')[-1])}',
                created_at: timestamp()
            }})
            """
            cypher_statements.append(doc_cypher)
        
        # 2. Create Chunk nodes
        for chunk in chunks:
            chunk_cypher = f"""
            CREATE (:Chunk {{
                id: '{chunk.chunk_id}',
                content: '{self._escape_cypher_string(chunk.content)}',
                source: '{self._escape_cypher_string(chunk.source)}',
                source_type: '{chunk.source_type}',
                chunk_index: {chunk.metadata.get('chunk_index', 0)},
                created_at: timestamp()
            }})
            """
            cypher_statements.append(chunk_cypher)
        
        # 3. Create Document-Chunk relationships
        for chunk in chunks:
            rel_cypher = f"""
            MATCH (d:Document {{id: '{self._escape_cypher_string(chunk.source)}'}})
            MATCH (c:Chunk {{id: '{chunk.chunk_id}'}})
            CREATE (c)-[:PART_OF]->(d)
            """
            cypher_statements.append(rel_cypher)
        
        # 4. Create Chunk-Chunk relationships
        for rel in chunk_relationships:
            rel_props = ", ".join([f"{k}: {v}" for k, v in rel["properties"].items()])
            rel_cypher = f"""
            MATCH (c1:Chunk {{id: '{rel["source_id"]}'}})
            MATCH (c2:Chunk {{id: '{rel["target_id"]}'}})
            CREATE (c1)-[:{rel["type"]} {{{rel_props}}}]->(c2)
            """
            cypher_statements.append(rel_cypher)
        
        # 5. Create Entity nodes
        for entity in all_entities:
            entity_cypher = f"""
            CREATE (:{entity.type} {{
                id: '{entity.id}',
                name: '{self._escape_cypher_string(entity.name)}',
                type: '{entity.type}',
                description: '{self._escape_cypher_string(entity.description or '')}',
                created_at: timestamp()
            }})
            """
            cypher_statements.append(entity_cypher)
        
        # 6. Create Chunk-Entity relationships
        for chunk in chunks:
            if not chunk.entities:
                continue
                
            for entity in chunk.entities:
                rel_cypher = f"""
                MATCH (c:Chunk {{id: '{chunk.chunk_id}'}})
                MATCH (e:{entity.type} {{id: '{entity.id}'}})
                CREATE (c)-[:HAS_ENTITY]->(e)
                """
                cypher_statements.append(rel_cypher)
        
        # 7. Create Entity-Entity relationships
        for rel in all_relationships:
            rel_cypher = f"""
            MATCH (s {{id: '{rel.source_entity_id}'}})
            MATCH (t {{id: '{rel.target_entity_id}'}})
            CREATE (s)-[:{rel.type} {{
                id: '{rel.id}',
                description: '{self._escape_cypher_string(rel.description or '')}'
            }}]->(t)
            """
            cypher_statements.append(rel_cypher)
        
        # 8. Create community detection Cypher statements
        community_cypher = """
        // Create entity projection for community detection
        CALL gds.graph.project('entity_graph',
            '*',
            {
                RELATED_TO: {
                    type: '*',
                    orientation: 'UNDIRECTED'
                }
            }
        )
        
        // Run Louvain community detection
        CALL gds.louvain.write('entity_graph', {
            writeProperty: 'community',
            relationshipWeightProperty: null
        })
        
        // Create community nodes
        MATCH (e)
        WHERE e.community IS NOT NULL
        WITH distinct e.community AS communityId, collect(e) AS communityMembers
        CREATE (c:__Community__ {
            id: 'community_' + communityId,
            name: 'Community ' + communityId,
            size: size(communityMembers),
            level: 0,
            created_at: timestamp()
        })
        
        // Create entity to community relationships
        WITH *
        UNWIND communityMembers AS entity
        MATCH (c:__Community__ {id: 'community_' + entity.community})
        CREATE (entity)-[:IN_COMMUNITY]->(c)
        
        // Clean up
        CALL gds.graph.drop('entity_graph')
        """
        cypher_statements.append(community_cypher)
        
        return cypher_statements
    
    def _escape_cypher_string(self, text: str) -> str:
        """Escape special characters for Cypher queries"""
        if not text:
            return ""
        return text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    
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