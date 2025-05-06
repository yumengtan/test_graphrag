from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Chunk:
    """Data class to represent a text chunk with metadata"""
    content: str
    source: str
    source_type: str
    chunk_id: str
    metadata: Dict[str, Any]
    entities: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "entities": self.entities,
            "relationships": self.relationships
        }


@dataclass
class Entity:
    """Data class to represent an extracted entity"""
    id: str
    type: str
    name: str
    properties: Dict[str, Any]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "properties": self.properties,
            "description": self.description
        }


@dataclass
class Relationship:
    """Data class to represent a relationship between entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    properties: Dict[str, Any]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "type": self.type,
            "properties": self.properties,
            "description": self.description
        }