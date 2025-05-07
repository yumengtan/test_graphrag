from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from base.chunk import Chunk


class TextChunker:
    """Handles chunking of text with explicit relationships between chunks"""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_text(self, text: str, source: str, source_type: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
        """Chunk text and create explicit relationships between chunks"""
        if metadata is None:
            metadata = {}
            
        chunks = self.text_splitter.split_text(text)
        chunk_objects = []
        relationships = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_type}_{hash(source)}_{i}"
            chunk_obj = Chunk(
                content=chunk,
                source=source,
                source_type=source_type,
                chunk_id=chunk_id,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                entities=None,
                relationships=None
            )
            chunk_objects.append(chunk_obj)
            
            # Create NEXT_CHUNK relationship with previous chunk
            if i > 0:
                prev_chunk_id = f"{source_type}_{hash(source)}_{i-1}"
                relationship = {
                    "source_id": prev_chunk_id,
                    "target_id": chunk_id,
                    "type": "NEXT_CHUNK",
                    "properties": {"sequence": i}
                }
                relationships.append(relationship)
                
            # Create SIMILAR relationships for overlapping chunks
            if self.chunk_overlap > 0 and i > 0:
                similarity = self._calculate_similarity(chunks[i-1], chunk)
                if similarity > 0.5:  # Arbitrary threshold
                    relationship = {
                        "source_id": f"{source_type}_{hash(source)}_{i-1}",
                        "target_id": chunk_id,
                        "type": "SIMILAR",
                        "properties": {"similarity_score": similarity}
                    }
                    relationships.append(relationship)
        
        return chunk_objects, relationships
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0