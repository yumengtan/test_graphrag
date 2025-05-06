from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from base.chunk import Chunk


class TextChunker:
    """Handles chunking of text for GraphRAG"""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_text(self, text: str, source: str, source_type: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text and create Chunk objects"""
        
        if metadata is None:
            metadata = {}
            
        chunks = self.text_splitter.split_text(text)
        chunk_objects = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_type}_{hash(source)}_{i}"
            chunk_obj = Chunk(
                content=chunk,
                source=source,
                source_type=source_type,
                chunk_id=chunk_id,
                metadata={**metadata, "chunk_index": i},
                entities=None,
                relationships=None
            )
            chunk_objects.append(chunk_obj)
            
        return chunk_objects