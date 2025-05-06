import logging
from typing import List, Optional, Dict, Any
import os
import re
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Neo4jExtractorWrapper:
    """Wrapper for extracting documents with consistent chunking parameters"""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):
        """Initialize the wrapper with chunking parameters
        
        Args:
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
        # Setup text splitter with consistent parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def extract_documents(self, source: str, source_type: str) -> List[Document]:
        """Extract documents based on source type with consistent chunking
        
        Args:
            source: Path or URL to document
            source_type: Type of source
            
        Returns:
            List of extracted documents with consistent chunk sizes
        """
        self.logger.info(f"Extracting documents from {source} of type {source_type}")
        
        try:
            # Extract raw documents using appropriate method
            raw_documents = self._extract_raw_documents(source, source_type)
            
            if not raw_documents:
                self.logger.warning(f"No documents extracted from {source}")
                return []
            
            # Apply consistent chunking
            chunked_documents = self._apply_chunking(raw_documents, source)
            
            self.logger.info(f"Extracted and chunked {len(chunked_documents)} documents from {source}")
            return chunked_documents
            
        except Exception as e:
            self.logger.error(f"Error extracting documents from {source}: {e}")
            return []
    
    def _extract_raw_documents(self, source: str, source_type: str) -> List[Document]:
        """Extract raw documents based on source type
        
        Args:
            source: Path or URL to document
            source_type: Type of source
            
        Returns:
            List of raw documents
        """
        if source_type == 'pdf':
            return self._extract_pdf(source)
        elif source_type == 'docx':
            return self._extract_docx(source)
        elif source_type == 'text':
            return self._extract_text(source)
        elif source_type == 'url':
            return self._extract_url(source)
        elif source_type == 'youtube':
            return self._extract_youtube(source)
        elif source_type == 'wikipedia':
            return self._extract_wikipedia(source)
        else:
            self.logger.warning(f"Unsupported source type: {source_type}")
            return []
    
    def _apply_chunking(self, documents: List[Document], source: str) -> List[Document]:
        """Apply consistent chunking to documents
        
        Args:
            documents: List of documents to chunk
            source: Original source path or URL
            
        Returns:
            List of documents with consistent chunk sizes
        """
        chunked_documents = []
        
        for i, doc in enumerate(documents):
            # Get document content
            text = doc.page_content
            
            # Skip empty documents
            if not text.strip():
                continue
            
            # Split text using RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_text(text)
            
            # Create new documents with chunks
            for j, chunk in enumerate(chunks):
                # Get metadata from document
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                metadata.update({
                    "chunk_index": j,
                    "doc_index": i,
                    "total_chunks": len(chunks),
                    "source": source
                })
                
                # Create new document
                chunked_doc = Document(page_content=chunk, metadata=metadata)
                chunked_documents.append(chunked_doc)
        
        return chunked_documents
    
    def _extract_pdf(self, source: str) -> List[Document]:
        """Extract documents from PDF file"""
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            return PyMuPDFLoader(source).load()
        except ImportError:
            self.logger.warning("PyMuPDFLoader not available, falling back to UnstructuredPDFLoader")
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                return UnstructuredPDFLoader(source).load()
            except ImportError:
                self.logger.error("PDF loaders not available")
                return []
        except Exception as e:
            self.logger.error(f"Error extracting PDF: {e}")
            return []
    
    def _extract_docx(self, source: str) -> List[Document]:
        """Extract documents from DOCX file"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(source).load()
        except ImportError:
            self.logger.warning("Docx2txtLoader not available, falling back to UnstructuredFileLoader")
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader
                return UnstructuredFileLoader(source).load()
            except ImportError:
                self.logger.error("DOCX loaders not available")
                return []
        except Exception as e:
            self.logger.error(f"Error extracting DOCX: {e}")
            return []
    
    def _extract_text(self, source: str) -> List[Document]:
        """Extract documents from text file"""
        try:
            from langchain_community.document_loaders import TextLoader
            return TextLoader(source, encoding='utf-8').load()
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return []
    
    def _extract_url(self, source: str) -> List[Document]:
        """Extract documents from URL"""
        try:
            from langchain_community.document_loaders import WebBaseLoader
            return WebBaseLoader(source, verify_ssl=False).load()
        except Exception as e:
            self.logger.error(f"Error extracting URL: {e}")
            return []
    
    def _extract_youtube(self, source: str) -> List[Document]:
        """Extract documents from YouTube video"""
        try:
            # Extract video ID
            match = re.search(r'(?:v=)([0-9A-Za-z_-]{11})\s*', source)
            if not match:
                parsed_url = urlparse(source)
                path = parsed_url.path.split('/')
                video_id = path[-1] if path else ""
            else:
                video_id = match.group(1)
            
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript segments
            full_text = " ".join([item['text'] for item in transcript])
            
            # Create a single document with the full transcript
            return [Document(page_content=full_text, metadata={"source": source, "video_id": video_id})]
            
        except Exception as e:
            self.logger.error(f"Error extracting YouTube transcript: {e}")
            return []
    
    def _extract_wikipedia(self, source: str) -> List[Document]:
        """Extract documents from Wikipedia article"""
        try:
            from langchain_community.document_loaders import WikipediaLoader
            
            # Extract article name from URL or use source as query
            if 'wikipedia.org' in source:
                parsed_url = urlparse(source)
                path = parsed_url.path.split('/')
                query = path[-1].replace('_', ' ') if len(path) > 1 else source
            else:
                query = source
            
            return WikipediaLoader(query=query, lang="en").load()
            
        except Exception as e:
            self.logger.error(f"Error extracting Wikipedia content: {e}")
            return []