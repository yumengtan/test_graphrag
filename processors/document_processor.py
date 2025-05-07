import os
import logging
import queue
from typing import Dict, Any, List, Tuple, Optional, Union
from urllib.parse import urlparse
import re

from base.chunk import Chunk
from chunkers.text_chunker import TextChunker
from extractors.tabular_extractor import TabularExtractor, CSVExtractor, ExcelExtractor
from models.local_llm import LocalLLM
from extractors.relationship_extractor import RelationshipExtractor
from processors.graph_processor import GraphProcessor
from processors.neo4j_wrapper import Neo4jExtractorWrapper

# Import Neo4j's document loaders
from processors.loader.local_file import get_documents_from_file_by_path
from processors.loader.web_pages import get_documents_from_web_page
from processors.loader.youtube import get_documents_from_youtube
from processors.loader.wikipedia import get_documents_from_Wikipedia

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Try to import Neo4j components
try:
    from langchain_community.graphs import Neo4jGraph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class DocumentProcessor:
    """Enhanced document processor with batched processing and progress tracking"""
    
    def __init__(
        self, 
        llm: Optional[LocalLLM] = None,
        chunk_size: int = 200, 
        chunk_overlap: int = 0,
        batch_size: int = 10,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        num_workers: int = 4
    ):
        self.llm = llm or LocalLLM()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Neo4j connection details
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.neo4j_graph = None
        
        # Initialize components
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_graph = Neo4jGraph(
                    url=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    database=self.neo4j_database
                )
            except Exception as e:
                logging.error(f"Error connecting to Neo4j: {e}")
        
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.relationship_extractor = RelationshipExtractor(self.llm, num_workers=num_workers)
        self.graph_processor = GraphProcessor(self.relationship_extractor)
        
        # For batch processing
        self.processing_queue = queue()
        self.results = {}
        self.failed_chunks = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, source: str) -> Dict[str, Any]:
        """Process document with batched processing and progress tracking"""
        self.logger.info(f"Processing document: {source}")
        
        # Determine source type
        source_type = self._detect_source_type(source)
        
        # Extract documents
        documents = self._extract_documents(source, source_type)
        
        # Preprocess documents
        processed_documents = self._preprocess_documents(documents)
        
        # Chunk documents with relationships
        chunks, chunk_relationships = self._chunk_documents_with_relationships(processed_documents, source, source_type)
        
        # Process chunks in batches
        total_chunks = len(chunks)
        processed_chunks = 0
        success_count = 0
        failure_count = 0
        
        # Add chunks to processing queue in batches
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i+self.batch_size]
            self.processing_queue.put((batch, i))
        
        # Process queue
        while not self.processing_queue.empty():
            batch, batch_index = self.processing_queue.get()
            
            try:
                # Process batch
                batch_results = self._process_batch(batch, chunk_relationships)
                
                # Update counts
                processed_chunks += len(batch)
                success_count += len(batch)
                
                # Store results
                self.results[f"batch_{batch_index}"] = batch_results
                
                progress = (processed_chunks / total_chunks) * 100
                self.logger.info(f"Progress: {progress:.2f}% ({processed_chunks}/{total_chunks} chunks)")
                
            except Exception as e:
                failure_count += len(batch)
                self.logger.error(f"Error processing batch {batch_index}: {e}")
                self.failed_chunks.extend(batch)
        
        # Generate graph
        cypher_statements = self.graph_processor.create_cypher_statements_with_communities(
            chunks, chunk_relationships, self.results
        )
        
        return {
            "source": source,
            "source_type": source_type,
            "total_chunks": total_chunks,
            "processed_chunks": processed_chunks,
            "success_count": success_count,
            "failure_count": failure_count,
            "chunks": [chunk.__dict__ for chunk in chunks],
            "cypher_statements": cypher_statements
        }
    
    def _process_batch(self, batch: List[Chunk], chunk_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of chunks in parallel"""
        # Extract entities and relationships using parallel extractor
        extraction_results = self.relationship_extractor.extract_entities_and_relationships_batch(batch)
        
        # Extract cross-chunk relationships
        cross_relationships = self.relationship_extractor.extract_cross_chunk_relationships(batch)
        
        return {
            "chunks": batch,
            "entities": extraction_results["entities"],
            "relationships": extraction_results["relationships"],
            "cross_relationships": cross_relationships
        }
    
    def _chunk_documents_with_relationships(self, documents: List[Document], source: str, source_type: str) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
        """Chunk documents and create relationships between chunks"""
        all_chunks = []
        all_relationships = []
        
        for i, doc in enumerate(documents):
            # Get document content
            text = doc.page_content
            
            # Get metadata
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            metadata.update({"doc_index": i})
            
            # Create chunks with relationships
            chunks, relationships = self.text_chunker.chunk_text(text, source, source_type, metadata)
            all_chunks.extend(chunks)
            all_relationships.extend(relationships)
            
        return all_chunks, all_relationships
    
    def extract_schema(self) -> str:
        """Extract schema from the Neo4j database
        
        Returns:
            Schema as a string
        """
        if not NEO4J_AVAILABLE or not self.neo4j_graph:
            return "Neo4j not available"
        
        try:
            # Refresh schema if method exists
            if hasattr(self.neo4j_graph, 'refresh_schema'):
                self.neo4j_graph.refresh_schema()
            
            # Return schema if available
            if hasattr(self.neo4j_graph, 'schema'):
                return self.neo4j_graph.schema
            else:
                return "Schema not available"
        except Exception as e:
            self.logger.error(f"Error extracting schema: {e}")
            return f"Error: {str(e)}"
    
    def generate_cypher_query(self, question: str, schema: str = None) -> str:
        """Generate a Cypher query from a natural language question
        
        Args:
            question: Natural language question
            schema: Optional Neo4j schema (if not provided, will be fetched)
            
        Returns:
            Cypher query
        """
        if not self.llm:
            self.logger.warning("LLM not available, cannot generate Cypher query")
            return ""
        
        # Get schema if not provided
        if not schema:
            schema = self.extract_schema()
        
        # Use the template from your testing script
        prompt = f"""You are an expert assistant that translates natural language questions into Cypher queries for querying a Neo4j graph database. 
Use the provided schema to generate an accurate and syntactically correct Cypher query.

### Schema:
{schema}

### Question:
{question}
"""
        
        response = self.llm.generate(prompt, max_tokens=512, temperature=0.1)
        
        # Extract the Cypher query from the response
        query = self._extract_cypher_query_from_response(response)
        
        return query

    def _extract_cypher_query_from_response(self, response: str) -> str:
        """Extract the Cypher query from the LLM response
        
        Args:
            response: LLM response
            
        Returns:
            Extracted Cypher query
        """
        # Look for "### Cypher Query:" marker
        if "### Cypher Query:" in response:
            self.logger.info("Found '### Cypher Query:' marker")
            query_part = response.split("### Cypher Query:")[1].strip()
            
            # Split by any potential markers that might follow
            for marker in ["###", "\n\n"]:
                if marker in query_part:
                    query_part = query_part.split(marker)[0].strip()
            
            return query_part.strip()
        
        # Look for common Cypher keywords
        cypher_keywords = ["MATCH", "CREATE", "MERGE", "RETURN", "WITH", "CALL"]
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(line.upper().startswith(kw) for kw in cypher_keywords):
                self.logger.info(f"Found line starting with Cypher keyword: {line}")
                return line.strip()
        
        # Look for "Cypher:" marker
        if "Cypher:" in response:
            self.logger.info("Found 'Cypher:' marker")
            query_part = response.split("Cypher:")[1].strip()
            
            # Get the first line or until the next marker
            if "\n" in query_part:
                query_part = query_part.split("\n")[0].strip()
                
            return query_part.strip()
        
        self.logger.warning("Could not extract a Cypher query using standard patterns")
        return ""
    
    def _detect_source_type(self, source: str) -> str:
        """Detect the type of source
        
        Args:
            source: Path or URL to document
            
        Returns:
            Source type (file, url, youtube, wikipedia)
        """
        # Check if it's a URL
        if source.startswith(('http://', 'https://')):
            # Check if it's a YouTube URL
            if 'youtube.com' in source or 'youtu.be' in source:
                return 'youtube'
            # Check if it's a Wikipedia URL
            elif 'wikipedia.org' in source:
                return 'wikipedia'
            else:
                return 'url'
        
        # Check if it's a file
        if os.path.isfile(source):
            file_extension = os.path.splitext(source)[1].lower()
            if file_extension in ['.csv']:
                return 'csv'
            elif file_extension in ['.xlsx', '.xls']:
                return 'xlsx'
            elif file_extension in ['.pdf']:
                return 'pdf'
            elif file_extension in ['.docx', '.doc']:
                return 'docx'
            elif file_extension in ['.txt', '.md', '.markdown']:
                return 'text'
            else:
                return 'file'
        
        # Default to text
        return 'text'
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess documents to ensure they have text content
        
        Args:
            documents: List of documents to preprocess
            
        Returns:
            List of preprocessed documents
        """
        preprocessed_documents = []
        
        for doc in documents:
            # Ensure document has page_content
            if not hasattr(doc, 'page_content') or not doc.page_content:
                # If document has no content, skip it
                continue
            
            # Add to preprocessed documents
            preprocessed_documents.append(doc)
        
        return preprocessed_documents

    def _extract_documents(self, source: str, source_type: str) -> List[Document]:
        """Extract documents based on source type
        
        Args:
            source: Path or URL to document
            source_type: Type of source
            
        Returns:
            List of extracted documents
        """
        try:
            if source_type == 'csv':
                # Check if file exists first
                if not os.path.exists(source):
                    self.logger.error(f"File not found: {source}")
                    return [Document(page_content=f"Error: File not found: {source}", 
                                    metadata={"source": source, "error": "File not found"})]
                
                # Extract data using our custom CSV extractor with hybrid approach
                text = self.csv_extractor.extract(source)
                # Split text using RecursiveCharacterTextSplitter immediately
                chunks = self.text_splitter.split_text(text)
                return [Document(page_content=chunk, metadata={"source": source, "chunk_index": i, "total_chunks": len(chunks)}) 
                        for i, chunk in enumerate(chunks)]
                
            elif source_type == 'xlsx':
                # Check if file exists first
                if not os.path.exists(source):
                    self.logger.error(f"File not found: {source}")
                    return [Document(page_content=f"Error: File not found: {source}", 
                                    metadata={"source": source, "error": "File not found"})]
                
                # Extract data using our custom Excel extractor with hybrid approach
                text = self.excel_extractor.extract(source)
                # Split text using RecursiveCharacterTextSplitter immediately
                chunks = self.text_splitter.split_text(text)
                return [Document(page_content=chunk, metadata={"source": source, "chunk_index": i, "total_chunks": len(chunks)}) 
                        for i, chunk in enumerate(chunks)]
            
            # For other document types, use built-in methods with fallbacks
            if source_type == 'youtube':
                try:
                    # YouTube extraction
                    match = re.search(r'(?:v=)([0-9A-Za-z_-]{11})\s*', source)
                    if not match:
                        parsed_url = urlparse(source)
                        path = parsed_url.path.split('/')
                        video_id = path[-1] if path else ""
                    else:
                        video_id = match.group(1)
                    
                    from youtube_transcript_api import YouTubeTranscriptApi
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    text = " ".join([item['text'] for item in transcript])
                    chunks = self.text_splitter.split_text(text)
                    return [Document(page_content=chunk, metadata={"source": source, "video_id": video_id, 
                                                            "chunk_index": i, "total_chunks": len(chunks)}) 
                        for i, chunk in enumerate(chunks)]
                except Exception as e:
                    self.logger.error(f"Error extracting YouTube transcript: {e}")
                    return [Document(page_content=f"Error extracting YouTube transcript: {str(e)}", 
                                    metadata={"source": source, "error": str(e)})]
                    
            elif source_type == 'wikipedia':
                try:
                    # Wikipedia extraction
                    from langchain_community.document_loaders import WikipediaLoader
                    if 'wikipedia.org' in source:
                        parsed_url = urlparse(source)
                        path = parsed_url.path.split('/')
                        query = path[-1].replace('_', ' ') if len(path) > 1 else source
                    else:
                        query = source
                    
                    wiki_docs = WikipediaLoader(query=query, lang="en").load()
                    documents = []
                    
                    for doc in wiki_docs:
                        chunks = self.text_splitter.split_text(doc.page_content)
                        for i, chunk in enumerate(chunks):
                            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                            metadata.update({"chunk_index": i, "total_chunks": len(chunks)})
                            documents.append(Document(page_content=chunk, metadata=metadata))
                    
                    return documents
                except Exception as e:
                    self.logger.error(f"Error extracting Wikipedia content: {e}")
                    return [Document(page_content=f"Error extracting Wikipedia content: {str(e)}", 
                                    metadata={"source": source, "error": str(e)})]
                
            elif source_type == 'url':
                try:
                    # Web page extraction
                    from langchain_community.document_loaders import WebBaseLoader
                    web_docs = WebBaseLoader(source, verify_ssl=False).load()
                    documents = []
                    
                    for doc in web_docs:
                        chunks = self.text_splitter.split_text(doc.page_content)
                        for i, chunk in enumerate(chunks):
                            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                            metadata.update({"chunk_index": i, "total_chunks": len(chunks)})
                            documents.append(Document(page_content=chunk, metadata=metadata))
                    
                    return documents
                except Exception as e:
                    self.logger.error(f"Error extracting web page content: {e}")
                    return [Document(page_content=f"Error extracting web page content: {str(e)}", 
                                    metadata={"source": source, "error": str(e)})]
            
            elif source_type == 'pdf' or source_type == 'docx' or source_type == 'text':
                # Check if file exists first
                if not os.path.exists(source):
                    self.logger.error(f"File not found: {source}")
                    return [Document(page_content=f"Error: File not found: {source}", 
                                    metadata={"source": source, "error": "File not found"})]
                    
                try:
                    # Use appropriate loader based on file type
                    if source_type == 'pdf':
                        from langchain_community.document_loaders import PyMuPDFLoader
                        docs = PyMuPDFLoader(source).load()
                    elif source_type == 'docx':
                        from langchain_community.document_loaders import Docx2txtLoader
                        docs = Docx2txtLoader(source).load()
                    else:  # Text files
                        from langchain_community.document_loaders import TextLoader
                        docs = TextLoader(source, encoding='utf-8').load()
                    
                    documents = []
                    for doc in docs:
                        chunks = self.text_splitter.split_text(doc.page_content)
                        for i, chunk in enumerate(chunks):
                            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                            metadata.update({"chunk_index": i, "total_chunks": len(chunks)})
                            documents.append(Document(page_content=chunk, metadata=metadata))
                    
                    return documents
                except Exception as e:
                    self.logger.error(f"Error extracting document content: {e}")
                    return [Document(page_content=f"Error extracting document content: {str(e)}", 
                                    metadata={"source": source, "error": str(e)})]
            
            # If we reach here, we don't have specific handling for this source type
            self.logger.warning(f"Unknown source type: {source_type}")
            return [Document(page_content=f"Error: Unknown source type: {source_type}", 
                            metadata={"source": source, "error": "Unknown source type"})]
                
        except Exception as e:
            self.logger.error(f"Unhandled error extracting documents: {e}")
            return [Document(page_content=f"Unhandled error extracting content: {str(e)}", 
                            metadata={"source": source, "error": str(e)})]
    
    def _chunk_documents(self, documents: List[Document], source: str, source_type: str) -> List[Chunk]:
        """Chunk documents using RecursiveCharacterTextSplitter
        
        Args:
            documents: List of documents to chunk
            source: Original source path or URL
            source_type: Type of source
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # If documents is empty, return empty list
        if not documents:
            return chunks
        
        for i, doc in enumerate(documents):
            # Get document content
            text = doc.page_content
            
            # Split text using RecursiveCharacterTextSplitter
            text_chunks = self.text_splitter.split_text(text)
            
            # Create Chunk objects
            for j, chunk_text in enumerate(text_chunks):
                chunk_id = f"{source_type}_{hash(source)}_{i}_{j}"
                
                # Get metadata from document
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                metadata.update({
                    "chunk_index": j,
                    "doc_index": i,
                    "total_chunks": len(text_chunks)
                })
                
                # Create Chunk object
                chunk = Chunk(
                    content=chunk_text,
                    source=source,
                    source_type=source_type,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    entities=None,
                    relationships=None
                )
                chunks.append(chunk)
        
        return chunks