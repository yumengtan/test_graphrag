# GraphRAG Document Processor

A modular Python system for processing various document types (URLs, YouTube videos, documents, CSV/XLSX) and creating graph representations for use with Neo4j and LLM-based graph analysis. Uses `RecursiveCharacterTextSplitter` with chunk size 200 and no overlap for consistent chunking across all document types.

## Overview

This system takes various input types and:
1. Extracts text content from the source
2. Chunks the text using RecursiveCharacterTextSplitter (chunk size 200, no overlap)
3. Uses a local LLM to extract entities and relationships
4. Generates Cypher statements for Neo4j graph database

## Key Features

- Support for multiple document types:
  - Local files (PDF, DOCX, TXT)
  - Web pages
  - YouTube videos
  - Wikipedia articles
  - CSV/XLSX files (custom implementation)
- Dynamic relationship and entity extraction
- Integration with Neo4j's LLM Graph Builder for most document types
- Custom implementation for CSV/XLSX processing
- Configurable chunking with RecursiveCharacterTextSplitter

## Project Structure

```
graphrag/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── chunk.py             # Data classes for chunks, entities, and relationships
│   ├── extractor.py         # Base extractor class
│   └── processor.py         # Base processor class
├── extractors/
│   ├── __init__.py
│   ├── text_extractor.py    # Plain text extractor
│   ├── web_extractor.py     # Web page extractor
│   ├── youtube_extractor.py # YouTube extractor
│   ├── pdf_extractor.py     # PDF extractor
│   ├── document_extractor.py # DOCX extractor
│   └── tabular_extractor.py # CSV/XLSX extractor
├── processors/
│   ├── __init__.py
│   ├── document_processor.py # Main document processing orchestrator
│   ├── graph_processor.py    # Cypher statement generator
│   ├── neo4j_wrapper.py      # Wrapper for Neo4j's existing extractors
│   ├── neo4j_integration.py  # Integration with Neo4j database
│   └── relationship_extractor.py # Entity and relationship extractor
├── chunkers/
│   ├── __init__.py
│   └── text_chunker.py      # Text chunking utilities
├── models/
│   ├── __init__.py
│   └── local_llm.py         # Interface for local LLM models
└── main.py                  # Main entry point
```

## Dependencies

- Python 3.8+
- langchain
- langchain_community
- langchain_core
- pytesseract
- pdf2image
- python-docx
- pandas
- requests
- beautifulsoup4
- youtube_transcript_api
- neo4j_llm_graph_builder (optional, for neo4j integration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/graphrag.git
cd graphrag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up a local LLM or API endpoint for entity extraction.

## Usage

### Command-line Interface

Process a document using the command-line interface:

```bash
python main.py --source "path/to/document.pdf" --chunk_size 200 --chunk_overlap 0 --output_dir output
```

Options:
- `--source`: Path or URL to the document to process
- `--model_path`: (Optional) Path to local LLM model
- `--endpoint`: (Optional) API endpoint for LLM
- `--chunk_size`: (Optional) Chunk size for text splitting (default: 200)
- `--chunk_overlap`: (Optional) Chunk overlap for text splitting (default: 0)
- `--output_dir`: (Optional) Output directory for results (default: 'output')
- `--neo4j_uri`: (Optional) Neo4j database URI
- `--neo4j_username`: (Optional) Neo4j username
- `--neo4j_password`: (Optional) Neo4j password
- `--import_to_neo4j`: (Optional) Import data directly into Neo4j
- `--clear_neo4j`: (Optional) Clear Neo4j graph before import

### Direct Neo4j Import

Import data directly into Neo4j:

```bash
python main.py --source "path/to/document.pdf" --import_to_neo4j --neo4j_uri "bolt://localhost:7687" --neo4j_username "neo4j" --neo4j_password "password"
```

You can also set environment variables for Neo4j credentials:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
python main.py --source "path/to/document.pdf" --import_to_neo4j
```

### Python API

```python
from processors.document_processor import DocumentProcessor
from models.local_llm import LocalLLM

# Initialize local LLM
llm = LocalLLM(model_path="path/to/model")

# Initialize document processor
processor = DocumentProcessor(
    llm=llm,
    chunk_size=200,
    chunk_overlap=0
)

# Process document
result = processor.process_document("path/to/document.pdf")

# Access results
chunks = result['chunks']
cypher_statements = result['cypher_statements']
```

## Integration with Neo4j's Extractors

This system is designed to work with Neo4j's existing document extractors while ensuring consistent chunking using RecursiveCharacterTextSplitter. The key components of this integration are:

### Neo4j Wrapper

The `Neo4jExtractorWrapper` class wraps Neo4j's existing document extractors and replaces their chunking mechanisms with our own RecursiveCharacterTextSplitter:

- It first extracts content using Neo4j's methods
- Then it applies our chunking parameters (chunk size 200, no overlap)
- This ensures consistent chunk sizes across all document types

### Direct Neo4j Database Integration

The system can directly import graph data into a Neo4j database using the `Neo4jIntegration` class:

- Executes Cypher statements to create nodes and relationships
- Can clear the graph before import if needed
- Can generate import scripts for later use

### CSV/XLSX Processing

Since Neo4j's LLM Graph Builder doesn't include processors for tabular data, we've implemented custom extractors:

- `CSVExtractor` and `ExcelExtractor` classes
- Intelligently detects potential primary keys and foreign key relationships
- Creates entities and relationships based on column names and values
- Generates Cypher statements compatible with Neo4j's graph data model

## Custom LLM Integration

The system is designed to work with any local LLM that can extract entities and relationships from text. By default, it includes a mock LLM implementation for development and testing.

To use a custom LLM, extend the `LocalLLM` class in `models/local_llm.py`.

## License

MIT License