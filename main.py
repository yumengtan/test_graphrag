import os
import argparse
import logging
import json
import time
import concurrent.futures
from queue import Queue
import math
from typing import List, Dict, Any, Tuple, Optional
import textwrap

from models.local_llm import LocalLLM
from base.chunk import Chunk, Entity, Relationship
from chunkers.text_chunker import TextChunker
from extractors.relationship_extractor import RelationshipExtractor
from processors.document_processor import DocumentProcessor
from processors.graph_processor import GraphProcessor

def main():
    """
    Main entry point for the optimized GraphRAG document processor
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process documents for GraphRAG')
    parser.add_argument('--source', type=str, help='Source file or URL')
    parser.add_argument('--model_path', type=str, default=None, help='Path to local LLM model')
    parser.add_argument('--model_type', type=str, default='safetensors', choices=['safetensors', 'gguf'], help='Model format type')
    parser.add_argument('--endpoint', type=str, default=None, help='API endpoint for LLM')
    parser.add_argument('--chunk_size', type=int, default=200, help='Chunk size for text splitting')
    parser.add_argument('--chunk_overlap', type=int, default=20, help='Chunk overlap for text splitting')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    # Import Neo4j if needed
    parser.add_argument('--import_to_neo4j', action='store_true', help='Import data directly into Neo4j')
    parser.add_argument('--clear_neo4j', action='store_true', help='Clear Neo4j graph before import')
    
    args = parser.parse_args()
    
    # If no source is provided, prompt for input
    if not args.source:
        args.source = input("Enter the path to your document file: ")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing components...")
        
        # Initialize local LLM
        llm = LocalLLM(
            model_path=args.model_path, 
            endpoint=args.endpoint,
            model_type=args.model_type
        )
        
        # Initialize optimized document processor
        processor = DocumentProcessor(
            llm=llm,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Process document with timing
        logger.info(f"Processing source: {args.source}")
        start_time = time.time()
        result = processor.process_document(args.source)
        process_time = time.time() - start_time
        
        # Report processing stats
        success_rate = (result["success_count"] / result["total_chunks"]) * 100 if result["total_chunks"] > 0 else 0
        logger.info(f"Document processing completed in {process_time:.2f} seconds")
        logger.info(f"Chunks: {result['success_count']}/{result['total_chunks']} ({success_rate:.2f}% success)")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results
        output_base = os.path.basename(args.source).replace(".", "_")
        
        # Save chunks
        chunks_file = os.path.join(args.output_dir, f"{output_base}_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(result['chunks'], f, indent=2)
        
        # Save Cypher statements
        cypher_file = os.path.join(args.output_dir, f"{output_base}_cypher.txt")
        with open(cypher_file, 'w', encoding='utf-8') as f:
            for statement in result['cypher_statements']:
                f.write(statement + "\n\n")
        
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()