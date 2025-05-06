import os
import argparse
import logging
import json
import time
from typing import List, Dict, Any
import textwrap
import csv

from processors.document_processor import DocumentProcessor
from models.local_llm import LocalLLM
from base.chunk import Chunk


def main():
    """
    Main entry point for the GraphRAG document processor
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process documents for GraphRAG')
    parser.add_argument('--source', type=str, help='Source file or URL')
    parser.add_argument('--model_path', type=str, default=None, help='Path to local LLM model')
    parser.add_argument('--model_type', type=str, default='safetensors', choices=['safetensors', 'gguf'], help='Model format type')
    parser.add_argument('--endpoint', type=str, default=None, help='API endpoint for LLM')
    parser.add_argument('--chunk_size', type=int, default=200, help='Chunk size for text splitting')
    parser.add_argument('--chunk_overlap', type=int, default=0, help='Chunk overlap for text splitting')  # Changed to 0
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    # Neo4j arguments
    parser.add_argument('--import_to_neo4j', action='store_true', help='Import data directly into Neo4j')
    parser.add_argument('--clear_neo4j', action='store_true', help='Clear Neo4j graph before import')
    
    # Query generation mode
    parser.add_argument('--query_mode', action='store_true', help='Enter query generation mode after processing')
    
    args = parser.parse_args()
    
    # If no source is provided, prompt for input
    if not args.source:
        args.source = input("Enter the path to your document file: ")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize local LLM
        logger.info(f"Initializing LLM with model path: {args.model_path}, type: {args.model_type}")
        llm = LocalLLM(
            model_path=args.model_path, 
            endpoint=args.endpoint,
            model_type=args.model_type
        )
        
        # Initialize document processor
        logger.info(f"Initializing document processor with chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
        processor = DocumentProcessor(
            llm=llm,
        )
        
        # Process document
        logger.info(f"Processing source: {args.source}")
        start_time = time.time()
        result = processor.process_document(args.source)
        process_time = time.time() - start_time
        logger.info(f"Document processing completed in {process_time:.2f} seconds")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results
        output_base = os.path.basename(args.source).replace(".", "_")
        
        # Save chunks
        chunks_file = os.path.join(args.output_dir, f"{output_base}_chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(result['chunks'], f, indent=2)
        logger.info(f"Saved chunks to {chunks_file}")
        
        # Save Cypher statements
        cypher_file = os.path.join(args.output_dir, f"{output_base}_cypher.txt")
        with open(cypher_file, 'w', encoding='utf-8') as f:
            for statement in result['cypher_statements']:
                f.write(statement + "\n\n")
        logger.info(f"Saved Cypher statements to {cypher_file}")
        
        # Extract schema if available
        schema = processor.extract_schema()
        if schema and schema != "Neo4j not available":
            logger.info("\nGenerated Schema:")
            logger.info(textwrap.fill(schema, 80))
        
        # Print summary
        print(f"\nProcessing complete for {args.source}")
        print(f"Source type: {result['source_type']}")
        print(f"Number of chunks: {len(result['chunks'])}")
        print(f"Number of Cypher statements: {len(result['cypher_statements'])}")
        print(f"Results saved to {args.output_dir}")
        
        # Query generation mode
        if args.query_mode:
            print("\nEntering query generation mode. Type 'exit' to quit.")
            
            # Setup output file for query results
            queries_file = os.path.join(args.output_dir, f"{output_base}_queries.csv")
            with open(queries_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['question', 'generated_query']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                query_num = 1
                while True:
                    user_input = input(f"\nQuestion {query_num}: ")
                    if user_input.lower() in ['exit', 'quit', 'done']:
                        break
                    
                    if user_input.strip():  # Only process non-empty questions
                        print(f"Generating Cypher query...")
                        start_time = time.time()
                        generated_query = processor.generate_cypher_query(user_input)
                        query_time = time.time() - start_time
                        
                        print(f"Generated Cypher query in {query_time:.2f} seconds:")
                        print(f"\n{generated_query}\n")
                        
                        # Save to CSV
                        writer.writerow({
                            'question': user_input,
                            'generated_query': generated_query
                        })
                        
                        # Increment question counter
                        query_num += 1
            
            print(f"\nQueries saved to {queries_file}")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()