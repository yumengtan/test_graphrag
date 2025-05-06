from typing import Dict, Any, Optional, List, Union
import json
import os
import logging


class LocalLLM:
    """Interface for local LLM models to be used for GraphRAG"""
    
    def __init__(
        self, 
        model_path: str = None, 
        endpoint: Optional[str] = None, 
        api_key: Optional[str] = None,
        model_type: str = 'safetensors'  # Added model_type parameter
    ):
        """Initialize the local LLM
        
        Args:
            model_path: Path to local model weights (if using local models)
            endpoint: API endpoint (if using hosted models)
            api_key: API key (if needed)
            model_type: Type of model format ('safetensors' or 'gguf')
        """
        self.model_path = model_path
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on configuration"""
        
        if self.endpoint:
            # Use API-based model
            self.logger.info(f"Using API-based model at {self.endpoint}")
            # No actual loading needed for API models
        else:
            # Use local model
            try:
                if self.model_type == 'safetensors':
                    self._load_safetensors_model()
                elif self.model_type == 'gguf':
                    self._load_gguf_model()
                else:
                    self.logger.warning(f"Unknown model type: {self.model_type}")
                    self._setup_mock_llm()
            except Exception as e:
                self.logger.error(f"Error loading local model: {e}")
                self.logger.warning("Falling back to mock LLM for development")
                self._setup_mock_llm()
    
    def _load_safetensors_model(self):
        """Load model from safetensors format"""
        try:
            # Check if transformers is installed
            import importlib
            if importlib.util.find_spec("transformers") is None:
                self.logger.warning("Transformers not installed. Install with: pip install transformers torch accelerate")
                self._setup_mock_llm()
                return
            
            # Import required modules
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Check for GPU availability
            if torch.cuda.is_available():
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                device_map = "auto"
            else:
                self.logger.warning("No GPU detected, using CPU (this will be slow)")
                device_map = "cpu"
            
            # Load model with quantization to reduce memory usage
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                trust_remote_code=True,
                load_in_8bit=True  # Use 8-bit quantization to reduce memory usage
            )
            
            # Create pipeline for easier inference
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            self.logger.info(f"Successfully loaded safetensors model from {self.model_path}")
            
        except ImportError as e:
            self.logger.error(f"Import error loading model: {e}")
            self._setup_mock_llm()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self._setup_mock_llm()
    
    def _load_gguf_model(self):
        """Load model from GGUF format"""
        try:
            # Check if llama-cpp-python is installed
            import importlib
            if importlib.util.find_spec("llama_cpp") is None:
                self.logger.warning("llama_cpp not installed. Install with: pip install llama-cpp-python")
                self._setup_mock_llm()
                return
            
            # Import required modules
            from llama_cpp import Llama
            
            # Load model
            self.logger.info(f"Loading GGUF model from {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,           # Context window size
                n_gpu_layers=-1,      # Use all GPU layers if available
                n_batch=512,          # Batch size for prompt processing
                temperature=0.1,      # Low temperature for deterministic outputs
                max_tokens=512,       # Maximum tokens to generate
                top_p=0.95,           # Top-p sampling parameter
                repeat_penalty=1.15,  # Equivalent to repetition_penalty
                f16_kv=True,          # Use float16 for key/value cache
            )
            
            self.logger.info(f"Successfully loaded GGUF model from {self.model_path}")
            
        except ImportError as e:
            self.logger.error(f"Import error loading model: {e}")
            self._setup_mock_llm()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self._setup_mock_llm()
    
    def _setup_mock_llm(self):
        """Set up a mock LLM for development and testing"""
        
        self.logger.warning("Using mock LLM for development")
        self.model = "mock"
        self.tokenizer = "mock"
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text based on prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated text
        """
        
        if self.endpoint:
            return self._generate_api(prompt, max_tokens, temperature)
        elif self.model_type == 'safetensors':
            return self._generate_safetensors(prompt, max_tokens, temperature)
        elif self.model_type == 'gguf':
            return self._generate_gguf(prompt, max_tokens, temperature)
        else:
            return self._mock_generate(prompt)
    
    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using API"""
        
        try:
            import requests
            
            # Prepare API request
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Send request to API endpoint
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            # Check for successful response
            response.raise_for_status()
            data = response.json()
            
            # Extract text from response (format depends on API)
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("text", "")
            elif "response" in data:
                return data["response"]
            else:
                self.logger.warning(f"Unexpected API response format: {data}")
                return self._mock_generate(prompt)
                
        except Exception as e:
            self.logger.error(f"API generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_safetensors(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using safetensors model"""
        
        if self.model == "mock":
            return self._mock_generate(prompt)
        
        try:
            import torch
            
            # Use pipeline if available
            if hasattr(self, 'pipe'):
                self.logger.info(f"Generating with pipeline")
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.01,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    return_full_text=False,
                )
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    if 'generated_text' in outputs[0]:
                        return outputs[0]['generated_text']
                    else:
                        return str(outputs[0])
                else:
                    return str(outputs)
            
            # Fall back to using model directly
            # Prepare input for model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Set generation parameters
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.1,
                "top_p": 0.95,
                "repetition_penalty": 1.15,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            
            # Check if model has chat method
            if hasattr(self.model, 'chat'):
                # Generate response using chat method
                with torch.no_grad():
                    outputs = self.model.chat(self.tokenizer, prompt, **generation_config)
                    return outputs
            else:
                # Generate response using generate method
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        **generation_config
                    )
                    
                    # Decode response, skipping the input tokens
                    response = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    return response
            
        except Exception as e:
            self.logger.error(f"Safetensors generation error: {e}")
            return self._mock_generate(prompt)
    
    def _generate_gguf(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using GGUF model"""
        
        if self.model == "mock":
            return self._mock_generate(prompt)
        
        try:
            # Generate with llama_cpp
            result = self.model(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.15,
            )
            
            if isinstance(result, dict) and "choices" in result:
                return result["choices"][0]["text"]
            elif isinstance(result, str):
                return result
            else:
                return str(result)
            
        except Exception as e:
            self.logger.error(f"GGUF generation error: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Generate mock responses for development and testing"""
        
        # Check if prompt is asking for entity extraction
        if "extract entities and relationships" in prompt.lower():
            return self._mock_entity_extraction(prompt)
        
        # Check if prompt is analyzing tabular data
        if "analyze this tabular data" in prompt.lower():
            return self._mock_tabular_analysis(prompt)
            
        # Check if prompt is for Cypher query generation
        if "translates natural language questions into Cypher queries" in prompt:
            return self._mock_cypher_generation(prompt)
        
        # Default mock response
        return f"Mock LLM response to: {prompt[:50]}..."
    
    def _mock_entity_extraction(self, prompt: str) -> str:
        """Generate mock entity extraction responses"""
        
        # Extract the text to analyze from the prompt
        text_start = prompt.find("Text to analyze:")
        if text_start != -1:
            text_to_analyze = prompt[text_start + len("Text to analyze:"):].strip()
        else:
            text_to_analyze = ""
        
        # Simple entity extraction based on capitalized words
        entities = []
        entity_id = 1
        detected_entities = set()
        
        # Simple naive entity detection just for demonstration
        for word in text_to_analyze.split():
            clean_word = word.strip(".,;:!?()[]{}\"'")
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1 and clean_word.lower() not in ["the", "a", "an", "and", "but", "or"]:
                if clean_word not in detected_entities:
                    entity_type = "Person"  # Default type
                    if "company" in text_to_analyze.lower() or "organization" in text_to_analyze.lower():
                        entity_type = "Organization"
                    elif "location" in text_to_analyze.lower() or "place" in text_to_analyze.lower():
                        entity_type = "Location"
                    
                    entities.append({
                        "id": f"entity_{entity_id}",
                        "type": entity_type,
                        "name": clean_word,
                        "description": f"Extracted from text: {clean_word}"
                    })
                    
                    detected_entities.add(clean_word)
                    entity_id += 1
        
        # Generate simple relationships
        relationships = []
        rel_id = 1
        
        # Connect entities that appear close to each other in text
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                relationships.append({
                    "id": f"rel_{rel_id}",
                    "source_entity_id": entities[i]["id"],
                    "target_entity_id": entities[i+1]["id"],
                    "type": "MENTIONED_WITH",
                    "description": f"Entities mentioned in proximity"
                })
                rel_id += 1
        
        # Create response JSON
        response = {
            "entities": entities,
            "relationships": relationships
        }
        
        return json.dumps(response, indent=2)
    
    def _mock_tabular_analysis(self, prompt: str) -> str:
        """Generate mock responses for tabular data analysis"""
        
        # Extract column names from prompt
        columns_start = prompt.find("Columns:")
        columns_end = prompt.find("\n", columns_start)
        
        if columns_start != -1 and columns_end != -1:
            columns_text = prompt[columns_start + len("Columns:"):columns_end].strip()
            columns = [col.strip() for col in columns_text.split(',')]
        else:
            columns = ["id", "name", "description"]
        
        # Create mock entities based on column names
        entities = []
        entity_types = set()
        
        # Determine entity types from column names
        for col in columns:
            col_lower = col.lower()
            
            # Skip common non-entity columns
            if col_lower in ["id", "created_at", "updated_at", "timestamp", "date"]:
                continue
                
            # Extract potential entity types
            if "_id" in col_lower:
                entity_type = col_lower.replace("_id", "").title()
                if entity_type and entity_type not in entity_types:
                    entity_types.add(entity_type)
                    entities.append({
                        "id": f"{entity_type.lower()}",
                        "type": entity_type,
                        "name": entity_type,
                        "properties": [p for p in columns if p.lower().startswith(entity_type.lower())],
                        "primary_key": f"{entity_type.lower()}_id"
                    })
        
        # If no specific entities found, create generic ones
        if not entities:
            entities = [
                {
                    "id": "main_entity",
                    "type": "MainEntity",
                    "name": "Main Entity",
                    "properties": [col for col in columns if not col.endswith("_id")],
                    "primary_key": next((col for col in columns if col.lower() == "id"), None)
                }
            ]
        
        # Create relationships
        relationships = []
        
        # Find potential relationships between entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Check for potential foreign keys
                    fk_col = f"{entity2['id']}_id"
                    if fk_col in [col.lower() for col in columns]:
                        relationships.append({
                            "source_entity": entity1["id"],
                            "target_entity": entity2["id"],
                            "type": f"HAS_{entity2['type'].upper()}",
                            "properties": []
                        })
        
        # Create mock response
        response = {
            "entities": entities,
            "relationships": relationships
        }
        
        return json.dumps(response, indent=2)
    
    def _mock_cypher_generation(self, prompt: str) -> str:
        """Generate mock Cypher query for Neo4j"""
        
        # Extract question from prompt
        question_start = prompt.find("### Question:")
        if question_start != -1:
            question = prompt[question_start + len("### Question:"):].strip()
        else:
            question = "Find all nodes"
        
        # Generate simple Cypher query based on question
        if "document" in question.lower():
            cypher_query = "MATCH (d:Document) RETURN d LIMIT 10"
        elif "relationship" in question.lower() or "related" in question.lower():
            cypher_query = "MATCH (d1)-[r]->(d2) RETURN d1, r, d2 LIMIT 10"
        elif "entity" in question.lower():
            cypher_query = "MATCH (e:Entity) RETURN e LIMIT 10"
        elif "chunk" in question.lower():
            cypher_query = "MATCH (c:Chunk) RETURN c LIMIT 10"
        else:
            # Create a query with a basic text match using the last word of the question
            last_word = question.replace("?", "").split()[-1]
            cypher_query = f"MATCH (n) WHERE n.content CONTAINS '{last_word}' RETURN n LIMIT 10"
        
        # Format the response like a real LLM would
        response = f"""Based on the provided schema and question, here is the Cypher query that will help answer the question:

### Cypher Query:
{cypher_query}

This query will search through the Neo4j database to find the relevant information based on the question. The query works by matching the appropriate nodes and relationships, and returning the results limited to 10 records.
"""
        
        return response