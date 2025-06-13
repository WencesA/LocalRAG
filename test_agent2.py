#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables
load_dotenv()

# Try the approach mentioned in the original files
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434')

# Set environment exactly as in the original DeepSeek_Agents files
os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'fake-key'

print(f"OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")

# Test with exact configuration from DeepSeek_Agents1.py
try:
    print("\n=== Testing with Original Configuration ===")
    
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "praison",
                "path": ".praison"
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen3:latest",
                "temperature": 0,
                "max_tokens": 8000,
                "ollama_base_url": "http://localhost:11434",
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": "http://localhost:11434",
                "embedding_dims": 1536
            },
        },
    }
    
    # Create a test text file
    test_file = "/tmp/test_knowledge.txt"
    with open(test_file, 'w') as f:
        f.write("The capital of France is Paris. France is a country in Europe.")
    
    agent = Agent(
        name="Knowledge Agent",
        instructions="You answer questions based on the provided knowledge.",
        knowledge=[test_file],
        knowledge_config=config,
        user_id="user1",
        llm="qwen3:latest"
    )
    
    result = agent.start("What is the capital of France?")
    print(f"Agent result: {result}")
    
except Exception as e:
    print(f"Agent error: {e}")
    import traceback
    traceback.print_exc()