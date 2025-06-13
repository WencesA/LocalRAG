#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables
load_dotenv()

# Set up environment
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434')
OPENAI_BASE_URL = f"{OLLAMA_ENDPOINT}/v1"
OPENAI_API_KEY = "fake-key"

os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

print(f"OLLAMA_ENDPOINT: {OLLAMA_ENDPOINT}")
print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

# Test simple agent without knowledge
try:
    print("\n=== Testing Simple Agent (No Knowledge) ===")
    simple_agent = Agent(
        name="Simple Agent",
        instructions="You are a helpful assistant.",
        llm="qwen3:latest"
    )
    
    result = simple_agent.start("What is 2+2?")
    print(f"Simple agent result: {result}")
    
except Exception as e:
    print(f"Simple agent error: {e}")

# Test agent with knowledge configuration
try:
    print("\n=== Testing Agent with Knowledge Config ===")
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "test_praison",
                "path": ".test_praison"
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen3:latest",
                "temperature": 0,
                "max_tokens": 8000,
                "ollama_base_url": OLLAMA_ENDPOINT,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": OLLAMA_ENDPOINT,
                "embedding_dims": 1536
            },
        },
    }
    
    # Create a test text file
    test_file = "/tmp/test_knowledge.txt"
    with open(test_file, 'w') as f:
        f.write("The capital of France is Paris. France is a country in Europe.")
    
    knowledge_agent = Agent(
        name="Knowledge Agent",
        instructions="You answer questions based on the provided knowledge.",
        knowledge=[test_file],
        knowledge_config=config,
        user_id="test_user",
        llm="qwen3:latest"
    )
    
    result = knowledge_agent.start("What is the capital of France?")
    print(f"Knowledge agent result: {result}")
    
except Exception as e:
    print(f"Knowledge agent error: {e}")
    import traceback
    traceback.print_exc()