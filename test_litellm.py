#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables
load_dotenv()

# Now with litellm installed, try different configurations
print("=== Test 1: LiteLLM with Ollama ===")

# LiteLLM format for ollama
os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'fake-key'

try:
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="ollama/qwen3:latest"  # LiteLLM format
    )
    
    result = agent.start("What is 2+2?")
    print(f"LiteLLM Ollama result: {result}")
    
except Exception as e:
    print(f"LiteLLM Ollama error: {e}")

# Test 2: Try without the base URL override
print("\n=== Test 2: Direct Ollama Model Name ===")

try:
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="qwen3:latest"
    )
    
    result = agent.start("What is 2+2?")
    print(f"Direct model result: {result}")
    
except Exception as e:
    print(f"Direct model error: {e}")

# Test 3: Try with knowledge using the working configuration
print("\n=== Test 3: Knowledge Agent with Working Config ===")

if os.path.exists("/tmp/test_knowledge.txt"):
    try:
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
        
        agent = Agent(
            name="Knowledge Agent",
            instructions="You answer questions based on the provided knowledge.",
            knowledge=["/tmp/test_knowledge.txt"],
            knowledge_config=config,
            user_id="test_user",
            llm="ollama/qwen3:latest"
        )
        
        result = agent.start("What is the capital of France?")
        print(f"Knowledge agent result: {result}")
        
    except Exception as e:
        print(f"Knowledge agent error: {e}")
        import traceback
        traceback.print_exc()
else:
    # Create test file
    with open("/tmp/test_knowledge.txt", 'w') as f:
        f.write("The capital of France is Paris. France is a country in Europe.")
    print("Created test knowledge file")