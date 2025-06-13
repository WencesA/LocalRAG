#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables
load_dotenv()

# Try using ollama:// protocol instead of http://
os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'ollama'  # Try different API key

print(f"OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")

# Test with just the llm parameter as string (like in original files)
try:
    print("\n=== Testing with LLM String Only ===")
    
    # Create a test text file
    test_file = "/tmp/test_knowledge.txt"
    with open(test_file, 'w') as f:
        f.write("The capital of France is Paris. France is a country in Europe.")
    
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="qwen3:latest"
    )
    
    result = agent.start("What is 2+2?")
    print(f"Simple agent result: {result}")
    
except Exception as e:
    print(f"Agent error: {e}")

# Try with no-key
try:
    print("\n=== Testing with Different API Key ===")
    os.environ['OPENAI_API_KEY'] = 'sk-no-key-required'
    
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="qwen3:latest"
    )
    
    result = agent.start("What is 2+2?")
    print(f"Agent result: {result}")
    
except Exception as e:
    print(f"Agent error: {e}")