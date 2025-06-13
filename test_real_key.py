#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables
load_dotenv()

# Test 1: Use real OpenAI key with OpenAI (to verify praisonaiagents works)
print("=== Test 1: Real OpenAI Key with OpenAI ===")
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

print(f"OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY'][:10]}...")

try:
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="gpt-4o-mini"  # Use cheaper OpenAI model
    )
    
    result = agent.start("What is 2+2?")
    print(f"OpenAI agent result: {result}")
    
except Exception as e:
    print(f"OpenAI agent error: {e}")

# Test 2: Use real OpenAI key with Ollama endpoint (might work)
print("\n=== Test 2: Real OpenAI Key with Ollama Endpoint ===")
os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

print(f"OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
print(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY'][:10]}...")

try:
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="qwen3:latest"
    )
    
    result = agent.start("What is 2+2?")
    print(f"Ollama with OpenAI key result: {result}")
    
except Exception as e:
    print(f"Ollama with OpenAI key error: {e}")

# Test 3: Check if there's a specific ollama provider
print("\n=== Test 3: Try Different LLM Provider Configuration ===")

try:
    # Try with a different approach - maybe the llm parameter expects something else
    agent = Agent(
        instructions="You are a helpful Assistant",
        llm="ollama/qwen3:latest"
    )
    
    result = agent.start("What is 2+2?")
    print(f"Ollama provider result: {result}")
    
except Exception as e:
    print(f"Ollama provider error: {e}")