# DeepSeek RAG Application

A Python-based RAG (Retrieval-Augmented Generation) application that allows you to upload directories of documents and query them using local LLM models through Ollama.

## Features

- **Directory-based document upload**: Select entire directories containing documents
- **Multi-format support**: PDF, Markdown (.md), and Text (.txt) files
- **Local LLM integration**: Works with Ollama models (tested with qwen3:latest, deepseek-r1:8b)
- **GUI Interface**: Easy-to-use Tkinter-based interface
- **RAG capabilities**: ChromaDB vector store with semantic search
- **Model selection**: Dropdown menu showing all available Ollama models

## Prerequisites

### 1. Ollama Installation

Install Ollama and pull the required models:

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai

# Pull required models
ollama pull qwen3:latest
ollama pull nomic-embed-text:latest
ollama pull deepseek-r1:8b  # Optional alternative model

# Start Ollama service
ollama serve
```

### 2. Python Environment

Create and activate a conda environment:

```bash
# Create environment
conda create -n DST python=3.12
conda activate DST
```

## Installation

1. **Clone/Download the repository**
```bash
git clone <repository-url>
cd DeepSeeKRAG
```

2. **Install Python dependencies**
```bash
pip install "praisonaiagents[llm]"
pip install "praisonaiagents[knowledge]"
pip install python-dotenv
pip install tkinter  # Usually included with Python
```

## Configuration

1. **Environment Setup**
   
   Create a `.env` file in the root directory:
```env
OLLAMA_ENDPOINT=http://127.0.0.1:11434
OPENAI_API_KEY=sk-your-openai-key-here  # Optional, for OpenAI models
CLAUDE_API_KEY=sk-ant-your-claude-key   # Optional, for Claude models
```

2. **Verify Ollama is running**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return a list of installed models
```

## Usage

### Running the Application

```bash
# Activate environment
conda activate DST

# Run the main application
python RAGApp.py
```

### Using the Interface

1. **Select LLM Model**: Choose from available Ollama models in the dropdown
2. **Select Directory**: Click "Select Directory" and choose a folder containing your documents
3. **Upload Documents**: Click "Upload Documents" to index the files (this may take a while)
4. **Query**: Enter your question and click "Submit"

### Alternative Interfaces

**Streamlit Web Interface:**
```bash
streamlit run app.py
```

**Command Line (Simple):**
```bash
python DeepSeek_Agents.py
```

## File Structure

```
DeepSeeKRAG/
├── RAGApp.py              # Main GUI application
├── app.py                 # Streamlit web interface
├── DeepSeek_Agents.py     # Simple command-line agent
├── DeepSeek_Agents1.py    # Agent with hardcoded PDF
├── DeepSeek_Agents2.py    # Streamlit agent with hardcoded PDF
├── Books/                 # Sample PDF documents
├── Results/              # Query results storage
├── Backup/               # Backup files
├── .env                  # Environment configuration
├── .praison/             # ChromaDB vector store
└── README.md             # This file
```

## Supported File Types

- **PDF**: `.pdf` files
- **Markdown**: `.md` files  
- **Text**: `.txt` files

The application recursively scans directories and subdirectories for these file types.

## Troubleshooting

### Common Issues

1. **"No response from model"**
   - Ensure Ollama is running: `ollama serve`
   - Check model is installed: `ollama list`
   - Verify model name format uses `ollama/model_name`

2. **API Key Errors**
   - The app uses "fake-key" for Ollama (this is normal)
   - Ensure `.env` file exists with correct `OLLAMA_ENDPOINT`

3. **Slow Performance**
   - Large documents take time to index
   - Consider using smaller models like `deepseek-r1:8b`
   - Ensure sufficient RAM for model and documents

4. **Import Errors**
   - Install missing dependencies: `pip install "praisonaiagents[llm]"`
   - Ensure conda environment is activated

### Debugging

Enable debug logging by running:
```bash
export PYTHONPATH=/path/to/project
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python RAGApp.py
```

## Technical Details

- **Vector Store**: ChromaDB with persistent storage in `.praison/`
- **Embeddings**: nomic-embed-text:latest model
- **LLM Framework**: praisonaiagents with LiteLLM backend
- **UI Framework**: Tkinter for desktop, Streamlit for web

## Dependencies

```
praisonaiagents[llm]>=0.0.61
praisonaiagents[knowledge]
python-dotenv
tkinter (built-in)
streamlit (optional, for web interface)
```

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify Ollama installation and model availability
3. Ensure all dependencies are installed correctly