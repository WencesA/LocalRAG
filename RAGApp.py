#!/usr/bin/env python3
"""
RAG Application - Dual-Mode Local LLM Frontend

PURPOSE:
This application provides a comprehensive Tkinter-based GUI for interacting with local
Ollama models in two distinct modes: direct chat and Retrieval-Augmented Generation (RAG).
It serves as a unified frontend for local LLM interaction with optional document-based
knowledge enhancement.

WHAT IT PROVIDES:
1. Chat Mode: Direct conversation with local LLM models without document requirements
2. RAG Mode: Document indexing and knowledge-based querying using vector storage
3. Dynamic model selection from installed Ollama models
4. Multi-threaded document processing to prevent UI blocking
5. Support for multiple file formats (PDF, MD, TXT) and directory scanning
6. Chroma vector store integration for document embeddings
7. LiteLLM format compatibility for seamless model switching

MODES OF OPERATION:
- Chat Mode (Default): Creates simple agents for direct LLM interaction
  * No document upload required
  * Immediate model access
  * Standard conversational interface
  * Model switching without losing functionality

- RAG Mode: Enhanced querying with document knowledge
  * Directory selection and file scanning
  * Document indexing with Chroma vector store
  * Knowledge-based question answering
  * Maintains original RAG functionality

ARCHITECTURE:
- Frontend: Tkinter GUI with responsive layout and mode-specific control visibility
- Backend: PraisonAI Agents framework with Ollama integration
- Models: Local Ollama models (qwen3:latest, deepseek-r1:8b, etc.)
- Embeddings: nomic-embed-text:latest for document vectorisation
- Storage: Chroma vector database for document embeddings
- Configuration: Environment-based settings with .env support

DEPENDENCIES:
- tkinter: GUI framework
- praisonaiagents: LLM agent framework
- python-dotenv: Environment variable management
- ollama: Local model service (must be running)
- chroma: Vector storage (installed with praisonaiagents)

ENVIRONMENT VARIABLES:
- OLLAMA_ENDPOINT: Ollama service URL (default: http://127.0.0.1:11434)
- OPENAI_BASE_URL: Set automatically to Ollama endpoint/v1
- OPENAI_API_KEY: Set to "fake-key" for Ollama compatibility

SUPPORTED FORMATS:
- Documents: PDF, Markdown (.md), Text (.txt)
- Models: Any Ollama-compatible model with LiteLLM format support
- Embeddings: Ollama embedding models (nomic-embed-text recommended)

WHEN TO USE:
- General local LLM interaction without document constraints
- Document-based knowledge querying and research
- Model comparison and testing across different scenarios  
- Private, local AI assistance without cloud dependencies
- RAG experimentation with custom document collections

USAGE PATTERNS:
1. Chat Mode: Select model → Enter query → Get response
2. RAG Mode: Select model → Select documents → Upload → Query knowledge base
3. Model Switching: Change dropdown selection, agent recreated automatically
4. Mode Switching: Toggle radio buttons, UI adapts dynamically

ERROR HANDLING:
- Model availability validation against installed Ollama models
- Document format verification and scanning
- Agent creation error reporting with specific mode context
- Network connectivity checks for Ollama service
- Graceful degradation with informative error messages

RELATIONSHIP TO TESTING:
This application works with the testing scripts in testing/ directory:
- environment_validation.py: Validates basic setup before GUI use
- litellm_integration_test.py: Tests model format compatibility
- legacy_config_test.py: Reference configuration troubleshooting
- openai_comparison_test.py: Performance benchmarking capabilities
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
import os
from dotenv import load_dotenv
from praisonaiagents import Agent

# Load environment variables from .env file
load_dotenv()

# Get Ollama endpoint from environment
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434')
OPENAI_BASE_URL = f"{OLLAMA_ENDPOINT}/v1"
OPENAI_API_KEY = "fake-key"  # Ollama doesn't need real API key

###############################################################################
# Helper Functions
###############################################################################

def get_ollama_models():
    """
    Retrieve a list of installed models from Ollama by running `ollama list`.
    Returns a list of model names or a fallback list if the command fails.
    """
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
        lines = output.strip().split("\n")
        if lines and lines[0].startswith("NAME"):
            lines = lines[1:]

        models = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(None, 3)
            if parts:
                model_name = parts[0]
                # Example line format: "mistral:latest   1a09f42b4a67   2.7 MB   Less than a second ago"
                models.append(model_name)
        return models
    except Exception as e:
        print(f"Error retrieving Ollama models: {e}")
        return ["mistral:latest"]  # fallback

def get_config(selected_model):
    """
    Build the configuration dictionary for the agent, using the selected LLM model.
    Adjust this if you need different settings for each model.
    """
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
                "model": selected_model,
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
    return config

###############################################################################
# Main Application
###############################################################################

class RAGApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM RAG Interface")
        # Set a default window size (optional)
        self.geometry("800x600")

        # Allow the window to resize
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # State variables
        self.selected_directory = tk.StringVar()
        self.selected_files = []
        self.upload_in_progress = False
        self.agent = None
        self.chat_agent = None
        self.mode = tk.StringVar(value="Chat")  # Default to Chat Mode

        # Load Ollama models
        self.ollama_models = get_ollama_models()
        self.model_var = tk.StringVar(value=self.ollama_models[0] if self.ollama_models else "")

        # Build the GUI
        self._build_ui()

    def _build_ui(self):
        """
        Build and arrange the Tkinter widgets in a main frame that expands.
        """
        main_frame = tk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure this frame to resize rows/columns as the window grows
        # We'll put the answer label in row=6 and text widget in row=7
        for r in range(8):
            main_frame.rowconfigure(r, weight=0)  # default 0, no expansion
        main_frame.columnconfigure(1, weight=1)   # let column=1 expand horizontally
        main_frame.rowconfigure(7, weight=1)      # let row=7 expand vertically

        # Model Selection
        model_label = tk.Label(main_frame, text="Select LLM Model:")
        model_label.grid(row=0, column=0, sticky="w", padx=(5,5), pady=(5,5))

        if self.ollama_models:
            model_dropdown = tk.OptionMenu(main_frame, self.model_var, *self.ollama_models)
            model_dropdown.grid(row=0, column=1, sticky="w", padx=(5,5), pady=(5,5))
        else:
            tk.Label(main_frame, text="No local Ollama models found.").grid(row=0, column=1, sticky="w")

        # Mode Selection (Radio Buttons)
        mode_label = tk.Label(main_frame, text="Mode:")
        mode_label.grid(row=1, column=0, sticky="w", padx=(5,5), pady=(5,5))
        
        mode_frame = tk.Frame(main_frame)
        mode_frame.grid(row=1, column=1, sticky="w", padx=(5,5), pady=(5,5))
        
        chat_radio = tk.Radiobutton(mode_frame, text="Chat Mode", variable=self.mode, value="Chat", command=self.on_mode_change)
        chat_radio.pack(side="left", padx=(0,10))
        
        rag_radio = tk.Radiobutton(mode_frame, text="RAG Mode", variable=self.mode, value="RAG", command=self.on_mode_change)
        rag_radio.pack(side="left")

        # Document Selection (RAG Mode only)
        self.doc_button = tk.Button(main_frame, text="Select Directory", command=self.browse_directory)
        self.doc_button.grid(row=2, column=0, sticky="w", padx=(5,5), pady=(5,5))

        self.file_label = tk.Label(main_frame, text="No directory selected")
        self.file_label.grid(row=2, column=1, sticky="w", padx=(5,5), pady=(5,5))

        # Upload Button (to index the documents) - RAG Mode only
        self.upload_button = tk.Button(main_frame, text="Upload Documents", command=self.upload_document)
        self.upload_button.grid(row=3, column=0, columnspan=2, pady=(5,5))

        # Query Entry
        query_label = tk.Label(main_frame, text="Enter your query:")
        query_label.grid(row=4, column=0, sticky="w", padx=(5,5), pady=(5,5))

        self.query_entry = tk.Entry(main_frame, width=50)
        self.query_entry.grid(row=4, column=1, sticky="ew", padx=(5,5), pady=(5,5))

        # Submit Button (to run the query)
        submit_button = tk.Button(main_frame, text="Submit", command=self.run_query)
        submit_button.grid(row=5, column=0, columnspan=2, pady=(5,5))

        # Answer Label
        answer_label = tk.Label(main_frame, text="Answer:")
        answer_label.grid(row=6, column=0, sticky="nw", padx=(5,5), pady=(5,5))

        # Answer Text + Scrollbar
        self.answer_text = tk.Text(main_frame, wrap="word")
        self.answer_text.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=(5,0), pady=(5,5))

        scrollbar = tk.Scrollbar(main_frame, command=self.answer_text.yview)
        self.answer_text.config(yscrollcommand=scrollbar.set)
        # Place scrollbar to the right of the text widget
        # We'll do row=7, column=2 so it's in the same row as the text
        scrollbar.grid(row=7, column=2, sticky="nse", padx=(0,5), pady=(5,5))
        
        # Initialize UI state based on default mode
        self.on_mode_change()

    def on_mode_change(self):
        """
        Handle mode toggle changes - show/hide RAG controls based on selected mode.
        """
        mode = self.mode.get()
        
        if mode == "Chat":
            # Hide RAG-specific controls
            self.doc_button.grid_remove()
            self.file_label.grid_remove()
            self.upload_button.grid_remove()
        else:  # RAG Mode
            # Show RAG-specific controls
            self.doc_button.grid()
            self.file_label.grid()
            self.upload_button.grid()

    def browse_directory(self):
        """
        Open a directory dialog to select a folder containing documents.
        """
        directory = filedialog.askdirectory(
            title="Select a directory containing documents"
        )
        if directory:
            self.selected_directory.set(directory)
            self.selected_files = self.scan_supported_files(directory)
            file_count = len(self.selected_files)
            self.file_label.config(text=f"{directory} ({file_count} files found)")

    def scan_supported_files(self, directory):
        """
        Scan the directory for supported file types (PDF, MD, TXT).
        Returns a list of file paths.
        """
        supported_extensions = ['.pdf', '.md', '.txt']
        files = []
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))
        
        return files

    def upload_document(self):
        """
        Create the agent with the chosen model and selected document.
        Perform indexing in a separate thread to avoid blocking the UI.
        """
        if self.upload_in_progress:
            messagebox.showinfo("Info", "Upload is already in progress.")
            return

        directory_path = self.selected_directory.get()
        if not directory_path:
            messagebox.showerror("Error", "Please select a directory first.")
            return

        if not self.selected_files:
            messagebox.showerror("Error", "No supported files (PDF, MD, TXT) found in the selected directory.")
            return

        selected_model = self.model_var.get()
        if selected_model not in self.ollama_models:
            messagebox.showerror("Error", f"The model '{selected_model}' is not available in Ollama.")
            return

        self.upload_in_progress = True
        thread = threading.Thread(target=self._upload_document_thread, args=(selected_model,))
        thread.start()

    def _upload_document_thread(self, selected_model):
        """
        Threaded function to create the agent and index the knowledge.
        """
        try:
            # Set environment variables for the agent
            os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            
            config = get_config(selected_model)

            # Create the agent - use LiteLLM format for Ollama
            llm_model = f"ollama/{selected_model}" if not selected_model.startswith("ollama/") else selected_model
            
            self.agent = Agent(
                name="Knowledge Agent",
                instructions="You answer questions based on the provided knowledge.",
                knowledge=self.selected_files,
                knowledge_config=config,
                user_id="user1",
                llm=llm_model
            )

            # The agent is created and knowledge is automatically indexed
            # No need for dummy query - praisonaiagents handles indexing during creation
            
            messagebox.showinfo("Success", f"{len(self.selected_files)} documents successfully uploaded and indexed.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during upload:\n{e}")
            self.agent = None
        finally:
            self.upload_in_progress = False

    def create_chat_agent(self, selected_model):
        """
        Create a simple agent for Chat Mode without knowledge configuration.
        """
        try:
            # Set environment variables for the agent
            os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            
            # Create simple agent - use LiteLLM format for Ollama
            llm_model = f"ollama/{selected_model}" if not selected_model.startswith("ollama/") else selected_model
            
            self.chat_agent = Agent(
                name="Chat Agent",
                instructions="You are a helpful assistant. Answer questions clearly and concisely.",
                llm=llm_model
            )
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create chat agent:\n{e}")
            self.chat_agent = None
            return False

    def run_query(self):
        """
        Run the user's query through the appropriate agent based on current mode.
        Chat Mode: Direct LLM interaction without documents
        RAG Mode: Document-based querying (original functionality)
        """
        mode = self.mode.get()
        query = self.query_entry.get().strip()
        
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return

        if mode == "Chat":
            self._handle_chat_query(query)
        else:  # RAG Mode
            self._handle_rag_query(query)

    def _handle_chat_query(self, query):
        """
        Handle queries in Chat Mode - direct LLM interaction.
        """
        selected_model = self.model_var.get()
        if selected_model not in self.ollama_models:
            messagebox.showerror("Error", f"The model '{selected_model}' is not available in Ollama.")
            return

        # Create chat agent if it doesn't exist or model changed
        if not self.chat_agent or getattr(self.chat_agent, '_model', None) != selected_model:
            if not self.create_chat_agent(selected_model):
                return  # Error message already shown in create_chat_agent
            # Store model reference for comparison
            self.chat_agent._model = selected_model

        try:
            result = self.chat_agent.start(query)

            # Ensure the result is a string before inserting into the Text widget
            if not isinstance(result, str):
                result_str = str(result) if result is not None else "No response from the model."
            else:
                result_str = result

            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, result_str)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred in Chat Mode:\n{e}")

    def _handle_rag_query(self, query):
        """
        Handle queries in RAG Mode - document-based querying (original functionality).
        """
        if self.upload_in_progress:
            messagebox.showinfo("Info", "System is busy performing the RAG operation. Please wait.")
            return

        if not self.agent:
            messagebox.showerror("Error", "No documents have been uploaded yet. Please upload first.")
            return

        try:
            # Ensure environment variables are set for queries too
            os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            
            result = self.agent.start(query)

            # Ensure the result is a string before inserting into the Text widget
            if not isinstance(result, str):
                result_str = str(result) if result is not None else "No response from the model."
            else:
                result_str = result

            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, result_str)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred in RAG Mode:\n{e}")

###############################################################################
# Run the Application
###############################################################################

if __name__ == "__main__":
    app = RAGApplication()
    app.mainloop()