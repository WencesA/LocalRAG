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
        # We'll put the answer label in row=5 and text widget in row=6
        for r in range(7):
            main_frame.rowconfigure(r, weight=0)  # default 0, no expansion
        main_frame.columnconfigure(1, weight=1)   # let column=1 expand horizontally
        main_frame.rowconfigure(6, weight=1)      # let row=6 expand vertically

        # Model Selection
        model_label = tk.Label(main_frame, text="Select LLM Model:")
        model_label.grid(row=0, column=0, sticky="w", padx=(5,5), pady=(5,5))

        if self.ollama_models:
            model_dropdown = tk.OptionMenu(main_frame, self.model_var, *self.ollama_models)
            model_dropdown.grid(row=0, column=1, sticky="w", padx=(5,5), pady=(5,5))
        else:
            tk.Label(main_frame, text="No local Ollama models found.").grid(row=0, column=1, sticky="w")

        # Document Selection
        doc_button = tk.Button(main_frame, text="Select Directory", command=self.browse_directory)
        doc_button.grid(row=1, column=0, sticky="w", padx=(5,5), pady=(5,5))

        self.file_label = tk.Label(main_frame, text="No directory selected")
        self.file_label.grid(row=1, column=1, sticky="w", padx=(5,5), pady=(5,5))

        # Upload Button (to index the documents)
        upload_button = tk.Button(main_frame, text="Upload Documents", command=self.upload_document)
        upload_button.grid(row=2, column=0, columnspan=2, pady=(5,5))

        # Query Entry
        query_label = tk.Label(main_frame, text="Enter your query:")
        query_label.grid(row=3, column=0, sticky="w", padx=(5,5), pady=(5,5))

        self.query_entry = tk.Entry(main_frame, width=50)
        self.query_entry.grid(row=3, column=1, sticky="ew", padx=(5,5), pady=(5,5))

        # Submit Button (to run the query)
        submit_button = tk.Button(main_frame, text="Submit", command=self.run_query)
        submit_button.grid(row=4, column=0, columnspan=2, pady=(5,5))

        # Answer Label
        answer_label = tk.Label(main_frame, text="Answer:")
        answer_label.grid(row=5, column=0, sticky="nw", padx=(5,5), pady=(5,5))

        # Answer Text + Scrollbar
        self.answer_text = tk.Text(main_frame, wrap="word")
        self.answer_text.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=(5,0), pady=(5,5))

        scrollbar = tk.Scrollbar(main_frame, command=self.answer_text.yview)
        self.answer_text.config(yscrollcommand=scrollbar.set)
        # Place scrollbar to the right of the text widget
        # We'll do row=6, column=2 so it's in the same row as the text
        scrollbar.grid(row=6, column=2, sticky="nse", padx=(0,5), pady=(5,5))

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

    def run_query(self):
        """
        Run the user's query through the already-uploaded agent.
        If the upload is still in progress, notify the user.
        """
        if self.upload_in_progress:
            messagebox.showinfo("Info", "System is busy performing the RAG operation. Please wait.")
            return

        if not self.agent:
            messagebox.showerror("Error", "No documents have been uploaded yet. Please upload first.")
            return

        query = self.query_entry.get().strip()
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return

        try:
            # Ensure environment variables are set for queries too
            os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            
            result = self.agent.start(query)

            # Ensure the result is a string before inserting into the Text widget.
            if not isinstance(result, str):
                result_str = str(result) if result is not None else "No response from the model."
            else:
                result_str = result

            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, result_str)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

###############################################################################
# Run the Application
###############################################################################

if __name__ == "__main__":
    app = RAGApplication()
    app.mainloop()