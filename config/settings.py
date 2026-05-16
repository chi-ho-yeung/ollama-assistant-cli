# ─────────────────────────────────────────────
# Assistant Configuration
# Edit these values to customize your assistant
# ─────────────────────────────────────────────

# Ollama model to use
# Available: "qwen3.5:4b", "qwen2.5:3b-instruct", "llama3.2:3b-instruct-q4_1"
MODEL = "qwen3.5:4b"

# Assistant name shown in the terminal
ASSISTANT_NAME = "Aria"

# Directory that file tools are allowed to read/write
# Use None to allow anywhere (not recommended)
import os
WORKSPACE_DIR = os.path.join(os.path.expanduser("~"), "assistant_workspace")

# Max number of conversation turns to keep in memory
MAX_HISTORY = 20

# Web search results to fetch per query
SEARCH_RESULTS = 5

# Script execution timeout in seconds
SCRIPT_TIMEOUT = 30

# Terminal colors (set False to disable)
USE_COLOR = True
