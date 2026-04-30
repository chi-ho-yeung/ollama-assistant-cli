# ollama-assistant-cli

A local, agentic CLI assistant powered by [Ollama](https://ollama.com) and [LangGraph](https://github.com/langchain-ai/langgraph). Built to run entirely on a laptop with **16GB RAM and 2GB VRAM** — no cloud API keys, no internet required.

The goal of this project is to explore how much useful work a small local LLM can do when given a narrow, well-defined task and a clear set of tools.

---

## Motivation

Most LLM assistant demos assume access to powerful cloud models. This project asks a different question:

> *Can a local model running on modest consumer hardware be genuinely useful for real daily tasks?*

The answer is yes — but only with the right model and careful prompt engineering. See [Model Recommendations](#model-recommendations) below.

---

## Features

- 🧠 **Agentic tool calling** via LangGraph — the model decides when and how to use tools
- ✅ **Todo management** — add, update, complete, delete tasks with natural language
- 📋 **Smart task formatting** — tasks are normalized to `(verb) (what) (context)` format for consistency
- 🔍 **Self-verification** — after any change, the model re-fetches the list and confirms the change was applied correctly
- 💬 **Persistent conversation history** — context is maintained across turns in a session
- ⌨️ **Input queuing** — type your next request while the model is still working
- ⏱️ **Progress indicator** — "Still working..." shown every 15 seconds for long operations
- 🔧 **`--think` flag** — enables reasoning output for models that support it (e.g. qwen3)

---

## Model Recommendations

Finding the right model for tool calling on constrained hardware took significant experimentation:

| Model | Result |
|---|---|
| `qwen2.5:7b-instruct` ✅ | **Recommended.** Best balance of speed, tool-call precision, and instruction following |
| `qwen2.5:3b-instruct` ⚠️ | Too small — imprecise tool call formatting, struggles to follow system prompts consistently |
| `qwen2.5-coder:7b` ⚠️ | Coder variant lacks the instruction-following precision needed for tool calling |
| `qwen3:4b` / `qwen3.5` ❌ | Thinking overhead makes responses too slow for interactive use on this hardware |

**Key insight:** For agentic applications on constrained hardware, model *precision* matters more than raw size. A well-tuned 7B instruct model outperforms larger or specialized variants for tool-calling tasks.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- ~5GB disk space for the recommended model

### Pull the recommended model

```bash
ollama pull qwen2.5:7b-instruct
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python assistant.py
```

### Options

```bash
python assistant.py --model qwen2.5:7b-instruct   # override model
python assistant.py --think                        # enable reasoning output (qwen3+ only)
```

### Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/tools` | List available agent tools |
| `/clear` | Clear conversation history |
| `/save` | Save conversation to file |
| `/ws` | Show workspace directory |
| `/exit`, `/quit`, `/bye` | Exit the app |

---

## Project Structure

```
ollama-assistant-cli/
├── assistant.py          # Main entry point and agent loop
├── system_prompt.txt     # Persona, instructions, and formatting rules for the LLM
├── requirements.txt      # Python dependencies
├── config/
│   └── settings.py       # Model name, workspace path, history limits
└── tools/
    └── agent_tools.py    # Tool implementations (manage_todo, web_search, etc.)
```

### Workspace

User data (todos, saved conversations) is stored in a separate workspace directory outside the repo, configured in `config/settings.py`:

```python
WORKSPACE_DIR = r"C:\Users\yourname\assistant_workspace"
```

---

## Context Management

Context efficiency is a core design goal. Small local LLMs have limited context windows, and filling them with stale history degrades response quality and speed.

This app uses a **32k context window** (configured in Ollama) and manages it actively:

- **Todo state compaction** — after every turn, all `manage_todo` tool call/result message pairs in LangGraph memory are replaced with a single concise snapshot of the current todo list. The model never needs to re-read the full history of how the list changed — only what it looks like now.
- **Conversation trimming** — general conversation history is capped at `MAX_HISTORY` turns, keeping the oldest exchanges from filling the window.
- **Narrow tool scope** — tools are purpose-built and return minimal JSON, not verbose prose, so tool results consume as few tokens as possible.
The goal: at any point in a session, the context window contains the system prompt, the current todo state (one snapshot), and recent conversation — nothing more.

### Performance

On a 16GB RAM / 2GB VRAM laptop using `qwen2.5:7b-instruct`:

- **First request** — \~45–60 seconds (Ollama loads the model into memory on first use)
- **Subsequent requests** — \~20–25 seconds per todo operation (model stays loaded between requests)

Keeping the model loaded between requests is important — use `ollama serve` and leave it running rather than letting it unload between sessions.

### About `ollama serve`

`ollama serve` starts the Ollama background server on `http://localhost:11434`. On Windows, installing Ollama typically registers it as a background service that starts automatically — you can verify it's running with:

```
tasklist | findstr ollama
```

By default, Ollama unloads a model from memory **5 minutes** after the last request. On next use it has to reload, causing the slow first-request delay. You can control this with the `OLLAMA_KEEP_ALIVE` environment variable:

```
OLLAMA_KEEP_ALIVE=15m    # keep loaded for 15 minutes (recommended for regular use)
OLLAMA_KEEP_ALIVE=1h     # keep loaded for 1 hour
OLLAMA_KEEP_ALIVE=-1     # keep loaded indefinitely until Ollama restarts
```

On Windows, set this as a permanent user environment variable via **System Properties → Environment Variables** so it applies every time Ollama starts.

You can also set the context window globally so you don't need to configure it per-app:

```
OLLAMA_NUM_CTX=32768
```

---

Edit `config/settings.py` to change the default model or workspace path:

```python
MODEL = "qwen2.5:7b-instruct"
WORKSPACE_DIR = r"C:\Users\yourname\assistant_workspace"
MAX_HISTORY = 20
```

---

## Roadmap

This project is in early stages. Planned features:

- **Smart suggestions** — based on your task list, the assistant proactively suggests what to focus on next
- **Goal breakdown** — given a task, suggest concrete steps to accomplish it
- **Monthly summary** — review and summarize tasks completed in the past month
- **Multi-session memory** — persist context across separate runs, not just within a session

---

## Design Philosophy

Rather than building a general-purpose assistant, this project takes a **narrow use case** approach — give the model a small, well-defined set of tools and measure whether it can reliably execute useful goals. This makes it easier to evaluate model capability, tune prompts, and ship something genuinely useful on modest hardware.

---

## License

MIT
