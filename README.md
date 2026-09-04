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
- 🚫 **No-think mode** — `/no_think` is injected automatically for qwen3-series models, suppressing reasoning for faster responses

---

## Model Recommendations

Finding the right model for tool calling on constrained hardware took significant experimentation:

| Model | Result |
|---|---|
| `qwen3.5:4b` ✅ | **Current.** Better formatting quality than qwen2.5 variants; reliable tool calling in no-think mode |
| `qwen2.5:7b-instruct` ✅ | Good alternative. Strong instruction following; slightly less natural output formatting |
| `qwen2.5:3b-instruct` ⚠️ | Too small — imprecise tool call formatting, struggles to follow system prompts consistently |
| `qwen2.5-coder:7b` ⚠️ | Coder variant lacks the instruction-following precision needed for tool calling |

**Key insight:** For agentic applications on constrained hardware, model *precision* matters more than raw size. A well-tuned small instruct model with `/no_think` suppression outperforms larger or specialized variants for tool-calling tasks.

---

## Thinking Mode

`qwen3.5` and other qwen3-series models support a hybrid thinking/non-thinking mode. **On this hardware (2GB VRAM, MX450), thinking mode is not recommended:**

- The model produces `<think>...</think>` reasoning blocks before answering
- On a 4b model, the reasoning budget is limited — it rarely produces better answers than no-think mode, especially for simple tasks like formatting a todo list or summarizing text
- Response time increases ~30% (90s → 120s for a startup todo list)
- The 4b model occasionally exhausts itself during the reasoning pass and outputs an empty response, requiring a retry
- The `/no_think` token is injected into every agent LLM call via `NoThinkWrapper` — this covers all internal agent calls (tool decision + response formatting), not just the user-visible message

**Thinking mode is worth exploring on better hardware** (dedicated GPU with 6GB+ VRAM, or a larger model like 9b+). On such hardware, thinking provides meaningful quality improvements for complex reasoning tasks — planning, multi-step logic, ambiguous instructions. For simple CRUD operations and formatting, even then the gain is marginal.

The codebase has full thinking-mode infrastructure (`NoThinkWrapper`, `/no_think` injection, `--think` flag history) that can be re-enabled when hardware permits.

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
python assistant.py --model qwen3.5:4b   # override model
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

This app explicitly sets `num_ctx=8192` (8k tokens) on every Ollama request. This is a deliberate performance choice — the default context window for qwen3-series models is 32k, which forces Ollama to allocate a much larger KV cache in VRAM at model load time even for short conversations. Fixing it at 8k reduces that allocation significantly, cutting first-request latency. 8k is more than sufficient for typical assistant chat sessions.

Context is also managed actively:

- **Todo state compaction** — after every turn, all `manage_todo` tool call/result message pairs in LangGraph memory are replaced with a single concise snapshot of the current todo list. The model never needs to re-read the full history of how the list changed — only what it looks like now.
- **Conversation trimming** — general conversation history is capped at `MAX_HISTORY` turns, keeping the oldest exchanges from filling the window.
- **Narrow tool scope** — tools are purpose-built and return minimal JSON, not verbose prose, so tool results consume as few tokens as possible.

The goal: at any point in a session, the context window contains the system prompt, the current todo state (one snapshot), and recent conversation — nothing more.

### Structured Output for Startup Todo Display

On startup, the assistant displays your current todo list using **LangChain's `.with_structured_output()`** rather than routing through the full ReAct agent loop.

The original approach sent a prompt through the agent, which required the model to: (1) decide to call the `manage_todo` tool, (2) call it, (3) format the result as readable output — three LLM steps, all unreliable on small models. Small local models frequently struggle with the ReAct loop: they may call the wrong tool, misformat the JSON, or produce empty output after exhausting themselves during reasoning.

The refactored approach:
1. Calls `manage_todo` directly in Python — no LLM involvement for data fetching
2. Parses the raw JSON directly into Python lists and dictionaries
3. Renders them with a deterministic formatter

This is faster (one Python call instead of multiple LLM round-trips), more reliable, and produces consistent output regardless of model quality.

Note: `.with_structured_output()` itself (which asks the LLM to emit JSON matching a schema) was evaluated but found to hang or produce garbage on small models like `qwen3.5:2b`. The final implementation avoids calling the LLM at all for startup todo rendering.

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

## Next Steps

The current architecture relies on the LLM to orchestrate multi-step flows (clarify → execute → validate → display). This works but is fragile on small models — they lose track of the sequence, skip steps, or echo raw JSON instead of a formatted response. The next phase replaces that with an explicit state machine built in LangGraph, where the model only handles the reasoning steps it's actually good at.

### Architecture: LangGraph State Machine

Rather than one agent loop that tries to do everything, each stage of a request becomes a dedicated node. The model participates only where judgment is needed; everything else is deterministic Python.

**Graph structure:**

```
[parse_intent]
      ↓
[clarify?] ──── needs info ────→ [ask_clarification] → (wait for user) → [parse_intent]
      ↓ has enough
[execute_tool]
      ↓
[validate]
      ↓
[validated?] ── mismatch ──→ [fix_attempt] → [execute_tool]
      ↓ correct              (max retries → [report_failure])
[display_result]
```

**What each node does:**

- **`parse_intent`** — LLM call with a tight prompt; returns structured JSON: action, target_id, missing fields, ambiguity flag. This is a classification task — well within small model capability.
- **`clarify`** — pure Python router. If `ambiguous=true` or required fields are missing, routes to `ask_clarification`. No model call.
- **`ask_clarification`** — LLM generates one focused question based on what's missing. Answer feeds back into `parse_intent` with accumulated context.
- **`execute_tool`** — pure Python. Calls `manage_todo` with structured args from state. Stores raw result. No model call.
- **`validate`** — Python for simple cases (deleted item absent from list, count changed as expected). LLM call only for complex cases (title match, content comparison). Stores `{"passed": bool, "mismatch": str}` in state.
- **`fix_attempt`** — LLM diagnoses the mismatch and produces a corrected tool call. Capped at 2 retries before routing to `report_failure`.
- **`display_result`** — pure Python formatter. No model call.

**Shared state dict** flows through every node:

```python
class AgentState(TypedDict):
    messages: list        # conversation history
    intent: dict          # output of parse_intent
    pending_action: dict  # args to pass to execute_tool
    tool_result: str      # raw response from manage_todo
    validation: dict      # {"passed": bool, "mismatch": str}
    retry_count: int
    final_response: str
```

**Why this is better for small models:**

Each LLM call has one job, a short context (just the relevant state slice), and a small output space. A 3B model that cannot reliably orchestrate a 5-step chain *can* reliably answer "what fields are missing?" or "does this list match what was requested?" — because those are narrow classification tasks, not planning tasks.

**Build order:**
1. `parse_intent` + `clarify` router — get the clarification loop working end-to-end
2. `execute_tool` + `display_result` — complete the happy path
3. `validate` + `fix_attempt` — add the retry loop once the happy path is solid; log mismatches to understand what actually fails

The free-form agent (web search, file writing, scripting) stays as-is alongside the state machine. Structured CRUD operations route through the graph; open-ended requests fall through to the existing loop.

---

## Design Philosophy

Rather than building a general-purpose assistant, this project takes a **narrow use case** approach — give the model a small, well-defined set of tools and measure whether it can reliably execute useful goals. This makes it easier to evaluate model capability, tune prompts, and ship something genuinely useful on modest hardware.

---

## License

MIT
