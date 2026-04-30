"""
tools/agent_tools.py
All tools available to the assistant.
"""

import os
import subprocess
import json
import datetime
from langchain.tools import tool
from ddgs import DDGS
from langchain_ollama import OllamaLLM

try:
    from config.settings import MODEL, WORKSPACE_DIR, SEARCH_RESULTS, SCRIPT_TIMEOUT
except ImportError:
    MODEL = "qwen3:4b"
    WORKSPACE_DIR = os.path.join(os.path.expanduser("~"), "assistant_workspace")
    SEARCH_RESULTS = 5
    SCRIPT_TIMEOUT = 30


# ── Safety helper ─────────────────────────────────────────────────────────────

def _safe_path(path: str) -> str:
    """Resolve path and enforce it stays within WORKSPACE_DIR."""
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE_DIR, path)
    resolved = os.path.realpath(os.path.expanduser(path))
    if WORKSPACE_DIR:
        allowed = os.path.realpath(WORKSPACE_DIR)
        os.makedirs(allowed, exist_ok=True)
        if not resolved.startswith(allowed):
            raise ValueError(
                f"'{path}' is outside the workspace '{WORKSPACE_DIR}'. "
                f"Use relative paths or filenames to stay inside the workspace."
            )
    return resolved

# ── File Tools ────────────────────────────────────────────────────────────────

MAX_READ_SIZE = 20000  # 20KB limit for full file reads

@tool
def read_file(path: str) -> str:
    """Read and return the full contents of a text file.
    Use relative paths (e.g. 'notes.txt') — they resolve to the workspace.
    If the file is larger than 20KB, it will be truncated.
    """
    try:
        safe = _safe_path(path)
        with open(safe, "r", encoding="utf-8") as f:
            content = f.read()
        
        size = len(content)
        if size > MAX_READ_SIZE:
            return (
                f"[{path}] is large ({size} chars). Showing first {MAX_READ_SIZE} chars:\n\n"
                f"{content[:MAX_READ_SIZE]}\n\n"
                f"... [TRUNCATED] ... Use 'read_file_part' to read specific lines."
            )
        return f"[{path}] ({size} chars):\n\n{content}" if content else f"[{path}] is empty."
    except FileNotFoundError:
        return f"Error: File not found — '{path}'"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def read_file_part(path: str, start_line: int, end_line: int) -> str:
    """Read a specific range of lines from a file.
    Args:
        path: Relative path to the file.
        start_line: The starting line number (1-indexed).
        end_line: The ending line number (inclusive).
    """
    try:
        safe = _safe_path(path)
        with open(safe, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if start_line < 1: start_line = 1
        if end_line > len(lines): end_line = len(lines)
        
        if start_line > end_line:
            return "Error: start_line cannot be greater than end_line."

        requested_lines = lines[start_line-1 : end_line]
        return f"[{path}] Lines {start_line}-{end_line}:\n\n" + "".join(requested_lines)
    except FileNotFoundError:
        return f"Error: File not found — '{path}'"
    except Exception as e:
        return f"Error reading file part: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates file and parent dirs if needed.
    Args:
        path: Filename or relative path (e.g. 'notes.txt' or 'reports/summary.md').
        content: The text content to write.
    """
    if "todos.json" in path:
        return "Error: Cannot manually edit todos.json. Use 'manage_todo' tool."
    try:
        safe = _safe_path(path)
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to '{path}' in workspace."
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def append_file(path: str, content: str) -> str:
    """Append content to an existing file (or create it if it doesn't exist).
    Args:
        path: Filename or relative path in the workspace.
        content: Text to append.
    """
    if "todos.json" in path:
        return "Error: Cannot manually edit todos.json. Use 'manage_todo' tool."
    try:
        safe = _safe_path(path)
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} characters to '{path}'."
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error appending to file: {e}"


@tool
def list_workspace(subfolder: str = "") -> str:
    """List files in the workspace directory (or a subfolder of it).
    Args:
        subfolder: Optional subfolder name inside the workspace. Leave blank for root.
    """
    try:
        target = _safe_path(subfolder) if subfolder else os.path.realpath(WORKSPACE_DIR)
        if not os.path.isdir(target):
            return f"No folder found at '{subfolder}'."
        entries = os.listdir(target)
        if not entries:
            return "Workspace is empty."
        lines = []
        for entry in sorted(entries):
            full = os.path.join(target, entry)
            marker = "/" if os.path.isdir(full) else ""
            size = f"  ({os.path.getsize(full)} bytes)" if os.path.isfile(full) else ""
            lines.append(f"  {entry}{marker}{size}")
        label = f"workspace/{subfolder}" if subfolder else "workspace"
        return f"Contents of {label}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing workspace: {e}"

# ── Script Execution Tool ─────────────────────────────────────────────────────

@tool
def run_script(command: str) -> str:
    """Run a shell command or Python script and return its output.
    
    WARNING: This tool executes arbitrary shell commands. Only run commands
    you trust. Avoid running commands that could delete or modify system files.
    
    Args:
        command: Shell command to execute (e.g. 'python script.py' or 'pip list').
    Note: Commands run from the workspace directory.
    """
    try:
        print(f"\n[Security Warning] Executing shell command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT,
            cwd=WORKSPACE_DIR,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"Output:\n{out}")
        if err:
            parts.append(f"Stderr:\n{err}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {SCRIPT_TIMEOUT}s."
    except Exception as e:
        return f"Error running command: {e}"


# ── Web Search Tool ───────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. No API key required.
    Args:
        query: What to search for.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=SEARCH_RESULTS))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "")
            snippet = r.get("body", "")[:250]
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching web: {e}"


# ── Summarize Tool ────────────────────────────────────────────────────────────

@tool
def summarize(text: str) -> str:
    """Summarize a long block of text using the local Ollama model.
    Args:
        text: Text to summarize.
    """
    try:
        llm = OllamaLLM(model=MODEL)
        prompt = (
            "Summarize the following text clearly and concisely. "
            "Focus on the key points.\n\n"
            f"{text}\n\nSummary:"
        )
        return llm.invoke(prompt)
    except Exception as e:
        return f"Error summarizing (is Ollama running with '{MODEL}'?): {e}"


@tool
def manage_todo(action: str, title: str = None, notes: str = None, due: str = None, todo_id: int = None, todo_ids: list = None) -> str:
    """Manage todo list items (add, list, complete, delete, update).
    Args:
        action: 'add', 'list', 'complete', 'delete', or 'update'.
        title: Title of the todo (for 'add' or 'update').
        notes: Notes to set on the todo (for 'add' or 'update'). Pass empty string to clear.
        due: Due date to set (for 'add' or 'update'). Pass empty string to clear.
        todo_id: Order number of a single todo to complete/delete/update.
        todo_ids: List of order numbers to complete/delete multiple at once.
    """
    file_path = os.path.join(WORKSPACE_DIR, "todos.json")
    
    # Initialize file
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)
            
    with open(file_path, "r") as f:
        todos = json.load(f)

    def reindex(todos):
        """Resequence order on incomplete items sorted by current order."""
        incomplete = sorted([t for t in todos if not t['completed']], key=lambda x: x.get('order', 99999))
        for i, t in enumerate(incomplete, 1):
            t['order'] = i
        for t in todos:
            if t['completed']:
                t['order'] = -1
        return todos

    if action == "add":
        if not title: return "Error: 'add' action requires a title."
        # Duplicate check — warn if a similar incomplete todo already exists
        existing_incomplete = [t for t in todos if not t['completed']]
        for t in existing_incomplete:
            if t['title'].strip().lower() == title.strip().lower():
                return f"Warning: An incomplete todo with the same title already exists (order #{t['order']}). Use action='update' with todo_id={t['order']} to modify it instead of adding a duplicate."
        new_order = max([t.get('order', 0) for t in existing_incomplete], default=0) + 1
        new_id = (max([t.get('id', 0) for t in todos], default=0) + 1)
        todos.append({
            "id": new_id, "order": new_order, "title": title, "notes": notes or "", 
            "due": due or "", "completed": False, "completed_at": None
        })
        with open(file_path, "w") as f: json.dump(todos, f, indent=2)
        return f"Added todo #{new_order}: {title}"

    elif action == "list":
        incomplete = [t for t in todos if not t['completed']]
        incomplete.sort(key=lambda x: x.get('order', 99999))
        if not incomplete: return "No incomplete todos."
        return json.dumps(incomplete, indent=2)

    elif action in ("complete", "delete"):
        # Normalise: accept todo_id or todo_ids
        ids = list(todo_ids) if todo_ids else ([todo_id] if todo_id is not None else [])
        if not ids:
            return f"Error: '{action}' requires todo_id or todo_ids."

        # Sort descending so removing higher-order items first doesn't shift lower ones
        ids_set = set(ids)
        matched = [t for t in todos if t.get('order') in ids_set and not t['completed']]
        if not matched:
            return f"Error: None of the specified todo order numbers were found: {ids}"

        if action == "complete":
            for t in matched:
                t['completed'] = True
                t['completed_at'] = datetime.datetime.now().isoformat()
        else:  # delete
            todos = [t for t in todos if t.get('order') not in ids_set or t['completed']]

        todos = reindex(todos)
        with open(file_path, "w") as f: json.dump(todos, f, indent=2)
        labels = ", ".join(f"#{i}" for i in sorted(ids))
        return f"{action.capitalize()}d todo(s) {labels}. Orders have been resequenced."

    elif action == "update":
        if todo_id is None:
            return "Error: 'update' action requires todo_id (order number)."
        found = next((t for t in todos if t.get('order') == todo_id and not t['completed']), None)
        if not found:
            return f"Error: Incomplete todo with order #{todo_id} not found."
        if title is not None:
            found['title'] = title
        if notes is not None:
            found['notes'] = notes
        if due is not None:
            found['due'] = due
        with open(file_path, "w") as f: json.dump(todos, f, indent=2)
        return f"Updated todo #{todo_id}: {found['title']}"

    return "Invalid action. Use 'add', 'list', 'complete', 'delete', or 'update'."


# ── Tool registry ─────────────────────────────────────────────────────────────

tools = [
    read_file,
    read_file_part,
    write_file,
    append_file,
    list_workspace,
    run_script,
    web_search,
    summarize,
    manage_todo,
]
