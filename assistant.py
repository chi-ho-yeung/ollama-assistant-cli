import sys
import os
import datetime
import threading
import time
import queue
import argparse
import json
import re
import ollama
from typing import List, Optional, Any, Dict

from config.settings import MODEL, ASSISTANT_NAME, MAX_HISTORY, WORKSPACE_DIR, USE_COLOR
from tools.agent_tools import tools, manage_todo

# ── Terminal Colors ───────────────────────────────────────────────────────────

class C:
    _on = USE_COLOR and sys.stdout.isatty()
    RESET   = "\033[0m"  if _on else ""
    BOLD    = "\033[1m"  if _on else ""
    DIM     = "\033[2m"  if _on else ""
    CYAN    = "\033[96m" if _on else ""
    GREEN   = "\033[92m" if _on else ""
    YELLOW  = "\033[93m" if _on else ""
    RED     = "\033[91m" if _on else ""
    BLUE    = "\033[94m" if _on else ""
    MAGENTA = "\033[95m" if _on else ""

# ── Utilities ────────────────────────────────────────────────────────────────

def format_todos(raw_data):
    if not raw_data or raw_data == "No incomplete todos.":
        return "No incomplete todos."
    try:
        items = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        lines = []
        for t in items:
            due = f" (Due: {t['due']})" if t.get('due') else ""
            lines.append(f"{t['order']}. {t['title']}{due}")
        return "\n".join(lines)
    except:
        return str(raw_data)

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def print_banner(model_name):
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗
║  {ASSISTANT_NAME} — Simple Local Assistant                                  ║
║  Model : {model_name:<68}║
║  Workspace : {WORKSPACE_DIR:<64}║
╚══════════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")

def print_help():
    print(f"""
{C.YELLOW}Commands:{C.RESET}
  {C.BOLD}/help{C.RESET}    Show this help
  {C.BOLD}/tools{C.RESET}   List available tools
  {C.BOLD}/clear{C.RESET}   Clear history
  {C.BOLD}/exit{C.RESET}    Quit
""")

# ── Tooling ──────────────────────────────────────────────────────────────────

def get_tool_map():
    return {t.name: t for t in tools}

def get_tools_description():
    lines = [f"- {t.name}: {t.description.split('\\n')[0][:100]}" for t in tools]
    return "\n".join(lines)

# ── Agent Logic ───────────────────────────────────────────────────────────────

_DECISION_SCHEMA = """
Respond ONLY with a JSON object:
{"tool_call": {"name": "tool_name", "arguments": {}}, "final_answer": null} 
OR
{"tool_call": null, "final_answer": "your response"}
"""

def run_turn(model, system_prompt, messages, tool_map):
    # Construct the prompt with the schema reminder at the end
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    full_messages.append({"role": "user", "content": _DECISION_SCHEMA})

    tool_calls_made = 0
    while tool_calls_made < 5:
        response = ollama.chat(
            model=model,
            messages=full_messages,
            options={"num_ctx": 8192, "temperature": 0.7}
        )
        
        raw_text = response['message']['content']
        clean_text = _strip_think(raw_text)

        try:
            # Extract JSON from response
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            if start == -1 or end == -1:
                return clean_text # Fallback to raw text

            decision = json.loads(clean_text[start:end+1])
            tc = decision.get("tool_call")
            fa = decision.get("final_answer")

            if fa:
                return fa
            
            if tc:
                tool_calls_made += 1
                name, args = tc["name"], tc.get("arguments", {})
                print(f"\n{C.DIM}  > Using tool: {name}...{C.RESET}")
                
                tool_fn = tool_map.get(name)
                if not tool_fn:
                    res = f"Error: tool {name} not found"
                else:
                    # LangChain tools use .invoke() or __call__
                    res = tool_fn.invoke(args) if hasattr(tool_fn, 'invoke') else tool_fn(args)
                
                # Special handling for todo mutating actions to keep it clean
                if name == "manage_todo" and args.get("action") in {"add", "complete", "delete", "update"}:
                    raw_list = manage_todo.invoke({"action": "list"})
                    return f"Done! Updated list:\n{format_todos(raw_list)}"

                full_messages.append({"role": "assistant", "content": f"Call {name} with {args}"})
                full_messages.append({"role": "user", "content": f"Tool result: {res}"})
                continue
                
        except Exception as e:
            return f"Error parsing LLM response: {e}\nRaw: {clean_text}"

    return "Max tool calls reached. Please try again."

# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()
    model_name = args.model

    print_banner(model_name)
    
    # Setup prompts
    with open("system_prompt.txt", "r") as f:
        sys_prompt = f.read().format(ASSISTANT_NAME=ASSISTANT_NAME, WORKSPACE_DIR=WORKSPACE_DIR)
    sys_prompt += f"\n\nAvailable tools:\n{get_tools_description()}"
    
    tool_map = get_tool_map()
    messages = []

    # Startup: list todos
    print(f"{C.BOLD}{C.MAGENTA}{ASSISTANT_NAME}:{C.RESET} Initializing... ", end="", flush=True)
    try:
        todos = manage_todo.invoke({"action": "list"})
        display = format_todos(todos)
        print(f"\n{display}\n")
        messages.append({"role": "assistant", "content": f"Current todos:\n{display}"})
    except Exception as e:

        print(f"Could not load todos: {e}\n")

    # Input handling
    input_queue = queue.Queue()
    def reader():
        import msvcrt
        while True:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwche()
                    if ch in ('\r', '\n'):
                        print()
                        # This is a simplification; in real app you'd buffer input
                        # For now, we just use a simple input() loop for brevity in this rewrite
                        pass 
                time.sleep(0.05)
            except: pass

    # Actually, let's use a standard input loop for the "simple" version 
    # to avoid threading complexity unless specifically needed.
    while True:
        try:
            user_input = input(f"{C.BOLD}{C.BLUE}You:{C.RESET} ").strip()
        except KeyboardInterrupt:
            break

        if not user_input: continue
        if user_input.lower() in ("/exit", "/quit", "/bye"): break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/tools":
            print(f"\n{C.YELLOW}Tools:{C.RESET}\n{get_tools_description()}\n")
            continue
        if user_input == "/clear":
            messages.clear()
            print(f"{C.GREEN}History cleared.{C.RESET}")
            continue

        messages.append({"role": "user", "content": user_input})
        
        # Agent Turn
        answer = run_turn(model_name, sys_prompt, messages, tool_map)
        answer = _strip_think(answer)
        
        print(f"\n{C.BOLD}{C.MAGENTA}{ASSISTANT_NAME}:{C.RESET} {answer}\n")
        messages.append({"role": "assistant", "content": answer})

        if len(messages) > MAX_HISTORY * 2:
            messages = messages[-(MAX_HISTORY * 2):]

    print(f"{C.DIM}Bye!{C.RESET}")

if __name__ == "__main__":
    main()
