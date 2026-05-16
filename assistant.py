"""
assistant.py  —  Personal CLI Assistant
Run: python assistant.py

Commands:
    /help   — show commands       /tools  — list tools
    /clear  — clear history       /save   — save conversation
    /ws     — show workspace      /exit   — quit
"""

import sys
import os
import datetime
import warnings
import threading
import time
import queue
import argparse
import json

# Suppress ResourceWarning for unclosed sockets on exit
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")

missing = []
try:
    from langchain_ollama import ChatOllama
except ImportError:
    missing.append("langchain-ollama")
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    missing.append("langgraph")
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.runnables import RunnableSerializable
except ImportError:
    missing.append("langchain-core")

if missing:
    print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
    print(f"Install: pip install {' '.join(missing)} duckduckgo-search\n")
    sys.exit(1)

from config.settings import MODEL, ASSISTANT_NAME, MAX_HISTORY, WORKSPACE_DIR, USE_COLOR
from tools.agent_tools import tools, manage_todo
import re

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


def print_banner(model_name):
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗
║  {ASSISTANT_NAME} — Personal Assistant                                             ║
║  Model : {model_name:<68}║
║  Workspace : {WORKSPACE_DIR:<64}║
╚══════════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")


def print_help():
    print(f"""
{C.YELLOW}Commands:{C.RESET}
  {C.BOLD}/help{C.RESET}    Show this help
  {C.BOLD}/tools{C.RESET}   List available tools
  {C.BOLD}/clear{C.RESET}   Clear conversation history
  {C.BOLD}/save{C.RESET}    Save conversation to workspace
  {C.BOLD}/ws{C.RESET}      Show workspace path
  {C.BOLD}/exit{C.RESET}    Quit
""")


def print_tools():
    print(f"\n{C.YELLOW}Loaded tools ({len(tools)}):{C.RESET}")
    for t in tools:
        desc = (t.description or "").split("\n")[0][:60]
        print(f"  {C.BOLD}{t.name:<18}{C.RESET} {C.DIM}{desc}{C.RESET}")
    print()


def save_conversation(messages: list):
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(WORKSPACE_DIR, f"conversation_{ts}.md")
    lines = [f"# Conversation — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"**You:** {msg.content}\n\n")
        elif isinstance(msg, AIMessage):
            lines.append(f"**{ASSISTANT_NAME}:** {msg.content}\n\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"{C.GREEN}Saved to: {filepath}{C.RESET}\n")

def timer_worker(stop_event):
    """Background thread to print elapsed time if the agent is still working."""
    start = time.time()
    while not stop_event.is_set():
        time.sleep(15)
        if not stop_event.is_set():
            elapsed = int(time.time() - start)
            print(f"\n{C.DIM}(Still working... {elapsed}s){C.RESET}", end="\r")

def print_thinking(msg):
    """Extract and print <think>...</think> blocks from a message in a dimmed style."""
    content = msg.content if isinstance(msg.content, str) else ""
    # Also check for thinking in additional_kwargs (some langchain versions put it there)
    thinking = None
    if hasattr(msg, "additional_kwargs"):
        thinking = msg.additional_kwargs.get("thinking") or msg.additional_kwargs.get("reasoning_content")
    if not thinking:
        import re
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
    if thinking:
        print(f"{C.DIM}┌─ Thinking ─────────────────────────────{C.RESET}")
        for line in thinking.strip().splitlines():
            print(f"{C.DIM}│ {line}{C.RESET}")
        print(f"{C.DIM}└────────────────────────────────────────{C.RESET}")


def compact_todo_context(agent, thread_config):
    """
    Replace all manage_todo tool call/result message pairs in LangGraph memory
    with a single concise snapshot of the current todo state.
    This keeps context lean — we only need to know the current list, not the
    full history of every add/remove/update operation this session.
    """
    try:
        from langchain_core.messages import ToolMessage, AIMessage as AI
        state = agent.get_state(thread_config)
        messages = state.values.get("messages", [])

        # Identify indexes of manage_todo tool call pairs (AIMessage with tool_calls + ToolMessage result)
        to_remove = set()
        for i, msg in enumerate(messages):
            if isinstance(msg, AI) and hasattr(msg, "tool_calls") and msg.tool_calls:
                if any(tc.get("name") == "manage_todo" for tc in msg.tool_calls):
                    to_remove.add(i)
                    # Also remove the corresponding ToolMessage result(s) that follow
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], ToolMessage):
                            to_remove.add(j)
                        else:
                            break

        if not to_remove:
            return  # nothing to compact

        # Fetch current todo state as the replacement snapshot
        todo_snapshot = manage_todo.invoke({"action": "list"})
        snapshot_msg = AI(content=f"[Context snapshot — current todo list]\n{todo_snapshot}")

        # Rebuild message list: keep non-todo messages, inject snapshot once
        compacted = [msg for i, msg in enumerate(messages) if i not in to_remove]
        # Insert snapshot just before the last HumanMessage so it's close to current context
        insert_at = len(compacted)
        for i in range(len(compacted) - 1, -1, -1):
            if isinstance(compacted[i], HumanMessage):
                insert_at = i
                break
        compacted.insert(insert_at, snapshot_msg)

        agent.update_state(thread_config, {"messages": compacted})
    except Exception:
        pass  # compaction is best-effort — never crash the main loop


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(content) // 4
    return total


class NoThinkWrapper(RunnableSerializable):
    """
    Wraps ChatOllama to prepend /no_think into the last human or tool message
    of every LLM call. This suppresses qwen3 reasoning on all internal agent
    calls (tool decision, response formatting) without modifying user-visible
    message content. summarize_with_thinking bypasses this wrapper entirely
    by calling the underlying llm directly with reasoning=True.
    """
    _llm: object = None

    def __init__(self, inner_llm):
        super().__init__()
        object.__setattr__(self, "_llm", inner_llm)

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def _inject(self, messages):
        from langchain_core.messages import HumanMessage as HM, ToolMessage
        msgs = list(messages)
        for i in reversed(range(len(msgs))):
            if isinstance(msgs[i], (HM, ToolMessage)):
                original = msgs[i].content if isinstance(msgs[i].content, str) else ""
                if not original.startswith("/no_think"):
                    msgs[i] = msgs[i].model_copy(update={"content": "/no_think " + original})
                break
        return msgs

    def invoke(self, messages, config=None, **kwargs):
        if kwargs.get("reasoning"):
            return self._llm.invoke(messages, **kwargs)
        return self._llm.invoke(self._inject(messages), config=config, **kwargs)

    def stream(self, messages, config=None, **kwargs):
        return self._llm.stream(self._inject(messages), config=config, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return NoThinkWrapper(self._llm.bind_tools(tools, **kwargs))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Personal Assistant CLI")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Ollama model to use (default: {MODEL})")
    args = parser.parse_args()
    
    print_banner(args.model)
    print(f"{C.DIM}Give me a minute while I start the LLM and pull your to-dos...{C.RESET}\n")

    try:
        # qwen3 supports /no_think token to suppress reasoning for faster responses.
        # Thinking is disabled by default; only summarize_with_thinking uses it explicitly.
        is_qwen3 = "qwen3" in args.model.lower()
        llm_kwargs = {"model": args.model, "temperature": 0.7}
        llm = ChatOllama(**llm_kwargs)
        if is_qwen3:
            llm = NoThinkWrapper(llm)
        active_tools = tools
    except Exception as e:
        print(f"\n{C.RED}Error: Could not connect to Ollama.\n{e}{C.RESET}")
        print("Make sure Ollama is running:  ollama serve")
        print(f"And the model is pulled:      ollama pull {args.model}\n")
        sys.exit(1)

    # Define the prompt to instruct the LLM to handle todo listing on startup.
    # This prompt tells the LLM to use the 'manage_todo' tool and format the output.
    # The LLM is assumed to have access to this tool through its configuration.
    # Load system prompt from file
    with open("system_prompt.txt", "r") as f:
        prompt_template = f.read()
    system_prompt = prompt_template.format(ASSISTANT_NAME=ASSISTANT_NAME, WORKSPACE_DIR=WORKSPACE_DIR)
    if is_qwen3:
        system_prompt = "/no_think\n" + system_prompt

    memory = MemorySaver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = create_react_agent(llm, active_tools, checkpointer=memory, prompt=system_prompt)
    thread_config = {"configurable": {"thread_id": "main-session"}}
    conversation_messages = []

    # On startup, ask the agent to display any outstanding to-do items
    todo_listing_prompt = (
        "Use the 'manage_todo' tool with action='list' to fetch all incomplete to-do items. "
        "Then display them using the todo formatting rules in your instructions. "
        "Do not say there was an error — if the tool returns data, display it. "
        "If the tool returns 'No incomplete todos', say so briefly."
    )
    conversation_messages.append(HumanMessage(content=todo_listing_prompt))

    # Run the agent once on startup to display the todo list
    print(f"{C.BOLD}{C.MAGENTA}{ASSISTANT_NAME}:{C.RESET} ", end="", flush=True)
    last_startup_content = ""
    stop_timer = threading.Event()
    timer_thread = threading.Thread(target=timer_worker, args=(stop_timer,), daemon=True)
    timer_thread.start()
    try:
        for event in agent.stream(
            {"messages": conversation_messages},
            config=thread_config,
            stream_mode="updates"
        ):
            for node_name, output in event.items():
                if "messages" in output:
                    msg = output["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_name = msg.tool_calls[0]['name']
                        print(f"\n{C.DIM}  > Using tool: {tool_name}...{C.RESET}")
                        print(f"{C.BOLD}{C.MAGENTA}{ASSISTANT_NAME}:{C.RESET} ", end="", flush=True)
                    elif isinstance(msg, AIMessage):
                        print_thinking(msg)
                        clean = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
                        if clean:
                            print(clean, end="", flush=True)
                            last_startup_content = clean
        stop_timer.set()
        print("\n")
        conversation_messages.append(AIMessage(content=last_startup_content))
    except Exception as e:
        stop_timer.set()
        print(f"\n{C.RED}(Could not load todos on startup: {e}){C.RESET}\n")

    # ── Input queue ───────────────────────────────────────────────────────────
    # Uses msvcrt on Windows for true non-blocking key reading so the user can
    # type while Aria is working. Characters are buffered into lines and queued.
    input_queue = queue.Queue()
    cancel_event = threading.Event()        # signals the agent loop to stop mid-stream
    _input_buf = []
    _prompt_shown = threading.Event()

    def input_reader():
        """Read keyboard input character by character (non-blocking on Windows)."""
        import msvcrt
        # Show initial prompt
        print(f"{C.BOLD}{C.BLUE}You:{C.RESET} ", end="", flush=True)
        _prompt_shown.set()
        while True:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwche()  # echo character immediately
                    if ch in ('\r', '\n'):  # Enter
                        print()  # newline after enter
                        line = "".join(_input_buf).strip()
                        _input_buf.clear()
                        if line:
                            # /stop and /cancel are handled immediately — set cancel flag right here
                            if line.lower() in ("/stop", "/cancel"):
                                cancel_event.set()
                                print(f"{C.YELLOW}(cancelling...){C.RESET}")
                            else:
                                input_queue.put(line)
                        # Only show next prompt if agent is idle (queue empty and not working)
                        # The main loop will re-show the prompt after processing
                    elif ch == '\x1b':  # Escape — cancel current request
                        cancel_event.set()
                        _input_buf.clear()
                        print(f"\n{C.YELLOW}(cancelling...){C.RESET}")
                    elif ch == '\x03':  # Ctrl+C
                        input_queue.put(None)
                        return
                    elif ch == '\x08':  # Backspace
                        if _input_buf:
                            _input_buf.pop()
                            # Erase character on terminal
                            print(' \b', end="", flush=True)
                    elif ch >= ' ':  # printable characters only
                        _input_buf.append(ch)
                else:
                    time.sleep(0.05)  # small sleep to avoid busy-waiting
            except Exception:
                time.sleep(0.05)

    reader_thread = threading.Thread(target=input_reader, daemon=True)
    reader_thread.start()
    _prompt_shown.wait()  # wait for first "You: " to appear before agent starts

    while True:
        # Drain any queued inputs first, then block for the next one
        user_input = input_queue.get()         # blocks until something arrives

        if user_input is None:
            print(f"\n{C.DIM}Bye!{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/exit", "/quit", "/bye"):
                print(f"{C.DIM}Bye!{C.RESET}"); break
            elif cmd == "/help":   print_help()
            elif cmd == "/tools":  print_tools()
            elif cmd == "/ws":
                print(f"{C.DIM}Workspace: {WORKSPACE_DIR}{C.RESET}\n")
            elif cmd == "/clear":
                conversation_messages.clear()
                thread_config = {"configurable": {"thread_id": f"session-{datetime.datetime.now().timestamp()}"}}
                print(f"{C.GREEN}Conversation cleared.{C.RESET}\n")
            elif cmd == "/save":
                save_conversation(conversation_messages) if conversation_messages else print(f"{C.DIM}Nothing to save yet.{C.RESET}\n")
            else:
                print(f"{C.RED}Unknown command '{cmd}'. Type /help.{C.RESET}\n")
            continue

        conversation_messages.append(HumanMessage(content=user_input))

        cancel_event.clear()  # reset before each new request
        try:
            # Start timer thread
            stop_timer = threading.Event()
            timer_thread = threading.Thread(target=timer_worker, args=(stop_timer,), daemon=True)
            timer_thread.start()
            
            start_time = time.time()
            
            # Keep track of whether we've printed the assistant's name prefix
            prefix_printed = False
            last_message_content = ""
            tool_call_count = 0
            MAX_TOOL_CALLS = 5  # prevent infinite tool-call loops
            
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=thread_config,
                stream_mode="updates"
            ):
                if cancel_event.is_set():
                    print(f"\n{C.YELLOW}(cancelled){C.RESET}\n")
                    break
                for node_name, output in event.items():
                    if "messages" in output:
                        msg = output["messages"][-1]
                        
                        # Handle tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                             tool_call_count += 1
                             if tool_call_count > MAX_TOOL_CALLS:
                                 print(f"\n{C.YELLOW}(too many tool calls — stopping to avoid a loop){C.RESET}\n")
                                 cancel_event.set()
                                 break
                             tool_name = msg.tool_calls[0]['name']
                             print(f"\n{C.DIM}  > Using tool: {tool_name}...{C.RESET}")
                             prefix_printed = False 
                        
                        # Handle actual assistant text
                        elif isinstance(msg, AIMessage):
                             print_thinking(msg)
                             clean = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
                             if clean:
                                 if not prefix_printed:
                                     print(f"{C.BOLD}{C.MAGENTA}{ASSISTANT_NAME}:{C.RESET} ", end="", flush=True)
                                     prefix_printed = True
                                 print(clean, end="", flush=True)
                                 last_message_content = clean

            # Stop timer thread
            stop_timer.set()
            
            # End timing and show context size
            elapsed = time.time() - start_time
            ctx_tokens = estimate_tokens(conversation_messages)
            ctx_pct = int((ctx_tokens / 32768) * 100)
            print(f"\n{C.DIM}({elapsed:.1f}s | ctx: ~{ctx_tokens:,} tokens / 32k — {ctx_pct}%){C.RESET}\n")
            
            conversation_messages.append(AIMessage(content=last_message_content))

            if len(conversation_messages) > MAX_HISTORY * 2:
                conversation_messages = conversation_messages[-(MAX_HISTORY * 2):]

            # Compact todo tool exchanges in LangGraph memory down to a single state snapshot
            compact_todo_context(agent, thread_config)
        except KeyboardInterrupt:
            stop_timer.set()
            print(f"\n{C.YELLOW}(interrupted){C.RESET}\n")
        except Exception as e:
            stop_timer.set()
            print(f" " * 30, end="\r")
            print(f"{C.RED}Error: {e}{C.RESET}\n")
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                print(f"{C.DIM}Is Ollama still running? Try: ollama serve{C.RESET}\n")

        # Re-show the You: prompt after response
        print(f"{C.BOLD}{C.BLUE}You:{C.RESET} {''.join(_input_buf)}", end="", flush=True)


if __name__ == "__main__":
    main()
