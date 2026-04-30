AI-Context Note: \`.md\` files are for AI parsing. Keep them brief, concise, and machine-readable to save tokens.

# Agent Project Context

This file provides important context about the agent project to help in development and task execution.

## Project Structure:

- **`assistant.py`**: The main script for the assistant's core functionality. It orchestrates the agent's behavior, processing inputs and generating outputs.
- **`config/`**: Directory for configuration files.
    - **`config/settings.py`**: Contains application-specific settings and configurations.
- **`tools/`**: Directory for utility scripts and agent tools.
    - **`tools/agent_tools.py`**: Houses custom tools and functions available to the agent.
- **`system_prompt.txt`**: Defines the agent's persona, instructions, and behavioral guidelines.
- **`requirements.txt`**: Lists all Python dependencies required for the project.

## Key Files and Their Purpose:

- **`assistant.py`**: The primary executable file. It orchestrates the agent's behavior, processing inputs and generating outputs. It utilizes the `manage_todo` tool to display current to-do items upon startup.
- **`config/settings.py`**: Holds environment-specific configurations, API keys, or general parameters.
- **`tools/agent_tools.py`**: Contains reusable functions or classes that the agent can call to perform specific actions. This file includes the implementation for the `manage_todo` tool.
- **`system_prompt.txt`**: Crucial for defining the agent's identity, its goals, and how it should interact with users or perform tasks. This is the foundational instruction set for the agent.
- **`requirements.txt`**: Essential for setting up the development environment. It ensures all necessary libraries are installed.

## To-Do Management Tool:

- The `assistant.py` script integrates with a `manage_todo` tool, which is implemented in `tools/agent_tools.py`.
- This tool is invoked to display existing to-do items when the assistant starts.
- It supports the following actions:
    - `add`: Adds a new todo item (requires `title`, optional `notes`, `due`).
    - `list`: Lists all incomplete todo items (returns JSON).
    - `complete`: Marks a todo as complete (requires `todo_id`).
    - `delete`: Removes a todo item (requires `todo_id`).

## Session Context Management:

- The application leverages `langgraph` and `MemorySaver` to manage session context and conversational memory.
- A unique `thread_id` is used to group interactions into distinct sessions.
- `MemorySaver` stores the *entire state* of a session as its latest version. This state encompasses:
    - The complete conversation history (user inputs and AI responses).
    - Results from tool calls.
    - The current status of any managed data, such as the todo list.
- When a new turn occurs, the agent loads this single, latest complete state from `MemorySaver`. It then updates this state with the new interaction and saves the *entire modified state* back. This ensures that context is preserved across turns, and only the most recent version of all conversational elements and managed data is retained for that session.

## Current Working Directory:

`C:\\ChiHo\\assistant`