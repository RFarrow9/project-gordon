"""
Project Gordon - Async Code Generation Agent

This agent helps automate code generation for the project-gordon codebase.
It understands your project structure and coding patterns.

Usage:
    python agents/codegen.py
"""

import asyncio
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock
)
from typing import Any


# ============================================================================
# CUSTOM MCP TOOLS - These give Claude special abilities for your project
# ============================================================================

@tool(
    "generate_module",
    "Generate a new Python module with boilerplate following project standards",
    {
        "module_name": str,
        "description": str,
        "include_tests": bool
    }
)
async def generate_module(args: dict[str, Any]) -> dict[str, Any]:
    """
    Custom tool to generate new modules following project-gordon patterns.

    This is just a placeholder - the actual code generation happens when
    Claude processes this request and uses Write/Edit tools.
    """
    module_name = args["module_name"]
    description = args["description"]
    include_tests = args.get("include_tests", True)

    instructions = f"""
    Create a new Python module: {module_name}
    Description: {description}

    Follow project-gordon standards:
    - Add docstrings to all functions
    - Use type hints
    - Include error handling
    - Follow existing code patterns in the project
    {"- Create test file in tests/ directory" if include_tests else ""}

    Structure:
    - Main module: {module_name}.py
    {"- Tests: tests/test_" + module_name + ".py" if include_tests else ""}
    """

    return {
        "content": [{
            "type": "text",
            "text": instructions
        }]
    }


@tool(
    "analyze_codebase",
    "Analyze the project structure and coding patterns",
    {
        "focus_area": str
    }
)
async def analyze_codebase(args: dict[str, Any]) -> dict[str, Any]:
    """
    Tool to help Claude understand your project before generating code.
    """
    focus = args["focus_area"]

    instructions = f"""
    Analyze the project-gordon codebase focusing on: {focus}

    Look at:
    - Existing patterns in similar modules
    - Import structures
    - Error handling approaches
    - Testing patterns
    - Documentation style

    Provide a summary of patterns to follow.
    """

    return {
        "content": [{
            "type": "text",
            "text": instructions
        }]
    }


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

async def create_codegen_agent():
    """
    Initialize the code generation agent with custom tools and project context.

    Returns:
        ClaudeSDKClient: Configured agent ready for code generation tasks
    """

    # Create MCP server with your custom tools
    codegen_server = create_sdk_mcp_server(
        name="project_gordon_codegen",
        version="1.0.0",
        tools=[generate_module, analyze_codebase]
    )

    # Configure the agent
    options = ClaudeAgentOptions(
        # Use Claude Code's system prompt (understands software engineering)
        system_prompt={
            "type": "preset",
            "preset": "claude_code"
        },

        # Add your custom MCP server
        mcp_servers={"gordon": codegen_server},

        # Allow the agent to use these tools
        allowed_tools=[
            "Read",           # Read existing files
            "Write",          # Create new files
            "Edit",           # Modify files
            "Bash",           # Run commands (git, pytest, etc.)
            "Glob",           # Find files
            "Grep",           # Search code
            "mcp__gordon__generate_module",    # Your custom tool
            "mcp__gordon__analyze_codebase"    # Your custom tool
        ],

        # Auto-accept edits (set to "ask" if you want approval)
        permission_mode="acceptEdits",

        # Load project context from .claude/ directory
        setting_sources=["project"]
    )

    return ClaudeSDKClient(options=options)


# ============================================================================
# EXAMPLE WORKFLOWS
# ============================================================================

async def simple_generation_example():
    """
    Example 1: Simple one-off code generation
    """
    print("=" * 70)
    print("EXAMPLE 1: Simple Code Generation")
    print("=" * 70)

    client = await create_codegen_agent()

    async with client:
        prompt = """
        Create a new utility module called 'audio_utils.py' that provides:
        - Function to normalize audio volume
        - Function to detect silence
        - Error handling for invalid audio data

        Follow the existing patterns in audio_handler.py
        """

        print(f"\n[>] Prompt: {prompt}\n")
        print("[*] Claude is generating code...\n")

        await client.query(prompt)

        # Stream the response
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}\n")


async def multi_step_example():
    """
    Example 2: Multi-step workflow with context retention

    This shows the POWER of the Agent SDK - Claude remembers previous steps!
    """
    print("=" * 70)
    print("EXAMPLE 2: Multi-Step Workflow with Memory")
    print("=" * 70)

    client = await create_codegen_agent()

    async with client:
        # Step 1: Generate code
        print("\n[>] Step 1: Generate initial module\n")
        await client.query("Create a new 'config_manager.py' module for managing app configuration")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:200]}...\n")

        # Step 2: Add tests (Claude remembers the code it just wrote!)
        print("\n[>] Step 2: Generate tests for the module we just created\n")
        await client.query("Now create comprehensive unit tests for the config_manager module")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:200]}...\n")

        # Step 3: Add documentation
        print("\n[>] Step 3: Add documentation\n")
        await client.query("Add detailed docstrings and a README for the config_manager")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:200]}...\n")


async def custom_tool_example():
    """
    Example 3: Using your custom MCP tools
    """
    print("=" * 70)
    print("EXAMPLE 3: Using Custom MCP Tools")
    print("=" * 70)

    client = await create_codegen_agent()

    async with client:
        prompt = """
        Use the generate_module tool to create a new 'voice_recognition' module
        with description 'Voice activity detection and speech recognition utilities'
        and include tests.
        """

        print(f"\n[>] Prompt: {prompt}\n")
        print("[*] Claude is using your custom tool...\n")

        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """
    Main entry point - choose which example to run
    """
    print("""
====================================================================
        Project Gordon - Code Generation Agent

  This agent automates code generation for your project
====================================================================
    """)

    print("Choose an example to run:")
    print("1. Simple code generation")
    print("2. Multi-step workflow (shows context retention)")
    print("3. Custom MCP tool usage")
    print("4. Interactive mode (ask your own questions)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        await simple_generation_example()
    elif choice == "2":
        await multi_step_example()
    elif choice == "3":
        await custom_tool_example()
    elif choice == "4":
        print("\n[*] Interactive Mode - Type your code generation requests")
        print("(Type 'exit' to quit)\n")

        client = await create_codegen_agent()
        async with client:
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == 'exit':
                    break

                await client.query(user_input)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"\nClaude: {block.text}\n")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
