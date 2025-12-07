"""
Quick demo of the Code Generation Agent

This script runs automatically without requiring input.
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


# Simple custom tool
@tool(
    "project_info",
    "Get information about project-gordon structure",
    {}
)
async def project_info(args):
    return {
        "content": [{
            "type": "text",
            "text": """
            Project Gordon Structure:
            - llm/ - LLM integration code
            - audio_handler.py - Audio processing
            - main.py - Main application
            - tts/, stt/, vad/ - Speech modules

            Follow these patterns when generating code:
            - Use type hints
            - Add docstrings
            - Include error handling
            """
        }]
    }


async def quick_demo():
    """
    Quick demonstration of the agent in action
    """
    print("="*70)
    print("PROJECT GORDON - AGENT SDK DEMO")
    print("="*70)
    print("\nThis demonstrates how the Agent SDK works...")
    print("\nCreating custom Claude agent with project-specific tools...")

    # Create MCP server
    demo_server = create_sdk_mcp_server(
        name="gordon_demo",
        version="1.0.0",
        tools=[project_info]
    )

    # Configure agent
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code"
        },
        mcp_servers={"gordon": demo_server},
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "mcp__gordon__project_info"
        ],
        permission_mode="ask",  # Will ask before making changes
        setting_sources=["project"]
    )

    print("\n[*] Initializing Claude agent...")

    try:
        client = ClaudeSDKClient(options=options)

        async with client:
            print("\n[*] Agent ready! Sending test query...\n")
            print("-"*70)

            # Simple query to test the agent
            prompt = """
            Analyze the project structure and suggest what utility module
            would be most useful to add. Don't create any files, just analyze
            and provide recommendations.
            """

            print(f"Query: {prompt.strip()}\n")
            print("-"*70)

            await client.query(prompt)

            print("\nClaude's Response:")
            print("-"*70)

            # Get the response
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\n{block.text}\n")

            print("-"*70)
            print("\n[SUCCESS] Demo completed!")
            print("\nWhat just happened?")
            print("1. Created a custom Claude agent with project-specific knowledge")
            print("2. Sent a query asking for analysis")
            print("3. Claude used its tools to analyze your project")
            print("4. Provided recommendations based on your codebase")
            print("\nNext steps:")
            print("- Run 'python agents/codegen.py' for interactive mode")
            print("- Customize the tools in codegen.py for your needs")
            print("- Try the multi-step example to see context retention")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nPossible issues:")
        print("1. ANTHROPIC_API_KEY not set - add to environment variables")
        print("2. No internet connection")
        print("3. claude-agent-sdk not installed - run: pip install claude-agent-sdk")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(quick_demo())
