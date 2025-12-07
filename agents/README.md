# Project Gordon - Code Generation Agents

This directory contains custom Claude agents for automating code generation.

## What's an Agent?

An agent is a **persistent Claude session** that:
- Remembers context across multiple requests (like a real conversation)
- Has custom tools specific to your project
- Can run asynchronously without blocking
- Can execute multi-step workflows automatically

## Quick Start

```bash
# Run the code generation agent
python agents/codegen.py
```

You'll see a menu with examples:
1. **Simple generation** - One-off code creation
2. **Multi-step workflow** - Shows how Claude remembers context
3. **Custom tools** - Uses your project-specific MCP tools
4. **Interactive mode** - Chat with Claude to generate code

## What Each Part Does

### 1. MCP Tools (`@tool` decorators)

These are like **custom commands** you're teaching Claude:

```python
@tool("generate_module", "Generate a new Python module...", {...})
async def generate_module(args):
    # This tells Claude WHAT to do, not HOW
    # Claude figures out the HOW using its existing tools (Read, Write, Edit)
    return instructions
```

**Think of it like:** Giving Claude a recipe card. When you say "use generate_module", Claude follows that recipe.

### 2. Agent Configuration (`ClaudeAgentOptions`)

This sets up **what Claude can do**:

```python
options = ClaudeAgentOptions(
    mcp_servers={"gordon": codegen_server},  # Your custom tools
    allowed_tools=["Read", "Write", ...],     # What Claude can use
    permission_mode="acceptEdits",            # Auto-approve changes
)
```

### 3. Session (`async with client`)

This is the **conversation**:

```python
async with client:
    await client.query("Create module X")
    # ... Claude creates it ...

    await client.query("Now add tests")
    # Claude REMEMBERS module X and creates matching tests!
```

Without a session, each query would be independent and Claude would forget.

## Example Workflow

Here's what happens when you run Example 2 (multi-step):

```
You: Create a config_manager.py module
Claude: *reads existing code patterns*
        *creates config_manager.py with similar style*
        ✓ Created config_manager.py

You: Now create tests for it
Claude: *remembers the config_manager.py it just created*
        *creates matching test_config_manager.py*
        ✓ Created tests

You: Add documentation
Claude: *remembers both files*
        *adds docstrings and README*
        ✓ Added docs
```

All in ONE session, Claude maintains context!

## Customizing for Your Project

### Add New Tools

Edit `codegen.py` and add new `@tool` functions:

```python
@tool(
    "generate_api_endpoint",
    "Create a new REST API endpoint",
    {"endpoint_name": str, "method": str}
)
async def generate_api_endpoint(args):
    return {
        "content": [{
            "type": "text",
            "text": f"Create {args['method']} endpoint: {args['endpoint_name']}"
        }]
    }
```

Then add to allowed_tools:
```python
"mcp__gordon__generate_api_endpoint"
```

### Change Permissions

```python
permission_mode="ask"  # Claude will ask before editing files
```

## Comparison: Agent SDK vs Regular Claude Code

| Feature | Claude Code (normal) | Agent SDK |
|---------|---------------------|-----------|
| Context retention | Single conversation | Persistent sessions |
| Custom tools | No | Yes (MCP) |
| Async operations | No | Yes |
| Multi-step workflows | Manual | Automated |
| Programmatic access | No | Yes |

## Use Cases

1. **Generate boilerplate** - Create new modules following project patterns
2. **Refactoring** - Update multiple files maintaining consistency
3. **Testing** - Generate tests for existing code
4. **Documentation** - Auto-generate docs from code
5. **CI/CD integration** - Run as part of build pipeline

## Next Steps

1. Run `python agents/codegen.py` and try Example 2
2. Try Interactive Mode (option 4) - ask it to generate something for your project
3. Customize the tools in `codegen.py` for your specific needs
4. Add more agents for different purposes (testing, refactoring, etc.)

## Troubleshooting

**Error: "API key not found"**
- Set `ANTHROPIC_API_KEY` environment variable
- Or create `.env` file with your key

**Error: "Tool not found"**
- Check tool name in `allowed_tools` matches MCP tool name
- Format: `mcp__<server_name>__<tool_name>`

**Claude not remembering context**
- Make sure you're using `async with client:` block
- All queries in the same block share context
