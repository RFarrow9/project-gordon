# Quick Start: Claude Agent SDK for Project Gordon

## What You've Got

You now have a **local Claude plugin** that can automate code generation for your project!

**Location:** `agents/codegen.py`

## What It Actually Does (Simple Explanation)

### Without Agent SDK (normal Claude Code):
```
You: "Create a module"
Claude: *creates module* ✓

You: "Add tests for that module"
Claude: "Which module?" ❌ (forgot)
```

### With Agent SDK (what you just set up):
```
You: "Create a module"
Agent: *creates module* ✓

You: "Add tests for that module"
Agent: *remembers the module* ✓ *creates matching tests* ✓

You: "Add docs"
Agent: *remembers both* ✓ *adds documentation* ✓
```

**The agent maintains context across multiple requests!**

## File Structure

```
agents/
├── codegen.py       ← Main agent (4 examples included)
├── demo.py          ← Simple demo to test it works
├── README.md        ← Detailed documentation
└── QUICKSTART.md    ← This file
```

## How to Use

### 1. Set Up API Key (Required)

The agent needs to connect to Claude's API:

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "your-api-key-here"
```

**Windows (CMD):**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**Or create a `.env` file:**
```
ANTHROPIC_API_KEY=your-api-key-here
```

Get your API key from: https://console.anthropic.com/

### 2. Test It Works

Run the simple demo (doesn't create files, just analyzes):

```bash
python agents/demo.py
```

This will:
- Create a custom Claude agent
- Ask it to analyze your project
- Show you recommendations
- Prove everything is working

### 3. Try Interactive Mode

```bash
python agents/codegen.py
```

Choose option **4** (Interactive mode) and try:
```
You: Create a simple logger utility in llm/logger.py
You: Add tests for it
You: Add docstrings
```

Watch how Claude remembers context between requests!

### 4. Try Multi-Step Example

```bash
python agents/codegen.py
```

Choose option **2** to see the agent:
1. Create a module
2. Remember it and create tests
3. Remember both and add docs

All in one continuous session!

## What Each File Does

### `codegen.py` - The Main Agent

**Key Parts:**

```python
# 1. CUSTOM TOOLS (@tool decorators)
@tool("generate_module", ...)
async def generate_module(args):
    # Teaches Claude a new command

# 2. AGENT SETUP
client = ClaudeSDKClient(options)
# Creates your custom Claude

# 3. SESSION (maintains memory)
async with client:
    await client.query("Step 1")
    await client.query("Step 2")  # Remembers Step 1!
```

### `demo.py` - Quick Test

Non-interactive demo you can run to verify setup.

## Real-World Use Cases

### Example 1: Generate Boilerplate

```python
async with client:
    await client.query("Create a new API client for OpenAI in llm/openai_client.py")
    # Claude creates it following your project patterns

    await client.query("Add retry logic with exponential backoff")
    # Claude updates the file it just created

    await client.query("Create tests")
    # Claude creates tests for the code it wrote
```

### Example 2: Refactoring

```python
async with client:
    await client.query("Extract all database code from main.py into a new db/ module")
    # Claude analyzes main.py and extracts DB code

    await client.query("Update all imports in other files")
    # Claude remembers what it moved and fixes imports

    await client.query("Add connection pooling")
    # Claude adds feature to the new module
```

### Example 3: Documentation

```python
async with client:
    await client.query("Analyze all files in llm/ and create a README")
    # Claude reads all files and creates comprehensive docs

    await client.query("Add usage examples for each function")
    # Claude adds examples based on actual code
```

## Key Concepts

### MCP Tools (`@tool` decorators)

These are **instructions** you give Claude:

```python
@tool("do_something", "description", {"param": str})
async def do_something(args):
    return {"content": [{"type": "text", "text": "Do XYZ"}]}
```

When you say "use do_something", Claude follows that recipe.

### Session Context

```python
async with client:
    # Everything inside here shares memory
    query("Do A")
    query("Now do B")  # Remembers A!
```

Leave the `async with` block and memory resets.

### Async

Allows long-running tasks without freezing:

```python
await client.query("Generate 10 modules")
# Can handle this without blocking
```

## Customization

### Add Your Own Tools

Edit `codegen.py`:

```python
@tool(
    "my_custom_tool",
    "What it does",
    {"arg1": str, "arg2": int}
)
async def my_custom_tool(args):
    return {
        "content": [{
            "type": "text",
            "text": f"Instructions for Claude: {args['arg1']}"
        }]
    }
```

Add to allowed tools:
```python
allowed_tools=[
    ...
    "mcp__gordon__my_custom_tool"
]
```

### Change Permissions

```python
permission_mode="ask"  # Claude asks before editing
permission_mode="acceptEdits"  # Auto-approve
```

## Troubleshooting

**Error: "API key not found"**
- Set `ANTHROPIC_API_KEY` environment variable

**Error: "Module not found: claude_agent_sdk"**
- Run: `pip install claude-agent-sdk`

**Agent doesn't remember context**
- Make sure queries are inside `async with client:` block

**Unicode errors**
- Fixed! (We replaced fancy characters with ASCII)

## Next Steps

1. **Run the demo**: `python agents/demo.py`
2. **Try interactive mode**: `python agents/codegen.py` → Option 4
3. **Customize tools**: Edit `codegen.py` and add project-specific tools
4. **Automate workflows**: Create custom agents for testing, docs, refactoring
5. **CI/CD integration**: Run agents in GitHub Actions

## Cost Warning

The Agent SDK uses Claude's API which costs money:
- Sonnet 3.5: ~$3 per million input tokens
- Each query uses tokens based on:
  - Your prompt
  - Files Claude reads
  - Code it generates

**Tip:** Start with `permission_mode="ask"` so you approve changes.

## Documentation

- Full docs: `agents/README.md`
- Agent SDK: https://platform.claude.com/docs/en/agent-sdk
- MCP Tools: https://code.claude.com/docs/en/mcp

## Summary

You now have:
- ✓ Custom Claude agent for your project
- ✓ Session-based context retention
- ✓ Custom MCP tools (extensible)
- ✓ Async code generation
- ✓ 4 working examples

**Try it now:**
```bash
python agents/demo.py
```
