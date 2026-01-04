# Gradio MCP Integration

This project is preconfigured to use a remote Gradio MCP server and also includes a local demo you can run optionally.

## VS Code / GitHub Copilot Chat (Remote)

- Config file: /.vscode/mcp.json
- Server: gradio → https://gradio-docs-mcp.hf.space/gradio_api/mcp/

Steps:
1) Reload VS Code (Cmd+Shift+P → "Developer: Reload Window").
2) Open Copilot Chat. Approve the "gradio" MCP server if prompted.
3) Ask Copilot Chat: "List tools for the 'gradio' MCP server".

## Cursor (Remote or Local)

- Config file: /.cursor/mcp.json
- Server: gradio → https://gradio-docs-mcp.hf.space/gradio_api/mcp/

If running a local demo (below), use your local base URL (http://127.0.0.1:7860/gradio_api/mcp/).

## Local MCP Demo (Optional)

Keep the main app intact while trying MCP locally.

1) Create/activate a dedicated virtual environment (recommended).
2) Install MCP-enabled Gradio:

```bash
pip install -r requirements-mcp.txt
```

3) Start the demo:

```bash
python scripts/gradio_mcp_demo.py
```

You should see:
- App: http://127.0.0.1:7860
- MCP (SSE): http://127.0.0.1:7860/gradio_api/mcp/sse

4) Point your MCP client (VS Code/Copilot, Cursor) at the local base MCP URL and list tools.

## Gradio Docs MCP Tools

The official Gradio Docs MCP server exposes two tools:
- `gradio_docs_mcp_load_gradio_docs`: No arguments; loads a concise /llms.txt-style summary of the latest Gradio docs.
- `gradio_docs_mcp_search_gradio_docs`: Takes a `query` string; returns relevant chunks from Gradio docs, guides, and demos.

## Notes
- We did not upgrade the main app's Gradio dependency to avoid breaking changes. The demo uses a separate requirements-mcp.txt.
- The remote Hugging Face Space may change availability; if it’s unavailable, use the local demo.
