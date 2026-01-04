import requests
import json
import os

BASE_URL = "http://localhost:7860"

def test_comparison():
    file_a = "/Users/airidas/Documents/KTU/P170M109 Computational Intelligence and Decision Making/project/data/synthetic/dataset/variation_01/variation_01_original.pdf"
    file_b = "/Users/airidas/Documents/KTU/P170M109 Computational Intelligence and Decision Making/project/data/synthetic/dataset/variation_01/variation_01_modified.pdf"

    print(f"Uploading {file_a}...")
    with open(file_a, "rb") as f:
        resp_a = requests.post(f"{BASE_URL}/gradio_api/upload", files={"files": f})
    path_a = resp_a.json()[0]
    print(f"Uploaded A: {path_a}")

    print(f"Uploading {file_b}...")
    with open(file_b, "rb") as f:
        resp_b = requests.post(f"{BASE_URL}/gradio_api/upload", files={"files": f})
    path_b = resp_b.json()[0]
    print(f"Uploaded B: {path_b}")

    # Now call the MCP tool via SSE JSON-RPC
    # We need to get the endpoint first
    import sseclient
    sse_resp = requests.get(f"{BASE_URL}/gradio_api/mcp/sse", stream=True)
    client = sseclient.SSEClient(sse_resp)
    it = client.events()
    ev = next(it)
    if ev.event != "endpoint":
        print("Error: No endpoint")
        return
    
    post_url = f"{BASE_URL}{ev.data}"
    
    # Initialize
    requests.post(post_url, json={
        "jsonrpc": "2.0", "method": "initialize", "id": 1,
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}
    })

    # Call compare_documents
    # Based on the schema, it takes an object with properties
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 3,
        "params": {
            "name": "compare_documents",
            "arguments": {
                "file_a": f"http://127.0.0.1:7860/gradio_api/file={path_a}",
                "file_b": f"http://127.0.0.1:7860/gradio_api/file={path_b}",
                "show_heat": False,
                "scanned_mode": False,
                "force_ocr_mode": False,
                "sensitivity": 0.82
            }
        }
    }
    
    print("\nCalling 'compare_documents' tool via MCP...")
    # This tool is a generator (yields), so we might get multiple messages
    requests.post(post_url, json=payload)
    
    print("Waiting for results...")
    for event in client.events():
        if event.event == "message":
            msg = json.loads(event.data)
            if msg.get("id") == 3:
                # This is the final result of the tool call
                # Note: Gradio MCP tools that yield usually return the LAST yield as the result,
                # or a notification. Actually, the Gradio MCP implementation handles yields
                # by sending notifications and a final result.
                print("\nReceived result from MCP:")
                print(json.dumps(msg, indent=2)[:1000] + "...") # Truncate for display
                break
            elif msg.get("method") == "notifications/tool_output":
                # Gradio-specific notification for yielding tools
                print(f"Progress Update: {msg['params']['content'][0]['text'][:200]}...")

if __name__ == "__main__":
    test_comparison()
