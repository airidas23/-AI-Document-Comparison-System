import requests
import sseclient
import json
import threading
import time

BASE_URL = "http://localhost:7860"
SSE_URL = f"{BASE_URL}/gradio_api/mcp/sse"

def get_tools_schema():
    response = requests.get(SSE_URL, stream=True)
    client = sseclient.SSEClient(response)
    iterator = client.events()
    first_event = next(iterator)
    if first_event.event != "endpoint":
        print("Failed to get endpoint")
        return
    
    post_url = f"{BASE_URL}{first_event.data}"
    
    # Initialize
    requests.post(post_url, json={
        "jsonrpc": "2.0", "method": "initialize", "id": 1,
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}
    })
    
    # List Tools
    resp = requests.post(post_url, json={
        "jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}
    })
    
    # We need to wait for the SSE message for id 2
    # Since we are just listing, we can just listen for a bit
    for event in client.events():
        if event.event == "message":
            msg = json.loads(event.data)
            if msg.get("id") == 2:
                tools = msg["result"]["tools"]
                for tool in tools:
                    if tool["name"] == "compare_documents":
                        print(json.dumps(tool, indent=2))
                break

if __name__ == "__main__":
    get_tools_schema()
