import requests
import sseclient
import json
import threading
import time
import sys

BASE_URL = "http://localhost:7860"
SSE_URL = f"{BASE_URL}/gradio_api/mcp/sse"

def listen_to_sse(client, events_list):
    """Listens for SSE events and appends them to a shared list."""
    try:
        for event in client.events():
            if event.event == "message":
                print(f"[SSE] Received message: {event.data[:100]}...") # Truncate for log cleanliness
                events_list.append(json.loads(event.data))
    except Exception as e:
        print(f"[SSE] Listener stopped: {e}")

def verify_mcp():
    print(f"Connecting to {SSE_URL}...")
    
    # Start SSE connection
    try:
        response = requests.get(SSE_URL, stream=True)
        client = sseclient.SSEClient(response)
    except Exception as e:
        print(f"Failed to connect to SSE: {e}")
        return

    message_endpoint = None
    
    # 1. Get endpoint (blocking read for first event)
    # We need to manually iterate to get the first event without consuming all
    iterator = client.events()
    try:
        first_event = next(iterator)
        if first_event.event == "endpoint":
            message_endpoint = first_event.data
            print(f"Message endpoint found: {message_endpoint}")
        else:
            print(f"Unexpected first event: {first_event.event}")
            return
    except StopIteration:
        print("Stream closed before endpoint received")
        return

    # Construct POST URL
    post_url = f"{BASE_URL}{message_endpoint}"
    
    # 2. Start listener thread for subsequent events (responses)
    events_received = []
    t = threading.Thread(target=listen_to_sse, args=(client, events_received), daemon=True)
    t.start()
    
    # 3. Send Initialize
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }
    
    print("\nSending 'initialize'...")
    requests.post(post_url, json=init_payload)
    
    # Wait for response ID 1
    time.sleep(2)
    
    # 4. Send Initialized
    notify_payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    requests.post(post_url, json=notify_payload)
    print("Sent 'initialized' notification")

    # 5. List Tools
    list_tools_payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2,
        "params": {}
    }
    
    print("\nSending 'tools/list'...")
    requests.post(post_url, json=list_tools_payload)
    
    # Wait for response ID 2
    time.sleep(2)
    
    # Analyze responses
    print("\n--- Analysis ---")
    tools_found = False
    for msg in events_received:
        if msg.get("id") == 2 and "result" in msg:
            result = msg["result"]
            if "tools" in result:
                print(f"SUCCESS: Found {len(result['tools'])} tools:")
                for tool in result['tools']:
                    print(f"- {tool['name']}: {tool.get('description', '')}")
                tools_found = True
    
    if not tools_found:
        print("WARNING: Did not receive tools/list response yet.")
        print("Received messages:", json.dumps(events_received, indent=2))

if __name__ == "__main__":
    verify_mcp()
