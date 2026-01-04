"""
Minimal Gradio MCP demo, based on the DataCamp guide.
- No external API keys required
- Exposes an MCP server via SSE at /gradio_api/mcp/sse

Usage:
  1) (Optional) Create a fresh virtualenv to avoid impacting the main app.
  2) Install MCP-enabled Gradio:  pip install -r requirements-mcp.txt
  3) Run:  python scripts/gradio_mcp_demo.py
  4) MCP URL will be printed, e.g. http://127.0.0.1:7860/gradio_api/mcp/sse

You can then point your MCP client (VS Code Copilot Chat, Cursor, etc.) to that URL.
"""
from __future__ import annotations

import gradio as gr


def check_weather(city: str) -> str:
    """
    Simple weather checker (mock data for demonstration).
    Returns a fixed lookup for a handful of cities so the tool
    is self-contained and reliable.
    """
    weather_data = {
        "london": "Cloudy, 15°C",
        "paris": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C",
        "sydney": "Sunny, 25°C",
    }
    city_key = (city or "").strip().lower()
    if not city_key:
        return "Please provide a city name."
    if city_key in weather_data:
        return f"Weather in {city.title()}: {weather_data[city_key]}"
    return (
        f"Weather data not available for {city}. Try: London, Paris, Tokyo, New York, or Sydney."
    )


demo = gr.Interface(
    fn=check_weather,
    inputs=[gr.Textbox(value="London", label="Enter city name")],
    outputs=[gr.Textbox(label="Weather Info")],
    title="Simple Weather Checker (MCP demo)",
    description="Enter a city name to check the weather (mock data).",
)


if __name__ == "__main__":
    # Enable the MCP server via SSE alongside the regular Gradio app
    demo.launch(mcp_server=True)
