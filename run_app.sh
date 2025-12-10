#!/bin/bash
# Run the application using the project's virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Check for venv
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Using virtual environment at .venv"
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "Virtual environment not found at .venv"
    echo "Please create one or ensure dependencies are installed."
    PYTHON="python3"
fi

# Run the app
echo "Starting app..."
"$PYTHON" "$PROJECT_ROOT/app.py"
