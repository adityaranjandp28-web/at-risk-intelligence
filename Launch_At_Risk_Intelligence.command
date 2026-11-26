#!/bin/bash
# -------------------------------------------------------------------
# One-Click Desktop Launcher for At-Risk Intelligence
# Double-click this file from Finder to launch the app in your browser!
# -------------------------------------------------------------------

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "====================================================="
echo "   Starting AT-RISK INTELLIGENCE Dashboard...        "
echo "====================================================="

# Check virtual environment
if [ -f "$DIR/.venv/bin/streamlit" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
    STREAMLIT_BIN="$DIR/.venv/bin/streamlit"
elif [ -f "$DIR/.venv-1/bin/streamlit" ]; then
    PYTHON_BIN="$DIR/.venv-1/bin/python"
    STREAMLIT_BIN="$DIR/.venv-1/bin/streamlit"
else
    PYTHON_BIN="python3"
    STREAMLIT_BIN="streamlit"
fi

"$STREAMLIT_BIN" run "$DIR/streamlit_app.py"
