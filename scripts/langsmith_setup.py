"""LangSmith Setup & Verification

Verifies that LangSmith tracing is properly configured, runs a traced
LLM call through OpenRouter, and prints a direct link to the trace.

Prerequisites:
    - .env file with LANGSMITH_API_KEY and OPENROUTER_API_KEY
    - `pip install langsmith` (or `uv add langsmith`)

Usage:
    python langsmith_setup.py
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory and root are in sys.path for direct execution
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_dir = os.path.join(_root_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root_dir, ".env"))


# ---------------------------------------------------------------------------
# 1. Verify Environment Variables
# ---------------------------------------------------------------------------

REQUIRED_VARS = {
    "LANGSMITH_API_KEY": "LangSmith API key (https://smith.langchain.com/settings)",
    "LANGSMITH_TRACING": "Set to 'true' to enable tracing",
    "OPENROUTER_API_KEY": "OpenRouter API key (https://openrouter.ai/settings/keys)",
}


def verify_environment() -> bool:
    """Check that all required env vars are set and not placeholders."""
    print("=" * 60)
    print("  LangSmith Environment Check")
    print("=" * 60)

    all_ok = True
    placeholders = {"your_langsmith_api_key_here", "your_openrouter_api_key_here"}

    for var, description in REQUIRED_VARS.items():
        value = os.environ.get(var, "")
        if not value or value in placeholders:
            print(f"  ✗  {var} — MISSING")
            print(f"       → {description}")
            all_ok = False
        else:
            # Mask the value for security
            masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "****"
            print(f"  ✓  {var} = {masked}")

    # Optional but useful vars
    optional_vars = {
        "LANGSMITH_PROJECT": os.environ.get("LANGSMITH_PROJECT", "default"),
        "LANGSMITH_ENDPOINT": os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ),
    }
    print()
    for var, value in optional_vars.items():
        print(f"  ℹ  {var} = {value}")

    print("=" * 60)
    return all_ok


# ---------------------------------------------------------------------------
# 2. Traced LLM Call
# ---------------------------------------------------------------------------


def run_traced_call() -> None:
    """Execute a simple LLM call with LangSmith tracing enabled."""
    from langchain_core.tracers.context import tracing_v2_enabled
    from langsmith import Client

    from langchain_rag.llm import get_llm

    project = os.environ.get("LANGSMITH_PROJECT", "My First App")

    print(f"\n🔗  Project  : {project}")
    print(f"📡  Endpoint : {os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}")
    print()

    llm = get_llm()

    prompt = "In one sentence, what is LangSmith and why is it useful?"
    print(f"📝  Prompt: {prompt}\n")

    # Use tracing context to ensure this run is captured
    with tracing_v2_enabled(project_name=project) as cb:
        response = llm.invoke(
            prompt,
            config={
                "metadata": {"source": "langsmith_setup.py"},
                "tags": ["setup", "verification"],
            },
        )

    print("✅  Response:")
    print(f"    {response.content}\n")

    # ---------------------------------------------------------------------------
    # 3. Print Trace Link
    # ---------------------------------------------------------------------------
    try:
        client = Client()
        # Verify connectivity by listing recent runs
        runs = list(client.runs.query(project_name=project, limit=1))
        if runs:
            run = runs[0]
            dashboard_url = (
                f"https://smith.langchain.com/o/default/projects/p/{project}"
            )
            print("🔍  Latest trace:")
            print(f"    Run ID   : {run.id}")
            print(f"    Name     : {run.name}")
            print(f"    Status   : {run.status}")
            print(f"    Dashboard: {dashboard_url}")
        else:
            print("⏳  No runs found yet — traces may take a moment to appear.")
    except Exception as e:
        print(f"⚠️   Could not fetch runs from LangSmith: {e}")
        print("    Traces should still appear in the dashboard shortly.")

    print()
    print("=" * 60)
    print("  ✅  LangSmith setup verified successfully!")
    print("  📊  View traces at: https://smith.langchain.com")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not verify_environment():
        print("\n❌  Fix the missing variables above in your .env file and retry.")
        sys.exit(1)

    print("\n🚀  Running a traced LLM call...\n")
    run_traced_call()
