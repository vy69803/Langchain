import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(
    model: str = "thinkingmachines/inkling:free",
    temperature: float = 0.7,
    **kwargs,
) -> ChatOpenAI:
    """Initialize and return the Thinking Machines model via OpenRouter.

    Args:
        model: OpenRouter model identifier (default: "thinkingmachines/inkling:free").
        temperature: Sampling temperature for generation.
        **kwargs: Additional arguments forwarded to ChatOpenAI.

    Returns:
        Configured ChatOpenAI instance connected to OpenRouter.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is missing or contains the default placeholder. "
            "Please update your .env file with a valid key from https://openrouter.ai/settings/keys"
        )

    headers = {
        "User-Agent": "Cline/3.0.0",
        "HTTP-Referer": "https://github.com/cline/cline",
        "X-Title": "Cline",
    }
    if "default_headers" in kwargs:
        headers.update(kwargs.pop("default_headers"))

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers=headers,
        **kwargs,
    )
