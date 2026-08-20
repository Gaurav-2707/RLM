"""
Unified LLM Client for the Recursive Labs RLM Architecture.

Supports all major backends, selected automatically based on available .env keys:
  - OpenAI        (GPT-4o):             OPENAI_API_KEY
  - Anthropic     (Claude):             ANTHROPIC_API_KEY
  - Gemini        (gemini-2.5-pro):     GENAI_API_KEY
  - Mistral       (mistral-large):      MISTRAL_API_KEY
  - Cohere        (command-r-plus):     COHERE_API_KEY
  - Azure OpenAI  (any deployment):     AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
  - Together AI   (fine-tune hosting):  TOGETHER_API_KEY
  - HuggingFace   (local transformers): HF_MODEL_PATH (no key required)
  - Ollama        (local Llama):        OLLAMA_MODEL (no key required, last fallback)

The wrapper never cares which LLM is underneath.
That is the point.
"""

import os
import re
import json
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

DEFAULT_PROVIDER = _DEFAULT_PROVIDER = (
    "openai"     if os.getenv("OPENAI_API_KEY")
    else "anthropic" if os.getenv("ANTHROPIC_API_KEY")
    else "gemini"    if os.getenv("GENAI_API_KEY")
    else "mistral"   if os.getenv("MISTRAL_API_KEY")
    else "cohere"    if os.getenv("COHERE_API_KEY")
    else "azure"     if os.getenv("AZURE_OPENAI_API_KEY")
    else "together"  if os.getenv("TOGETHER_API_KEY")
    else "huggingface" if os.getenv("HF_MODEL_PATH")
    else "ollama"
)


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if provider == "mistral":
        return os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    if provider == "cohere":
        return os.getenv("COHERE_MODEL", "command-r-plus")
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    if provider == "together":
        return os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    if provider == "huggingface":
        return os.getenv("HF_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
    raw = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
    return raw.replace("ollama/", "") if raw.startswith("ollama/") else raw


DEFAULT_MODEL = _default_model_for_provider(DEFAULT_PROVIDER)


def _detect_provider() -> str:
    """Auto-detect the best available LLM provider from environment keys.

    Priority: OpenAI > Gemini > Ollama (local).

    Returns:
        str: One of "openai", "gemini", or "ollama".
    """
    return DEFAULT_PROVIDER


def _detect_model(provider: str) -> str:
    """Return the default model name for the given provider.

    Args:
        provider: One of "openai", "gemini", or "ollama".

    Returns:
        str: Model identifier string.
    """
    return _default_model_for_provider(provider)


def extract_json_from_text(text: str) -> str:
    """Aggressively extract a JSON object from raw LLM output.

    Handles: markdown backticks, leading prose, trailing newlines,
    nested arrays wrapping a single dict, and non-string values.

    Args:
        text: Raw string output from an LLM completion.

    Returns:
        str: A clean JSON string ready for json.loads().

    Raises:
        ValueError: If no JSON object could be extracted.
    """
    if not isinstance(text, str):
        text = json.dumps(text)

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()

    # Try to extract the first {...} block with a regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


class LLMClient:
    """Provider-agnostic LLM client for the RLM Runtime.

    The RLM Runtime never cares which LLM is underneath.
    Automatically selects the best available provider from environment keys.

    Priority: OpenAI > Anthropic > Gemini > Mistral > Cohere > Azure >
              Together AI > HuggingFace (local) > Ollama (local fallback)

    Example:
        client = LLMClient()                          # auto-detect
        client = LLMClient(provider="anthropic")      # force Claude
        client = LLMClient(model="llama3:8b")         # force local model
        result = client.completion("Fix this bug...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.provider = (provider or _detect_provider()).lower()
        self.model = model or _detect_model(self.provider)

        # ── OpenAI ────────────────────────────────────────────────────────
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set in .env")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Run: uv pip install openai") from exc
            self.client = OpenAI(api_key=self.api_key)

        # ── Anthropic (Claude) ────────────────────────────────────────────
        elif self.provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in .env")
            try:
                import anthropic as _anthropic
            except ImportError as exc:
                raise ImportError("Run: uv pip install anthropic") from exc
            self.client = _anthropic.Anthropic(api_key=self.api_key)

        # ── Gemini ────────────────────────────────────────────────────────
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GENAI_API_KEY")
            if not self.api_key:
                raise ValueError("GENAI_API_KEY not set in .env")
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError("Run: uv pip install google-genai") from exc
            self._genai = genai
            self.client = genai.Client(api_key=self.api_key)

        # ── Mistral ───────────────────────────────────────────────────────
        elif self.provider == "mistral":
            self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY not set in .env")
            try:
                from mistralai import Mistral
            except ImportError as exc:
                raise ImportError("Run: uv pip install mistralai") from exc
            self.client = Mistral(api_key=self.api_key)

        # ── Cohere ────────────────────────────────────────────────────────
        elif self.provider == "cohere":
            self.api_key = api_key or os.getenv("COHERE_API_KEY")
            if not self.api_key:
                raise ValueError("COHERE_API_KEY not set in .env")
            try:
                import cohere
            except ImportError as exc:
                raise ImportError("Run: uv pip install cohere") from exc
            self.client = cohere.ClientV2(api_key=self.api_key)

        # ── Azure OpenAI ──────────────────────────────────────────────────
        elif self.provider == "azure":
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            if not self.api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env")
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise ImportError("Run: uv pip install openai") from exc
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )

        # ── Together AI (cheap inference + fine-tune hosting) ─────────────
        elif self.provider == "together":
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("TOGETHER_API_KEY not set in .env")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Run: uv pip install openai") from exc
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1",
            )

        # ── HuggingFace local (transformers pipeline) ─────────────────────
        elif self.provider == "huggingface":
            self.api_key = None
            model_path = model or os.getenv("HF_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
            self.model = model_path
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError("Run: uv pip install transformers accelerate") from exc
            self.client = pipeline(
                "text-generation",
                model=model_path,
                device_map="auto",
                trust_remote_code=True,
            )

        # ── Ollama (local, last fallback) ─────────────────────────────────
        elif self.provider == "ollama":
            self.api_key = "ollama-dummy"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Run: uv pip install openai") from exc
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                f"Choose from: openai, anthropic, gemini, mistral, cohere, "
                f"azure, together, huggingface, ollama"
            )

    def completion(
        self,
        messages: "list[dict] | str",
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Send a completion request and return the model's text response.

        Enforces JSON-object output mode where supported. For Gemini,
        adds a system instruction to guarantee JSON output.

        Args:
            messages: Either a string (converted to user message) or a list
                of {"role": ..., "content": ...} dicts.
            max_tokens: Optional cap on output tokens.
            response_format: e.g. {"type": "json_object"} (OpenAI/Ollama).
            **kwargs: Extra keyword args forwarded to the underlying API.

        Returns:
            str: The raw text content of the model's response.

        Raises:
            RuntimeError: On any API or network error.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        want_json = (response_format or {}).get("type") == "json_object"

        try:
            # ── OpenAI ────────────────────────────────────────────────────
            if self.provider == "openai":
                call_kwargs = dict(kwargs)
                if want_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **call_kwargs,
                )
                return response.choices[0].message.content

            # ── Gemini ────────────────────────────────────────────────────
            elif self.provider == "gemini":
                genai = self._genai

                # Prepend a JSON-only system instruction when needed
                if want_json:
                    json_instruction = {
                        "role": "user",
                        "content": (
                            "IMPORTANT SYSTEM INSTRUCTION: You MUST respond with "
                            "ONLY a raw JSON object. No markdown, no backticks, no prose."
                        ),
                    }
                    messages = [json_instruction] + list(messages)

                history = []
                for msg in messages[:-1]:
                    role = "model" if msg.get("role") == "assistant" else "user"
                    content = msg.get("content", "")
                    if content:
                        history.append(
                            genai.types.Content(
                                role=role,
                                parts=[genai.types.Part(text=content)],
                            )
                        )

                config = None
                if max_tokens:
                    config = genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens
                    )

                chat = self.client.chats.create(
                    model=self.model,
                    config=config,
                    history=history,
                )

                last_msg = messages[-1].get("content", "") if messages else ""
                response = chat.send_message(last_msg)

                if hasattr(response, "text") and response.text is not None:
                    return response.text
                elif hasattr(response, "parts") and response.parts:
                    text = "".join(
                        p.text for p in response.parts
                        if hasattr(p, "text") and p.text is not None
                    )
                    if text:
                        return text
                # Fallback: empty string instead of None (prevents downstream TypeError)
                return str(response) if response is not None else ""

            # ── Anthropic (Claude) ────────────────────────────────────────
            elif self.provider == "anthropic":
                # Separate system messages from conversation messages
                system_content = ""
                conv_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system_content += msg.get("content", "") + "\n"
                    else:
                        conv_messages.append(msg)

                if want_json and system_content:
                    system_content += "\nYou MUST respond with ONLY a raw JSON object. No markdown, no prose."
                elif want_json:
                    system_content = "You MUST respond with ONLY a raw JSON object. No markdown, no prose."

                call_kwargs: dict = {}
                if system_content.strip():
                    call_kwargs["system"] = system_content.strip()
                if max_tokens:
                    call_kwargs["max_tokens"] = max_tokens
                else:
                    call_kwargs["max_tokens"] = 4096  # Anthropic requires max_tokens

                response = self.client.messages.create(
                    model=self.model,
                    messages=conv_messages,
                    **call_kwargs,
                )
                return response.content[0].text

            # ── Mistral ───────────────────────────────────────────────────
            elif self.provider == "mistral":
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                )
                return response.choices[0].message.content

            # ── Cohere ────────────────────────────────────────────────────
            elif self.provider == "cohere":
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                )
                return response.message.content[0].text

            # ── Azure OpenAI (OpenAI-compatible) ──────────────────────────
            elif self.provider == "azure":
                call_kwargs = dict(kwargs)
                if want_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                return response.choices[0].message.content

            # ── Together AI (OpenAI-compatible) ───────────────────────────
            elif self.provider == "together":
                call_kwargs = dict(kwargs)
                if want_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                return response.choices[0].message.content

            # ── HuggingFace local ─────────────────────────────────────────
            elif self.provider == "huggingface":
                # Flatten messages to a single prompt string
                prompt = "\n".join(
                    f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                    for m in messages
                )
                prompt += "\nASSISTANT:"
                outputs = self.client(
                    prompt,
                    max_new_tokens=max_tokens or 512,
                    do_sample=False,
                )
                generated = outputs[0]["generated_text"]
                # Strip the prompt prefix
                if generated.startswith(prompt):
                    generated = generated[len(prompt):]
                return generated.strip()

            # ── Ollama (OpenAI-compatible endpoint) ───────────────────────
            elif self.provider == "ollama":
                call_kwargs = dict(kwargs)
                if want_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **call_kwargs,
                )
                return response.choices[0].message.content

        except Exception as exc:
            raise RuntimeError(
                f"[LLMClient/{self.provider}] completion failed: {exc}"
            ) from exc
