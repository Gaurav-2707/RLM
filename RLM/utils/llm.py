import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


class LLMClient:
    """Thin wrapper around different LLM providers.

    The API is intentionally kept minimal so that the rest of the
    repository doesn't need to care which backend is being used.

    Currently supports ``openai`` and ``gemini`` (via the
    ``google.genai`` package).  The provider is inferred from
    ``provider`` argument or the model name, but can be forced explicitly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ollama/llama3",
        provider: Optional[str] = None,
    ):
        self.model = model

        # determine provider
        if provider:
            self.provider = provider.lower()
        else:
            if model.lower().startswith("ollama/"):
                self.provider = "ollama"
            elif model.lower().startswith("gemini"):
                self.provider = "gemini"
            else:
                self.provider = "openai"

        if self.provider == "ollama":
            self.api_key = "ollama-dummy"
            # Strip the ollama/ prefix as the ollama server just expects "llama3"
            if self.model.startswith("ollama/"):
                self.model = self.model[7:]
                
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai package is required for Ollama provider") from exc
            
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            # Re-map provider to openai so the completion logic can be reused natively
            self.provider = "openai"
        else:
            # prefer explicit environment variables; fall back to either name
            self.api_key = api_key or os.getenv("GENAI_API_KEY")
            if not self.api_key:
                raise ValueError("LLM API key is required; set OPENAI_API_KEY or GENAI_API_KEY")

            if self.provider == "openai":
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise ImportError("openai package is required for OpenAI provider") from exc
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "gemini":
                try:
                    from google import genai
                except ImportError as exc:
                    raise ImportError(
                        "google-genai package is required for Gemini provider"
                    ) from exc
                self.client = genai.Client(api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported provider '{self.provider}'")

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        # normalize message form
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **kwargs,
                )
                return response.choices[0].message.content
            else:  # gemini with google.genai API
                # Convert OpenAI message format to genai history format
                # genai only accepts "user" and "model" roles (not "system")
                from google import genai
                
                history = []
                for msg in messages[:-1]:  # all but last message go to history
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # genai only accepts "user" and "model" roles
                    # Convert "system" to "user"
                    if role == "system":
                        role = "user"
                    
                    if content:  # only add non-empty messages
                        history.append(
                            genai.types.Content(
                                role=role,
                                parts=[genai.types.Part(text=content)]
                            )
                        )
                
                # Create chat with history
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
                
                # Send the last message and get response
                last_msg = messages[-1].get("content", "") if messages else ""
                response = chat.send_message(last_msg)
                
                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'parts') and response.parts:
                    # Extract text from parts
                    text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
                    return ''.join(text_parts) if text_parts else str(response)
                else:
                    return str(response)
                    
        except Exception as e:
            raise RuntimeError(f"Error during {self.provider} completion: {str(e)}")
            