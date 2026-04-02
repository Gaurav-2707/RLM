import time
import random
import logging
import re
from typing import Optional, Dict, Any, List, Union
from google import genai
from pydantic import BaseModel, Field

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeminiInterface")

class GeminiConfig(BaseModel):
    """Configuration for Gemini API calls."""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 2048
    candidate_count: int = 1

class GeminiInterface:
    """
    Robust wrapper for the Gemini API with exponential backoff, 
    circuit breakers, and token tracking.
    """
    def __init__(self, api_key: str, config: Optional[GeminiConfig] = None):
        self.api_key = api_key
        self.config = config or GeminiConfig()
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

        # Circuit Breaker State
        self.failure_count = 0
        self.max_failures = 5
        self.circuit_open = False
        self.last_failure_time = 0
        self.recovery_timeout = 60  # seconds

        # Token metrics
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _check_circuit(self):
        """Checks if the circuit is open and if it should be reset."""
        if self.circuit_open:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker attempting recovery (closing).")
                self.circuit_open = False
                self.failure_count = 0
            else:
                remaining = int(self.recovery_timeout - (time.time() - self.last_failure_time))
                raise RuntimeError(f"Circuit breaker is OPEN. API calls suspended for {remaining}s.")

    def _record_success(self):
        """Resets failure count on successful call."""
        self.failure_count = 0
        if self.circuit_open:
            logger.info("Circuit breaker CLOSED after successful call.")
            self.circuit_open = False

    def _record_failure(self):
        """Increments failure count and opens circuit if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.error(f"Circuit breaker OPEN after {self.failure_count} consecutive failures.")

    def generate_content(
        self, 
        prompt: Union[str, List[Dict[str, Any]]], 
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Sends a request to Gemini with 429 handling and circuit breaker logic.
        """
        self._check_circuit()

        max_retries = 5
        base_delay = 2.0  # seconds
        
        # Handle structured input if provided as a list of dicts (messages style)
        formatted_contents = prompt
        
        for attempt in range(max_retries):
            try:
                # Prepare generation config
                gen_config = genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    max_output_tokens=self.config.max_output_tokens,
                    candidate_count=self.config.candidate_count,
                    response_mime_type="application/json" if "json" in str(prompt).lower() else "text/plain"
                )

                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=formatted_contents,
                    config=gen_config
                )

                # Track tokens
                if response.usage_metadata:
                    self.total_input_tokens += response.usage_metadata.prompt_token_count
                    self.total_output_tokens += response.usage_metadata.candidates_token_count

                self._record_success()
                
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                
                return response.text

            except Exception as e:
                err_str = str(e).lower()
                # 429 Resource Exhausted / Rate Limit
                if "429" in err_str or "resource_exhausted" in err_str or "rate_limit" in err_str:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit (429). Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                
                # Other errors count towards circuit breaker
                self._record_failure()
                logger.error(f"Gemini API error on attempt {attempt+1}: {e}")
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Final failure after {max_retries} attempts: {e}")
                
                # For non-429 errors, we still retry a bit but with less sleep
                time.sleep(1)

        raise RuntimeError("Max retries exceeded for Gemini API.")

    def get_token_summary(self) -> Dict[str, int]:
        """Returns the accumulated token counts."""
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
            "total": self.total_input_tokens + self.total_output_tokens
        }

    def reset_metrics(self):
        """Resets token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
