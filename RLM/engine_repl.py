"""
Engine REPL adapter.

Wraps RLMEngine (3-step Decompose→Refine→Synthesise pipeline) and exposes:
  - run(problem)           → final answer string
  - get_repl_function()    → deep_reason() callable for REPLEnv globals injection
  - get_steps()            → full step history from last run
  - reset()                → clear state
"""

from RLM.repl import Sub_RLM
from RLM.engine.rlm_engine import RLMEngine


class EngineREPL:
    """
    Adapter that connects RLMEngine to the REPL environment.

    Parameters
    ----------
    model : str
        LLM model string (e.g. "gemini-2.5-flash").
    api_key : str, optional
        API key override. If None, reads from environment variable.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.sub_rlm = Sub_RLM(model=model)
        self.engine = RLMEngine(self.sub_rlm)
        self._last_run_history: list = []

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def run(self, problem: str) -> str:
        """
        Run the 3-step Decompose→Refine→Synthesise pipeline.

        Parameters
        ----------
        problem : str
            The sub-problem or question to reason over.

        Returns
        -------
        str
            The final synthesised answer only (not the full step dict).
        """
        result = self.engine.run(problem)
        self._last_run_history = result.get("steps", [])
        return result.get("final_output", "")

    def get_repl_function(self) -> callable:
        """
        Return a ``deep_reason(problem)`` callable suitable for injection
        into ``REPLEnv.globals``.

        The model can call ``deep_reason("complex sub-question")`` inside a
        ``\`\`\`repl`` block to trigger the full 3-step reasoning pipeline.
        """
        def deep_reason(problem: str) -> str:
            """Use deep_reason(problem) for complex multi-step sub-problems requiring structured reasoning."""
            return self.run(problem)

        return deep_reason

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_steps(self) -> list:
        """Return the full step-by-step history dict from the last run() call."""
        return list(self._last_run_history)

    def reset(self):
        """Reset engine history and underlying RLM state."""
        self.engine.reset()
        self._last_run_history = []
