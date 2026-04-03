"""
IntegratedRLM — RLM_REPL wired with ACC + Memory + Engine.

Drop-in replacement for RLM_REPL that activates ACC, Memory, and Engine
via simple boolean flags. When all flags are False, behaviour is identical
to plain RLM_REPL.

Usage:
    from RLM.integrated_repl import IntegratedRLM

    rlm = IntegratedRLM(
        model="ollama/llama3",
        enable_acc=True,
        enable_memory=True,
        enable_engine=True,
    )
    answer = rlm.completion(context=..., query=...)
"""

from typing import Dict, List, Optional, Any

from RLM.rlm_repl import RLM_REPL
from RLM.repl import REPLEnv
from RLM.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import RLM.utils.utils as utils


class IntegratedRLM(RLM_REPL):
    """
    Enhanced RLM_REPL that wires ACC, Memory, and Engine into the loop.

    Parameters
    ----------
    model : str
        Root LLM model string.
    recursive_model : str
        Sub-LLM model string (used by REPL's llm_query and Engine).
    enable_acc : bool
        If True, uses AdaptiveComputeController to set max_iterations
        dynamically based on query complexity (depth 1→5, 2→10, 3→20 iters).
    enable_memory : bool
        If True, retrieves relevant past experiences before each run and
        stores the result afterwards. Also injects memory_retrieve() into
        the REPL globals so the model can query memory mid-execution.
    enable_engine : bool
        If True, injects deep_reason(problem) into the REPL globals, giving
        the model access to the 3-step Decompose→Refine→Synthesise pipeline.
    memory_capacity : int
        Maximum episodic memories to retain (only relevant if enable_memory).
    """

    _DEPTH_TO_ITERS = {1: 5, 2: 10, 3: 20}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ollama/llama3",
        recursive_model: str = "ollama/llama3",
        max_iterations: int = 10,
        enable_logging: bool = False,
        enable_acc: bool = False,
        enable_memory: bool = False,
        enable_engine: bool = False,
        memory_capacity: int = 200,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
            enable_logging=enable_logging,
        )

        self.enable_acc = enable_acc
        self.enable_memory = enable_memory
        self.enable_engine = enable_engine

        # Lazy-init adapters
        self._acc_adapter = None
        self._memory_adapter = None
        self._engine_adapter = None

        # ACC episode report from last completion()
        self.last_acc_report = None
        # Depth chosen for last completion()
        self.last_depth = None

        if enable_acc:
            from RLM.acc import AdaptiveComputeController, ComplexityScorer
            self._acc_controller = AdaptiveComputeController()
            self._scorer = ComplexityScorer()

        if enable_memory:
            from RLM.memory_repl import MemoryREPL
            self._memory_adapter = MemoryREPL(capacity=memory_capacity)

        if enable_engine:
            from RLM.engine_repl import EngineREPL
            self._engine_adapter = EngineREPL(model=recursive_model)

    # ------------------------------------------------------------------
    # Override setup_context to wire in plugins
    # ------------------------------------------------------------------

    def setup_context(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ):
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Build system messages
        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)

        # --- Memory pre-retrieval —-
        if self.enable_memory and self._memory_adapter:
            memory_context = self._memory_adapter.retrieve_as_context(query, top_k=3)
            if memory_context:
                self.messages.append({
                    "role": "user",
                    "content": f"The following are previous problems you have solved. Use them ONLY as inspiration for your reasoning methodology or syntax. DO NOT confuse them with your current task!\n\n[System Memory]\n{memory_context}",
                })

        # Build plugins dict for REPLEnv
        plugins: Dict[str, Any] = {}
        if self.enable_memory and self._memory_adapter:
            plugins["memory_retrieve"] = self._memory_adapter.get_repl_function()
        if self.enable_engine and self._engine_adapter:
            plugins["deep_reason"] = self._engine_adapter.get_repl_function()

        # Init REPL env
        context_data, context_str = utils.convert_context_for_repl(context)
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            plugins=plugins if plugins else None,
        )

        # --- ACC episode start ---
        if self.enable_acc:
            self._acc_controller.new_episode()

        return self.messages

    # ------------------------------------------------------------------
    # Override completion to apply ACC depth per iteration
    # ------------------------------------------------------------------

    def completion(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ) -> str:
        self.messages = self.setup_context(context, query)

        # Determine max_iterations from ACC before the loop starts
        if self.enable_acc:
            score = self._scorer.score(query or "", context=str(context)[:5000])
            depth = self._acc_controller.select_depth(score)
            self.last_depth = depth
            if depth == 0:
                return "Error: ACC budget exhausted before query could run."
            self._max_iterations = self._DEPTH_TO_ITERS.get(depth, self._max_iterations)

        # Standard REPL loop
        for iteration in range(self._max_iterations):
            response = self.llm.completion(
                self.messages + [next_action_prompt(query, iteration)]
            )
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)

            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env,
                    self.repl_env_logger, self.logger,
                )
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": "You responded with:\n" + response,
                })

            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )
            if final_answer:
                self.logger.log_final_response(final_answer)
                self._post_completion(query or "", final_answer)
                return final_answer

        # Exhausted iterations — force final answer
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer_raw = self.llm.completion(self.messages)
        
        # Try to parse out FINAL() or FINAL_VAR() from the raw forced response
        parsed_answer = utils.check_for_final_answer(final_answer_raw, self.repl_env, self.logger)
        final_answer = parsed_answer if parsed_answer else final_answer_raw.strip("* ")

        self.logger.log_final_response(final_answer)
        self._post_completion(query or "", final_answer)
        return final_answer

    # ------------------------------------------------------------------
    # Post-completion hooks (memory store, ACC report)
    # ------------------------------------------------------------------

    def _post_completion(self, query: str, answer: str):
        """Called after every successful completion to store memory and close ACC episode."""
        if self.enable_acc:
            self.last_acc_report = self._acc_controller.end_episode()

        if self.enable_memory and self._memory_adapter:
            # Store this QA pair with a neutral-positive outcome
            reasoning_summary = f"Ran REPL loop, depth={getattr(self, 'last_depth', 'N/A')}"
            self._memory_adapter.store(
                query=query,
                reasoning=reasoning_summary,
                action="repl_completion",
                outcome=f"Answered: {answer[:200]}",
                outcome_score=0.6,  # Default neutral-positive; can be updated with EM score later
            )
