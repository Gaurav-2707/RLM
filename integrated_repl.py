"""
IntegratedRLM — RLM_REPL wired with ACC + Memory + Engine.

Drop-in replacement for RLM_REPL that activates ACC, Memory, and Engine
via simple boolean flags. When all flags are False, behaviour is identical
to plain RLM_REPL.

Usage:
    from RLM.integrated_repl import IntegratedRLM

    rlm = IntegratedRLM(
        model="ollama/llama3.1:8b",
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
from RLM.utils.tracing import TraceStorage
from RLM.utils.llm import DEFAULT_MODEL


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

    _DEPTH_TO_ITERS = {1: 8, 2: 12, 3: 20}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        recursive_model: str = None,
        max_iterations: int = 50,
        enable_logging: bool = False,
        enable_acc: bool = False,
        enable_memory: bool = False,
        enable_engine: bool = False,
        memory_capacity: int = 200,
        memory_path: str = "rlm_memory.json",
        force_iterations: bool = False,
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
        self.force_iterations = force_iterations
        self._current_iteration = 0

        self.tracer = TraceStorage()

        # Lazy-init adapters
        self._acc_adapter = None
        self._memory_adapter = None
        self._engine_adapter = None

        # ACC episode report from last completion()
        self.last_acc_report = None
        # Depth chosen for last completion()
        self.last_depth = None

        if enable_acc:
            from RLM.acc.controller import AdaptiveComputeController
            from RLM.acc import ComplexityScorer
            self._acc_controller = AdaptiveComputeController()
            self._complexity_scorer = ComplexityScorer()
            self._max_iterations = 5  # Base iteration limit for True Adaptive Compute

        self._code_history: List[str] = []

        if enable_memory:
            from RLM.memory_repl import MemoryREPL
            self._memory_adapter = MemoryREPL(capacity=memory_capacity)
            self._memory_path = memory_path
            self._memory_adapter.load(self._memory_path)

        if enable_engine:
            from RLM.engine_repl import EngineREPL
            self._engine_adapter = EngineREPL(model=recursive_model)

        # Standardized context cap for experiments
        self.preview_len = 2000

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

        # --- Memory pre-retrieval ---
        memory_content = ""
        if self.enable_memory and self._memory_adapter:
            memory_context, conflicts = self._memory_adapter.retrieve_as_context(query, top_k=1)
            if memory_context:
                demonstration = f"\n\n[Demonstration: Successful Reasoning Trace]\n{memory_context}"
                self.messages[0]["content"] += demonstration
            
            # Conflict Awareness
            if conflicts:
                warning_text = "\n".join(conflicts)
                memory_content += f"\n\n!!! PREVIOUS FAILURES (AVOID) !!!\n{warning_text}"

        # --- Context Injection (Visibility Bridge) ---
        context_data, context_str = utils.convert_context_for_repl(context)
        preview = (context_str[:self.preview_len] + "...") if len(context_str) > self.preview_len else context_str
        
        # Inject memory and preview into the first user message for 8B models
        # This prevents system prompt 'drowning'.
        self.messages.append({
            "role": "user", 
            "content": f"[Global Context Preview (First {self.preview_len} chars)]\n{preview}{memory_content}\n"
                       f"\nTask: {query}"
        })

        # Build plugins dict for REPLEnv
        plugins: Dict[str, Any] = {}
        if self.enable_memory and self._memory_adapter:
            plugins["memory_retrieve"] = self._memory_adapter.get_repl_function()
        if self.enable_engine and self._engine_adapter:
            plugins["deep_reason"] = self._engine_adapter.get_repl_function()
            
        if self.enable_acc:
            def extend_budget(reason: str):
                self._max_iterations += 5
                return f"Budget extended. Max iterations is now {self._max_iterations}. Reason: {reason}"
            plugins["extend_budget"] = extend_budget

        self._final_answer_submitted = None
        self._snapshot_answers = {}
        self._snapshot_confidences = {}
        
        def submit_final_answer(answer: str, confidence: float = 1.0):
            words = str(answer).split()
            if len(words) > 6:
                return "ERROR: Final answer must be an exact entity, number, or short phrase. Retry."
            
            current_iter = getattr(self, '_current_iteration', 0)
            
            gate_result = {"exit": True}
            if self.enable_acc and not self.force_iterations:
                gate_result = self._acc_controller.should_exit(
                    current_answer=str(answer),
                    retrieved_context=context_str[:5000],
                    llm_client=self.llm,
                    confidence=float(confidence),
                    iteration=current_iter
                )
            
            if not gate_result["exit"] and not self.force_iterations:
                return "ERROR: Answer not supported by context. You must search for more information."
            
            self._snapshot_answers[current_iter] = str(answer)
            self._snapshot_confidences[current_iter] = float(confidence)
            
            if self.force_iterations:
                return f"Snapshot answer '{answer}' recorded with confidence {confidence}. FORCE_ITERATIONS is active. You must continue verifying or exploring alternative approaches."
            else:
                if gate_result.get("rollback"):
                    peak_iter = gate_result.get("peak_iteration")
                    rolled_back_answer = self._snapshot_answers.get(peak_iter, answer)
                    self._final_answer_submitted = str(rolled_back_answer)
                    return f"ACC ROLLBACK: Confidence dropped. Rolling back to peak answer '{rolled_back_answer}' from iteration {peak_iter}. Exit triggered."
                else:
                    self._final_answer_submitted = str(answer)
                    return f"Final answer '{answer}' accepted with confidence {confidence}."

        plugins["submit_final_answer"] = submit_final_answer

        # Init REPL env
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            plugins=plugins if plugins else None,
        )

        # --- ACC episode start ---
        if self.enable_acc:
            score = self._complexity_scorer.score(query, context=context_str)
            self._acc_controller.new_episode(complexity_score=score)

        # --- REPL Code History reset ---
        self._code_history = []

        # --- Tracer reset ---
        self.tracer.reset()
        self.tracer.set_metadata({
            "model": self.model,
            "recursive_model": self.recursive_model,
            "enable_acc": self.enable_acc,
            "enable_memory": self.enable_memory,
            "enable_engine": self.enable_engine
        })
        self.tracer.set_query(query)

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
        
        # Add system prompt enforcement for search requirement and structured output
        enforcement_msg = (
            "SYSTEM MANDATE: You MUST NOT surrender prematurely. If you lack information, you must actively "
            "search using the provided tools. Concluding without any tool usage is strictly forbidden.\n"
            "If you realize the problem is complex, you may call `extend_budget(reason)` to request more compute.\n"
            "To end the episode (or submit an intermediate answer), you MUST use the `submit_final_answer(answer: str, confidence: float)` tool. "
            "Confidence should be between 0.0 and 1.0."
        )
        self.messages.insert(0, {"role": "system", "content": enforcement_msg})

        self._responses_history = []
        
        # Standard REPL loop
        for iteration in range(self._max_iterations):
            self._current_iteration = iteration + 1
            response = self.llm.completion(
                self.messages + [next_action_prompt(query, iteration)]
            )
            self._responses_history.append(response)
            
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)

            context_length = sum(len(str(m.get("content", ""))) for m in self.messages)
            
            self.tracer.add_repl_step(
                iteration=iteration + 1,
                response=response,
                code=None,
                stdout=None,
                stderr=None,
                engine_history=None,
                context_length=context_length,
                snapshot_answer=self._snapshot_answers.get(iteration + 1),
                confidence=self._snapshot_confidences.get(iteration + 1)
            )

            if code_blocks:
                # Loop Detection: If we've seen this exact code block before
                merged_code = "\n".join(code_blocks)
                if merged_code in self._code_history:
                    self.logger.log_tool_execution("LOOP_DETECT", "Detected loop, interrupting")
                    loop_warning = "ERROR: You already tried this action. You are looping. Try a completely different approach or submit your best guess now."
                    self.messages.append({"role": "user", "content": loop_warning})
                    continue

                self._code_history.append(merged_code)

                # Capture current execution count to track new executions in this iteration
                prev_exec_count = self.repl_env_logger.execution_count

                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env,
                    self.repl_env_logger, self.logger,
                )

                if getattr(self, '_final_answer_submitted', None) is not None and not self.force_iterations:
                    final_answer = self._final_answer_submitted
                    self.logger.log_final_response(final_answer)
                    self._post_completion(query or "", final_answer)
                    return final_answer

                # Capture traceability for each execution triggered by the model
                # Note: If multiple code blocks exist, we update the trace. 
                # For simplicity in Phase 0, we'll just track the last execution's info in the entry
                # or create multiple entries if needed. Let's do multiple entries for precision.
                
                # Remove the placeholder if we are adding specific execution steps
                self.tracer.repl_history.pop()

                for i in range(prev_exec_count, self.repl_env_logger.execution_count):
                    exec_info = self.repl_env_logger.executions[i]
                    
                    engine_history = None
                    if self.enable_engine and self._engine_adapter:
                        engine_history = self._engine_adapter.get_steps()

                    self.tracer.add_repl_step(
                        iteration=iteration + 1,
                        response=response,
                        code=exec_info.code,
                        stdout=exec_info.stdout,
                        stderr=exec_info.stderr,
                        engine_history=engine_history,
                        context_length=context_length,
                        snapshot_answer=self._snapshot_answers.get(iteration + 1),
                        confidence=self._snapshot_confidences.get(iteration + 1)
                    )
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": "You responded with:\n" + response,
                })

        # --- Salvage Step: Handle case where max_iterations reached without submit_final_answer() ---
        if self.force_iterations and self._snapshot_answers:
            last_iter = max(self._snapshot_answers.keys())
            salvaged_answer = self._snapshot_answers[last_iter]
        elif getattr(self, '_final_answer_submitted', None) is not None:
            salvaged_answer = self._final_answer_submitted
        else:
            salvaged_answer = "unknown"
            
        self.logger.log_final_response(salvaged_answer)
        self._post_completion(query or "", salvaged_answer)
        return salvaged_answer

    # ------------------------------------------------------------------
    # Post-completion hooks (memory store, ACC report)
    # ------------------------------------------------------------------

    def _post_completion(self, query: str, answer: str):
        """Called after every successful completion to store memory and close ACC episode."""
        if self.enable_acc:
            self.last_acc_report = self._acc_controller.end_episode()

        if self.enable_memory and self._memory_adapter:
            # Store this QA pair with a neutral-positive outcome initially
            reasoning_summary = f"Ran REPL loop, depth={getattr(self, 'last_depth', 'N/A')}"
            self._memory_adapter.store(
                query=query,
                reasoning=reasoning_summary,
                action="repl_completion",
                outcome=f"Answered: {answer[:200]}",
                outcome_score=0.6,
            )
            # Persist immediately
            self._memory_adapter.save(self._memory_path)

        # Finalise tracer
        self.tracer.set_predicted_answer(answer)
        if self.enable_acc:
            self.tracer.set_acc_data({
                "depth_selected": self.last_depth,
                "records": [str(r) for r in self._acc_controller.records]
            })

    def update_last_memory(self, judge_score: int):
        """
        Feedback loop: Update the score of the most recent memory entry.
        Maps 1-5 judge score to -1.0 to 1.0 range.
        """
        if not (self.enable_memory and self._memory_adapter):
            return
            
        # Map 1-5 -> -1.0 to 1.0 (approximately)
        # 1 -> -1.0, 2 -> -0.5, 3 -> 0.0, 4 -> 0.5, 5 -> 1.0
        normalized_score = (judge_score - 3) / 2.0
        
        if self._memory_adapter.system.memories:
            self._memory_adapter.system.memories[-1].outcome_score = normalized_score
            self._memory_adapter.save(self._memory_path)
            self.logger.info(f"Memory Updated: Last entry score set to {normalized_score} (Judge: {judge_score})")
