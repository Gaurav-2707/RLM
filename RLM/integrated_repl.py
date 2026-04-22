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
            from RLM.acc import AdaptiveComputeController, SemanticComplexityScorer
            self._acc_controller = AdaptiveComputeController()
            self._scorer = SemanticComplexityScorer()

        self._code_history: List[str] = []

        if enable_memory:
            from RLM.memory_repl import MemoryREPL
            self._memory_adapter = MemoryREPL(capacity=memory_capacity)
            self._memory_path = memory_path
            self._memory_adapter.load(self._memory_path)

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

        # --- Memory pre-retrieval ---
        memory_content = ""
        if self.enable_memory and self._memory_adapter:
            memory_context, conflicts = self._memory_adapter.retrieve_as_context(query, top_k=3)
            if memory_context:
                memory_content = f"\n\n[REASONING EXAMPLES]\n{memory_context}"
            
            # Conflict Awareness
            if conflicts:
                warning_text = "\n".join(conflicts)
                memory_content += f"\n\n!!! PREVIOUS FAILURES (AVOID) !!!\n{warning_text}"

        # --- Context Injection (Visibility Bridge) ---
        context_data, context_str = utils.convert_context_for_repl(context)
        preview_len = 5000
        preview = (context_str[:preview_len] + "...") if len(context_str) > preview_len else context_str
        
        # Inject memory and preview into the first user message for 8B models
        # This prevents system prompt 'drowning'.
        self.messages.append({
            "role": "user", 
            "content": f"[Global Context Preview (First {preview_len} chars)]\n{preview}{memory_content}\n"
                       f"\nTask: {query}"
        })

        # Build plugins dict for REPLEnv
        plugins: Dict[str, Any] = {}
        if self.enable_memory and self._memory_adapter:
            plugins["memory_retrieve"] = self._memory_adapter.get_repl_function()
        if self.enable_engine and self._engine_adapter:
            plugins["deep_reason"] = self._engine_adapter.get_repl_function()

        # Init REPL env
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            plugins=plugins if plugins else None,
        )

        # --- ACC episode start ---
        if self.enable_acc:
            self._acc_controller.new_episode()

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

        # Determine max_iterations from ACC before the loop starts
        if self.enable_acc:
            score = self._scorer.score(query or "", context=str(context)[:5000])
            depth = self._acc_controller.select_depth(score)
            self.last_depth = depth
            if depth == 0:
                return "Error: ACC budget exhausted before query could run."
            self._max_iterations = self._DEPTH_TO_ITERS.get(depth, self._max_iterations)

        self._responses_history = []
        
        # Standard REPL loop
        for iteration in range(self._max_iterations):
            response = self.llm.completion(
                self.messages + [next_action_prompt(query, iteration)]
            )
            self._responses_history.append(response)
            
            # Research Grade: Semantic Drift Detection
            # If responses are wildly different, it signals confusion -> boost depth
            if self.enable_acc and len(self._responses_history) > 2:
                recent = self._responses_history[-3:]
                drift = self._calculate_drift(recent)
                if drift > 0.7: # High drift threshold
                    # Force ACC to depth 3 if it was lower
                    if self.last_depth < 3:
                        self.logger.info(f"HIGH DRIFT DETECTED ({drift:.2f}). Boosting ACC depth to 3.")
                        self.last_depth = 3
                        self._max_iterations = self._DEPTH_TO_ITERS.get(3, self._max_iterations)
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)

            # --- Trace capture (Initial response) ---
            # We add a placeholder entry that we'll update if code is executed
            trace_entry_idx = len(self.tracer.repl_history)
            self.tracer.add_repl_step(
                iteration=iteration + 1,
                response=response,
                code=None,
                stdout=None,
                stderr=None,
                engine_history=None
            )

            if code_blocks:
                # Loop Detection: If we've seen this exact code block before twice, force FINAL
                merged_code = "\n".join(code_blocks)
                repeat_count = self._code_history.count(merged_code)
                if repeat_count >= 2:
                    # Force extraction — don't waste more iterations
                    self.logger.log_tool_execution("LOOP_DETECT", "Detected 2+ repeated blocks, forcing extraction")
                    salvaged = self._force_final_extraction(query or "")
                    self._post_completion(query or "", salvaged)
                    return salvaged
                elif repeat_count == 1:
                    loop_warning = (
                        "WARNING: You already ran this exact code. It will produce the same output. "
                        "Do NOT repeat it again. Instead: either use different search terms, "
                        "call llm_query() to synthesize from what you already know, "
                        "or write FINAL(your answer) RIGHT NOW as plain text."
                    )
                    self.messages.append({"role": "user", "content": loop_warning})

                self._code_history.append(merged_code)

                # Capture current execution count to track new executions in this iteration
                prev_exec_count = self.repl_env_logger.execution_count

                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env,
                    self.repl_env_logger, self.logger,
                )

                # Capture traceability for each execution triggered by the model
                # Note: If multiple code blocks exist, we update the trace. 
                # For simplicity in Phase 0, we'll just track the last execution's info in the entry
                # or create multiple entries if needed. Let's do multiple entries for precision.
                
                # Remove the placeholder if we are adding specific execution steps
                self.tracer.repl_history.pop(trace_entry_idx)

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
                        engine_history=engine_history
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

        # --- Salvage Step: Handle case where max_iterations reached without FINAL() ---
        salvaged_answer = self._force_final_extraction(query or "")
        self.logger.log_final_response(salvaged_answer)
        self._post_completion(query or "", salvaged_answer)
        return salvaged_answer

    def _force_final_extraction(self, query: str) -> str:
        """
        One last attempt to extract a clean, concise answer from the reasoning history.
        Strictly enforces a short output to prevent F1 degradation.
        """
        import re as _re
        extract_prompt = (
            "Based on ALL research and code outputs above, give the FINAL ANSWER to: "
            f'"{query}"\n\n'
            "Rules:\n"
            "- Output ONLY the answer (a name, date, yes/no, or very short phrase).\n"
            "- Do NOT write code, sentences, or explanations.\n"
            "- Do NOT use FINAL() here — just write the raw answer.\n"
            "- Maximum 10 words.\n\nAnswer:"
        )
        extraction_messages = self.messages + [{"role": "user", "content": extract_prompt}]
        raw = self.llm.completion(extraction_messages).strip()

        # Strip any code fence leakage
        raw = raw.replace("```repl", "").replace("```python", "").replace("```", "").strip()

        # Strip FINAL() wrappers if the model still uses them
        m = _re.search(r'FINAL(?:_VAR)?\s*\(([^)]+)\)', raw)
        if m:
            raw = m.group(1).strip("'\" ")
            
        # Strip FINAL ANSWER: wrappers 
        m = _re.search(r'(?i)FINAL\s*ANSWER\s*:\s*(.*)', raw)
        if m:
            raw = m.group(1).strip("'\" ")

        # Trim to first meaningful sentence if still too verbose
        if len(raw.split()) > 15:
            # Take first sentence or clause
            first_sentence = _re.split(r'[.!?\n]', raw)[0].strip()
            if first_sentence:
                raw = first_sentence

        from string import punctuation
        raw = raw.strip(punctuation + " \n\t")

        # Logical reasoning labels often come in brackets like [entailment] or (A)
        # Try to extract content inside common brackets if raw is short
        if 1 <= len(raw) <= 20:
            m_bracket = _re.search(r'[\[\(\{\<]([a-zA-Z0-9_-]+)[\]\)\}\>]', raw)
            if m_bracket:
                raw = m_bracket.group(1)

        return raw or "unknown"

    def _calculate_drift(self, responses: List[str]) -> float:
        """
        Calculates semantic drift between the last 3 turns.
        1.0 = completely different topics, 0.0 = convergent logic.
        """
        if not hasattr(self, "_scorer") or len(responses) < 2:
            return 0.0
        
        import numpy as np
        embeddings = [self._scorer._get_embedding(r[:1000]) for r in responses]
        
        # Calculate pairwise cosine distances
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(embeddings)
        
        # Avg distance (1 - similarity) between consecutive steps
        drift = 1.0 - np.mean([sims[i, i+1] for i in range(len(responses)-1)])
        return float(drift)

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
