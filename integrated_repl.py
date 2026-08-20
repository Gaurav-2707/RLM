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

import os
from typing import Dict, List, Optional, Any

from RLM.rlm_repl import RLM_REPL
from RLM.repl import REPLEnv
from RLM.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import RLM.utils.utils as utils
from RLM.utils.tracing import TraceStorage
from RLM.utils.llm import DEFAULT_MODEL
from RLM.utils.trajectory import TrajectoryCollector


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
        enable_tdrl: bool = False,
        repo_path: Optional[str] = None,
        test_command: Optional[str] = None,
        contribute_traces: bool = False,
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
        self.enable_tdrl = enable_tdrl
        self.repo_path = repo_path
        self.test_command = test_command
        self.contribute_traces = contribute_traces
        self.force_iterations = force_iterations
        self._current_iteration = 0

        self.tracer = TraceStorage()

        # Trajectory collector — always on, contribute flag controls upload
        from RLM.utils.trajectory import TrajectoryCollector
        self._trajectory_collector = TrajectoryCollector(contribute=contribute_traces)

        # TDRL: file snapshots and env
        self._file_snapshots: Dict[str, str] = {}
        self._tdrl_env = None
        if enable_tdrl:
            if not repo_path:
                raise ValueError("enable_tdrl=True requires repo_path to be set")
            if not test_command:
                raise ValueError("enable_tdrl=True requires test_command to be set")

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

        # ── TDRL plugins: edit_file, run_tests, rollback ─────────────────
        if self.enable_tdrl:
            import subprocess
            import glob as _glob

            def _snapshot_repo():
                """Snapshot all Python files in the repo to RAM."""
                self._file_snapshots = {}
                for fpath in _glob.glob(
                    os.path.join(self.repo_path, "**", "*.py"), recursive=True
                ):
                    try:
                        with open(fpath, "r", encoding="utf-8") as _f:
                            self._file_snapshots[fpath] = _f.read()
                    except Exception:
                        pass

            # Take initial snapshot
            _snapshot_repo()

            def edit_file(file_path: str, new_content: str) -> str:
                """
                Write new_content to file_path.
                Automatically snapshots the file first so rollback() can restore it.
                Returns: confirmation string or error.
                """
                abs_path = os.path.join(self.repo_path, file_path) if not os.path.isabs(file_path) else file_path
                # Snapshot before editing
                if abs_path not in self._file_snapshots:
                    try:
                        with open(abs_path, "r", encoding="utf-8") as _f:
                            self._file_snapshots[abs_path] = _f.read()
                    except FileNotFoundError:
                        self._file_snapshots[abs_path] = ""  # new file
                try:
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as _f:
                        _f.write(new_content)
                    return f"✓ Edited {file_path} ({len(new_content)} chars)"
                except Exception as e:
                    return f"✗ Edit failed: {e}"

            def run_tests() -> str:
                """
                Run the test command and return structured output.
                Returns: pass/fail summary + stdout (last 3000 chars).
                """
                try:
                    result = subprocess.run(
                        self.test_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=self.repo_path,
                        timeout=120,
                    )
                    output = (result.stdout + result.stderr)[-3000:]
                    passed = result.returncode == 0
                    status = "✓ TESTS PASSED" if passed else "✗ TESTS FAILED"
                    return f"{status}\n\n{output}"
                except subprocess.TimeoutExpired:
                    return "✗ TEST TIMEOUT (120s)"
                except Exception as e:
                    return f"✗ Test runner error: {e}"

            def rollback(reason: str = "reverting failed edit") -> str:
                """
                Physically restore all files to their pre-edit snapshots.
                Use this when run_tests() fails and you want to try a different approach.
                Returns: list of restored files.
                """
                restored = []
                for fpath, content in self._file_snapshots.items():
                    try:
                        if content == "":
                            if os.path.exists(fpath):
                                os.remove(fpath)
                            restored.append(f"deleted {os.path.basename(fpath)}")
                        else:
                            with open(fpath, "w", encoding="utf-8") as _f:
                                _f.write(content)
                            restored.append(os.path.basename(fpath))
                    except Exception as e:
                        restored.append(f"failed to restore {fpath}: {e}")
                # Re-snapshot clean state
                _snapshot_repo()
                summary = ", ".join(restored) if restored else "nothing to restore"
                return f"↩ Rollback complete. Restored: {summary}. Reason: {reason}"

            plugins["edit_file"] = edit_file
            plugins["run_tests"] = run_tests
            plugins["rollback"] = rollback

        self._final_answer_submitted = None
        self._messages_snapshots = {}
        self._snapshot_answers = {}
        self._snapshot_confidences = {}
        
        def submit_final_answer(answer: str, confidence: float = 1.0):
            words = str(answer).split()
            if len(words) > 6:
                return "ERROR: Final answer must be an exact entity, number, or short phrase. Retry."
            
            current_iter = getattr(self, '_current_iteration', 0)
            
            self._snapshot_answers[current_iter] = str(answer)
            self._snapshot_confidences[current_iter] = float(confidence)

            executed_code = self._code_history[-1] if self._code_history else None
            gate_result = {"exit": True}
            if self.enable_acc and not self.force_iterations:
                gate_result = self._acc_controller.should_exit(
                    current_answer=str(answer),
                    retrieved_context=context_str[:5000],
                    llm_client=self.llm,
                    confidence=float(confidence),
                    iteration=current_iter,
                    executed_code=executed_code
                )
            
            if not gate_result["exit"] and not self.force_iterations and not gate_result.get("rollback"):
                return "ERROR: Answer not supported by context. You must search for more information."
            
            if self.force_iterations:
                return f"Snapshot answer '{answer}' recorded with confidence {confidence}. FORCE_ITERATIONS is active. You must continue verifying or exploring alternative approaches."
            else:
                if gate_result.get("rollback"):
                    peak_iter = gate_result.get("peak_iteration")
                    rolled_back_answer = self._snapshot_answers.get(peak_iter, answer)
                    
                    if gate_result.get("terminal_rollback", True):
                        self._final_answer_submitted = str(rolled_back_answer)
                        return f"ACC TERMINAL ROLLBACK: Confidence dropped. Rolling back to peak answer '{rolled_back_answer}' from iteration {peak_iter}. Exit triggered."
                    else:
                        if peak_iter in self._messages_snapshots:
                            import copy
                            self.messages = copy.deepcopy(self._messages_snapshots[peak_iter])
                        
                        rollback_msg = (
                            f"ACC DIRECTED ROLLBACK: Confidence fell to {confidence:.2f} (peak was {self._snapshot_confidences.get(peak_iter, 1.0):.2f}). "
                            f"We have rolled back the execution context to the state of iteration {peak_iter} where you obtained the peak answer '{rolled_back_answer}'. "
                            f"Reason: {gate_result['reason']}. Please execute a different reasoning path or run query search tool to verify info."
                        )
                        self.messages.append({"role": "user", "content": rollback_msg})
                        return f"WARNING: ACC Rollback triggered. Context restored to iteration {peak_iter}. {rollback_msg}"
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

        # ── Start trajectory collection ───────────────────────────────────
        self._trajectory_collector.start(
            model_provider=getattr(self.llm, "provider", "unknown"),
            model_name=getattr(self.llm, "model", "unknown"),
            task_description=query or "",
            repo_path=self.repo_path or "",
            test_command=self.test_command or "",
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
            import copy
            self._messages_snapshots[iteration + 1] = copy.deepcopy(self.messages)
            
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

    def _post_completion(self, query: str, answer: str, success: bool = True):
        """Called after every completion to store memory, close ACC, save trajectory."""
        if self.enable_acc:
            self.last_acc_report = self._acc_controller.end_episode()

        if self.enable_memory and self._memory_adapter:
            reasoning_summary = f"Ran REPL loop, depth={getattr(self, 'last_depth', 'N/A')}"
            self._memory_adapter.store(
                query=query,
                reasoning=reasoning_summary,
                action="repl_completion",
                outcome=f"Answered: {answer[:200]}",
                outcome_score=0.6 if success else -0.5,
            )
            self._memory_adapter.save(self._memory_path)

        # ── Trajectory serialization (training data collection) ───────────
        if self._trajectory_collector.active is not None:
            final_diff = ""
            if self.enable_tdrl:
                import difflib
                diffs = []
                for fpath, before in self._file_snapshots.items():
                    if os.path.exists(fpath):
                        with open(fpath, "r", encoding="utf-8") as _f:
                            after = _f.read()
                        if before != after:
                            diffs.extend(difflib.unified_diff(
                                before.splitlines(), after.splitlines(),
                                fromfile=f"a/{os.path.basename(fpath)}",
                                tofile=f"b/{os.path.basename(fpath)}",
                                lineterm="",
                            ))
                final_diff = "\n".join(diffs)

            path = self._trajectory_collector.finish(
                success=success,
                final_diff=final_diff,
            )
            if path:
                self.logger.log_final_response(f"[Trajectory saved: {path}]")

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

        if self.enable_acc and hasattr(self, '_acc_controller') and getattr(self._acc_controller, '_conformal', None) is not None:
            is_correct = (judge_score >= 4)
            last_iter = self._current_iteration
            last_conf = self._snapshot_confidences.get(last_iter, 1.0)
            self._acc_controller.update_conformal(is_correct, last_conf, last_iter)
            self.logger.info(f"Conformal Updated: Online observation added (Correct: {is_correct}, Conf: {last_conf}, Iter: {last_iter})")
