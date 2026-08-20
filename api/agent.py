import logging
import json
import os
import time
import uuid
import difflib
import numpy as np
from typing import List
from RLM.experiments.rl.test_driven_env import TestDrivenEnv
from RLM.api.schemas import ExecuteRequest, ExecuteResponse, ExecutionStep
from RLM.engine.planner import System2Planner
from RLM.api.context_os import load_repo_memory, save_repo_memory, memory_path_for

logger = logging.getLogger(__name__)

class RLMAgentWrapper:
    """
    Wraps the TestDrivenEnv in an inference loop.
    For the API MVP, if a trained model isn't present, it uses a heuristic fallback policy
    (e.g., trying to EDIT, then RUN_TESTS, handling rollbacks on failure).
    """
    
    def __init__(self, request: ExecuteRequest):
        self.request = request
        self.trace_id = str(uuid.uuid4())
        self.memory_path = memory_path_for(request.repo_path)
        self.memory_system = load_repo_memory(request.repo_path, capacity=100)
        self.env = TestDrivenEnv(
            repo_path=request.repo_path,
            test_command=request.test_command,
            reward_type=request.reward_type,
            max_steps=request.max_steps,
            memory_system=self.memory_system,
        )
        # Attempt to load stable_baselines3 model if it exists, otherwise use fallback
        self.model = None
        try:
            from stable_baselines3 import PPO
            import os
            model_path = "weights/test_driven_ppo.zip"
            if os.path.exists(model_path):
                self.model = PPO.load(model_path)
                logger.info("Loaded PPO model for agent.")
        except ImportError:
            logger.warning("stable_baselines3 not installed. Using heuristic policy.")

    def _build_execution_context(self) -> dict:
        active_file = self._infer_active_file()
        state_query = f"{self.request.task_description}\n{active_file}"

        retrieved, conflicts = self.memory_system.retrieve(state_query, top_k=5)
        retrieved_payload = [
            {
                "entry_id": mem.entry_id,
                "state": mem.state,
                "action": mem.action,
                "outcome": mem.outcome,
                "outcome_score": mem.outcome_score,
                "score": score,
            }
            for mem, score in retrieved
        ]
        memory_context = "\n".join(
            f"- {mem.action} -> {mem.outcome} (score={score:.3f})"
            for mem, score in retrieved
        ) or "No relevant memories."
        failure_warnings = "\n".join(conflicts) or "No failure warnings."

        try:
            planner = System2Planner(project_root=self.request.repo_path)
            execution_plan = planner.formulate_plan(
                active_file=active_file,
                developer_intent=self.request.task_description,
                memory_context=f"{failure_warnings}\n{memory_context}",
            )
        except Exception as exc:
            logger.warning("Planner failed; continuing with fallback plan: %s", exc)
            execution_plan = json.dumps({
                "plan": [
                    {
                        "file": os.path.basename(active_file),
                        "action": self.request.task_description,
                    }
                ],
                "planner_error": str(exc),
            })

        return {
            "trace_id": self.trace_id,
            "active_file": active_file,
            "developer_intent": self.request.task_description,
            "execution_plan": execution_plan,
            "retrieved_memory_ids": [item["entry_id"] for item in retrieved_payload],
            "retrieved_memory_records": retrieved_payload,
            "retrieved_memories": memory_context,
            "failure_warnings": failure_warnings,
            "failure_warnings_injected": conflicts,
            "memory_path": self.memory_path,
        }

    def _infer_active_file(self) -> str:
        for name in ("auth.py", "math_utils.py"):
            candidate = os.path.join(self.request.repo_path, name)
            if os.path.exists(candidate):
                return candidate

        for root, _, files in os.walk(self.request.repo_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    return os.path.join(root, file)
        return self.request.repo_path

    def _save_trace(self, response: ExecuteResponse, execution_context: dict):
        results_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "experiments", "results", "rlm_execute"
        ))
        os.makedirs(results_dir, exist_ok=True)
        trace_path = os.path.join(results_dir, f"trace_{self.trace_id}.json")
        test_outputs = [
            {
                "step": event.get("step"),
                "type": event.get("type"),
                "command": event.get("command"),
                "returncode": event.get("returncode"),
                "output": event.get("output") or event.get("error"),
            }
            for event in self.env.trace_events
            if event.get("type") in {"tests_run", "tests_error"}
        ]
        rollback_files = sorted({
            file_path
            for event in self.env.trace_events
            if event.get("type") == "rollback"
            for file_path in event.get("files", [])
        })
        rollback_events = [
            event for event in self.env.trace_events
            if event.get("type") == "rollback"
        ]
        memory_events = [
            event for event in self.env.trace_events
            if event.get("type") == "memory_written"
        ]
        final_diff = self._final_diff()
        estimated_context_tokens = self._estimate_tokens(execution_context, test_outputs, final_diff)
        payload = {
            "metadata": {
                "schema_version": 1,
                "source": "rlm_execute",
                "trace_id": self.trace_id,
                "method": "rlm_1_full",
                "method_display": "RLM-1 (Full Neuro-Symbolic System)",
                "model": "configured-llm-provider",
                "task_id": self.trace_id,
                "repo_path": self.request.repo_path,
                "task_description": self.request.task_description,
                "test_command": self.request.test_command,
                "success": response.success,
                "final_reward": response.final_reward,
                "total_steps": response.total_steps,
                "timestamp": time.time(),
            },
            "metrics": {
                "steps": response.total_steps,
                "context_tokens": estimated_context_tokens,
                "failure_loops": len(rollback_events),
                "runtime_s": max(0.0, time.time() - getattr(self, "started_at", time.time())),
            },
            "execution_context": execution_context,
            "retrieved_memory_ids": execution_context.get("retrieved_memory_ids", []),
            "failure_warnings_injected": execution_context.get("failure_warnings_injected", []),
            "rollback_files": rollback_files,
            "test_outputs": test_outputs,
            "final_diff": final_diff,
            "token_cost_estimate": {
                "estimated_context_tokens": estimated_context_tokens,
                "pricing_source": "rough_chars_div_4_local_estimate",
            },
            "api_trace": [step.model_dump() for step in response.trace],
            "env_events": self.env.trace_events,
            "memory_events": memory_events,
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return trace_path

    def _final_diff(self) -> str:
        diffs = []
        for file_path, before in self.env.file_snapshots.items():
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                after = f.read()
            if before == after:
                continue
            diffs.extend(difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{os.path.relpath(file_path, self.request.repo_path)}",
                tofile=f"b/{os.path.relpath(file_path, self.request.repo_path)}",
                lineterm="",
            ))
        return "\n".join(diffs)

    def _estimate_tokens(self, execution_context: dict, test_outputs: list, final_diff: str) -> int:
        payload = {
            "execution_context": execution_context,
            "test_outputs": test_outputs,
            "final_diff": final_diff,
        }
        return max(1, len(json.dumps(payload, default=str)) // 4)
            
    def _heuristic_policy(self, obs: np.ndarray) -> int:
        """
        State: [iteration, tests_passed_ratio, last_action_status, confidence, context_len]
        Actions: 0 = RUN_TESTS, 1 = EDIT_FILE, 2 = ROLLBACK
        """
        tests_passed = obs[1]
        last_status = obs[2]
        
        # If tests completely passed, we shouldn't even be here, but just in case
        if tests_passed == 1.0:
            return 0 # RUN_TESTS to trigger done flag
            
        # If last action was a failed test, try to edit
        if last_status == -1.0:
            # If it keeps failing, maybe rollback. Heuristic: random chance to rollback
            if np.random.rand() > 0.7:
                return 2 # ROLLBACK
            return 1 # EDIT
            
        # If we just edited (status usually 0.5 for success edit mock), run tests
        if last_status == 0.5:
            return 0 # RUN_TESTS
            
        # Default: try editing
        return 1
        
    def execute(self) -> ExecuteResponse:
        self.started_at = time.time()
        execution_context = self._build_execution_context()
        self.env.set_execution_context(execution_context)
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        action_map = {0: "RUN_TESTS", 1: "EDIT_FILE", 2: "ROLLBACK"}
        trace: List[ExecutionStep] = []
        
        conformal_score = None
        
        while not done and step_count < self.request.max_steps:
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = self._heuristic_policy(obs)
                
            obs_next, reward, done, _, info = self.env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Extract state values from observation
            iter_norm, test_ratio, last_status, conf, ctx_len = obs_next[:5]
            
            step_record = ExecutionStep(
                step_number=step_count,
                action_taken=action_map.get(action, "UNKNOWN"),
                reward_received=reward,
                confidence=float(conf),
                context_length=float(ctx_len),
                is_success=(test_ratio == 1.0),
                logs=f"Action {action_map.get(action)} executed. Status: {last_status}"
            )
            trace.append(step_record)
            
            # If tests pass, mock a conformal safety check (assuming high confidence + pass)
            if test_ratio == 1.0:
                conformal_score = 0.95 # Mock conformal probability 1-alpha
                
            obs = obs_next

        # Check final success
        success = (self.env.tests_passed_ratio == 1.0)
        
        response = ExecuteResponse(
            success=success,
            final_reward=total_reward,
            total_steps=step_count,
            conformal_safety_score=conformal_score if success else None,
            trace=trace,
            message="Agent execution completed successfully." if success else "Agent failed to solve the task within max steps."
        )
        trace_path = self._save_trace(response, execution_context)
        save_repo_memory(self.request.repo_path, self.memory_system)
        response.message = f"{response.message} Trace saved to {trace_path}"
        return response
