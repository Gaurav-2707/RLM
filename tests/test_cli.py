"""
Smoke tests for the `rlm` CLI (cli.py).

Tests call cli.main() directly with sys.argv patched.
Heavy commands (benchmark, serve) are shallow-mocked so tests stay fast.

Key: CLI uses lazy imports inside each cmd_* function. Always patch at
the *source* module (e.g. "RLM.integrated_repl.IntegratedRLM"), not
at "RLM.cli.IntegratedRLM" which doesn't exist until cmd_run() runs.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from RLM.cli import main as cli_main


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(*argv, expect_exit=False):
    """
    Call cli_main() with given argv. Returns SystemExit code or 0 on success.
    """
    with patch.object(sys, "argv", ["rlm"] + list(argv)):
        try:
            cli_main()
            return 0
        except SystemExit as e:
            code = e.code or 0
            if not expect_exit and code != 0:
                raise
            return code


# ── help / meta ───────────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help_flag_exits_zero(self):
        code = _run("--help", expect_exit=True)
        assert code == 0

    def test_no_subcommand_exits_nonzero(self):
        code = _run(expect_exit=True)
        assert code != 0

    def test_each_subcommand_help_exits_zero(self):
        for cmd in ("init", "status", "run", "benchmark", "contribute", "serve"):
            code = _run(cmd, "--help", expect_exit=True)
            assert code == 0, f"`rlm {cmd} --help` exited {code}"


# ── init ──────────────────────────────────────────────────────────────────────

class TestCmdInit:
    def test_init_runs_with_mocked_graph(self):
        """rlm init should run without crashing when graph is mocked."""
        with tempfile.TemporaryDirectory() as tmp:
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = 0
            mock_graph.number_of_edges.return_value = 0
            with patch("RLM.memory.graph.SemanticContextGraph", return_value=mock_graph), \
                 patch("RLM.cli._load_config", return_value={}), \
                 patch("RLM.cli._save_config"):
                _run("init", "--repo", tmp)

    def test_init_saves_config(self):
        """rlm init must call _save_config."""
        with tempfile.TemporaryDirectory() as tmp:
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = 3
            mock_graph.number_of_edges.return_value = 2
            with patch("RLM.memory.graph.SemanticContextGraph", return_value=mock_graph), \
                 patch("RLM.cli._load_config", return_value={}), \
                 patch("RLM.cli._save_config") as mock_save:
                _run("init", "--repo", tmp)
            assert mock_save.called, "_save_config must be called by cmd_init"


# ── status ────────────────────────────────────────────────────────────────────

class TestCmdStatus:
    def test_status_no_crash_empty_state(self):
        """rlm status with no trajectories and no graph must not crash."""
        with patch("RLM.cli._load_config", return_value={}), \
             patch("RLM.utils.trajectory.trajectory_stats",
                   return_value={"total": 0, "verified": 0, "success_rate": 0.0}), \
             patch("os.path.exists", return_value=False):
            try:
                _run("status")
            except Exception as e:
                pytest.fail(f"rlm status crashed unexpectedly: {e}")

    def test_status_no_crash_with_stats(self):
        """rlm status with mock stats must not crash."""
        stats = {
            "total": 10, "verified": 8, "successful": 6,
            "success_rate": 0.75, "contributed": 3,
            "with_rollbacks": 2, "avg_steps": 5.0,
            "avg_rollbacks": 0.5, "models_seen": ["gpt-4o"],
        }
        with patch("RLM.cli._load_config", return_value={"model": "gpt-4o"}), \
             patch("RLM.utils.trajectory.trajectory_stats", return_value=stats), \
             patch("os.path.exists", return_value=False):
            _run("status")


# ── contribute ────────────────────────────────────────────────────────────────

class TestCmdContribute:
    def test_contribute_enable_sets_flag(self):
        """rlm contribute --enable must save contribute_traces=True."""
        with patch("RLM.cli._load_config", return_value={"contribute_traces": False}), \
             patch("RLM.cli._save_config") as mock_save:
            _run("contribute", "--enable")
        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][0]
        assert saved_cfg.get("contribute_traces") is True

    def test_contribute_disable_clears_flag(self):
        """rlm contribute --disable must save contribute_traces=False."""
        with patch("RLM.cli._load_config", return_value={"contribute_traces": True}), \
             patch("RLM.cli._save_config") as mock_save:
            _run("contribute", "--disable")
        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][0]
        assert saved_cfg.get("contribute_traces") is False


# ── run ───────────────────────────────────────────────────────────────────────

class TestCmdRun:
    def test_run_calls_completion(self):
        """rlm run 'task' must invoke IntegratedRLM and call completion."""
        mock_rlm = MagicMock()
        mock_rlm.completion.return_value = "Fixed the bug."
        with patch("RLM.integrated_repl.IntegratedRLM", return_value=mock_rlm), \
             patch("RLM.cli._load_config", return_value={}):
            with tempfile.TemporaryDirectory() as tmp:
                _run("run", "fix the auth bug", "--repo", tmp)
        mock_rlm.completion.assert_called_once()

    def test_run_test_command_enables_tdrl(self):
        """--test-command must propagate enable_tdrl=True."""
        captured = {}
        def _init(*args, **kwargs):
            captured.update(kwargs)
            m = MagicMock()
            m.completion.return_value = "done"
            return m
        with patch("RLM.integrated_repl.IntegratedRLM", side_effect=_init), \
             patch("RLM.cli._load_config", return_value={}):
            with tempfile.TemporaryDirectory() as tmp:
                _run("run", "fix bug", "--repo", tmp, "--test-command", "pytest -q")
        assert captured.get("enable_tdrl") is True
        assert captured.get("test_command") == "pytest -q"

    def test_run_no_test_command_disables_tdrl(self):
        """Without --test-command, enable_tdrl must be False."""
        captured = {}
        def _init(*args, **kwargs):
            captured.update(kwargs)
            m = MagicMock()
            m.completion.return_value = "done"
            return m
        with patch("RLM.integrated_repl.IntegratedRLM", side_effect=_init), \
             patch("RLM.cli._load_config", return_value={}):
            with tempfile.TemporaryDirectory() as tmp:
                _run("run", "fix bug", "--repo", tmp)
        assert captured.get("enable_tdrl") is False


# ── serve ─────────────────────────────────────────────────────────────────────

class TestCmdServe:
    def test_serve_calls_uvicorn(self):
        """rlm serve must call uvicorn.run with the correct app path."""
        import uvicorn as uvicorn_mod
        with patch.object(uvicorn_mod, "run") as mock_run:
            _run("serve")
        mock_run.assert_called_once()
        call_args_str = str(mock_run.call_args)
        assert "RLM.api.main" in call_args_str

    def test_serve_uses_port_8000(self):
        """rlm serve must bind to port 8000."""
        import uvicorn as uvicorn_mod
        with patch.object(uvicorn_mod, "run") as mock_run:
            _run("serve")
        kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
        assert kwargs.get("port") == 8000
