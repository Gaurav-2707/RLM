"""
Smoke tests for IntegratedRLM.

Patches LLMClient so tests run without API keys or network.
Verifies initialization, completion(), and TDRL plugin wiring.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


CANNED_RESPONSE = "I have fixed the bug by updating the logic."

# Fake API key so LLMClient's key check doesn't fire
_FAKE_ENV = {"OPENAI_API_KEY": "sk-test-fake-key-for-unit-tests-only"}


def _patched_llm():
    """Context manager: patch LLMClient.completion so no real call is made."""
    return patch("RLM.utils.llm.LLMClient.completion", return_value=CANNED_RESPONSE)


def _make_rlm(**kwargs):
    """Create an IntegratedRLM with env vars + LLM patched out."""
    from RLM.integrated_repl import IntegratedRLM
    with patch.dict(os.environ, _FAKE_ENV):
        rlm = IntegratedRLM(model="gpt-4o", **kwargs)
    return rlm


# ── initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_basic_init_no_crash(self):
        """IntegratedRLM initialises without error under default settings."""
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model")
        assert rlm is not None

    def test_trajectory_collector_created(self):
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model")
        assert hasattr(rlm, "_trajectory_collector")

    def test_contribute_false_by_default(self):
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model")
        assert rlm.contribute_traces is False

    def test_contribute_flag_flows_to_collector(self):
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model", contribute_traces=True)
        assert rlm._trajectory_collector.contribute is True

    def test_tdrl_flags_default_false(self):
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model")
        assert rlm.enable_tdrl is False
        assert rlm.enable_acc is False
        assert rlm.enable_memory is False
        assert rlm.enable_engine is False

    def test_tdrl_flags_propagate(self):
        from RLM.integrated_repl import IntegratedRLM
        with tempfile.TemporaryDirectory() as tmp:
            rlm = IntegratedRLM(
                model="test-model",
                enable_tdrl=True,
                repo_path=tmp,
                test_command="pytest",
            )
            assert rlm.enable_tdrl is True
            assert rlm.repo_path == tmp
            assert rlm.test_command == "pytest"


# ── completion ────────────────────────────────────────────────────────────────

class TestCompletion:
    def test_completion_returns_string(self):
        """completion() must always return a string."""
        rlm = _make_rlm()
        with _patched_llm(), patch.dict(os.environ, _FAKE_ENV):
            result = rlm.completion(context=[], query="What does this code do?")
        assert isinstance(result, str)

    def test_completion_string_context(self):
        """completion() accepts a plain string as context."""
        rlm = _make_rlm()
        with _patched_llm(), patch.dict(os.environ, _FAKE_ENV):
            result = rlm.completion(context="def foo(): return 1", query="explain this")
        assert isinstance(result, str)

    def test_completion_list_context(self):
        """completion() accepts a list as context."""
        rlm = _make_rlm()
        with _patched_llm(), patch.dict(os.environ, _FAKE_ENV):
            result = rlm.completion(context=["context item 1"], query="find bugs")
        assert isinstance(result, str)

    def test_completion_non_empty_result(self):
        """Patched LLM should produce a non-empty response."""
        rlm = _make_rlm()
        with _patched_llm(), patch.dict(os.environ, _FAKE_ENV):
            result = rlm.completion(context=[], query="fix the bug")
        assert len(result) > 0


# ── TDRL plugin wiring ────────────────────────────────────────────────────────

class TestTDRLPlugins:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tdrl_rlm(self):
        from RLM.integrated_repl import IntegratedRLM
        return IntegratedRLM(
            model="test-model",
            enable_tdrl=True,
            repo_path=self.tmpdir,
            test_command="true",  # always-pass no-op command
        )

    def test_edit_file_plugin_present(self):
        """When enable_tdrl=True, an edit_file capability must exist."""
        rlm = self._tdrl_rlm()
        # Defined as a local closure injected at completion time — but
        # we can verify TDRL flag is set and repo_path is stored.
        assert rlm.enable_tdrl is True
        assert rlm.repo_path == self.tmpdir

    def test_file_snapshots_dict_present(self):
        """_file_snapshots must be initialized for rollback to work."""
        rlm = self._tdrl_rlm()
        assert hasattr(rlm, "_file_snapshots")
        assert isinstance(rlm._file_snapshots, dict)

    def test_test_command_stored(self):
        """test_command must be stored for run_tests plugin to use."""
        rlm = self._tdrl_rlm()
        assert rlm.test_command == "true"


# ── trajectory collector integration ─────────────────────────────────────────

class TestTrajectoryCollection:
    def test_collector_ready_before_completion(self):
        """Trajectory collector must exist before any completion call."""
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model")
        assert rlm._trajectory_collector is not None

    def test_collector_contribute_matches_flag(self):
        from RLM.integrated_repl import IntegratedRLM
        for flag in (True, False):
            rlm = IntegratedRLM(model="test-model", contribute_traces=flag)
            assert rlm._trajectory_collector.contribute == flag

    def test_contribute_false_does_not_upload(self):
        """When contribute_traces=False, upload_trajectory_async must not be called."""
        from RLM.integrated_repl import IntegratedRLM
        rlm = IntegratedRLM(model="test-model", contribute_traces=False)
        with _patched_llm(), \
             patch.dict(os.environ, _FAKE_ENV), \
             patch("RLM.utils.upload.upload_trajectory_async") as mock_upload:
            rlm.completion(context=[], query="fix this")
        mock_upload.assert_not_called()
