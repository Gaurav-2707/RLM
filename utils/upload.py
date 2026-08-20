"""
Trajectory Upload Pipeline — the data flywheel plumbing.

When a user opts in (contribute_traces=True), verified trajectories are
anonymized and uploaded to the Recursive Labs training endpoint. This is
what converts product usage into RLM-0 training data.

Design constraints:
  1. Never block the main execution loop (async upload)
  2. Re-verify PII strip before every upload (don't trust cached state)
  3. Retry with backoff (transient network failures are common)
  4. Validate server acknowledgement (checksum round-trip)
  5. Gracefully degrade: failed upload = warning, not crash

Usage (auto-wired into TrajectoryCollector when contribute=True):

    from RLM.utils.upload import upload_trajectory
    success = upload_trajectory(traj, endpoint="https://data.recursivelabs.ai/v1/contribute")

Or async (non-blocking, used inside IntegratedRLM._post_completion):

    from RLM.utils.upload import upload_trajectory_async
    upload_trajectory_async(traj, endpoint)   # fire-and-forget
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Optional

from RLM.utils.trajectory import RLMTrajectory

logger = logging.getLogger(__name__)

# Default endpoint — overridden by ~/.rlm/config.json "upload_endpoint"
DEFAULT_ENDPOINT = "https://data.recursivelabs.ai/v1/contribute"

# Retry config
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0   # seconds; doubles each retry (1s, 2s, 4s)
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30


def _load_endpoint() -> str:
    """Read upload endpoint from ~/.rlm/config.json, fall back to default."""
    cfg_path = os.path.expanduser("~/.rlm/config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            ep = cfg.get("upload_endpoint", "").strip()
            if ep:
                return ep
        except Exception:
            pass
    return DEFAULT_ENDPOINT


_PII_PATTERNS = [
    # API keys: sk-..., Bearer token, generic long hex/base64 secrets
    (re.compile(r'sk-[A-Za-z0-9\-_]{20,}'), '<REDACTED_API_KEY>'),
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*'), '<REDACTED_BEARER>'),
    # Email addresses
    (re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'), '<REDACTED_EMAIL>'),
    # Absolute file paths with usernames: /Users/name, /home/name, C:\Users\name
    (re.compile(r'/Users/[^/\s,\'"]+'), '/Users/<REDACTED>'),
    (re.compile(r'/home/[^/\s,\'"]+'), '/home/<REDACTED>'),
    (re.compile(r'C:\\[Uu]sers\\[^\\\s,\'"]+'), 'C:\\Users\\<REDACTED>'),
    # Generic long tokens that look like secrets (32+ hex chars)
    (re.compile(r'\b[0-9a-fA-F]{32,}\b'), '<REDACTED_TOKEN>'),
]


def _strip_pii(payload: dict) -> dict:
    """
    Re-verify PII anonymization on the serialized payload.
    Recursively redacts API keys, emails, absolute paths, and long tokens.
    """
    def _clean(value):
        if isinstance(value, str):
            for pattern, replacement in _PII_PATTERNS:
                value = pattern.sub(replacement, value)
            return value
        elif isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_clean(item) for item in value]
        return value

    return _clean(payload)


def _checksum(payload: dict) -> str:
    """SHA-256 of the canonical JSON serialization."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def upload_trajectory(
    traj: RLMTrajectory,
    endpoint: Optional[str] = None,
) -> bool:
    """
    Upload a single verified trajectory to the training endpoint.

    Steps:
      1. Serialize to dict
      2. Strip PII (re-verify, don't trust existing anonymization)
      3. Compute checksum
      4. POST with retry (3x, exponential backoff)
      5. Validate server echoes back matching checksum
      6. Return True on success, False on any failure

    Parameters
    ----------
    traj     : The verified trajectory to upload. Must have verified=True.
    endpoint : Upload URL. Falls back to ~/.rlm/config.json or DEFAULT_ENDPOINT.

    Returns
    -------
    bool : True if accepted by server, False otherwise.
    """
    if not traj.verified:
        logger.warning("upload_trajectory: skipping unverified trajectory %s", traj.trajectory_id)
        return False

    ep = endpoint or _load_endpoint()

    try:
        payload = asdict(traj)
    except Exception as e:
        logger.warning("upload_trajectory: serialization failed for %s: %s", traj.trajectory_id, e)
        return False

    payload = _strip_pii(payload)
    checksum = _checksum(payload)
    body = json.dumps({"trajectory": payload, "checksum": checksum}, ensure_ascii=True).encode("utf-8")

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                ep,
                data=body,
                headers={"Content-Type": "application/json", "X-RLM-Version": "0.1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT + _READ_TIMEOUT) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                server_checksum = resp_body.get("checksum", "")
                if server_checksum == checksum:
                    logger.debug("upload_trajectory: %s accepted", traj.trajectory_id)
                    return True
                else:
                    logger.warning(
                        "upload_trajectory: checksum mismatch for %s (local=%s server=%s)",
                        traj.trajectory_id, checksum[:8], server_checksum[:8],
                    )
                    return False
        except urllib.error.HTTPError as e:
            logger.warning("upload_trajectory: HTTP %d on attempt %d for %s",
                           e.code, attempt + 1, traj.trajectory_id)
        except Exception as e:
            logger.warning("upload_trajectory: attempt %d failed for %s: %s",
                           attempt + 1, traj.trajectory_id, e)
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_BACKOFF_BASE * (2 ** attempt))

    logger.warning("upload_trajectory: all retries exhausted for %s", traj.trajectory_id)
    return False


def upload_trajectory_async(
    traj: RLMTrajectory,
    endpoint: Optional[str] = None,
) -> None:
    """
    Fire-and-forget async upload in a daemon thread.

    Failures are logged as warnings but never propagate to the caller.
    The main execution loop is never blocked.

    Parameters
    ----------
    traj     : Trajectory to upload.
    endpoint : Upload URL (optional, same fallback as upload_trajectory).
    """
    def _worker():
        try:
            ok = upload_trajectory(traj, endpoint=endpoint)
            if ok:
                logger.info("Trajectory %s uploaded successfully", traj.trajectory_id)
            else:
                logger.warning("Trajectory %s upload failed (will not retry)", traj.trajectory_id)
        except Exception as e:
            logger.warning("Trajectory upload error (trajectory_id=%s): %s",
                           traj.trajectory_id, e)

    t = threading.Thread(target=_worker, daemon=True, name=f"rlm-upload-{traj.trajectory_id[:8]}")
    t.start()


def batch_upload(
    trajs: list,
    endpoint: Optional[str] = None,
    min_score: float = 0.0,
) -> dict:
    """
    Upload a list of trajectories, returning a summary dict.

    Parameters
    ----------
    trajs     : List of RLMTrajectory objects.
    endpoint  : Upload URL (optional).
    min_score : If > 0, only upload trajectories whose traj_weight >= min_score.
                Useful for filtering low-quality contributions before upload.

    Returns
    -------
    dict with keys: total, uploaded, failed, skipped
    """
    from RLM.training.scorer import TrajectoryScorer
    scorer = TrajectoryScorer() if min_score > 0 else None
    corpus: list = []

    total = len(trajs)
    uploaded = 0
    failed = 0
    skipped = 0

    for traj in trajs:
        if scorer is not None:
            score = scorer.score(traj, corpus)
            if score < min_score:
                logger.debug("batch_upload: skipping %s (score=%.3f < %.3f)",
                             traj.trajectory_id, score, min_score)
                skipped += 1
                continue
        ok = upload_trajectory(traj, endpoint=endpoint)
        if ok:
            uploaded += 1
            corpus.append(traj)   # progressive novelty update
        else:
            failed += 1

    logger.info("batch_upload: total=%d uploaded=%d failed=%d skipped=%d",
                total, uploaded, failed, skipped)
    return {"total": total, "uploaded": uploaded, "failed": failed, "skipped": skipped}
