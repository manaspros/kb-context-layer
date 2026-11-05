"""OpenAI-compatible chat client: stdlib only, disk-cached by request hash, retried, rate-safe.

Rate safety: at most MAX_CONCURRENT requests in flight, at most TOKENS_PER_MIN tokens in any rolling
minute (reserved up front, trued-up on completion), a hard TOTAL_CAP on tokens spent, and a full stop
(everything stays cached) on any 429 that asks for a long wait or names a policy / org-level limit.
"""
import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from kb.common import ROOT

CACHE = ROOT / "data" / "cache"
PARAMS = {"temperature": 0, "max_tokens": 8000}  # 4000 left 8-17% of outputs empty (reasoning exhausts it); chosen on tune
LEGACY_MAX_TOKENS = 4000  # the cap before round 2
MAX_CONCURRENT = 6
TOKENS_PER_MIN = 400_000
RESERVE_SLACK = 48_000  # MAX_CONCURRENT * max_tokens: worst-case overshoot before true-up
EXPECTED_COMPLETION = 1500  # reserved per call until the real usage is known
SPENT_BEFORE_ROUND = 10_983_194  # cache total when the round-2 budget started (ledger: spent() - this)
TOTAL_CAP = SPENT_BEFORE_ROUND + 10_000_000
LONG_RETRY_AFTER = 120
POLICY_429 = re.compile(r"polic|organi[sz]ation|\borg\b|org-wide|org_|workspace", re.I)

_sem = threading.Semaphore(MAX_CONCURRENT)
_lock = threading.Lock()
_window: deque[list] = deque()  # [timestamp, tokens] per call in the last minute
_state = {"total": None, "stop": None}


class StopRun(RuntimeError):
    """Raised for a policy/long 429 or the token cap; the caller must stop and report."""


def model_tag() -> str:
    """Last two path components of LLM_MODEL; the only form of the model name we record."""
    return "/".join(os.environ["LLM_MODEL"].split("/")[-2:])


def spent() -> int:
    """Total tokens ever spent through this cache (prior runs included), plus this process."""
    with _lock:
        if _state["total"] is None:
            docs = (json.loads(f.read_text()) for f in CACHE.glob("*.json"))
            _state["total"] = sum(sum(d["usage"].values()) for d in docs if not d.get("carried"))
        return _state["total"]


def _reserve(est: int) -> list:
    while True:
        with _lock:
            if _state["stop"]:
                raise StopRun(_state["stop"])
            now = time.time()
            while _window and now - _window[0][0] > 60:
                _window.popleft()
            if sum(t for _, t in _window) + est <= TOKENS_PER_MIN - RESERVE_SLACK:
                entry = [now, est]
                _window.append(entry)
                return entry
        time.sleep(0.5)


def _stop(reason: str) -> None:
    with _lock:
        _state["stop"] = _state["stop"] or reason
    raise StopRun(reason)


def chat(messages: list[dict]) -> dict:
    """Return {"text", "usage": {prompt_tokens, completion_tokens}, "cached"}."""
    model = os.environ["LLM_MODEL"]
    key = hashlib.sha256(json.dumps([model, messages, PARAMS], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return {**json.loads(path.read_text()), "cached": True}
    if PARAMS["max_tokens"] > LEGACY_MAX_TOKENS and "reasoning" not in PARAMS:
        # A response that stopped by itself under the old 4000 cap is a valid response under a higher cap, so it is
        # carried over (no tokens spent, not counted by spent()); only truncated or empty responses are re-issued.
        old = CACHE / (hashlib.sha256(json.dumps(
            [model, messages, {**PARAMS, "max_tokens": LEGACY_MAX_TOKENS}], sort_keys=True).encode()).hexdigest() + ".json")
        if old.exists():
            d = json.loads(old.read_text())
            if d["text"].strip() and d["usage"]["completion_tokens"] < LEGACY_MAX_TOKENS:
                path.write_text(json.dumps({**d, "carried": True}))
                return {**d, "cached": True}
    headers = {"Content-Type": "application/json", "User-Agent": "kb-context-layer/0.1",
               "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
               **json.loads(os.environ.get("LLM_HEADERS") or "{}")}
    body = json.dumps({"model": model, "messages": messages, **PARAMS}).encode()
    req = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=body, headers=headers)
    est = len(body) // 3 + EXPECTED_COMPLETION
    with _sem:
        for attempt in range(10):
            if spent() >= TOTAL_CAP:
                _stop(f"token cap {TOTAL_CAP} reached")
            entry = _reserve(est)
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                entry[1] = 0
                if e.code == 429:
                    wait = e.headers.get("Retry-After", "")
                    wait = float(wait) if re.fullmatch(r"\d+(\.\d+)?", wait or "") else 0.0
                    text = e.read(4000).decode("utf-8", "replace")
                    if wait > LONG_RETRY_AFTER or POLICY_429.search(text):
                        _stop(f"429 stop: retry-after={wait:g}s, policy-like body={bool(POLICY_429.search(text))}")
                    time.sleep(min(max(wait, 2 ** attempt), 120))
                    continue
                if e.code < 500 or attempt == 9:
                    raise RuntimeError(f"LLM HTTP {e.code}") from None
            except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError):
                entry[1] = 0
                if attempt == 9:
                    raise RuntimeError("LLM network error") from None
            time.sleep(min(2 ** attempt, 60))
        else:
            raise RuntimeError("LLM retries exhausted")
        u = data.get("usage") or {}
        used = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        with _lock:
            entry[0], entry[1] = time.time(), used
            _state["total"] = (_state["total"] or 0) + used
    out = {"text": (data["choices"][0]["message"].get("content") or ""),
           "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                     "completion_tokens": u.get("completion_tokens", 0)},
           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return {**out, "cached": False}
