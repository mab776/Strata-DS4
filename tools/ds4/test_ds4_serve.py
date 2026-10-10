#!/usr/bin/env python3
"""tools/ds4/test_ds4_serve.py - (mab776 M5) the gate of tools/ds4/ds4_serve.cpp on the mini fixtures (CPU only).

    tools/ds4/run_serve_gate.sh                       fixtures + this file in the strata-ds4-pytest:py312 container
    python3 tools/ds4/test_ds4_serve.py -v [Class]    from the repo root, with the env below

Env: DS4_SERVE_BIN / DS4_GEN_BIN / DS4_SAMPLER_BIN (default build-ds4-cpu/{ds4_serve_cpu,ds4_generate_cpu,
test_ds4_sampler}), DS4_FX (fixture dir, default build-ds4-cpu/fx; a missing fixture is generated, which needs gguf-py:
third_party/llama.cpp/gguf-py).

The pass rule (DESIGN.md section 8): "equal" = the logits row behind every token bit-identical (DS4_DUMP_TOKLOGITS) AND
the same tokens; a reference is never empty.  Every engine runs with DS4_CPU_FA_REF=1 (ggml-cpu's tiled flash-attn
takes 64+ queries and rounds differently; with the reference kernel a chunk is the decode math exactly), --threads 4
and --ctx 4096 for both binaries (the compressed caches are sized from --ctx).  Fixtures: `tame` (the numerically
tame mini model), `swa16` (its sliding window shrunk to 16: the raw ring wraps at 19 tokens), `tamex` (tame with
routed experts that compute something, tools/ds4/make_mini_serve.py: the expert tier reaches the logits).

  T1 Parity      fresh ds4_serve == ds4_generate_cpu, decode loop and chunks 1/7/16/33/64
  T2 Reuse       a reused request == the same request run fresh; RESUME and the decision == the spec model below
  T3 Twenty      20 mixed requests in one process: each == fresh, same PID, flat RSS, same open files
  T4 Stop        STOP -> DONE ... cancel in < 1 s (decode and prefill), stale STOP ignored, QUIT/EOF exit < 2 s
  T5 Server      through serve.server.StrataEngine and over HTTP (Service + serve(port=0))
  T6 Sampler     tools/ds4/test_ds4_sampler + Philox against a transcription in Python
  TErrors        refusals: one line each, the engine stays up and keeps its live state
"""
from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BUILD = Path(os.environ.get("DS4_BUILD", ROOT / "build-ds4-cpu"))
SERVE = os.environ.get("DS4_SERVE_BIN", str(BUILD / "ds4_serve_cpu"))
GEN = os.environ.get("DS4_GEN_BIN", str(BUILD / "ds4_generate_cpu"))
SAMPLER = os.environ.get("DS4_SAMPLER_BIN", str(BUILD / "test_ds4_sampler"))
FX = Path(os.environ.get("DS4_FX", BUILD / "fx"))
CTX = 4096
THREADS = os.environ.get("DS4_THREADS", "4")
ENV = {k: v for k, v in os.environ.items() if k not in ("DS4_DUMP_TOKLOGITS", "DS4_SERVE_TEST_DELAY_MS")}
ENV["DS4_CPU_FA_REF"] = "1"
PROTO = re.compile(r"^(INFO|READY|RESUME|PP|T|DONE|ERR|SERR) ")
V = 64          # the mini vocabulary
TURN = 63       # the user-turn token (--turn-token): random prompts never contain it, tests place it by hand

FIXTURES = {   # name -> (generator, extra env)
    "tame": ("bench/ds4-2026-10-06/dense/make_mini_tame.py", {}),
    "swa16": ("bench/ds4-2026-10-06/dense/make_mini_tame.py", {"DS4_SWA": "16"}),
    "tamex": ("tools/ds4/make_mini_serve.py", {}),
    "mtp": ("tools/ds4/make_mini_mtp.py", {}),     # mab776 M7b: the mini MTP head (pairs with tame)
}
_fx_lock = threading.Lock()


def fixture(name: str) -> Path:
    path = FX / f"{name}.gguf"
    with _fx_lock:
        if not path.exists():
            FX.mkdir(parents=True, exist_ok=True)
            script, extra = FIXTURES[name]
            tmp = FX / f"{name}.tmp.gguf"
            subprocess.run([sys.executable, str(ROOT / script), str(tmp)], check=True, env=dict(os.environ, **extra),
                           stdout=subprocess.DEVNULL)
            tmp.rename(path)
    return path


def prompt(n: int, seed: int, avoid=(TURN,)) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < n:
        t = rng.randrange(V)
        if t not in avoid:
            out.append(t)
    return out


def other(t: int) -> int:
    """A token that is not `t` (and not the turn token)."""
    return (t + 1) % (V - 1)


# ------------------------------------------------------------------------------------------------ references

_refs: dict = {}


def ref(fx: str, ids: list[int], max_new: int, chunk: int = 0, stop=()) -> tuple[list[int], bytes]:
    """ds4_generate_cpu on the full prompt: (tokens, logits rows)."""
    key = (fx, tuple(ids), max_new, chunk, tuple(stop))
    if key in _refs:
        return _refs[key]
    with tempfile.TemporaryDirectory() as d:
        rows = os.path.join(d, "rows")
        cmd = [GEN, "-m", str(fixture(fx)), "--backend", "cpu", "--experts", "cpu", "--threads", THREADS, "--ctx",
               str(CTX), "--ids", ",".join(map(str, ids)), "-n", str(max_new)]
        if chunk:
            cmd += ["--prefill-chunk", str(chunk)]
        if stop:
            cmd += ["--stop", ",".join(map(str, stop))]
        out = subprocess.run(cmd, env=dict(ENV, DS4_DUMP_TOKLOGITS=rows), capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise AssertionError("ds4_generate_cpu failed: " + out.stderr[-3000:])
        toks = [int(x) for x in out.stdout.split()]
        data = Path(rows).read_bytes()
    if not toks or len(data) != len(toks) * V * 4:
        raise AssertionError(f"empty or short reference ({len(toks)} tokens, {len(data)} bytes)")
    _refs[key] = (toks, data)
    return toks, data


# ------------------------------------------------------------------------------------------------ the raw driver

class Res:
    def __init__(self):
        self.resume = None
        self.pp: list[tuple[int, int, float, float]] = []
        self.tokens: list[int] = []
        self.done: list[str] | None = None
        self.err: str | None = None
        self.rows = b""
        self.t_done = None

    @property
    def finish(self):
        return self.done[5] if self.done else None

    @property
    def gen(self):
        return int(self.done[1])

    @property
    def reused(self):
        return int(self.done[8])

    @property
    def prompt_read(self):
        return int(self.done[14])


class Serve:
    """ds4_serve over its pipes (what serve/server.py does), every stdout line checked against the protocol."""

    DECISION = re.compile(r"GEN #(\d+): .* -> (append|rewind|checkpoint|reset)(?: (last|turn|user))?, reused (\d+)")

    def __init__(self, fx: str = "tame", *args, env: dict | None = None, dump: bool = True, start: bool = True):
        self.fx = fx
        self.dir = tempfile.mkdtemp(prefix="ds4serve-")
        e = dict(ENV)
        if dump:
            e["DS4_DUMP_TOKLOGITS"] = os.path.join(self.dir, "rows")
        e.update(env or {})
        self.cmd = [SERVE, "--serve", "-m", str(fixture(fx)), "--backend", "cpu", "--experts", "cpu", "--threads",
                    THREADS, "--ctx", str(CTX), *map(str, args)]
        self.logf = open(os.path.join(self.dir, "stderr.log"), "w")
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.logf, env=e,
                                  text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        self.bad: list[str] = []
        self.seq = 0
        self.info: dict = {}
        self.ctx = None
        self.can_stop = False
        threading.Thread(target=self._pump, daemon=True).start()
        if start:
            self.handshake()

    def _pump(self):
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if not PROTO.match(line):
                self.bad.append(line)
            self.q.put(line)
        self.q.put(None)

    def handshake(self, timeout=120):
        while True:
            line = self.line(timeout)
            if line is None:
                raise AssertionError("the engine exited before READY: " + self.log_tail())
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            elif line.startswith("READY "):
                f = line.split()
                self.ctx = int(f[1])
                self.can_stop = "stop" in f[2:]
                return

    def line(self, timeout=120):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"no line from the engine in {timeout} s: " + self.log_tail()) from None

    def send(self, text: str):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    @staticmethod
    def gen_line(ids, max_new, keys=""):
        return f"GEN {max_new}{' ' + keys if keys else ''} " + ",".join(map(str, ids))

    def gen(self, ids, max_new, keys="", timeout=600) -> Res:
        self.send(self.gen_line(ids, max_new, keys))
        self.seq += 1
        return self.result(timeout)

    def result(self, timeout=600, on_line=None) -> Res:
        r, seq = Res(), self.seq
        while True:
            line = self.line(timeout)
            if line is None:
                raise AssertionError("the engine exited during a request: " + self.log_tail())
            if line.startswith("RESUME "):
                assert r.resume is None and not r.pp and not r.tokens, "RESUME must come first, once"
                r.resume = int(line.split()[1])
            elif line.startswith("PP "):
                f = line.split()
                assert f[1].isdigit() and f[2].isdigit(), line
                r.pp.append((int(f[1]), int(f[2]), float(f[3]), float(f[4])))
            elif line.startswith("T "):
                r.tokens.append(int(line[2:]))
            elif line.startswith("DONE "):
                r.done = line.split()
                assert len(r.done) == 15, line
                r.t_done = time.monotonic()
            elif line.startswith(("ERR ", "SERR ")):
                r.err = line
                r.t_done = time.monotonic()
            if on_line is not None:
                on_line(line, r)
            if r.done is not None or r.err is not None:
                break
        rows = os.path.join(self.dir, f"rows.{seq}")
        if os.path.exists(rows):
            r.rows = Path(rows).read_bytes()
            os.remove(rows)
        return r

    def log_text(self) -> str:
        self.logf.flush()
        return Path(self.dir, "stderr.log").read_text(errors="replace")

    def log_tail(self, n=25) -> str:
        return "\n".join(self.log_text().splitlines()[-n:])

    def decision(self, seq: int):
        for m in self.DECISION.finditer(self.log_text()):
            if int(m.group(1)) == seq:
                return m.group(2), m.group(3), int(m.group(4))
        raise AssertionError(f"no decision logged for GEN #{seq}: " + self.log_tail())

    def rss_kib(self) -> int:
        for line in Path(f"/proc/{self.p.pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        raise AssertionError("no VmRSS")

    def fds(self) -> int:
        return len(os.listdir(f"/proc/{self.p.pid}/fd"))

    def quiet(self, seconds=0.3):
        try:
            line = self.q.get(timeout=seconds)
        except queue.Empty:
            return
        raise AssertionError(f"unexpected line {line!r}")

    def close(self, timeout=30) -> int:
        rc = self.p.poll()
        if rc is None:
            try:
                self.send("QUIT")
                self.p.stdin.close()
            except OSError:
                pass
            rc = self.p.wait(timeout)
        left = []
        while True:
            try:
                line = self.q.get(timeout=5)
            except queue.Empty:
                break
            if line is None:
                break
            left.append(line)
        self.logf.close()
        self.p.stdout.close()
        if self.bad:
            raise AssertionError(f"non-protocol lines on stdout: {self.bad[:5]}")
        if left:
            raise AssertionError(f"lines after the last request: {left[:5]}")
        shutil.rmtree(self.dir, ignore_errors=True)
        return rc


# ------------------------------------------------------------------------------------------------ the spec model

class Spec:
    """DESIGN.md section 4, in Python: which RESUME each request must get, and what the engine holds afterwards.
    Mutation runs set DS4_TEST_SPEC_K / DS4_TEST_SPEC_KEEP_ABOVE=1 / DS4_TEST_SPEC_KEEP_RESET=1 so the spec makes the
    mutated engine's choice and the logits comparison (not a decision mismatch) has to catch the mutant."""
    K = int(os.environ.get("DS4_TEST_SPEC_K", "8"))                       # Ds4Dense::max_tokens() (4 before M7c)
    KEEP_ABOVE = os.environ.get("DS4_TEST_SPEC_KEEP_ABOVE") == "1"        # mutation: checkpoints above R are kept
    KEEP_RESET = os.environ.get("DS4_TEST_SPEC_KEEP_RESET") == "1"        # mutation: a reset keeps the checkpoints

    def __init__(self, turn_token: int = -1, ckpt: bool = True):
        self.live: list[int] = []
        self.ck: dict[str, list[int]] = {}
        self.slots = (["last", "turn"] + (["user"] if turn_token >= 0 else [])) if ckpt else []
        self.turn = turn_token

    def choose(self, ids):
        n, L = len(ids), len(self.live)
        lcp = 0
        while lcp < L and lcp < n and self.live[lcp] == ids[lcp]:
            lcp += 1
        best = ["reset", 0, None]

        def offer(kind, R, slot):
            if R > best[1]:
                best[:] = [kind, R, slot]

        if L >= 1 and lcp == L and L <= n - 1:
            offer("append", L, None)
        m = min(lcp, n - 1)
        if m >= 1 and 1 <= L - m <= self.K:
            offer("rewind", m, None)
        for s in self.slots:
            c = self.ck.get(s)
            if c and 1 <= len(c) <= n - 1 and ids[:len(c)] == c:
                offer("checkpoint", len(c), s)
        return tuple(best)

    def update(self, ids, choice, res: Res, ckpt: bool = True):
        kind, R, slot = choice
        n = len(ids)
        if kind == "rewind":
            self.live = self.live[:R]
        elif kind == "checkpoint":
            self.live = list(self.ck[slot])
        elif kind == "reset":
            self.live = []
            if not self.KEEP_RESET:
                self.ck = {}
        if not self.KEEP_ABOVE:
            self.ck = {s: c for s, c in self.ck.items() if len(c) <= R}
        reached = R + res.prompt_read if res.finish == "cancel" else n
        ckpt = ckpt and bool(self.slots)
        user_at = -1
        if ckpt and self.turn >= 0:
            for i in range(n - 2, R, -1):
                if ids[i] == self.turn:
                    user_at = i
                    break
        cslot = "last" if kind in ("append", "rewind") else "turn"
        for s in ([user_at] if user_at > R else []) + [n - 1]:
            if ckpt and s >= 1 and reached >= s:
                self.ck["user" if s == user_at else cslot] = list(ids[:s])
        if res.finish == "cancel" and reached < n:
            self.live = list(ids[:reached])
        else:
            self.live = list(ids) + (res.tokens if res.finish == "cancel" else res.tokens[:-1])


# ------------------------------------------------------------------------------------------------ shared checks

class Case(unittest.TestCase):
    maxDiff = 2000

    def same(self, r: Res, want: tuple[list[int], bytes], what=""):
        toks, rows = want
        self.assertIsNone(r.err, f"{what}: {r.err}")
        self.assertTrue(rows, "empty reference")
        if r.tokens != toks or r.rows != rows:
            k = next((i for i in range(min(len(r.tokens), len(toks)))
                      if r.tokens[i] != toks[i] or r.rows[i * V * 4:(i + 1) * V * 4] != rows[i * V * 4:(i + 1) * V * 4]),
                     min(len(r.tokens), len(toks)))
            self.fail(f"{what}: differs from fresh at token {k}: got {r.tokens[:k + 3]} want {toks[:k + 3]} "
                      f"(rows {len(r.rows)} vs {len(rows)} bytes)")

    def step(self, eng: Serve, spec: Spec, ids, max_new, keys="", ckpt=True, compare=True, stop=(), what=""):
        """One request: RESUME and the logged decision == the spec, the output == a fresh ds4_generate_cpu."""
        choice = spec.choose(ids)
        r = eng.gen(ids, max_new, keys)
        self.assertIsNone(r.err, f"{what}: {r.err}")
        kind, slot, R = eng.decision(eng.seq)
        self.assertEqual((r.resume, kind, slot), (choice[1], choice[0], choice[2]), f"{what}: decision vs spec")
        self.assertEqual(r.reused, r.resume)
        if r.finish != "cancel":
            self.assertEqual(r.prompt_read, len(ids) - r.resume)
            steps = r.gen - 1   # decode steps: the last token is never fed
            self.assertEqual((int(r.done[9]), int(r.done[10]), int(r.done[12])), (0, 4 * steps, 4 * steps),
                             f"{what}: DONE hits/lookups/file_tier are not this request's")
        if compare:
            self.same(r, ref(eng.fx, ids, max_new, stop=stop), what or f"GEN #{eng.seq}")
        spec.update(ids, choice, r, ckpt)
        return r


# ------------------------------------------------------------------------------------------------ T1

class T1Parity(Case):
    """Criterion 1: a fresh ds4_serve gives ds4_generate_cpu's tokens and logits rows, bit for bit."""

    def run_fresh(self, fx, ids, max_new, *args):
        eng = Serve(fx, *args)
        try:
            r = eng.gen(ids, max_new)
            self.assertEqual(r.resume, 0)
            self.assertEqual(r.finish, "length")
            self.assertEqual(r.prompt_read, len(ids))
            self.assertEqual(r.pp[-1][:2], (len(ids), len(ids)))
            self.same(r, ref(fx, ids, max_new), f"{fx} n={len(ids)} {args}")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_premise_chunk_equals_loop(self):
        """ds4_generate's own chunked prefill == its decode loop under DS4_CPU_FA_REF (FINDINGS s20) - the premise."""
        ids = prompt(150, 1)
        self.assertEqual(ref("tame", ids, 8, chunk=64), ref("tame", ids, 8))
        self.assertEqual(ref("tamex", ids, 8, chunk=33), ref("tamex", ids, 8))

    def test_decode_loop(self):
        for fx in ("tame", "swa16", "tamex"):
            for n in (1, 8, 40, 150, 600):
                with self.subTest(fx=fx, n=n):
                    self.run_fresh(fx, prompt(n, 100 + n), 24)
        for max_new in (1, 64):
            with self.subTest(max_new=max_new):
                self.run_fresh("tame", prompt(40, 7), max_new)

    def test_chunks(self):
        for fx in ("tame", "swa16", "tamex"):
            for chunk in (1, 7, 16, 33, 64):
                for n in (8, 40, 150, 600):
                    if chunk == 1 and n > 150:
                        continue
                    with self.subTest(fx=fx, chunk=chunk, n=n):
                        self.run_fresh(fx, prompt(n, 100 + n), 24, "--prefill-chunk", chunk, "--chunk-min", 1)


# ------------------------------------------------------------------------------------------------ T2

class T2Reuse(Case):
    """Criterion 2: a request that reuses state == the same request run fresh; RESUME == the spec's choice."""

    def test_append_rewind_restore_h12(self):
        eng, spec = Serve("tame"), Spec()
        try:
            p1 = prompt(40, 11)
            self.step(eng, spec, p1, 24, what="first turn (reset)")
            live = spec.live
            self.step(eng, spec, live + [5, 9, 13], 12, what="append")
            for k in (1, 2, 3, 4):   # rewind k positions in place
                live = spec.live
                ids = live[:len(live) - k] + [other(live[len(live) - k])] + prompt(3, 20 + k)
                choice = spec.choose(ids)
                self.assertEqual(choice[:2], ("rewind", len(live) - k), f"drop {k}")
                self.step(eng, spec, ids, 10, what=f"rewind {k}")
            live = spec.live   # K + 1 positions back: beyond the rings' slack -> the newest checkpoint below it
            d = Spec.K + 1
            ids = live[:len(live) - d] + [other(live[len(live) - d])] + prompt(3, 30)
            r7 = ids
            self.step(eng, spec, ids, 24, what=f"drop {d} -> checkpoint")
            self.assertEqual(eng.decision(eng.seq)[0], "checkpoint")
            self.step(eng, spec, r7, 24, what="regenerate -> turn checkpoint at n-1")
            ids = r7[:len(r7) - 1] + [other(r7[-1])] + prompt(6, 31)   # H12: the template's </think> for <think>
            self.assertEqual(spec.choose(ids)[:2], ("checkpoint", len(r7) - 1))
            self.step(eng, spec, ids, 16, what="H12 next turn")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_user_turn_edit(self):
        """--turn-token: an edited last user message resumes at the last turn token."""
        sys_p = prompt(12, 40)
        p1 = sys_p + [TURN] + prompt(10, 41) + [TURN] + prompt(14, 42)   # turn tokens at 12 and 23
        for args, want in ((("--turn-token", TURN), ("checkpoint", 23)), ((), ("reset", 0))):
            with self.subTest(args=args):
                eng, spec = Serve("tame", *args), Spec(TURN if args else -1)
                try:
                    self.step(eng, spec, p1, 16, what="turn 1")
                    edited = p1[:30] + prompt(9, 43)   # the last user message changed after its 6th token
                    self.assertEqual(spec.choose(edited)[:2], want)
                    self.step(eng, spec, edited, 16, what="edited last message")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_reset_after_conversation(self):
        """A reset after >= 11 tokens: the CSA block 0's missing previous half must be SENTINEL again."""
        for fx in ("tame", "tamex"):
            with self.subTest(fx=fx):
                eng, spec = Serve(fx), Spec()
                try:
                    self.step(eng, spec, prompt(40, 50), 24, what="conversation A")
                    b = prompt(30, 51)
                    self.assertEqual(spec.choose(b)[0], "reset")
                    self.step(eng, spec, b, 24, what="unrelated B (reset)")
                    self.step(eng, spec, prompt(9, 52), 8, what="short C (reset)")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_aba(self):
        """A, a continuation of A (checkpoint `last`), unrelated B (reset), then A again: B's reset must have dropped
        A's checkpoints - restoring one would read B's compressed rows."""
        eng, spec = Serve("tame", "--prefill-chunk", 64, "--chunk-min", 8), Spec()
        try:
            a1 = prompt(300, 60)
            self.step(eng, spec, a1, 6, what="A1")
            a2 = spec.live + prompt(5, 61)
            self.step(eng, spec, a2, 6, what="A2 (append)")
            self.assertIn("last", spec.ck)
            self.step(eng, spec, prompt(310, 62), 6, what="B (reset)")
            a3 = a2 + prompt(1, 64) + prompt(4, 63)
            self.step(eng, spec, a3, 8, what="A3 after B")
            self.assertEqual(eng.decision(eng.seq)[0], "reset")   # after the logits: a mutant must fail on them
        finally:
            self.assertEqual(eng.close(), 0)

    def test_continuation_keeps_the_turn_checkpoint(self):
        """The DeepSeek turn pattern: turn 1 ends inside its thinking, a #1053 close-retry pass continues it (append),
        then turn 2 renders turn 1's `<think>` as `</think>`: it diverges at n1 - 1 and must resume there - from the
        `turn` checkpoint, which the continuation pass (role `last`) did not overwrite."""
        eng, spec = Serve("tame", "--prefill-chunk", 16, "--chunk-min", 4), Spec()
        try:
            p1 = prompt(70, 145)
            self.step(eng, spec, p1, 12, what="turn 1")
            retry = spec.live + prompt(3, 146)                 # + "\n</think>\n\n"
            self.step(eng, spec, retry, 12, what="#1053 continuation")
            self.step(eng, spec, spec.live + prompt(2, 147), 12, what="a second continuation")
            turn2 = p1[:69] + [other(p1[69])] + prompt(20, 148)
            self.assertEqual(spec.choose(turn2)[:3], ("checkpoint", 69, "turn"))
            self.step(eng, spec, turn2, 12, what="turn 2")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_stale_checkpoint_dropped(self):
        """A continuation leaves `last` above an edit's resume point; the edit rewrites those positions, so a later
        resend of the continuation must NOT restore `last` (its compressed rows now hold the edit's tokens)."""
        for args in ((), ("--prefill-chunk", 16, "--chunk-min", 1)):
            with self.subTest(args=args):
                eng, spec = Serve("tame", "--turn-token", TURN, *args), Spec(TURN)
                try:
                    p1 = prompt(40, 140) + [TURN] + prompt(19, 141)   # user turn at 40, n-1 = 59
                    self.step(eng, spec, p1, 10, what="turn 1")
                    cont = spec.live + prompt(5, 142)
                    self.step(eng, spec, cont, 10, what="continuation (checkpoint last)")
                    self.assertIn("last", spec.ck)
                    edit = p1[:45] + [other(p1[45])] + prompt(14, 143)
                    self.assertEqual(spec.choose(edit)[:2], ("checkpoint", 40))
                    self.step(eng, spec, edit, 10, what="edit (resumes at the user turn, below `last`)")
                    self.step(eng, spec, cont, 10, what="the continuation's prompt again")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_long_prompt_near_the_limit(self):
        """4000 tokens through 64-token chunks (the CSA cache's last blocks), then a continuation up to the limit."""
        eng, spec = Serve("tame", "--prefill-chunk", 64, "--chunk-min", 8), Spec()
        try:
            p = prompt(4000, 150)
            self.step(eng, spec, p, 40, what="4000 tokens")
            cont = spec.live + prompt(9, 151)
            self.assertEqual(len(cont) + 40 + 8, CTX)
            self.step(eng, spec, cont, 40, what="continuation to the context limit")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_ckpt0(self):
        eng, spec = Serve("tame"), Spec()
        try:
            p = prompt(30, 70)
            self.step(eng, spec, p, 12, keys="ckpt=0", ckpt=False, what="one-shot (no checkpoint)")
            self.assertEqual(spec.ck, {})
            self.step(eng, spec, p, 12, what="the same again: nothing to restore -> reset")
            cont = spec.live + prompt(4, 71)
            self.step(eng, spec, cont, 12, keys="ckpt=0", ckpt=False, what="continuation, ckpt=0")
            nxt = p[:29] + [other(p[29])] + prompt(5, 72)
            self.assertEqual(spec.choose(nxt)[:2], ("checkpoint", 29))
            self.step(eng, spec, nxt, 12, what="next turn from the turn checkpoint")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_continuations_after_stop_length_cancel(self):
        """#123/#1053-style passes: prompt + what the engine generated (+ extra), after each way a reply can end."""
        base = prompt(25, 80)
        stop_tok = ref("tame", base, 12)[0][4]   # the 5th token of the plain reply ends it here
        eng, spec = Serve("tame", "--stop", stop_tok), Spec()
        try:
            r = self.step(eng, spec, base, 12, stop=(stop_tok,), what="ends on the stop id")
            self.assertEqual(r.finish, "stop")
            self.assertEqual(r.tokens[-1], stop_tok)
            cont = base + r.tokens[:-1] + prompt(3, 81)   # the generated stop token is not part of the new prompt
            self.assertEqual(spec.choose(cont)[0], "append")
            r = self.step(eng, spec, cont, 6, stop=(stop_tok,), what="continuation after stop")
            # a length ending: the server continues with EVERY generated token (the last one was never fed); a stop
            # ending: without the stop token
            cont2 = cont + (r.tokens if r.finish == "length" else r.tokens[:-1]) + prompt(2, 82)
            self.assertEqual(spec.choose(cont2)[0], "append")
            self.step(eng, spec, cont2, 6, stop=(stop_tok,), what=f"continuation after {r.finish}")
            # a cancel after a few tokens: GEN and STOP in one write -> the read stops early; then the remainder
            long = spec.live + prompt(40, 83)
            choice = spec.choose(long)
            eng.send(eng.gen_line(long, 30) + "\nSTOP")
            eng.seq += 1
            r = eng.result()
            self.assertEqual(r.finish, "cancel")
            self.assertEqual(r.resume, choice[1])
            spec.update(long, choice, r)
            resend = long + prompt(2, 84)
            self.step(eng, spec, resend, 6, stop=(stop_tok,), what="resend after a cancelled read")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_chunked_resume_points(self):
        """Chunked reads from resume points that are not multiples of the chunk, mixed with decode-loop spans."""
        eng, spec = Serve("tame", "--prefill-chunk", 16, "--chunk-min", 5), Spec()
        try:
            p = prompt(37, 90)
            self.step(eng, spec, p, 9, what="37 tokens")
            nxt = p[:36] + [other(p[36])] + prompt(50, 91)   # R = 36, then 50 tokens: 16+16+16+2
            self.assertEqual(spec.choose(nxt)[:2], ("checkpoint", 36))
            self.step(eng, spec, nxt, 9, what="restore 36, chunks")
            self.step(eng, spec, spec.live + prompt(3, 92), 7, what="append 3 (decode loop)")
            self.step(eng, spec, spec.live + prompt(21, 93), 7, what="append 21 (chunk + loop)")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_hca_block_and_ring(self):
        """Around the HCA layer's first block (position 127 completes it) and its 131-slot ring."""
        for args in ((), ("--prefill-chunk", 33, "--chunk-min", 1)):
            with self.subTest(args=args):
                eng, spec = Serve("tame", *args), Spec()
                try:
                    p = prompt(130, 100)
                    self.step(eng, spec, p, 6, what="130 tokens")                  # live 135, turn ck at 129
                    self.step(eng, spec, p, 6, what="regenerate (restore at 129)")
                    live = spec.live                                              # 135
                    ids = live[:131] + [other(live[131])] + prompt(2, 101)        # rewind 4 to 131
                    self.assertEqual(spec.choose(ids)[:2], ("rewind", 131))
                    self.step(eng, spec, ids, 6, what="rewind to 131")
                    ids2 = p[:126] + [other(p[126])] + prompt(8, 102)            # before the block completes
                    self.step(eng, spec, ids2, 6, what="diverge at 126")
                    q = prompt(128, 103)
                    self.step(eng, spec, q, 5, what="128 tokens (block 0 completes inside the prompt)")
                    self.step(eng, spec, q[:127] + [other(q[127])] + prompt(3, 104), 6, what="restore at 127")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_swa16_wrap(self):
        """The 16-token window: the raw ring (16 + K - 1 slots) wraps inside the prompts; rewinds 1..4 and restores across
        it."""
        for args in ((), ("--prefill-chunk", 7, "--chunk-min", 1)):
            with self.subTest(args=args):
                eng, spec = Serve("swa16", *args), Spec()
                try:
                    p = prompt(23, 110)
                    self.step(eng, spec, p, 10, what="23 tokens")
                    for k in (1, 2, 3, 4):
                        live = spec.live
                        ids = live[:len(live) - k] + [other(live[len(live) - k])] + prompt(4, 111 + k)
                        self.assertEqual(spec.choose(ids)[0], "rewind")
                        self.step(eng, spec, ids, 8, what=f"rewind {k}")
                    live = spec.live
                    d = Spec.K + 2
                    ids = live[:len(live) - d] + [other(live[len(live) - d])] + prompt(20, 116)
                    self.assertEqual(spec.choose(ids)[0], "checkpoint")
                    self.step(eng, spec, ids, 8, what="restore across the wrap")
                    self.step(eng, spec, prompt(40, 117), 8, what="reset")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_tamex_reuse(self):
        """The same reuse paths with routed experts that reach the logits (decode run + chunk run_chunk)."""
        eng, spec = Serve("tamex", "--prefill-chunk", 16, "--chunk-min", 4, "--turn-token", TURN), Spec(TURN)
        try:
            p = prompt(50, 120) + [TURN] + prompt(30, 121)
            self.step(eng, spec, p, 10, what="turn 1")
            self.step(eng, spec, spec.live + prompt(20, 122), 10, what="append")
            live = spec.live
            self.step(eng, spec, live[:len(live) - 3] + prompt(5, 123), 10, what="rewind 3")
            self.assertEqual(spec.choose(p[:80] + [other(p[80])] + prompt(9, 124))[:2], ("checkpoint", 80))
            self.step(eng, spec, p[:80] + [other(p[80])] + prompt(9, 124), 10, what="H12 restore")
            self.step(eng, spec, p[:60] + prompt(12, 125), 10, what="edit after the turn token")
            self.step(eng, spec, prompt(70, 126), 10, what="reset")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_sampled_continued_equals_fresh(self):
        """A seeded sampled request draws Philox(seed, position): continued == fresh, and two fresh runs agree."""
        keys = ("temperature=0.8 top_k=0 top_p=0.9 seed=1234",
                "temperature=1.3 top_k=20 min_p=0.05 penalty_last_n=64 penalty_repeat=1.3 penalty_present=0.5 seed=99")
        for k in keys:
            with self.subTest(keys=k):
                p = prompt(40, 130)
                eng = Serve("tame")
                try:
                    r1 = eng.gen(p, 16)
                    cont = p + r1.tokens[:-1] + prompt(3, 131)
                    r2 = eng.gen(cont, 24, k)
                    self.assertEqual(r2.resume, len(p) + 15)
                finally:
                    self.assertEqual(eng.close(), 0)
                fresh = []
                for _ in range(2):
                    f = Serve("tame")
                    try:
                        fresh.append(f.gen(cont, 24, k))
                    finally:
                        self.assertEqual(f.close(), 0)
                self.assertEqual(fresh[0].resume, 0)
                self.assertEqual((fresh[0].tokens, fresh[0].rows), (fresh[1].tokens, fresh[1].rows))
                self.assertEqual((r2.tokens, r2.rows), (fresh[0].tokens, fresh[0].rows))
                greedy = ref("tame", cont, 24)[0]
                self.assertNotEqual(r2.tokens, greedy, "the sampled request decoded greedily")


# ------------------------------------------------------------------------------------------------ T3

class T3Twenty(Case):
    """Criterion 3: 20 requests in one process, no reload: each == fresh, the same PID, flat RSS, same open files."""

    def test_twenty(self):
        eng, spec = Serve("tamex", "--prefill-chunk", 16, "--chunk-min", 8, "--turn-token", TURN), Spec(TURN)
        a, b, c = prompt(60, 200) + [TURN] + prompt(20, 201), prompt(90, 202), prompt(33, 203) + [TURN] + prompt(5, 204)
        pid, rss, fds = eng.p.pid, {}, {}
        try:
            n = 0

            def go(ids, max_new, what, **kw):
                nonlocal n
                r = self.step(eng, spec, ids, max_new, what=f"#{n + 1} {what}", **kw)
                n += 1
                if n in (5, 20):
                    rss[n], fds[n] = eng.rss_kib(), eng.fds()
                return r

            go(a, 12, "A1")
            go(spec.live + prompt(4, 205), 8, "A1 continuation")
            go(b, 10, "B1 (reset)")
            go(spec.live + prompt(12, 206), 10, "B2")
            go(c, 9, "C1 (reset)")
            go(c[:38] + [other(c[38])] + prompt(6, 207), 9, "C2 (H12)")
            live = spec.live
            go(live[:len(live) - 2] + prompt(3, 208), 9, "C3 rewind 2")
            go(a, 12, "A again (reset)")
            go(a[:80] + [other(a[80])] + prompt(10, 209), 12, "A next turn")
            go(a[:66] + prompt(15, 210), 8, "A edited last message")
            go(b + prompt(2, 211), 10, "B resend (reset)")
            # a cancelled read: GEN + STOP in one write, then the resend continues where the read stopped
            long = b + prompt(150, 212)
            choice = spec.choose(long)
            eng.send(eng.gen_line(long, 20) + "\nSTOP")
            eng.seq += 1
            r = eng.result()
            self.assertEqual((r.finish, r.resume), ("cancel", choice[1]))
            spec.update(long, choice, r)
            n += 1
            go(long, 10, "the cancelled prompt again")
            go(spec.live + prompt(30, 213), 10, "B continuation")
            go(c, 9, "C (reset)")
            live = spec.live
            go(live[:len(live) - 4] + prompt(2, 214), 9, "C rewind 4")
            go(c + prompt(9, 215), 9, "C variant (regen-like)")
            go(a + prompt(5, 216), 12, "A (reset)")
            go(spec.live + prompt(3, 217), 12, "A continuation")
            self.assertEqual(n, 19)
            go(prompt(200, 218), 14, "D long (reset)")
            self.assertEqual(n, 20)
            self.assertEqual(eng.p.pid, pid)
            self.assertIsNone(eng.p.poll())
            print(f"\n20 requests: pid {pid}, RSS after #5 / #20 = {rss[5]} / {rss[20]} KiB, fds {fds}", file=sys.stderr)
            self.assertLessEqual(rss[20], rss[5] * 1.02 + 8 * 1024, f"RSS grew: {rss}")
            self.assertEqual(fds[20], fds[5], f"open files: {fds}")
        finally:
            self.assertEqual(eng.close(), 0)


# ------------------------------------------------------------------------------------------------ T4

class T4Stop(Case):
    """Criterion 4: STOP ends a request in < 1 s (decode and prompt read), a stale STOP is ignored, QUIT/EOF exit fast."""

    def test_stop_during_decode(self):
        eng = Serve("tame", env={"DS4_SERVE_TEST_DELAY_MS": "20"})
        try:
            p = prompt(30, 300)
            seen = []

            def on(line, r):
                if line.startswith("T ") and len(r.tokens) == 5 and not seen:
                    seen.append(time.monotonic())
                    eng.send("STOP")
            eng.send(eng.gen_line(p, 400))
            eng.seq += 1
            r = eng.result(on_line=on)
            self.assertEqual(r.finish, "cancel")
            self.assertLess(r.t_done - seen[0], 1.0)
            print(f"\nSTOP during decode -> DONE in {1000 * (r.t_done - seen[0]):.1f} ms", file=sys.stderr)
            self.assertLess(r.gen, 400)
            self.assertEqual(r.prompt_read, len(p))
            # every emitted token was fed: the continuation appends to all of them
            spec = Spec()
            spec.live = p + r.tokens
            cont = p + r.tokens + prompt(3, 301)
            r2 = eng.gen(cont, 5)
            self.assertEqual(r2.resume, len(p) + len(r.tokens))
            self.same(r2, ref("tame", cont, 5), "after a cancelled decode")
        finally:
            self.assertEqual(eng.close(), 0)

    def _stop_in_prefill(self, args, delay, n, pp_before_stop):
        eng = Serve("tame", *args, env={"DS4_SERVE_TEST_DELAY_MS": str(delay)})
        try:
            p = prompt(n, 310 + n)
            sent = []

            def on(line, r):
                if line.startswith("PP ") and len(r.pp) == pp_before_stop and not sent:
                    sent.append(time.monotonic())
                    eng.send("STOP")
            eng.send(eng.gen_line(p, 8))
            eng.seq += 1
            r = eng.result(on_line=on)
            self.assertEqual(r.finish, "cancel")
            self.assertEqual(r.gen, 0)
            self.assertLess(r.t_done - sent[0], 1.0)
            print(f"\nSTOP during the prompt read {args or '(decode loop)'} -> DONE in "
                  f"{1000 * (r.t_done - sent[0]):.1f} ms, {r.prompt_read} of {n} read", file=sys.stderr)
            self.assertLess(r.prompt_read, n - r.resume)
            self.assertGreater(r.prompt_read, 0)
            if args:   # chunks: the stop is seen right after the chunk the last PP reported
                self.assertEqual(r.prompt_read, r.pp[-1][0] - r.resume)
            else:      # the decode loop reads on until the next token's check
                self.assertGreaterEqual(r.prompt_read, r.pp[-1][0] - r.resume)
            r2 = eng.gen(p, 8)   # the same prompt again: it continues where the read stopped
            self.assertEqual(r2.resume, r.prompt_read)
            self.same(r2, ref("tame", p, 8), "after a cancelled read")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_stop_during_chunked_prefill(self):
        self._stop_in_prefill(("--prefill-chunk", 16, "--chunk-min", 1), 100, 600, 3)

    def test_stop_during_decode_loop_prefill(self):
        self._stop_in_prefill((), 8, 300, 1)

    def test_gen_and_stop_in_one_write(self):
        eng = Serve("tame")
        try:
            p = prompt(50, 320)
            eng.send(eng.gen_line(p, 100) + "\nSTOP")
            eng.seq += 1
            r = eng.result()
            self.assertEqual((r.resume, r.finish), (0, "cancel"))
            self.assertLess(r.gen, 100)
            # the engine is in step: a normal request after it
            r2 = eng.gen(p, 6)
            self.assertEqual(r2.finish, "length")
            self.same(r2, ref("tame", p, 6))
        finally:
            self.assertEqual(eng.close(), 0)

    def test_stale_and_idle_stop(self):
        eng = Serve("tame")
        try:
            p = prompt(20, 330)
            r = eng.gen(p, 8)
            self.assertEqual(r.finish, "length")
            eng.send("STOP")          # what the server sends after a reply that ended on a stop id: stale
            eng.quiet(0.3)            # nothing printed for it
            r = eng.gen(prompt(25, 331), 8)
            self.assertEqual((r.finish, r.gen), ("length", 8))
            eng.send("STOP")
            eng.send("STOP")
            eng.quiet(0.3)
            r = eng.gen(prompt(25, 332), 8)
            self.assertEqual((r.finish, r.gen), ("length", 8))
            self.same(r, ref("tame", prompt(25, 332), 8))
        finally:
            self.assertEqual(eng.close(), 0)

    def test_stop_storm(self):
        """60 requests with STOPs at random moments - none, at once, inside the prompt read, inside the decode, after
        DONE (stale) - over 3 conversations and their continuations: every request is one terminal line, RESUME and the
        decision follow the spec, and every output (a cancelled one too) is the fresh run's, or a prefix of it."""
        rng = random.Random(20261010)
        eng, spec = Serve("tame", "--prefill-chunk", 16, "--chunk-min", 4,
                          env={"DS4_SERVE_TEST_DELAY_MS": "2"}), Spec()
        stats = {"cancel": 0, "stop": 0, "length": 0}
        try:
            convs = [prompt(rng.randrange(20, 160), 900 + i) for i in range(3)]
            for i in range(60):
                base = spec.live if spec.live and rng.random() < 0.5 else rng.choice(convs)
                cut = rng.randrange(max(1, len(base) - 6), len(base) + 1)
                ids = base[:cut] + prompt(rng.randrange(1, 40), 1000 + i)
                max_new = rng.randrange(1, 30)
                choice = spec.choose(ids)
                when = rng.choice([None, None, 0.0, 0.002, 0.01, 0.03, 0.08])
                eng.send(eng.gen_line(ids, max_new) + ("\nSTOP" if when == 0.0 else ""))
                eng.seq += 1
                if when:
                    time.sleep(when)
                    eng.send("STOP")
                r = eng.result()
                self.assertIsNone(r.err, f"#{i}")
                kind, slot, R = eng.decision(eng.seq)
                self.assertEqual((r.resume, kind, slot), (choice[1], choice[0], choice[2]), f"#{i} decision")
                toks, rows = ref("tame", ids, max_new)
                if r.finish == "cancel":
                    self.assertEqual(r.tokens, toks[:len(r.tokens)], f"#{i} cancelled tokens")
                    self.assertEqual(r.rows, rows[:len(r.rows)], f"#{i} cancelled rows")
                else:
                    self.assertEqual(r.tokens, toks, f"#{i}")
                    self.assertEqual(r.rows, rows, f"#{i} rows")
                stats[r.finish] += 1
                spec.update(ids, choice, r)
                if rng.random() < 0.3:
                    eng.send("STOP")   # stale: after DONE, before the next GEN
            self.assertGreater(stats["cancel"], 5, stats)
            self.assertGreater(stats["length"], 5, stats)
            print(f"\nstop storm: {stats}", file=sys.stderr)
        finally:
            self.assertEqual(eng.close(), 0)

    def _exit_during(self, how, phase):
        args = ("--prefill-chunk", 16, "--chunk-min", 1) if phase == "prefill" else ()
        eng = Serve("tame", *args, env={"DS4_SERVE_TEST_DELAY_MS": "200" if phase == "prefill" else "30"})
        p = prompt(600 if phase == "prefill" else 20, 340)
        eng.send(eng.gen_line(p, 500))
        eng.seq += 1
        while True:
            line = eng.line(30)
            if line.startswith("PP " if phase == "prefill" else "T "):
                break
        t0 = time.monotonic()
        if how == "QUIT":
            eng.send("QUIT")
        eng.p.stdin.close()
        rc = eng.p.wait(10)
        dt = time.monotonic() - t0
        self.assertEqual(rc, 0)
        self.assertLess(dt, 2.0, f"{how} during {phase}: exit took {dt:.2f} s")
        print(f"\n{how} during {phase}: exit in {1000 * dt:.0f} ms", file=sys.stderr)
        eng.logf.close()
        self.assertFalse(eng.bad)
        shutil.rmtree(eng.dir, ignore_errors=True)

    def test_quit_and_eof_exit_fast(self):
        for how in ("QUIT", "EOF"):
            for phase in ("prefill", "decode"):
                with self.subTest(how=how, phase=phase):
                    self._exit_during(how, phase)

    def test_exit_during_load(self):
        """The server gave up while the model loaded (QUIT / its end of the pipe closed): exit without READY."""
        eng = Serve("tame", start=False)
        eng.send("QUIT")
        eng.p.stdin.close()
        self.assertEqual(eng.p.wait(30), 0)
        self.assertIsNone(eng.q.get(timeout=5))
        eng.logf.close()


# ------------------------------------------------------------------------------------------------ errors

class TErrors(Case):
    """Every refusal is one line, the engine stays up, and the live state is untouched (the next request reuses it)."""

    def test_refusals_keep_state(self):
        eng, spec = Serve("tame"), Spec()
        try:
            p = prompt(30, 400)
            self.step(eng, spec, p, 10, what="before")
            pid = eng.p.pid
            bad = ["GEN 0 1", "GEN 5", "GEN -3 1,2", "GEN abc 1,2", "GEN 5 1,,2,x", "GEN 5 1,64", "GEN 5 -1,2",
                   "GEN 5 temperature=nan 1,2", "GEN 5 top_p=inf 1,2", f"GEN {CTX - 8 - 2 + 1} 1,2",
                   "GEN 99999999999999999999999 1,2", "GENI 5 /tmp/x.emb 1,2", "VRAM", "VRAM 512", "SAVE /tmp/s",
                   "RESTORE /tmp/s", "BOGUS", "BGEN 0 5 1,2"]
            for line in bad:
                eng.send(line)
                if line.startswith("GEN"):
                    eng.seq += 1
                r = eng.result()
                self.assertIsNotNone(r.err, line)
                self.assertIsNone(r.resume, line)
                if line.startswith(("SAVE", "RESTORE")):
                    self.assertEqual(r.err, "SERR invalid 0 this engine keeps no disk sessions")
                else:
                    self.assertTrue(r.err.startswith("ERR "), r.err)
                    self.assertFalse(r.err.startswith("ERR verify"), r.err)
            texts = {"GEN 0 1": "ERR bad request: max_new", "GEN 5": "ERR bad request: token list was empty",
                     "GEN 5 1,64": "ERR a token id is outside the vocabulary",
                     f"GEN {CTX - 8 - 2 + 1} 1,2": f"ERR prompt (2 tokens) + max_new ({CTX - 9}) exceeds the context ({CTX})",
                     "GENI 5 /tmp/x.emb 1,2": "ERR this engine was started without --vision",
                     "VRAM": "ERR this engine was started without --vram-elastic"}
            for line, want in texts.items():
                eng.send(line)
                if line.startswith("GEN"):
                    eng.seq += 1
                self.assertEqual(eng.result().err, want)
            eng.send("")                      # a blank line: nothing
            eng.quiet(0.3)
            self.assertEqual(eng.p.pid, pid)
            # the refusals left the live state alone: the conversation continues (append) and matches fresh
            cont = spec.live + prompt(4, 401)
            self.assertEqual(spec.choose(cont)[:2], ("append", len(p) + 9))
            self.step(eng, spec, cont, 6, what="append after the refusals")
            # accepted edge cases: 1,,2 is two ids (upstream's parser), a 30-digit seed saturates, unknown keys skip
            r = eng.gen([1, 2], 3, "seed=123456789012345678901234567890 pin=1 cvec=0 spec_min_p=0.5 foo=bar")
            self.assertEqual(r.finish, "length")
            r = eng.gen([1, 2], CTX - 8 - 2)   # exactly at the context limit: accepted, every token generated
            self.assertIsNone(r.err)
            self.assertEqual(r.gen, CTX - 10)
            self.assertEqual(r.finish, "length")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_engine_failure_resets(self):
        """An engine error mid-request (DS4_SERVE_TEST_FAIL=<pos>): one ERR line (after any T lines), the state is reset
        and the next request starts over and matches fresh."""
        for args in ((), ("--prefill-chunk", 16, "--chunk-min", 1)):
            with self.subTest(args=args):
                eng, spec = Serve("tame", *args, env={"DS4_SERVE_TEST_FAIL": "50"}), Spec()
                try:
                    p = prompt(30, 410)
                    self.step(eng, spec, p, 10, what="below the failing position")
                    r = eng.gen(spec.live + prompt(20, 411), 5)   # the prompt read reaches position 50
                    self.assertEqual(r.err, "ERR prefill failed at 50: test failure (DS4_SERVE_TEST_FAIL)"
                                     if not args else "ERR prefill failed: test failure (DS4_SERVE_TEST_FAIL)")
                    self.assertEqual(r.tokens, [])
                    spec.live, spec.ck = [], {}
                    self.step(eng, spec, p, 10, what="after a prefill failure: starts over")
                    q = prompt(40, 412)
                    r = eng.gen(q, 20)                           # decode reaches position 50 after 10 tokens
                    self.assertEqual(r.err, "ERR decode failed at pos 50: test failure (DS4_SERVE_TEST_FAIL)")
                    self.assertEqual(len(r.tokens), 11)
                    spec.live, spec.ck = [], {}
                    self.assertEqual(spec.choose(q)[0], "reset")
                    self.step(eng, spec, q, 10, what="after a decode failure: starts over")
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_context_clamp_and_flags(self):
        e = Serve("tame", "--ctx", 100000)   # the last --ctx wins: 100000, clamped to the model's 4096
        try:
            self.assertEqual(e.ctx, CTX)
            self.assertEqual(e.info.get("context"), CTX)
            self.assertEqual(e.info.get("engine"), "ds4-serve-0.3")
            self.assertTrue(e.can_stop)
            self.assertNotIn("batch_slots", e.info)
        finally:
            self.assertEqual(e.close(), 0)
        # --vram-reserve-mib (serve/server.py's Settings page appends it): the margin, or ignored beside --vram-margin
        for args, want in ((("--vram-reserve-mib", 700), "--vram-reserve-mib 700 -> --vram-margin 0.68 GiB"),
                           (("--vram-reserve-mib", 700, "--vram-margin", 1.5),
                            "--vram-reserve-mib 700 ignored: --vram-margin 1.50 GiB is given"),
                           (("--vram-margin", 1.5, "--vram-reserve-mib", 700),
                            "--vram-reserve-mib 700 ignored: --vram-margin 1.50 GiB is given")):
            with self.subTest(args=args):
                e = Serve("tame", *args)
                try:
                    self.assertEqual(e.ctx, CTX)
                    self.assertIn(want, e.log_text())
                    self.same(e.gen(prompt(12, 420), 4), ref("tame", prompt(12, 420), 4))
                finally:
                    self.assertEqual(e.close(), 0)
        for args, want in ((("--slots", "auto"), "--expert-slots"), (("--mtp-resident",), "need --mtp FILE"),
                           (("--mtp-keep", "3"), "need --mtp FILE"), (("--temp", "0.7"), "greedy"), (("--ids", "1,2"), "one-shot"),
                           (("--bogus",), "unknown argument --bogus"), (("--prefill-chunk", "9000"), "--prefill-chunk")):
            with self.subTest(args=args):
                out = subprocess.run([SERVE, "--serve", "-m", str(fixture("tame")), "--backend", "cpu", "--experts",
                                      "cpu", "--ctx", "512", *args], env=ENV, stdin=subprocess.DEVNULL,
                                     capture_output=True, text=True, timeout=60)
                self.assertEqual(out.returncode, 2)
                self.assertEqual(out.stdout, "")
                self.assertIn(want, out.stderr)


# ------------------------------------------------------------------------------------------------ T5

class Tok64:
    """A test tokenizer inside the mini model's 64 ids: printable ASCII 32..89 <-> ids 0..57, the six specials of the
    server's ByteTokenizer at 58..63 (so the server's stop ids, im_end and endoftext, are 59 and 60)."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    @property
    def control_tokens(self):
        return list(self.SPECIALS)

    def encode(self, text, parse_special=False, plain=()):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i) and not any(a <= i < b for a, b in plain):
                    out.append(58 + k)
                    i += len(s)
                    break
            else:
                c = ord(text[i])
                out.append(c - 32 if 32 <= c < 90 else c % 58)
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        return "".join(self.SPECIALS[t - 58] if t >= 58 else chr(32 + t) for t in ids)


class T5Server(Case):
    """Criterion 5: serve/server.py's own StrataEngine class drives ds4_serve: handshake, generate, cancel and drain,
    refusals that keep the engine, close() in < 20 s; then the HTTP server on top of it."""

    def engine(self, *args, env=None):
        from serve.server import StrataEngine
        self.tmp = tempfile.mkdtemp(prefix="ds4srv-")
        log = os.path.join(self.tmp, "engine.log")
        e = dict(ENV, **(env or {}))
        eng = StrataEngine(SERVE, ["-m", str(fixture("tame")), "--backend", "cpu", "--experts", "cpu", "--threads",
                                   THREADS, "--max-context", str(CTX), *map(str, args)], log=log, env=e)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        return eng

    def collect(self, eng, ids, max_new, sampling=None, cancel=None):
        return [t for t in eng.generate(ids, max_new, sampling or {}, cancel or threading.Event()) if t is not None]

    def test_strata_engine(self):
        from serve.server import SessionRefused
        eng = self.engine()
        try:
            self.assertEqual(eng.max_context, CTX)
            self.assertTrue(eng.can_stop)
            self.assertEqual(eng.info["engine"], "ds4-serve-0.3")
            self.assertEqual(eng.info["version"], "ds4-serve-0.3")
            self.assertEqual(eng.batch, 0)
            pid = eng.proc.pid
            p = prompt(40, 500)
            toks = self.collect(eng, p, 12)
            self.assertEqual(toks, ref("tame", p, 12)[0])
            self.assertEqual((eng.last["generated"], eng.last["prompt_tokens"], eng.last["finish"], eng.last["reused"],
                              eng.last["prompt_read"]), (12, 40, "length", 0, 40))
            cont = p + toks[:-1] + prompt(3, 501)
            toks2 = self.collect(eng, cont, 6)
            self.assertEqual(toks2, ref("tame", cont, 6)[0])
            self.assertEqual(eng.last["reused"], 51)
            self.assertEqual(eng.reused, 51)
            # refusals: ValueError, the engine stays and keeps step
            with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
                self.collect(eng, [1, 64], 4)
            with self.assertRaisesRegex(ValueError, "exceeds the context"):
                self.collect(eng, [1, 2], CTX)
            with self.assertRaisesRegex(ValueError, "vram-elastic"):
                eng.vram(512)
            with self.assertRaises(SessionRefused) as cm:
                eng.session_file("save", os.path.join(self.tmp, "s.bin"))
            self.assertEqual(cm.exception.kind, "invalid")
            self.assertTrue(eng.alive())
            self.assertEqual(eng.proc.pid, pid)
            # a consumer that leaves early: the generator's finally sends STOP and drains to DONE
            gen = eng.generate(prompt(30, 502), 2000, {}, threading.Event())
            got = []
            while len(got) < 3:
                t = next(gen)
                if t is not None:
                    got.append(t)
            gen.close()
            self.assertEqual(eng.last["finish"], "cancel")
            self.assertTrue(all(isinstance(t, int) for t in got))
            toks3 = self.collect(eng, prompt(25, 503), 5)
            self.assertEqual(toks3, ref("tame", prompt(25, 503), 5)[0])
            self.assertEqual(eng.proc.pid, pid)
        finally:
            t0 = time.monotonic()
            eng.close()
            self.assertLess(time.monotonic() - t0, 20)
            self.assertIsNone(eng.proc)

    def test_cancel_event_during_prompt_read(self):
        eng = self.engine("--prefill-chunk", 16, "--chunk-min", 1, env={"DS4_SERVE_TEST_DELAY_MS": "100"})
        try:
            cancel = threading.Event()
            threading.Timer(0.5, cancel.set).start()
            t0 = time.monotonic()
            out = self.collect(eng, prompt(600, 510), 8, cancel=cancel)   # returns once cancelled
            self.assertEqual(out, [])
            self.assertLess(time.monotonic() - t0, 3.0)
            # the next request drains nothing stale and works (the generator's finally sent STOP and read DONE)
            self.assertEqual(eng.last["finish"], "cancel")
            self.assertLess(eng.last["prompt_read"], 600)
            toks = self.collect(eng, prompt(20, 511), 4)
            self.assertEqual(toks, ref("tame", prompt(20, 511), 4)[0])
        finally:
            t0 = time.monotonic()
            eng.close()
            self.assertLess(time.monotonic() - t0, 20)

    def test_http(self):
        """Service + serve(port=0) over ds4_serve: two turns (cached_tokens == the engine's RESUME), streaming, and a
        client that hangs up mid-stream (the server sends STOP; the next request works)."""
        from serve.frontend import ChatTemplate
        from serve.server import Service, serve
        eng = self.engine("--turn-token", 58, env={"DS4_SERVE_TEST_DELAY_MS": "5"})
        svc = Service(eng, Tok64(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(body, stream=False, timeout=120):
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=timeout)
        try:
            msgs = [{"role": "system", "content": "SYSTEM PROMPT FOR THE MINI MODEL."},
                    {"role": "user", "content": "HELLO THERE, HOW ARE YOU?"}]
            with post({"model": "m", "max_tokens": 12, "messages": msgs}) as r:
                one = json.loads(r.read())
            reused1 = one["usage"]["prompt_tokens_details"]["cached_tokens"]
            self.assertEqual(reused1, eng.last["reused"])
            msg = one["choices"][0]["message"]
            msgs2 = msgs + [{"role": "assistant", "content": msg.get("content") or "OK",
                             "reasoning_content": msg.get("reasoning_content") or ""},
                            {"role": "user", "content": "AND NOW?"}]
            with post({"model": "m", "max_tokens": 8, "messages": msgs2}) as r:
                two = json.loads(r.read())
            reused2 = two["usage"]["prompt_tokens_details"]["cached_tokens"]
            self.assertEqual(reused2, eng.last["reused"])
            self.assertGreater(reused2, 0, "turn 2 reused nothing")
            # streaming
            with post({"model": "m", "max_tokens": 6, "stream": True, "messages": msgs2}) as r:
                events = [line for line in r.read().decode().splitlines() if line.startswith("data: ")]
            self.assertEqual(events[-1], "data: [DONE]")
            # a client that hangs up mid-stream
            pid = eng.proc.pid
            r = post({"model": "m", "max_tokens": 3000, "stream": True, "messages": msgs2 + [
                {"role": "assistant", "content": "X"}, {"role": "user", "content": "A LONG ONE PLEASE"}]})
            r.readline()
            r.readline()
            r.close()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and (eng.last or {}).get("finish") != "cancel":
                time.sleep(0.1)
            self.assertEqual(eng.last.get("finish"), "cancel")
            with post({"model": "m", "max_tokens": 4, "messages": msgs}) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(eng.proc.pid, pid)
        finally:
            httpd.shutdown()
            httpd.server_close()
            eng.close()


# ------------------------------------------------------------------------------------------------ T6

# ------------------------------------------------------------------------------------------------ T7 (mab776 M6)

class T7Park(Case):
    """M6 conversation parking (--park-mib): a conversation the engine switched away from comes back from host RAM;
    the request then reads only what is new and equals the same request run fresh, bit for bit (the compressed rows
    past a parked position keep the other conversation's bytes: invisible, as after a reset)."""

    ARGS = ("--park-mib", 64, "--park-min", 16, "--prefill-chunk", 64, "--chunk-min", 8)
    PARKED = re.compile(r" parked (\d+) tokens \([^,]+, [0-9.]+ MiB = (\d+) B")

    def gen_log(self, eng, seq):
        for line in eng.log_text().splitlines():
            if f"GEN #{seq}:" in line:
                return line
        self.fail(f"no GEN #{seq} line: " + eng.log_tail())

    def turn(self, eng, ids, max_new, want_resume, what, unparked=None):
        r = eng.gen(ids, max_new)
        self.assertIsNone(r.err, what)
        self.assertEqual(r.resume, want_resume, f"{what}: RESUME ({self.gen_log(eng, eng.seq)})")
        self.assertEqual(r.prompt_read, len(ids) - r.resume, f"{what}: prompt read")
        if unparked is not None:
            self.assertEqual("unparked" in self.gen_log(eng, eng.seq), unparked, f"{what}: unpark")
        self.same(r, ref(eng.fx, ids, max_new), what)
        return r

    @staticmethod
    def next_turn(prev, seed, k=5):
        """the next turn of a conversation whose last prompt was `prev`: DeepSeek's template turns that prompt's last
        token (<think>) into </think> - it diverges at n - 1 -, then the answer and the new user message follow"""
        return prev[:-1] + [other(prev[-1])] + prompt(k, seed)

    def test_aba(self):
        for fx in ("tame", "tamex"):
            with self.subTest(fx=fx):
                eng = Serve(fx, *self.ARGS)
                try:
                    a1 = prompt(300, 70)
                    self.turn(eng, a1, 6, 0, "A1")
                    b1 = prompt(200, 71)
                    self.turn(eng, b1, 6, 0, "B1 (A parked)", unparked=False)
                    m = self.PARKED.search(self.gen_log(eng, eng.seq))
                    self.assertTrue(m, "A was not parked")
                    self.assertEqual(int(m.group(1)), len(a1) + 5)   # max_new 6: the last token is never fed
                    a2 = self.next_turn(a1, 72)
                    self.turn(eng, a2, 8, len(a1) - 1, "A2 (unparked: its turn checkpoint)", unparked=True)
                    b2 = self.next_turn(b1, 73)
                    self.turn(eng, b2, 8, len(b1) - 1, "B2 (unparked)", unparked=True)
                finally:
                    self.assertEqual(eng.close(), 0)

    def test_abca_and_append(self):
        """three parked conversations; a return that APPENDS to a parked live state (a continuation pass)"""
        eng = Serve("tame", *self.ARGS)
        try:
            a1, b1, c1 = prompt(250, 80), prompt(180, 81), prompt(120, 82)
            ra = self.turn(eng, a1, 6, 0, "A1")
            a_live = a1 + ra.tokens[:-1]
            self.turn(eng, b1, 6, 0, "B1")
            self.turn(eng, c1, 6, 0, "C1")
            self.turn(eng, a_live + prompt(3, 83), 8, len(a_live), "A continued (unparked, append)", unparked=True)
            self.turn(eng, self.next_turn(b1, 84), 8, len(b1) - 1, "B2", unparked=True)
            self.turn(eng, self.next_turn(c1, 85), 8, len(c1) - 1, "C2", unparked=True)
        finally:
            self.assertEqual(eng.close(), 0)

    def test_budget_evicts_oldest(self):
        """room for one parked conversation: B's park evicts A's, so A comes back as a reset (read again, still exact)
        while B comes back from RAM - taken out of the list before A2 is parked to make room (else A2's park would
        evict the very conversation the request is switching to).  C is too short to be parked (--park-min)."""
        a1, b1, c1 = prompt(260, 90), prompt(240, 91), prompt(9, 92)
        probe = Serve("tame", *self.ARGS)
        try:
            self.turn(probe, a1, 6, 0, "probe A1")
            self.turn(probe, b1, 6, 0, "probe B1")
            sa = int(self.PARKED.search(self.gen_log(probe, probe.seq)).group(2))
            self.turn(probe, c1, 6, 0, "probe C1")
            sb = int(self.PARKED.search(self.gen_log(probe, probe.seq)).group(2))
        finally:
            self.assertEqual(probe.close(), 0)
        eng = Serve("tame", *self.ARGS, env={"DS4_SERVE_TEST_PARK_BYTES": str(max(sa, sb) + min(sa, sb) // 2)})
        try:
            self.turn(eng, a1, 6, 0, "A1")
            self.turn(eng, b1, 6, 0, "B1 (A parked)")
            self.turn(eng, c1, 6, 0, "C1 (B parked, A evicted)")
            self.assertIn("evicting a parked conversation", eng.log_text())
            self.turn(eng, self.next_turn(a1, 93), 8, 0, "A2: evicted -> read again", unparked=False)
            self.turn(eng, self.next_turn(b1, 94), 8, len(b1) - 1, "B2: from RAM", unparked=True)
        finally:
            self.assertEqual(eng.close(), 0)

    def test_off_by_default(self):
        """without --park-mib nothing is parked: A after B is a reset (M5's behaviour)"""
        eng = Serve("tame", "--prefill-chunk", 64, "--chunk-min", 8)
        try:
            a1 = prompt(200, 95)
            self.turn(eng, a1, 6, 0, "A1")
            self.turn(eng, prompt(150, 96), 6, 0, "B1")
            self.turn(eng, self.next_turn(a1, 97), 8, 0, "A2: reset", unparked=False)
            self.assertNotIn(" parked ", eng.log_text().replace("parked 0 (", ""))
        finally:
            self.assertEqual(eng.close(), 0)


# ------------------------------------------------------------------------------------------------ T8 (mab776 M7b)

class T8Verify(Case):
    """M7b speculative decoding in serve mode (--mtp): greedy requests give plain greedy's tokens AND logits rows, bit
    for bit (on the CPU a 2-token pass is bit-identical to one-token decoding), whatever the drafts - the mini MTP's own
    (always rejected), an oracle's (always accepted) or a mix; through chunked prompts and reuse."""

    def eng(self, *args, **kw):
        return Serve("tame", "--mtp", str(fixture("mtp")), *args, **kw)

    def oracle_file(self, eng, toks):
        path = os.path.join(eng.dir, "oracle")
        Path(path).write_text("\n".join(map(str, toks)) + "\n")
        return path

    def run_case(self, ids, max_new, corrupt=None, args=(), stop=()):
        want = ref("tame", ids, max_new, stop=stop)
        env = {}
        if corrupt is not None:
            plain = ref("tame", ids, 40)[0]   # the drafts: plain greedy's tokens
            d = tempfile.mkdtemp(prefix="ds4oracle-")
            Path(d, "o").write_text("\n".join(map(str, plain)) + "\n")
            env = {"DS4_SERVE_TEST_ORACLE": os.path.join(d, "o"), "DS4_SERVE_TEST_CORRUPT": str(corrupt)}
        sargs = [*args]
        if stop:
            sargs += ["--stop", ",".join(map(str, stop))]
        eng = self.eng(*sargs, env=env)
        try:
            r = eng.gen(ids, max_new)
            self.same(r, want, f"n={len(ids)} max_new={max_new} corrupt={corrupt} {args} stop={stop}")
            return r
        finally:
            self.assertEqual(eng.close(), 0)

    def test_drafts(self):
        for i, ids in enumerate((prompt(8, 200), prompt(3, 201), prompt(40, 202))):
            with self.subTest(i=i, drafts="mtp"):
                r = self.run_case(ids, 40)
                self.assertGreater(int(r.done[7]), 0, "no draft offered")
            for c in (0, 3, 2):
                with self.subTest(i=i, corrupt=c):
                    r = self.run_case(ids, 40, corrupt=c)
                    if c == 0:   # every draft right: each pass emits 2
                        self.assertEqual(int(r.done[6]), int(r.done[7]), "an oracle draft was rejected")

    def test_max_new_cuts_and_stop(self):
        ids = prompt(8, 210)
        for n in (1, 2, 3, 7, 8):
            with self.subTest(max_new=n):
                self.run_case(ids, n, corrupt=0)
        plain = ref("tame", ids, 40)[0]
        for k in (5, 6, 9):
            for c in (0, 3):
                with self.subTest(stop_at=k, corrupt=c):
                    self.run_case(ids, 40, corrupt=c, stop=(plain[k - 1],))

    def test_chunked_prompt(self):
        for n in (150, 600):
            for c in (None, 0, 3):
                with self.subTest(n=n, corrupt=c):
                    self.run_case(prompt(n, 220 + n), 24, corrupt=c, args=("--prefill-chunk", 64, "--chunk-min", 8))

    def test_chain(self):
        """M7c: K chained drafts per pass (--mtp-k K; the pass carries up to K + 1 = 8 tokens).  Greedy == plain greedy,
        tokens and logits, whatever the drafts: the MTP's own (row 0 rejects, the chain is never examined), all right
        (every pass emits K + 1), or a rejection at each chain position in turn (corrupt every 2nd / 3rd / 5th)."""
        ids = prompt(8, 240)
        for k in (2, 3, 5, 7):
            for c in (None, 0, 2, 3, 5):
                with self.subTest(k=k, corrupt=c):
                    r = self.run_case(ids, 40, corrupt=c, args=("--mtp-k", k))
                    acc, drafted = int(r.done[6]), int(r.done[7])
                    self.assertEqual(drafted % k, 0, "drafts are offered K at a time")
                    if c == 0:   # every draft right: only the last pass may leave drafts unexamined (max_new)
                        self.assertLess(drafted - acc, k, "an oracle draft was rejected")

    def test_chain_cuts_chunks_and_key(self):
        """K drafts with max_new / stop cutting a pass at every chain position, through chunked prompts (the MTP window
        then runs 8-token passes), and a request's own mtp_k key over --mtp-k."""
        ids = prompt(8, 250)
        plain = ref("tame", ids, 40)[0]
        for k in (3, 6):
            for n in (1, 2, 3, 4, 5, 7, 9):
                with self.subTest(k=k, max_new=n):
                    self.run_case(ids, n, corrupt=0, args=("--mtp-k", k))
            for at in (5, 6, 9):
                for c in (0, 3):
                    with self.subTest(k=k, stop_at=at, corrupt=c):
                        self.run_case(ids, 40, corrupt=c, args=("--mtp-k", k), stop=(plain[at - 1],))
        for n in (150, 600):
            for c in (None, 0, 3):
                with self.subTest(n=n, corrupt=c):
                    self.run_case(prompt(n, 260 + n), 24, corrupt=c,
                                  args=("--mtp-k", 4, "--prefill-chunk", 64, "--chunk-min", 8))
        eng = self.eng()   # --mtp-k 1; requests ask for 4, the default, and none (plain decoding)
        try:
            for mk, want_k in (("mtp_k=4", 4), ("", 1), ("mtp_k=0", 0)):
                a = prompt(20, 270 + want_k)
                r = eng.gen(a, 16, mk)
                self.same(r, ref("tame", a, 16), f"key {mk!r}")
                last = [x for x in eng.log_text().splitlines() if "ds4_serve: prompt" in x][-1]
                if want_k == 0:
                    self.assertEqual(int(r.done[7]), 0, "mtp_k=0 drafted")
                    self.assertIn("drafts off", last)
                else:
                    self.assertEqual(int(r.done[7]) % want_k, 0)
                    self.assertIn(f"(K={want_k},", last, f"key {mk!r}: the log line's K")
        finally:
            self.assertEqual(eng.close(), 0)

    def test_reuse(self):
        """append and the next turn's checkpoint restore with the MTP head on: each request == the same one fresh"""
        eng = self.eng("--prefill-chunk", 64, "--chunk-min", 8)
        try:
            a1 = prompt(300, 230)
            r1 = eng.gen(a1, 12)
            self.same(r1, ref("tame", a1, 12), "A1")
            a2 = a1 + r1.tokens[:-1] + prompt(5, 231)   # a continuation: what the state may hold, then new tokens
            r2 = eng.gen(a2, 12)
            self.same(r2, ref("tame", a2, 12), "A2 (append)")
            a3 = a2[:-1] + [other(a2[-1])] + prompt(6, 232)   # the next turn: turn checkpoint at n - 1
            r3 = eng.gen(a3, 12)
            self.assertEqual(r3.resume, len(a2) - 1)
            self.same(r3, ref("tame", a3, 12), "A3 (turn checkpoint)")
        finally:
            self.assertEqual(eng.close(), 0)


def py_philox(seed: int, counter: int) -> float:
    """sampler.cu's philox_uniform, transcribed independently of ds4_sampler.hpp."""
    M = 0xFFFFFFFF
    c0, c1, c2, c3 = counter & M, (counter >> 32) & M, seed & M, (seed >> 32) & M
    for i in range(10):
        hi0, lo0 = ((0x9E3779B9 * c0) >> 32) & M, (0x9E3779B9 * c0) & M
        hi1, lo1 = ((0xBB67AE85 * c2) >> 32) & M, (0xBB67AE85 * c2) & M
        c0, c1, c2, c3 = hi1 ^ c1 ^ i, lo1, hi0 ^ c3 ^ 0, lo0
    return struct.unpack("f", struct.pack("f", (c0 >> 8) * (1.0 / 16777216.0)))[0]


class T6Sampler(Case):
    def test_cpp_checks(self):
        out = subprocess.run([SAMPLER], capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])

    def test_philox_transcription(self):
        cases = [(0, 0), (1, 0), (12345, 7), (2**63 + 12345, 2**40 + 3), (2**64 - 1, 2**64 - 1), (99, 4095)]
        for seed in {s for s, _ in cases}:
            counters = [c for s, c in cases if s == seed] + list(range(20))
            out = subprocess.run([SAMPLER, "--philox", str(seed), *map(str, counters)], capture_output=True, text=True,
                                 timeout=60)
            got = [float(x) for x in out.stdout.split()]
            want = [py_philox(seed, c) for c in counters]
            for c, g, w in zip(counters, got, want):
                self.assertAlmostEqual(g, w, places=8, msg=f"philox({seed}, {c})")


if __name__ == "__main__":
    unittest.main()
