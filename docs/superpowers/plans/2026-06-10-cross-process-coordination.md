# Cross-Process Scheduler Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend RequestScheduler's p1 (contention-retry) solo execution across OS processes via a two-flock gate+pool file lock, so the 8-worker rlm-trainer benchmark stops surfacing cross-process context-500 failures.

**Architecture:** New `CrossProcessGate` class (lm_repl/clients/coordination.py) wraps two flock files per server key. `RequestScheduler` calls `gate.enter(priority)` after local admission and `gate.exit(priority)` before local release. Plumbed opt-in via `scheduler_coordination_dir` through RLM -> LMHandler. Spec: docs/superpowers/specs/2026-06-10-cross-process-coordination-design.md (read it first).

**Tech Stack:** Python 3.14, fcntl.flock, asyncio (LOCK_NB poll loop for the async path), pytest with multiprocessing spawn-context children. Test command prefix: `.venv/bin/python -m pytest` from `/home/potto/src/lm-repl`.

**Conventions (from CLAUDE.md):** No em dashes anywhere. No Co-Authored-By trailers on commits. Never save working files to the repo root.

**Async test idiom in this suite:** sync `def test_...()` containing `async def main(): ...` then `asyncio.run(main())` (see tests/test_scheduler.py:385-393). Follow it; do not add pytest.mark.asyncio.

---

### Task 1: CrossProcessGate sync core

**Files:**
- Create: `lm_repl/clients/coordination.py`
- Create: `tests/test_coordination.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coordination.py`. Multiprocessing children MUST be module-level functions (spawn context pickles by qualified name). Events must come from the same context object.

```python
"""Tests for CrossProcessGate (two-flock gate+pool cross-process coordination)."""

import fcntl
import multiprocessing as mp
import os
import threading
import time

import pytest

from lm_repl.clients.coordination import CrossProcessGate
from lm_repl.clients.scheduler import Priority

KEY = "testkey0000000000"


# ---- module-level workers for spawn-context children ----

def _hold_pool_sh(dir_, key, acquired_evt, release_evt):
    gate = CrossProcessGate(dir_, key)
    gate.enter(Priority.NORMAL)
    acquired_evt.set()
    release_evt.wait(15)
    gate.exit(Priority.NORMAL)


def _hold_pool_sh_forever(dir_, key, acquired_evt):
    gate = CrossProcessGate(dir_, key)
    gate.enter(Priority.NORMAL)
    acquired_evt.set()
    time.sleep(60)


def _hold_gate_ex_raw(dir_, key, acquired_evt, release_evt):
    # Simulates a p1 that is WAITING (holds gate EX, not yet pool EX).
    fd = os.open(os.path.join(dir_, f"{key}.gate"), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    acquired_evt.set()
    release_evt.wait(15)
    os.close(fd)


# ---- same-process semantics ----

def test_normal_enters_are_shared(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.enter(Priority.NORMAL)
    gate.enter(Priority.NORMAL)  # second SH must not block
    gate.exit(Priority.NORMAL)
    gate.exit(Priority.NORMAL)


def test_p1_enter_exit_roundtrip(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.enter(Priority.CONTENTION_RETRY)
    gate.exit(Priority.CONTENTION_RETRY)
    # Reacquirable afterwards
    gate.enter(Priority.NORMAL)
    gate.exit(Priority.NORMAL)


def test_exit_without_enter_is_noop(tmp_path):
    gate = CrossProcessGate(tmp_path, KEY)
    gate.exit(Priority.NORMAL)
    gate.exit(Priority.CONTENTION_RETRY)


def test_unwritable_dir_raises(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(RuntimeError, match="coordination unavailable"):
            CrossProcessGate(ro / "locks", KEY)
    finally:
        os.chmod(ro, 0o700)


# ---- cross-process semantics ----

def test_p1_waits_for_other_process_share(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_pool_sh, args=(str(tmp_path), KEY, acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate = CrossProcessGate(tmp_path, KEY)
        entered = threading.Event()

        def p1():
            gate.enter(Priority.CONTENTION_RETRY)
            entered.set()

        threading.Thread(target=p1, daemon=True).start()
        time.sleep(0.3)
        assert not entered.is_set()  # blocked: child holds pool SH
        release.set()
        assert entered.wait(15)
        gate.exit(Priority.CONTENTION_RETRY)
    finally:
        release.set()
        child.join(15)
    assert child.exitcode == 0


def test_waiting_p1_blocks_new_normal_admissions(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_gate_ex_raw, args=(str(tmp_path), KEY, acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate = CrossProcessGate(tmp_path, KEY)
        entered = threading.Event()

        def normal():
            gate.enter(Priority.NORMAL)
            entered.set()

        threading.Thread(target=normal, daemon=True).start()
        time.sleep(0.3)
        assert not entered.is_set()  # blocked at the gate doorway
        release.set()
        assert entered.wait(15)
        gate.exit(Priority.NORMAL)
    finally:
        release.set()
        child.join(15)
    assert child.exitcode == 0


def test_crash_releases_locks(tmp_path):
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()
    child = ctx.Process(target=_hold_pool_sh_forever, args=(str(tmp_path), KEY, acquired))
    child.start()
    assert acquired.wait(15)
    child.kill()
    child.join(15)

    gate = CrossProcessGate(tmp_path, KEY)
    done = threading.Event()

    def p1():
        gate.enter(Priority.CONTENTION_RETRY)
        done.set()

    threading.Thread(target=p1, daemon=True).start()
    assert done.wait(15)  # the dead child's flock vanished with its fds
    gate.exit(Priority.CONTENTION_RETRY)


def test_different_server_keys_do_not_couple(tmp_path):
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_gate_ex_raw, args=(str(tmp_path), "keyaaaa", acquired, release))
    child.start()
    try:
        assert acquired.wait(15)
        gate_b = CrossProcessGate(tmp_path, "keybbbb")
        entered = threading.Event()

        def normal():
            gate_b.enter(Priority.NORMAL)
            entered.set()

        threading.Thread(target=normal, daemon=True).start()
        assert entered.wait(5)  # different key: no coupling
        gate_b.exit(Priority.NORMAL)
    finally:
        release.set()
        child.join(15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coordination.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'lm_repl.clients.coordination'`

- [ ] **Step 3: Write the implementation**

Create `lm_repl/clients/coordination.py`:

```python
"""Cross-process admission gate for RequestScheduler (two-flock gate+pool).

Design: docs/superpowers/specs/2026-06-10-cross-process-coordination-design.md.

Two lock files per server key in a shared coordination directory:

    <dir>/<key>.gate  - doorway. Normal requests hold SH momentarily on the
                        way in; a p1 holds EX for its whole run, which freezes
                        new admissions machine-wide (the cross-process
                        _waiting_p1 rule) and serializes p1s globally.
    <dir>/<key>.pool  - the in-flight set. Normal requests hold SH for the
                        request duration; a p1 takes EX, granted only when
                        every holder drains (the cross-process _active == 0
                        rule).

Crash cleanup is the kernel's: flock drops when an fd closes, including on
process death. The gate distinguishes only p1 vs everything else; p2-p5
ordering stays in-process. Same-host processes only (flock does not span
machines, and network filesystems are explicitly out of scope).
"""

import fcntl
import logging
import os
import threading
from pathlib import Path

from lm_repl.clients.scheduler import Priority

log = logging.getLogger(__name__)


class CrossProcessGate:
    """Two-flock readers-writer gate with writer preference.

    enter()/aenter() acquire for one request; exit() releases one acquisition
    (non-blocking fd closes, so both sync and async paths use it). Normal
    requests' pool fds are fungible: exit(NORMAL) closes any one of this
    process's SH holds, which the kernel treats identically.
    """

    def __init__(self, coordination_dir: str | Path, server_key: str):
        self._dir = Path(coordination_dir)
        self._gate_path = self._dir / f"{server_key}.gate"
        self._pool_path = self._dir / f"{server_key}.pool"
        self._mu = threading.Lock()
        self._pool_fds: list[int] = []  # one SH fd per in-flight normal request
        self._p1_fds: tuple[int, int] | None = None  # (gate_fd, pool_fd) of the active p1
        # Fail fast: surface an unwritable dir or a no-flock filesystem at
        # construction, not on request N.
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            for path in (self._gate_path, self._pool_path):
                fd = self._open(path)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass  # held EX by a live p1 elsewhere: flock works here
                finally:
                    os.close(fd)
        except OSError as e:
            raise RuntimeError(
                f"cross-process coordination unavailable at {self._dir}: {e}"
            ) from e

    @staticmethod
    def _open(path: Path) -> int:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)

    def enter(self, priority: int) -> None:
        """Blocking acquisition for one request. Releases partial holds and
        re-raises on failure, leaving no lock behind."""
        if priority == Priority.CONTENTION_RETRY:
            gate_fd = self._open(self._gate_path)
            try:
                fcntl.flock(gate_fd, fcntl.LOCK_EX)
                pool_fd = self._open(self._pool_path)
                try:
                    fcntl.flock(pool_fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(pool_fd)
                    raise
            except BaseException:
                os.close(gate_fd)
                raise
            with self._mu:
                self._p1_fds = (gate_fd, pool_fd)
        else:
            gate_fd = self._open(self._gate_path)
            try:
                fcntl.flock(gate_fd, fcntl.LOCK_SH)
                pool_fd = self._open(self._pool_path)
                try:
                    fcntl.flock(pool_fd, fcntl.LOCK_SH)
                except BaseException:
                    os.close(pool_fd)
                    raise
            finally:
                # The gate is only the doorway: release it whether or not the
                # pool acquisition succeeded.
                os.close(gate_fd)
            with self._mu:
                self._pool_fds.append(pool_fd)

    def exit(self, priority: int) -> None:
        """Release one acquisition. Never raises: it sits in finally paths,
        and the locks are released by the fd close regardless."""
        try:
            if priority == Priority.CONTENTION_RETRY:
                with self._mu:
                    fds, self._p1_fds = self._p1_fds, None
                if fds is not None:
                    gate_fd, pool_fd = fds
                    os.close(pool_fd)
                    os.close(gate_fd)
            else:
                with self._mu:
                    pool_fd = self._pool_fds.pop() if self._pool_fds else None
                if pool_fd is not None:
                    os.close(pool_fd)
        except OSError as e:
            log.warning("gate exit failed (locks still released on close): %s", e)
```


- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coordination.py -v`
Expected: all 8 tests PASS. If spawn children fail with import errors, the fix is running pytest from the repo root (children inherit sys.path).

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 385 passed, 8 skipped (377 + 8 new)

- [ ] **Step 6: Commit**

```bash
git add lm_repl/clients/coordination.py tests/test_coordination.py
git commit -m "feat: CrossProcessGate two-flock gate+pool for cross-process p1 exclusivity"
```

---

### Task 2: Async acquisition (poll loop) and cancellation safety

**Files:**
- Modify: `lm_repl/clients/coordination.py` (add `aenter` and `_apoll`)
- Modify: `tests/test_coordination.py` (append async tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coordination.py` (note the suite's idiom: sync test wrapping `asyncio.run(main())`):

```python
import asyncio

# ---- async path ----

def test_aenter_roundtrip(tmp_path):
    async def main():
        gate = CrossProcessGate(tmp_path, KEY)
        await gate.aenter(Priority.NORMAL)
        gate.exit(Priority.NORMAL)
        await gate.aenter(Priority.CONTENTION_RETRY)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())


def test_aenter_p1_waits_for_share(tmp_path):
    async def main():
        gate = CrossProcessGate(tmp_path, KEY)
        gate.enter(Priority.NORMAL)
        task = asyncio.create_task(gate.aenter(Priority.CONTENTION_RETRY))
        await asyncio.sleep(0.2)
        assert not task.done()  # polling: pool SH still held
        gate.exit(Priority.NORMAL)
        await asyncio.wait_for(task, 5)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())


def test_aenter_cancellation_releases_partial(tmp_path):
    async def main():
        # Hold gate EX via a raw fd (flock conflicts across fds even within
        # one process), so aenter(NORMAL) blocks polling at the doorway.
        raw_fd = os.open(tmp_path / f"{KEY}.gate", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(raw_fd, fcntl.LOCK_EX)
        gate = CrossProcessGate(tmp_path, KEY)
        task = asyncio.create_task(gate.aenter(Priority.NORMAL))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate._pool_fds == []
        assert gate._p1_fds is None
        fcntl.flock(raw_fd, fcntl.LOCK_UN)
        os.close(raw_fd)
        # Gate still fully usable after the cancellation
        await gate.aenter(Priority.CONTENTION_RETRY)
        gate.exit(Priority.CONTENTION_RETRY)

    asyncio.run(main())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coordination.py -k aenter -v`
Expected: 3 FAIL with `AttributeError: 'CrossProcessGate' object has no attribute 'aenter'`

- [ ] **Step 3: Implement aenter**

Add `import asyncio` to coordination.py's imports and this constant after `log = ...`:

```python
# Async acquisition polls LOCK_NB at this interval instead of blocking a
# thread: a cancelled task cannot interrupt a blocking flock in an executor
# thread, and that thread would eventually acquire a lock nobody releases.
POLL_INTERVAL = 0.025
```

Add to `CrossProcessGate` (after `enter`):

```python
    async def aenter(self, priority: int) -> None:
        """Async acquisition: LOCK_NB poll loop (POLL_INTERVAL) instead of a
        blocking flock in an executor thread, so task cancellation can never
        strand a lock in a thread nobody joins. On any failure, including
        CancelledError, partial holds are released before re-raising."""
        op = fcntl.LOCK_EX if priority == Priority.CONTENTION_RETRY else fcntl.LOCK_SH
        gate_fd = self._open(self._gate_path)
        try:
            await self._apoll(gate_fd, op)
            pool_fd = self._open(self._pool_path)
            try:
                await self._apoll(pool_fd, op)
            except BaseException:
                os.close(pool_fd)
                raise
        except BaseException:
            os.close(gate_fd)
            raise
        if priority == Priority.CONTENTION_RETRY:
            with self._mu:
                self._p1_fds = (gate_fd, pool_fd)
        else:
            os.close(gate_fd)
            with self._mu:
                self._pool_fds.append(pool_fd)

    @staticmethod
    async def _apoll(fd: int, op: int) -> None:
        while True:
            try:
                fcntl.flock(fd, op | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                await asyncio.sleep(POLL_INTERVAL)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coordination.py -v`
Expected: all 11 PASS

- [ ] **Step 5: Commit**

```bash
git add lm_repl/clients/coordination.py tests/test_coordination.py
git commit -m "feat: async gate acquisition via cancellation-safe LOCK_NB poll loop"
```

---

### Task 3: RequestScheduler gate integration

**Files:**
- Modify: `lm_repl/clients/scheduler.py`
- Modify: `tests/test_scheduler.py` (append stub-gate tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scheduler.py`:

```python
# =============================================================================
# Cross-process gate integration (stub gate; real flock tested in
# tests/test_coordination.py)
# =============================================================================


class _StubGate:
    """Duck-typed CrossProcessGate: records calls, optionally fails enter."""

    def __init__(self, fail_enter=False):
        self.calls = []
        self.fail_enter = fail_enter
        self.aenter_started = threading.Event()
        self.block_aenter = False

    def enter(self, priority):
        self.calls.append(("enter", priority))
        if self.fail_enter:
            raise RuntimeError("gate boom")

    async def aenter(self, priority):
        self.calls.append(("aenter", priority))
        self.aenter_started.set()
        if self.block_aenter:
            await asyncio.Event().wait()  # parks forever until cancelled
        if self.fail_enter:
            raise RuntimeError("gate boom")

    def exit(self, priority):
        self.calls.append(("exit", priority))


def test_gate_enter_after_admission_exit_before_release():
    g = _StubGate()
    s = RequestScheduler(max_concurrent=2, gate=g)
    s.acquire(Priority.NORMAL)
    assert s.active == 1
    s.release(Priority.NORMAL)
    assert s.active == 0
    assert g.calls == [("enter", Priority.NORMAL), ("exit", Priority.NORMAL)]


def test_gate_failure_rolls_back_local_slot():
    g = _StubGate(fail_enter=True)
    s = RequestScheduler(max_concurrent=2, gate=g)
    with pytest.raises(RuntimeError, match="gate boom"):
        s.acquire(Priority.NORMAL)
    assert s.active == 0
    # Scheduler stays usable once the gate recovers
    g.fail_enter = False
    s.acquire(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_gate_failure_rollback_unblocks_waiters():
    g = _StubGate(fail_enter=True)
    s = RequestScheduler(max_concurrent=1, gate=g)
    with pytest.raises(RuntimeError):
        s.acquire(Priority.NORMAL)
    # The failed acquire must not leave a phantom active slot
    g.fail_enter = False
    admitted = threading.Event()

    def second():
        s.acquire(Priority.NORMAL)
        admitted.set()

    threading.Thread(target=second, daemon=True).start()
    assert admitted.wait(2)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_async_gate_failure_rolls_back_local_slot():
    async def main():
        g = _StubGate(fail_enter=True)
        s = RequestScheduler(max_concurrent=2, gate=g)
        with pytest.raises(RuntimeError, match="gate boom"):
            await s.aacquire(Priority.NORMAL)
        assert s.active == 0

    asyncio.run(main())


def test_cancel_during_gate_wait_rolls_back_local_slot():
    async def main():
        g = _StubGate()
        g.block_aenter = True
        s = RequestScheduler(max_concurrent=2, gate=g)
        task = asyncio.create_task(s.aacquire(Priority.NORMAL))
        await asyncio.sleep(0.05)
        assert g.aenter_started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert s.active == 0

    asyncio.run(main())


def test_no_gate_means_no_gate_calls():
    s = RequestScheduler(max_concurrent=2)
    s.acquire(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py -k gate -v`
Expected: FAIL with `TypeError: RequestScheduler.__init__() got an unexpected keyword argument 'gate'` (test_no_gate_means_no_gate_calls passes already)

- [ ] **Step 3: Implement scheduler integration**

In `lm_repl/clients/scheduler.py`:

3a. Add the TYPE_CHECKING import after the existing imports (coordination.py imports `Priority` from this module, so the runtime import must stay one-directional):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lm_repl.clients.coordination import CrossProcessGate
```

3b. Extend `__init__` (signature and body):

```python
    def __init__(
        self,
        max_concurrent: int = 8,
        aging_interval: float | None = 30.0,
        gate: "CrossProcessGate | None" = None,
    ):
        """
        Args:
            max_concurrent: Most requests allowed in flight at once. Match the
                server's slot count (llama-server --parallel).
            aging_interval: Seconds of queue wait worth one priority level for
                p2-p5 waiters (an old LOW eventually outranks a fresh HIGH).
                None disables aging (strict priority, FIFO within a level).
            gate: Optional CrossProcessGate extending p1 exclusivity across OS
                processes that share one server. Entered after local
                admission, exited before local release. None keeps behavior
                purely in-process.
        """
        self._max_concurrent = max_concurrent
        self._aging_interval = aging_interval
        self._gate = gate
        self._lock = threading.Lock()
        self._active = 0
        self._active_p1 = 0
        self._waiting_p1 = 0
        self._seq = 0
        self._sync_waiters: list[_Waiter] = []
        self._async_waiters: list[_AsyncWaiter] = []
```

3c. Rework `acquire` and add `_release_local`; `release` delegates. The rollback MUST use `_release_local`, not `release`: `release` calls `gate.exit`, which would pop another request's pool fd for a gate we never entered.

```python
    # -- sync interface --

    def acquire(self, priority: int = Priority.NORMAL) -> None:
        with self._lock:
            if self._can_dispatch(priority):
                self._admit(priority)
                waiter = None
            else:
                self._seq += 1
                band, vkey = self._sort_fields(priority)
                waiter = _Waiter(priority, band, vkey, self._seq)
                if priority == Priority.CONTENTION_RETRY:
                    self._waiting_p1 += 1
                heapq.heappush(self._sync_waiters, waiter)

        if waiter is not None:
            waiter.event.wait()

        if self._gate is not None:
            try:
                self._gate.enter(priority)
            except BaseException:
                # Cross-process acquisition failed after local admission: give
                # the local slot back so counters never skew. Local-only
                # release: we never entered the gate.
                self._release_local(priority)
                raise

    def _release_local(self, priority: int) -> None:
        with self._lock:
            self._active -= 1
            if priority == Priority.CONTENTION_RETRY:
                self._active_p1 -= 1
            self._dispatch_next()

    def release(self, priority: int = Priority.NORMAL) -> None:
        if self._gate is not None:
            self._gate.exit(priority)
        self._release_local(priority)
```

3d. Rework `aacquire`/`arelease` the same way (the existing CancelledError handling around `waiter.event.wait()` is preserved verbatim; the gate step is appended after admission):

```python
    # -- async interface --

    async def aacquire(self, priority: int = Priority.NORMAL) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._can_dispatch(priority):
                self._admit(priority)
                waiter = None
            else:
                self._seq += 1
                band, vkey = self._sort_fields(priority)
                waiter = _AsyncWaiter(priority, band, vkey, self._seq, loop)
                if priority == Priority.CONTENTION_RETRY:
                    self._waiting_p1 += 1
                heapq.heappush(self._async_waiters, waiter)

        if waiter is not None:
            try:
                await waiter.event.wait()
            except asyncio.CancelledError:
                with self._lock:
                    if waiter.dispatched:
                        # Dispatch already admitted us; give the slot back since
                        # this task will never run a request or release.
                        self._active -= 1
                        if priority == Priority.CONTENTION_RETRY:
                            self._active_p1 -= 1
                    else:
                        # Still queued: mark for lazy removal by _dispatch_next.
                        waiter.cancelled = True
                        if priority == Priority.CONTENTION_RETRY:
                            self._waiting_p1 -= 1
                    # Either branch can unblock other waiters (a freed slot, or a
                    # vanished waiting-p1 that was gating admissions).
                    self._dispatch_next()
                raise

        if self._gate is not None:
            try:
                await self._gate.aenter(priority)
            except BaseException:
                # Includes CancelledError while polling the gate: aenter has
                # already closed its partial fds; give the local slot back.
                self._release_local(priority)
                raise

    async def arelease(self, priority: int = Priority.NORMAL) -> None:
        if self._gate is not None:
            # exit() only closes fds; it never blocks, so no executor needed.
            self._gate.exit(priority)
        self._release_local(priority)
```

- [ ] **Step 4: Run the scheduler suite**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py -v`
Expected: all PASS (34 existing + 6 new). The existing tests exercise the restructured acquire/aacquire with gate=None; any failure there means the restructure changed semantics - fix before proceeding.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 394 passed, 8 skipped

- [ ] **Step 6: Commit**

```bash
git add lm_repl/clients/scheduler.py tests/test_scheduler.py
git commit -m "feat: RequestScheduler enters/exits cross-process gate around local slots"
```

---

### Task 4: Plumbing through OpenAIClient, LMHandler, RLM

**Files:**
- Modify: `lm_repl/clients/openai.py` (store base_url; one line plus context)
- Modify: `lm_repl/core/lm_handler.py:142-176`
- Modify: `lm_repl/core/rlm.py:73-74` area (param, docstring, store, pass-through at line ~226)
- Modify: `tests/test_scheduler.py` (append plumbing tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scheduler.py`. Check the file's existing LMHandler-construction tests first and reuse their client-stub pattern if one exists; otherwise:

```python
# =============================================================================
# Coordination plumbing (LMHandler / RLM)
# =============================================================================

import hashlib

from lm_repl.core.rlm import RLM


def _stub_client(base_url="http://127.0.0.1:8080/v1"):
    c = MagicMock()
    c.model_name = "m"
    c.base_url = base_url
    return c


def test_lmhandler_builds_gate_keyed_by_base_url(tmp_path):
    h = LMHandler(
        _stub_client(),
        scheduler_max_concurrent=4,
        scheduler_coordination_dir=tmp_path,
    )
    assert h.scheduler is not None
    assert h.scheduler._gate is not None
    key = hashlib.sha256(b"http://127.0.0.1:8080/v1").hexdigest()[:16]
    assert (tmp_path / f"{key}.gate").exists()
    assert (tmp_path / f"{key}.pool").exists()


def test_lmhandler_no_dir_no_gate():
    h = LMHandler(_stub_client(), scheduler_max_concurrent=4)
    assert h.scheduler is not None
    assert h.scheduler._gate is None


def test_coordination_dir_requires_scheduler(tmp_path):
    with pytest.raises(ValueError, match="scheduler_max_concurrent"):
        LMHandler(_stub_client(), scheduler_coordination_dir=tmp_path)


def test_openai_client_stores_base_url():
    client = OpenAIClient(
        api_key="x", model_name="m", base_url="http://127.0.0.1:8080/v1"
    )
    assert client.base_url == "http://127.0.0.1:8080/v1"


def test_rlm_stores_coordination_dir(tmp_path):
    rlm = RLM(
        backend_kwargs={"model_name": "m", "api_key": "x"},
        scheduler_max_concurrent=4,
        scheduler_coordination_dir=tmp_path,
    )
    assert rlm.scheduler_coordination_dir == tmp_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py -k "coordination or base_url or builds_gate or no_gate" -v`
Expected: FAIL with `TypeError` (unexpected keyword `scheduler_coordination_dir`) and `AttributeError` for `client.base_url`

- [ ] **Step 3: Implement**

3a. `lm_repl/clients/openai.py`, in `OpenAIClient.__init__` right after `self.scheduler = scheduler` (line 112):

```python
        self.scheduler = scheduler
        # Kept for consumers that need the target server's identity (e.g.
        # LMHandler keys the cross-process gate by base_url).
        self.base_url = base_url
```

3b. `lm_repl/core/lm_handler.py`. Add imports at the top of the file:

```python
import hashlib
from pathlib import Path

from lm_repl.clients.coordination import CrossProcessGate
```

Extend `__init__` signature (after `scheduler_aging_interval`):

```python
        scheduler_max_concurrent: int | None = None,
        scheduler_aging_interval: float | None = 30.0,
        scheduler_coordination_dir: str | Path | None = None,
```

Replace the scheduler-construction block (lines 161-175) with:

```python
        # One scheduler shared by every client that targets the same server, so the
        # priority queue (and p1 exclusivity) spans all traffic. Match
        # scheduler_max_concurrent to the server's slot count (llama-server --parallel).
        # None disables scheduling entirely (previous behavior).
        # scheduler_aging_interval: seconds of queue wait worth one priority level
        # (anti-starvation); None disables aging.
        # scheduler_coordination_dir: opt-in cross-process gate extending p1
        # exclusivity to other OS processes targeting the same base_url
        # (lock files keyed by sha256(base_url)). Requires the scheduler.
        if scheduler_coordination_dir is not None and scheduler_max_concurrent is None:
            raise ValueError(
                "scheduler_coordination_dir requires scheduler_max_concurrent "
                "(no scheduler, no cross-process gate)"
            )
        self.scheduler: RequestScheduler | None = None
        if scheduler_max_concurrent is not None:
            gate = None
            if scheduler_coordination_dir is not None:
                key_src = str(getattr(client, "base_url", None) or "default")
                server_key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
                gate = CrossProcessGate(scheduler_coordination_dir, server_key)
            self.scheduler = RequestScheduler(
                max_concurrent=scheduler_max_concurrent,
                aging_interval=scheduler_aging_interval,
                gate=gate,
            )
            for c in (client, other_backend_client):
                if c is not None and hasattr(c, "scheduler"):
                    c.scheduler = self.scheduler
```

Note: `other_backend_client` shares the gate, consistent with already sharing the scheduler (documented spec decision).

3c. `lm_repl/core/rlm.py`. Add to `__init__` signature after `scheduler_aging_interval` (line 74):

```python
        scheduler_coordination_dir: str | Path | None = None,
```

Add `from pathlib import Path` to imports if not present. Add to the docstring after the `scheduler_aging_interval` entry:

```
            scheduler_coordination_dir: If set (with scheduler_max_concurrent), directory of
                cross-process lock files extending contention-retry (p1) exclusivity across
                OS processes that target the same server. Opt-in; same host only. None
                (default) keeps coordination in-process.
```

Store it next to the other scheduler params (line ~151):

```python
        self.scheduler_max_concurrent = scheduler_max_concurrent
        self.scheduler_aging_interval = scheduler_aging_interval
        self.scheduler_coordination_dir = scheduler_coordination_dir
```

Pass it in `_spawn_completion_context` (line ~226):

```python
        lm_handler = LMHandler(
            client,
            other_backend_client=other_backend_client,
            scheduler_max_concurrent=self.scheduler_max_concurrent,
            scheduler_aging_interval=self.scheduler_aging_interval,
            scheduler_coordination_dir=self.scheduler_coordination_dir,
        )
```

SRLM forwards `**kwargs` to RLM (lm_repl/core/srlm.py), so it needs no change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scheduler.py tests/test_coordination.py -q`
Expected: all PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 399 passed, 8 skipped

- [ ] **Step 6: Commit**

```bash
git add lm_repl/clients/openai.py lm_repl/core/lm_handler.py lm_repl/core/rlm.py tests/test_scheduler.py
git commit -m "feat: plumb scheduler_coordination_dir through RLM and LMHandler"
```

---

### Task 5: Live harness multi-process mode

**Files:**
- Modify: `scripts/test_scheduler_live.py`

No unit tests; this is the live harness itself. The mode must work WITHOUT the server too (it fails fast if the server is down, which is fine).

- [ ] **Step 1: Add the multiproc mode**

Add `import hashlib`, `import multiprocessing`, `import os` to the imports. Add after `toolarge_test()`:

```python
def _multiproc_worker(idx, n_per_proc, toks, mc, coord_dir):
    """One OS process: own gate-equipped scheduler, unique prompts."""
    from lm_repl.clients.coordination import CrossProcessGate

    key = hashlib.sha256(BASE_URL.encode()).hexdigest()[:16]
    gate = CrossProcessGate(coord_dir, key)
    scheduler = RequestScheduler(max_concurrent=mc, gate=gate)
    client = make_client(scheduler)

    async def run():
        async def do_one(i):
            # Per-process uniq offset: prompts must be unique across ALL
            # processes or prefix-cache reuse masks KV pressure.
            uniq = idx * 10000 + i
            prompt = make_prompt(toks, uniq=uniq)
            t0 = time.time()
            try:
                result = await client.acompletion(prompt)
                print(f"  [p{idx}:{i}] OK ({time.time() - t0:.1f}s): {(result or '')[:40]!r}")
                return "ok"
            except Exception as e:
                print(f"  [p{idx}:{i}] FAIL ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:100]}")
                return "fail"

        results = await asyncio.gather(*[do_one(i) for i in range(n_per_proc)])
        assert scheduler.active == 0, "Slot leak!"
        return results.count("fail")

    sys.exit(min(asyncio.run(run()), 250))


def multiproc_test(n_procs=4, n_per_proc=4, toks=12000, mc=2,
                   coord_dir="/tmp/lm-repl-coord-live"):
    """N OS processes share one llama-server through the cross-process gate.

    Collectively oversubscribes the unified KV pool so the server mass-kills
    in-flight requests; every process's p1 retries must drain GLOBALLY (the
    other processes' traffic included) and succeed. Success: zero failures.
    """
    print(f"\n=== MULTIPROC TEST: {n_procs} procs x {n_per_proc} reqs, "
          f"~{toks} tok, mc={mc}/proc, dir={coord_dir} ===")
    os.makedirs(coord_dir, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_multiproc_worker, args=(i, n_per_proc, toks, mc, coord_dir))
        for i in range(n_procs)
    ]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total_fails = sum(p.exitcode or 0 for p in procs)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    print(f"Total failures across processes: {total_fails}")
    print("PASSED" if total_fails == 0 else "FAILED")
```

Add to the `__main__` dispatch (before the `else`):

```python
    elif mode == "multiproc":
        np_ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        npp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        toks = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
        mc = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        multiproc_test(np_, npp, toks, mc)
```

Update the usage line to `[smoke|contention|toolarge|multiproc|all]`.

- [ ] **Step 2: Syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/test_scheduler_live.py').read())"`
Expected: silent success

- [ ] **Step 3: Live verification (server required)**

Server prep (see CLAUDE.md llama-server section; restart between contention runs - prompt cache):

```bash
~/src/local-ai/scripts/llama-server.sh stop || true
# confirm 0 listeners on 8080 and VRAM near idle before relaunch
~/src/local-ai/scripts/llama-server.sh start
curl -X POST http://127.0.0.1:8080/models/load -d '{"model":"gemma-4-12b-it-cpt-sft-kb-v2-slerp"}'
```

Then:

```bash
cd /home/potto/src/lm-repl
rm -rf /tmp/lm-repl-coord-live
.venv/bin/python scripts/test_scheduler_live.py multiproc 4 4 12000 2
```

Expected: 16/16 OK, "retrying at p1" lines in multiple `[pN:...]` streams, "Total failures across processes: 0", PASSED. The pre-fix behavior (to compare if desired: comment out the gate in `_multiproc_worker`) shows cross-process collisions surfacing as FAIL lines.

- [ ] **Step 4: Commit**

```bash
git add scripts/test_scheduler_live.py
git commit -m "test: multi-process live mode exercising the cross-process gate"
```

---

### Task 6: rlm-trainer benchmark flag and full benchmark re-run

**Files (separate repo `~/src/rlm-trainer`, separate git conventions - GH issue backlog, `--trailer "Github-Issue:#<n>"` when issue-linked):**
- Modify: `~/src/rlm-trainer/benchmark.py` (has UNCOMMITTED `--scheduler-max-concurrent` changes from the prior session; this task commits both together)

- [ ] **Step 1: Add the flag**

Mirror the existing `scheduler_max_concurrent` plumbing exactly (benchmark.py lines 81, 225, 269, 347, 349):

- `run_benchmark(...)` signature: add `scheduler_coordination_dir: str | None = None` after `scheduler_max_concurrent`.
- params dict (line ~269): add `"scheduler_coordination_dir": scheduler_coordination_dir,`.
- SRLM construction (line ~81): add `scheduler_coordination_dir=params.get("scheduler_coordination_dir"),`.
- argparse (after line 347):

```python
    parser.add_argument("--scheduler-coordination-dir", default=None, help="Shared lock dir extending p1 contention-retry exclusivity across worker processes (requires --scheduler-max-concurrent)")
```

- main call (line ~349): add `scheduler_coordination_dir=args.scheduler_coordination_dir`.

- [ ] **Step 2: Commit in rlm-trainer**

```bash
cd ~/src/rlm-trainer
git add benchmark.py
git commit -m "feat: scheduler flags for unified-KV contention (per-process cap + cross-process coordination dir)"
```

(Check `gh issue list` for a matching open issue; if one exists, add `--trailer "Github-Issue:#<n>"`.)

- [ ] **Step 3: Benchmark re-run (server + user awareness required; ~48+ min)**

Pre-flight, all from CLAUDE.md ops rules:

```bash
ps -eo cmd | grep '[.]venv/bin/python'        # MUST be empty (orphan check)
# restart llama-server clean (explicit PID kill, never pkill -f), reload model
cd ~/src/rlm-trainer
mv benchmarks/gemma-4-12b-it-cpt-sft-kb-v2-slerp.json benchmarks/gemma-4-12b-it-cpt-sft-kb-v2-slerp.c8-pregate.json   # resume-by-slug skips done tasks otherwise
scripts/gpu_telemetry.sh start logs/bench-v2-c8-gate-gpu.csv
.venv/bin/python benchmark.py --model gemma-4-12b-it-cpt-sft-kb-v2-slerp \
  --url http://127.0.0.1:8080/v1 --concurrency 8 --tasks-dir tasks_subset \
  --scheduler-max-concurrent 4 \
  --scheduler-coordination-dir /tmp/lm-repl-bench-coord \
  2>&1 | tee logs/bench-v2-c8-gate.log
scripts/gpu_telemetry.sh stop
```

Success criteria (from the spec):
- Surfaced ctx-500 task failures: ~0 (was 14). Count: `grep -c "Context size has been exceeded" logs/bench-v2-c8-gate.log` distinguishing absorbed retries from task-level failures in the results json.
- Timeouts not materially worse than 11; wall-clock in the same ballpark as 48 min (a regression here means p1 drains are freezing admissions too often - report, do not tune blindly).
- After the run: `curl -s http://127.0.0.1:8080/metrics | grep requests_processing` shows 0 (no runaway slots).

- [ ] **Step 4: Record results**

Compare against `benchmarks/gemma-4-12b-it-cpt-sft-kb-v2-slerp.c1-baseline.json` (correctness) and the c8 pre-gate run (failures/timeouts/wall-clock). Summarize in the final report to the user.

---

### Task 7: Final verification and push (lm-repl)

- [ ] **Step 1: Full suite**

Run: `cd /home/potto/src/lm-repl && .venv/bin/python -m pytest tests/ -q`
Expected: 399 passed, 8 skipped

- [ ] **Step 2: Push**

```bash
git log --oneline origin/main..HEAD   # review: spec + 4-5 feature/test commits
git push
```

- [ ] **Step 3: Close out backlog**

Check `gh issue list -R ClearBridgeRIP/rlm-trainer` for issues about benchmark contention failures; close with a comment referencing the commits if one matches.
