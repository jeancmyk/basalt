"""serve/tests/test_winjob.py - job object kills the child when the parent's handle closes."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.engine.winjob import contain
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
ok = contain(child)
print(ok, child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""


def running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestWinJob(unittest.TestCase):
    def test_children_end_with_a_killed_server(self):
        python = getattr(sys, "_base_executable", None) or sys.executable
        parent = subprocess.Popen([python, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        child = grandchild = None
        try:
            ok, child, grandchild = parent.stdout.readline().split()
            child, grandchild = int(child), int(grandchild)
            self.assertEqual(ok, "True")
            self.assertTrue(running(child) and running(grandchild))
            parent.kill()
            parent.wait(10)
            deadline = time.monotonic() + 10
            while (running(child) or running(grandchild)) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(running(child), "the child outlived the server")
            self.assertFalse(running(grandchild), "the grandchild outlived the server")
        finally:
            if parent.poll() is None:
                parent.kill()
            for pid in (child, grandchild):
                if pid and running(pid):
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)

    def test_elsewhere_or_without_a_process_it_is_a_no_op(self):
        from serve.engine.winjob import contain
        self.assertFalse(contain(None))


# A job created without JOB_OBJECT_LIMIT_BREAKAWAY_OK does not ignore CREATE_BREAKAWAY_FROM_JOB: it fails
# CreateProcess with ERROR_ACCESS_DENIED, which reads as the engine binary being unreadable. A supervisor
# that runs the server inside its own job (llama-swap and similar) is exactly that case.
JOIN_RESTRICTIVE_JOB = r"""
import ctypes, ctypes.wintypes as w, sys

class Limits(ctypes.Structure):
    _fields_ = [("user_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("LimitFlags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("active", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("sched", w.DWORD)]

class Extended(ctypes.Structure):
    _fields_ = [("base", Limits), ("io", ctypes.c_byte * 48), ("proc_mem", ctypes.c_size_t),
                ("job_mem", ctypes.c_size_t), ("peak_proc", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateJobObjectW.restype = w.HANDLE
k32.GetCurrentProcess.restype = w.HANDLE
k32.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
k32.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]

job = k32.CreateJobObjectW(None, None)
if not job:
    print("SKIP create", flush=True); raise SystemExit(0)
info = Extended()
info.base.LimitFlags = 0                     # no BREAKAWAY_OK, no SILENT_BREAKAWAY_OK
if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
    print("SKIP limits", flush=True); raise SystemExit(0)
if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
    # already inside someone else's job: this case cannot be built from here
    print("SKIP assign %d" % ctypes.get_last_error(), flush=True); raise SystemExit(0)
print("JOINED", flush=True)
"""

CHILD = "import time; time.sleep(30)"


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestBreakaway(unittest.TestCase):
    def _in_a_restrictive_job(self):
        python = getattr(sys, "_base_executable", None) or sys.executable
        return subprocess.run([python, "-c", JOIN_RESTRICTIVE_JOB + "\nprint('ok', flush=True)"],
                              capture_output=True, text=True)

    def test_popen_survives_a_job_that_forbids_breakaway(self):
        from serve.engine import winjob
        probe = self._in_a_restrictive_job()
        if "SKIP" in probe.stdout:
            self.skipTest(f"cannot join a job here: {probe.stdout.strip()}")
        self.assertIn("JOINED", probe.stdout)

        # the child that reproduces the supervisor's situation and asks winjob to spawn the engine
        script = JOIN_RESTRICTIVE_JOB + f"""
sys.path.insert(0, {str(ROOT)!r})
from serve.engine import winjob
try:
    p = winjob.popen([{sys.executable!r}, "-c", {CHILD!r}])
    print("SPAWNED", p.pid, flush=True)
    import time; time.sleep(5)
    p.kill(); p.wait(10)
except OSError as e:
    print("DENIED", e.winerror, flush=True)
"""
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        self.assertIn("JOINED", out.stdout, out.stderr)
        self.assertNotIn("DENIED", out.stdout, f"the spawn was refused: {out.stdout}")
        self.assertIn("SPAWNED", out.stdout, f"winjob.popen did not start the child: {out.stdout} {out.stderr}")

    def test_the_bare_flag_really_is_refused_so_the_fallback_is_not_cargo_cult(self):
        """Guards the premise: without the fallback the spawn fails, so removing it would break it again."""
        script = JOIN_RESTRICTIVE_JOB + f"""
import subprocess
try:
    p = subprocess.Popen([{sys.executable!r}, "-c", {CHILD!r}], creationflags=0x01000000)
    print("ACCEPTED", flush=True); p.kill(); p.wait(10)
except OSError as e:
    print("DENIED", e.winerror, flush=True)
"""
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        if "SKIP" in out.stdout:
            self.skipTest("cannot join a job here")
        self.assertIn("DENIED 5", out.stdout,
                      f"expected ERROR_ACCESS_DENIED from the bare flag, got: {out.stdout} {out.stderr}")


@unittest.skipUnless(os.name == "nt", "power throttling is a Windows setting")
class TestNoThrottle(unittest.TestCase):
    def test_a_contained_child_is_opted_out_of_power_throttling(self):
        from serve.engine import winjob
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
        finally:
            child.kill()
            child.wait(10)


if __name__ == "__main__":
    unittest.main()
